"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import http.client
import json
import io
import os
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, EngineStuck, MockEngine, Service, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Lifecycle(unittest.TestCase):
    def test_close_cleans_up_after_a_broken_stdin_pipe(self):
        for running in (False, True):
            with self.subTest(running=running):
                engine = StrataEngine("missing-executable", [], lazy=True)
                read_fd, write_fd = os.pipe()
                stdin = os.fdopen(write_fd, "w")
                self.addCleanup(stdin.close)
                os.close(read_fd)                       # the engine's end of the pipe is already gone
                stdin.write("pending")                  # buffered: flush() and close() will hit the broken pipe
                proc = mock.Mock(stdin=stdin, stdout=io.StringIO())
                self.addCleanup(proc.stdout.close)
                proc.poll.side_effect = [None if running else 0, 0]
                engine.proc, engine.pump, engine.log = proc, mock.Mock(), io.StringIO()
                self.addCleanup(engine.log.close)
                engine.ended, engine.progress, engine.last = False, (1, 2), {"generated": 1}

                engine.close()

                self.assertTrue(stdin.closed)
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(engine.log.closed)
                engine.pump.join.assert_called_once_with(timeout=2)
                self.assertIsNone(engine.proc)
                self.assertTrue(engine.ended)
                self.assertIsNone(engine.progress)
                self.assertEqual(engine.last, {})
                if running:
                    proc.terminate.assert_called_once()
                    proc.wait.assert_called_once_with(timeout=5)
                else:
                    proc.terminate.assert_not_called()
                    proc.wait.assert_not_called()
                proc.kill.assert_not_called()
                engine.close()                          # repeated cleanup is harmless

    def test_close_sends_eof_before_waiting_for_windows_reader(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        engine.close()
        self.assertIsNone(engine.proc)
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_close_terminates_before_it_kills(self):
        # an engine that does not end on QUIT is terminated first (as the unload always did); kill is the last resort
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 20), 0]
        engine.proc = proc
        engine.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--max-context", "16384"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        engine.close()

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_model_detail_endpoint_answers_for_every_id_it_serves(self):
        # GET /v1/models/{id}: the same entry /v1/models lists, so a client can ask whether one id is really served
        # here instead of reading the whole list (Visual Studio's model picker)
        self.svc.set_aliases(["strata-alias"])
        listed = {m["id"]: m for m in self.request("/v1/models")[1]["data"]}
        self.assertEqual(self.engine.starts, 0)          # asking for a model never loads the engine
        for model_id in (self.svc.model, "strata-alias"):
            code, detail = self.request(f"/v1/models/{model_id}")
            self.assertEqual(code, 200, model_id)
            for key, value in listed[model_id].items():    # the same entry the list has under that id
                self.assertEqual(detail[key], value, model_id)
            self.assertEqual(detail["id"], model_id)
            self.assertEqual(detail["object"], "model")
            self.assertEqual(detail["status"]["value"], "unloaded")   # listed, loads on use
            self.assertEqual(detail["meta"]["n_ctx"], 4096)
            self.assertEqual(detail["owned_by"], "strata")  # OpenAI's two fields, for a strict SDK client
            self.assertIsInstance(detail["created"], int)
        self.assertEqual(self.request("/v1/models/another-model")[0], 404)
        self.assertEqual(self.request("/v1/models/")[0], 200)   # the trailing slash is the list, as everywhere else
        self.assertEqual(self.engine.starts, 0)

    def test_bare_models_path_still_only_lists(self):
        # llama.cpp's /models has never taken an id, and nothing asks for one there: /models/{id} stays the plain
        # 404 it always was, so the only detail endpoint is the versioned one
        self.assertEqual(self.request(f"/models/{self.svc.model}")[0], 404)
        self.assertEqual(self.request("/models")[1]["data"][0]["id"], self.svc.model)

    def test_model_detail_endpoint_is_empty_while_the_engine_is_gone(self):
        # an engine that died (and nothing loads it on use): it is not offered, so its id is not found either
        self.engine.unloaded = False
        self.assertEqual(self.request("/v1/models")[1]["data"], [])
        self.assertEqual(self.request(f"/v1/models/{self.svc.model}")[0], 404)
        self.engine.loaded = True
        code, detail = self.request(f"/v1/models/{self.svc.model}")
        self.assertEqual(code, 200)
        self.assertEqual(detail["status"]["value"], "loaded")

    def test_model_detail_endpoint_needs_the_api_key_like_the_list(self):
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request(f"/v1/models/{self.svc.model}")[0], 401)
        self.assertEqual(self.request(f"/v1/models/{self.svc.model}",
                                      headers={"Authorization": "Bearer local-secret"})[0], 200)

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)

    def sse_connection(self):
        """An open GET /models/sse (llama.cpp's model status feed, as a front-end that watches it uses)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        self.addCleanup(conn.close)
        conn.request("GET", "/models/sse")
        return conn, conn.getresponse()

    def read_sse(self, response):
        """The next SSE record on an open /models/sse: one line plus the blank line that ends it."""
        line = response.readline()
        self.assertEqual(response.readline(), b"\n")
        return line

    def test_models_sse_streams_load_and_unload_events(self):
        conn, response = self.sse_connection()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        self.request("/v1/load", {"model": self.svc.model})
        events = [json.loads(self.read_sse(response).removeprefix(b"data: ")) for _ in range(2)]
        self.assertEqual([e["event"] for e in events], ["status_change", "status_change"])
        self.assertEqual([e["data"]["status"] for e in events], ["loading", "loaded"])
        self.assertEqual(events[1]["data"]["info"]["id"], self.svc.model)
        self.request("/v1/unload", {})
        event = json.loads(self.read_sse(response).removeprefix(b"data: "))
        self.assertEqual(event["event"], "status_change")
        self.assertEqual(event["data"], {"status": "unloaded", "exit_code": 0})

    def test_models_sse_reports_a_failed_load_as_unloaded(self):
        conn, response = self.sse_connection()

        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        self.engine.exit_code = lambda: 3
        self.assertEqual(self.request("/v1/load", {"model": self.svc.model})[0], 503)
        events = [json.loads(self.read_sse(response).removeprefix(b"data: ")) for _ in range(2)]
        self.assertEqual([e["data"]["status"] for e in events], ["loading", "unloaded"])
        self.assertEqual(events[1]["data"]["exit_code"], 3)

    def test_models_sse_heartbeats_while_quiet(self):
        with mock.patch("serve.server.KEEPALIVE_S", 0.2):    # the usual 10 s cadence, sped up for the test
            conn, response = self.sse_connection()
            self.assertEqual(response.status, 200)
            self.assertEqual(self.read_sse(response), b": keep-alive\n")

    def test_models_sse_requires_the_api_key(self):
        self.svc.api_key = "local-secret"
        conn, response = self.sse_connection()
        self.assertEqual(response.status, 401)
        conn.request("GET", "/models/sse", headers={"Authorization": "Bearer local-secret"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")


if __name__ == "__main__":
    unittest.main()
