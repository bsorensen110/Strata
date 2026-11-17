"""Compiler/GPU-free checks of the opt-in decode quality arms."""
import contextlib
import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quant_quality as qq


class ProbeQualityTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="tmp-fp8-quality-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = pathlib.Path(temp.name)
        self.config = self.root / "engine.json"
        self.config.write_text(json.dumps({"exe": "not-launched", "args": ["--spec", "8", "--kv", "q4_0"],
                                           "env": {"STRATA_HIP_FP8_ATTN": "1"}}))
        self.ids = self.root / "ids.json"
        self.ids.write_text(json.dumps({"seg_tokens": 3, "vocab_size": 3, "segments": [
            {"name": "english", "ids": [0, 1, 2]}, {"name": "code", "ids": [2, 1, 0]}]}))

    def run_arm(self, arm, evidence="selected: packed conversion"):
        out = self.root / f"{arm}.tsv"
        args = SimpleNamespace(config=str(self.config), ids=str(self.ids), out=str(out), arm=arm,
                               env=["STRATA_HIP_FP8_ATTN=0", "UNCHANGED=keep"], min_free_gb=0,
                               gpu_experts=False, wrap=None)
        calls = []

        def fake_engine(command, **kwargs):
            env = kwargs["env"]
            calls.append((command, env))
            if env["STRATA_HIP_FP8_ATTN"] == "1" and evidence:
                kwargs["stderr"].write(f"strata fp8-attn probe: device 0 gfx1201: {evidence}\n")

            class Engine:
                stdout = iter(["READY\n", "DONE\n", "DONE\n"])
                returncode = 0

                def __init__(self):
                    self.stdin = self
                    self.index = 0

                def write(self, text):
                    if text.startswith("GEN"):
                        with open(env["STRATA_LOGPOS"], "a") as log:
                            for pos in range(2):
                                fields = [str(pos), "1", "-1", "1", "-1", "1", "-10", "-1"]
                                fields += [f"{i}:-{i + 1}.0" for i in range(qq.TOPK)]
                                log.write("\t".join(fields) + "\n")
                        values = (2.0, 1.0, -1.0) if arm == "fp8-attn" and self.index else (1.0, 2.0, -1.0)
                        pathlib.Path(f"{env['STRATA_DUMP_FIRST_LOGITS']}.{self.index}").write_bytes(struct.pack("<3f", *values))
                        self.index += 1

                def flush(self):
                    pass

                def wait(self, timeout=None):
                    return self.returncode

                def kill(self):
                    self.returncode = -9

            return Engine()

        with mock.patch.object(qq.subprocess, "Popen", side_effect=fake_engine), \
                mock.patch.object(qq, "mem_available_gb", return_value=1000), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qq.cmd_run(args), 0)
        return out, calls[0]

    def test_arms_force_single_query_and_report_full_vocabulary_kl(self):
        ref, (command, env) = self.run_arm("current")
        self.assertEqual(command.count("--spec"), 1)
        self.assertEqual(command[command.index("--spec") + 1], "1")
        self.assertEqual(command[command.index("--kv") + 1], "fp16")
        self.assertEqual(env["STRATA_HIP_FP8_ATTN"], "0")
        self.assertEqual(env["UNCHANGED"], "keep")
        run, (_, env) = self.run_arm("fp8-attn")
        self.assertEqual(env["STRATA_HIP_FP8_ATTN"], "1")
        result = qq.compare_first_logits(ref, run)
        self.assertEqual(result["route"], "selected")
        self.assertEqual(result["texts"]["english"]["kl_at_0"], 0)
        self.assertTrue(result["texts"]["english"]["top1"])
        self.assertGreater(result["texts"]["code"]["kl_at_0"], 0)
        self.assertFalse(result["texts"]["code"]["top1"])
        self.assertEqual(result["top1"], 0.5)
        report = self.root / "quality.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(qq.cmd_compare(SimpleNamespace(ref=str(ref), runs=[str(run)], segments=True,
                                                             first=True, json=str(report))), 0)
        self.assertIn("full-vocabulary KL@0, route=selected", output.getvalue())
        self.assertEqual(json.loads(report.read_text())[str(run)]["first_token"]["top1"], 0.5)

    def test_refusal_is_labelled_and_no_route_evidence_is_an_error(self):
        ref, _ = self.run_arm("current")
        run, _ = self.run_arm("fp8-attn", "declined: this device image lacks the conversion builtins")
        self.assertIn("null probe", qq.compare_first_logits(ref, run)["route"])
        with self.assertRaisesRegex(SystemExit, "no decode route evidence"):
            self.run_arm("fp8-attn", evidence="")
        self.assertFalse(pathlib.Path(f"{run}.seg.json").exists(), "failed rerun retained stale success metadata")

    def test_missing_nonfinite_or_mismatched_logits_cannot_pass(self):
        ref, _ = self.run_arm("current")
        run, _ = self.run_arm("fp8-attn")
        first = pathlib.Path(f"{run}.first.0")
        first.write_bytes(struct.pack("<3f", 0, float("nan"), 1))
        with self.assertRaisesRegex(SystemExit, "nonfinite"):
            qq.compare_first_logits(ref, run)
        first.unlink()
        with self.assertRaisesRegex(SystemExit, "missing first-token"):
            qq.compare_first_logits(ref, run)
        first.write_bytes(struct.pack("<2f", 0, 1))
        with self.assertRaisesRegex(SystemExit, "vocabulary sizes"):
            qq.compare_first_logits(ref, run)
        metadata = pathlib.Path(f"{run}.seg.json")
        data = json.loads(metadata.read_text())
        data["input_ids_sha256"] = "another text"
        metadata.write_text(json.dumps(data))
        with self.assertRaisesRegex(SystemExit, "input IDs"):
            qq.compare_first_logits(ref, run)


if __name__ == "__main__":
    unittest.main()
