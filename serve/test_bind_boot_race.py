"""The boot race on a user-scope systemd unit (observed on the R9700 box, 2026-10-25): the unit
starts before DHCP has handed this PC its configured address, so the preflight listen fails with
EADDRNOTAVAIL and the start died at `ap.error(...)`.  Eleven failed starts after one reboot; the
unit's Restart=on-failure eventually won, so the fix belongs in the server: an address that does
not exist YET is worth waiting for, while a port that is TAKEN is a real conflict and must stay a
loud, immediate failure.

    python -m serve.test_bind_boot_race
"""
import errno
import socket
import unittest

import serve.server as server


def os_error(code: int) -> OSError:
    e = OSError("simulated bind failure")
    e.errno = code
    return e


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class PreflightBind(unittest.TestCase):
    def test_an_available_address_binds_at_once(self):
        waited = server.preflight_bind("127.0.0.1", free_port(), seconds=1.0, step=0.05)
        self.assertLess(waited, 1.0)

    def test_an_address_this_pc_does_not_have_is_waited_out(self):
        # TEST-NET-1 is not routed to this machine: bind answers EADDRNOTAVAIL, exactly as a
        # not-yet-configured LAN address does at boot.
        with self.assertRaises(OSError) as caught:
            server.preflight_bind("192.0.2.1", 8086, seconds=0.4, step=0.05)
        self.assertTrue(server.bind_is_transient(caught.exception))
        code = getattr(caught.exception, "winerror", None) or caught.exception.errno
        self.assertEqual(code, errno.EADDRNOTAVAIL)

    def test_the_retry_returns_when_the_address_lands(self):
        calls = []

        def factory():
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                raise os_error(errno.EADDRNOTAVAIL)
            return server.Server(("127.0.0.1", free_port()), server.BaseHTTPRequestHandler)

        waited = server.preflight_bind("192.168.1.156", 8086, seconds=5.0, step=0.05, factory=factory)
        self.assertEqual(len(calls), 3)
        self.assertLess(waited, 5.0)

    def test_a_taken_port_is_never_retried(self):
        taken = free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
            squatter.bind(("127.0.0.1", taken))
            squatter.listen(1)
            calls = []

            def factory():
                calls.append(len(calls) + 1)
                raise os_error(errno.EADDRINUSE)

            with self.assertRaises(OSError) as caught:
                server.preflight_bind("127.0.0.1", taken, seconds=5.0, step=0.05, factory=factory)
            self.assertEqual(len(calls), 1, "a conflict must fail at once, not after a wait")
            self.assertFalse(server.bind_is_transient(caught.exception))

    def test_a_windows_address_error_is_the_same_race(self):
        # Windows reports "the requested address is not valid in its context" as winerror 10049.
        self.assertTrue(server.bind_is_transient(os_error(10049)))
        self.assertTrue(server.bind_is_transient(os_error(errno.EADDRNOTAVAIL)))
        self.assertFalse(server.bind_is_transient(os_error(errno.EACCES)))

    def test_the_message_for_the_race_still_names_the_cause(self):
        # listen_problem stays the failure text: the retry must not invent a vaguer one.
        m = server.listen_problem("192.168.1.156", 8086, os_error(errno.EADDRNOTAVAIL))
        self.assertIn("192.168.1.156:8086", m)
        self.assertIn("not an address of this PC", m)


if __name__ == "__main__":
    unittest.main()
