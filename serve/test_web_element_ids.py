"""Every element id the page scripts touch must exist in the page (serve/web).

app.js reads `$("id")` while it starts up. A missing element is null, and the first assignment
to it (`$("s-ws-readmax").oninput = ...`) throws before the page's first poll, so the header pill
stays on "Connecting…" for good. This check reads the sources and fails, naming each id the
scripts use that the HTML does not define (and the script does not create itself).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

SERVE = Path(__file__).resolve().parent
WEB = SERVE / "web"

PAGES = {"index.html": WEB / "index.html", "monitor.html": WEB / "monitor.html"}
ID_IN_HTML = re.compile(r'\bid="([^"]+)"')
SCRIPT_SRC = re.compile(r'<script[^>]+src="([^"]+\.js)"')
LOOKUP = re.compile(r'\$\(\s*["\']([^"\']+)["\']\s*\)|getElementById\(\s*["\']([^"\']+)["\']\s*\)')
CREATED = re.compile(r'\.id\s*=\s*["\']([^"\']+)["\']|\bid="([^"$]+)"')


def scripts_of(page: Path) -> list[Path]:
    found = []
    for src in SCRIPT_SRC.findall(page.read_text(encoding="utf-8")):
        for base in (SERVE, WEB):
            candidate = (base / src.lstrip("/")).resolve()
            if candidate.is_file():
                found.append(candidate)
                break
        else:
            raise AssertionError(f"{page.name} loads {src}, which is not in serve/ or serve/web/")
    return found


class PageElementIds(unittest.TestCase):
    def check(self, page_name: str) -> None:
        page = PAGES[page_name]
        defined = set(ID_IN_HTML.findall(page.read_text(encoding="utf-8")))
        missing: dict[str, list[str]] = {}
        for script in scripts_of(page):
            text = script.read_text(encoding="utf-8", errors="replace")
            created = {a or b for a, b in CREATED.findall(text)}
            used = {a or b for a, b in LOOKUP.findall(text)}
            for ident in sorted(used - defined - created):
                missing.setdefault(ident, []).append(script.name)
        detail = "\n".join(f"  {ident}  (used by {', '.join(sorted(set(where)))})"
                           for ident, where in sorted(missing.items()))
        self.assertFalse(missing, f"{page_name} is missing element ids its scripts use "
                                  f"(first null lookup throws at startup):\n{detail}")

    def test_index_html_defines_every_id_app_js_uses(self):
        self.check("index.html")

    def test_monitor_html_defines_every_id_monitor_js_uses(self):
        self.check("monitor.html")


if __name__ == "__main__":
    unittest.main()
