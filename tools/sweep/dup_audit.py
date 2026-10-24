#!/usr/bin/env python3
"""Audit a merge sweep for duplicated statements left at union seams.

Reads the list of files to check from stdin (one path per line), so it can be
pointed at exactly what the sweep touched.

Detectors:
  MARKERS  conflict markers survived a commit            (must be zero)
  ADJ      identical non-trivial line repeated back-to-back   -> unclosed brace risk
  NEAR     identical non-trivial line repeated within 4 lines -> double-applied call
  FN       identical function definition line twice           -> duplicate definition
"""
import re, sys, collections

MINLEN = 8
ALLOW = re.compile(r'^(else|break;|continue;|return;|\}|\};|\{|\}\);|#endif|#else|;|fi|done)$')
TRIVIAL = re.compile(r'^[\s{}();,#]*$')
FN = re.compile(r'^(?:static\s+)?(?:inline\s+)?[A-Za-z_][\w:<>,&*\s]*?\b(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$')
KEYWORDS = ('if', 'for', 'while', 'switch', 'return', 'catch', 'else')


def trivial(s):
    t = s.strip()
    return (not t) or len(t) < MINLEN or bool(TRIVIAL.match(t)) or bool(ALLOW.match(t))


def main():
    files = [l.strip() for l in sys.stdin if l.strip().endswith(
        ('.cpp', '.hpp', '.h', '.inc', '.metal', '.py'))]
    hits = collections.Counter()
    out = []
    for p in files:
        try:
            raw = open(p, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        if '<<<<<<<' in raw and not p.endswith('dup_audit.py'):
            out.append(f"MARKERS {p}")
            hits['MARKERS'] += 1
        lines = raw.split('\n')
        for i in range(len(lines) - 1):
            if lines[i] == lines[i + 1] and not trivial(lines[i]):
                out.append(f"ADJ  {p}:{i+1}  {lines[i].strip()[:110]}")
                hits['ADJ'] += 1
        for i in range(len(lines)):
            if trivial(lines[i]):
                continue
            for j in range(i + 2, min(i + 5, len(lines))):
                if lines[i] == lines[j] and not trivial(lines[j]):
                    out.append(f"NEAR {p}:{i+1}~{j+1}  {lines[i].strip()[:110]}")
                    hits['NEAR'] += 1
                    break
        seen = {}
        for i, ln in enumerate(lines):
            m = FN.match(ln)
            if not m or ln.lstrip().startswith(KEYWORDS):
                continue
            sig = ln.strip()          # exact signature text: overloads differ, duplicates do not
            if sig in seen:
                out.append(f"FN   {p}:{i+1} (dup of {seen[sig]})  {sig[:100]}")
                hits['FN'] += 1
            else:
                seen[sig] = i + 1
    print('\n'.join(out) if out else 'clean: no duplicated adjacent statements in the swept files')
    print('--- summary:', dict(hits) or 'none', '---')


main()
