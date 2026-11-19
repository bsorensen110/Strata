#!/usr/bin/env python3
"""Analyze DF2 per-position acceptance telemetry and selector candidate recall."""

import argparse
import json
import math
import re
import sys
from pathlib import Path


HIST_RE = re.compile(r"\[accept-hist\] req=(\d+) (.+)")
PROMPT_RE = re.compile(r"strata serve: prompt (\d+) tokens =")
ENGINE_START_RE = re.compile(r"engine started:")
DEFAULT_EPOCH = 0  # Synthetic records without an epoch share this epoch.
BUCKETS = (
    ("<50k", lambda n: n < 50_000),
    ("50-150k", lambda n: 50_000 <= n <= 150_000),
    (">150k", lambda n: n > 150_000),
)


def parse_hist_line(line):
    match = HIST_RE.search(line)
    if not match:
        return None
    fields = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", match.group(2)))
    reached = [int(x) for x in fields["df2_pos_reached"].split(",")]
    accepted = [int(x) for x in fields["df2_pos_accept"].split(",")]
    if len(reached) != 7 or len(accepted) != 7:
        raise ValueError("accept-hist lines must contain seven DF2 positions")
    if any(a > r for a, r in zip(accepted, reached)):
        raise ValueError("accepted count exceeds reached count")
    return {
        "req": int(match.group(1)),
        "epoch": DEFAULT_EPOCH,
        "reached": reached,
        "accepted": accepted,
        "lookup_rows": int(fields["lookup_rows"]),
        "lookup_accept": int(fields["lookup_accept"]),
        "chain_rows": int(fields["chain_rows"]),
        "chain_accept": int(fields["chain_accept"]),
    }


def read_engine_log(path):
    records = []
    epoch = DEFAULT_EPOCH
    pending_prompt = None
    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if ENGINE_START_RE.search(line):
                epoch += 1
                pending_prompt = None
            prompt = PROMPT_RE.search(line)
            if prompt:
                pending_prompt = int(prompt.group(1))
            hist = parse_hist_line(line)
            if hist is not None:
                hist["epoch"] = epoch
                hist["prompt_tokens"] = pending_prompt
                records.append(hist)
                pending_prompt = None
    return records


def read_dumps(paths):
    rounds = []
    for path in paths:
        with open(path, encoding="utf-8") as dump:
            first = dump.readline()
            try:
                header = json.loads(first)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid geometry header: {exc}") from exc
            if header.get("v") != 1 or header.get("type") != "header":
                raise ValueError(f"{path}: expected a v1 geometry header")
            for line_number, line in enumerate(dump, 2):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if rec.get("v") != 1 or rec.get("type") != "round":
                    raise ValueError(f"{path}:{line_number}: expected a v1 round record")
                if len(rec.get("candidate_tokens", [])) != 7 or len(rec.get("candidate_scores", [])) != 7:
                    raise ValueError(f"{path}:{line_number}: expected 7 candidate rows")
                rounds.append(rec)
    return rounds


def wilson(successes, trials):
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z * z / trials
    mid = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def print_curve(title, records):
    print(title)
    if not records:
        print("  no accept-hist samples")
        return
    print("  pos  accepted/reached  conditional  95% Wilson CI")
    for i in range(7):
        reached = sum(r["reached"][i] for r in records)
        accepted = sum(r["accepted"][i] for r in records)
        if reached == 0:
            print(f"  {i + 1:>3}  {accepted}/{reached}             n/a         n/a")
            continue
        low, high = wilson(accepted, reached)
        print(f"  {i + 1:>3}  {accepted}/{reached:<14} {accepted / reached:>7.1%}     [{low:.1%}, {high:.1%}]")


def report(records, rounds):
    print_curve("DF2 conditional acceptance", records)
    for label, belongs in BUCKETS:
        bucket = [r for r in records if r["prompt_tokens"] is not None and belongs(r["prompt_tokens"])]
        print_curve(f"Prompt bucket {label}", bucket)

    df2_reached = sum(sum(r["reached"]) for r in records)
    df2_accepted = sum(sum(r["accepted"]) for r in records)
    lookup_rows = sum(r["lookup_rows"] for r in records)
    lookup_accepted = sum(r["lookup_accept"] for r in records)
    chain_rows = sum(r["chain_rows"] for r in records)
    chain_accepted = sum(r["chain_accept"] for r in records)
    df2_rate = f"{df2_accepted}/{df2_reached} ({df2_accepted / df2_reached:.1%})" if df2_reached else "n/a"
    lookup_rate = f"{lookup_accepted}/{lookup_rows} ({lookup_accepted / lookup_rows:.1%})" if lookup_rows else "n/a"
    chain_rate = f"{chain_accepted}/{chain_rows} ({chain_accepted / chain_rows:.1%})" if chain_rows else "n/a"
    blended_rows = lookup_rows + chain_rows
    blended_accepted = lookup_accepted + chain_accepted
    blended_rate = (f"{blended_accepted}/{blended_rows} ({blended_accepted / blended_rows:.1%})"
                    if blended_rows else "n/a")
    print("Blended vs DF2-only (source counters)")
    print(f"  DF2 conditional rows: {df2_rate}; blended suffix+chain rows: {blended_rate} "
          f"(suffix {lookup_rate}; chain {chain_rate})")

    rank_counts = [0] * 16
    misses = eligible = 0
    for rec in rounds:
        pos = rec.get("first_rejected_position")
        token = rec.get("first_rejected_token")
        if not isinstance(pos, int) or pos < 0 or pos >= rec.get("df2_rows", 7) or token is None:
            continue
        candidates = rec["candidate_tokens"][pos]
        eligible += 1
        try:
            rank_counts[candidates.index(token)] += 1
        except ValueError:
            misses += 1
    print(f"Selector recall at first rejected DF2 position (n={eligible})")
    for k in (1, 2, 4, 8, 16):
        hits = sum(rank_counts[:k])
        print(f"  Recall@{k}: {hits}/{eligible} ({hits / eligible:.1%})" if eligible else f"  Recall@{k}: n/a")
    distribution = ", ".join(f"rank {i + 1}={n}" for i, n in enumerate(rank_counts) if n)
    print(f"  rank distribution: {distribution or 'none'}; not in top-16={misses}")

    long_context = [r for r in records if r["prompt_tokens"] is not None and r["prompt_tokens"] > 150_000]
    request_count = len({(r.get("epoch", DEFAULT_EPOCH), r["req"]) for r in long_context})
    tail_reached = [sum(r["reached"][i] for r in long_context) for i in range(1, 7)]
    min_reached = min(tail_reached)
    if request_count < 5 or min_reached < 100:
        print(f"VERDICT: undetermined (long-context requests={request_count}, "
              f"min position-2..7 reaches={min_reached}/100)")
        return
    tail = [sum(r["accepted"][i] for r in long_context) / n
            for i, n in enumerate(tail_reached, 1)]
    mean = sum(tail) / len(tail)
    verdict = "flat" if mean >= 0.84 else "steep" if mean < 0.80 else "undetermined"
    print(f"VERDICT: {verdict} (mean positions 2-7={mean:.1%}, n={sum(tail_reached)})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine_log", type=Path, help="stderr log containing prompt and [accept-hist] lines")
    parser.add_argument("candidate_dumps", nargs="+", type=Path, help="one or more v1 candidate JSONL dumps")
    args = parser.parse_args(argv)
    try:
        records = read_engine_log(args.engine_log)
        rounds = read_dumps(args.candidate_dumps)
        report(records, rounds)
    except (OSError, KeyError, ValueError, ZeroDivisionError) as exc:
        print(f"accept_curve.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
