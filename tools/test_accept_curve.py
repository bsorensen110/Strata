import contextlib
import io
from pathlib import Path
import tempfile

import accept_curve


FIXTURES = Path(__file__).parent / "testdata"


def report_text(records, rounds=()):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        accept_curve.report(records, rounds)
    return out.getvalue()


def synthetic_records(accepted, reached=20, prompt_tokens=150_001):
    return [
        {
            "req": req,
            "prompt_tokens": prompt_tokens,
            "reached": [reached] * 7,
            "accepted": [accepted] * 7,
            "lookup_rows": 0,
            "lookup_accept": 0,
            "chain_rows": 0,
            "chain_accept": 0,
        }
        for req in range(5)
    ]


def hist_line(req=1):
    return (
        f"[accept-hist] req={req} df2_pos_reached=1,1,1,1,1,1,1 "
        "df2_pos_accept=1,1,1,1,1,1,1 lookup_rows=0 lookup_accept=0 "
        "chain_rows=0 chain_accept=0\n"
    )


def read_log_text(text):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "engine.log"
        path.write_text(text, encoding="utf-8")
        return accept_curve.read_engine_log(path)


def main():
    records = accept_curve.read_engine_log(FIXTURES / "accept_curve.log")
    rounds = accept_curve.read_dumps([FIXTURES / "accept_curve.jsonl"])
    assert [r["prompt_tokens"] for r in records] == [40_000, 180_000]
    assert len(rounds) == 2
    assert rounds[0]["candidate_tokens"][1][1] == rounds[0]["first_rejected_token"]
    text = report_text(records, rounds)
    assert "Prompt bucket <50k" in text
    assert "Prompt bucket >150k" in text
    assert "Recall@1: 0/2 (0.0%)" in text
    assert "Recall@2: 1/2 (50.0%)" in text
    assert "VERDICT: undetermined (long-context requests=1, min position-2..7 reaches=0/100)" in text

    smoke = synthetic_records(1, reached=1, prompt_tokens=11)[:1]
    assert "VERDICT: undetermined (long-context requests=0, min position-2..7 reaches=0/100)" in report_text(smoke)

    for accepted, reached, verdict, mean in (
        (20, 20, "flat", "100.0%"),
        (15, 20, "steep", "75.0%"),
        (21, 25, "flat", "84.0%"),
        (20, 25, "undetermined", "80.0%"),
    ):
        samples = synthetic_records(accepted, reached=reached)
        expected = f"VERDICT: {verdict} (mean positions 2-7={mean}, n={reached * 30})"
        assert expected in report_text(samples)

    for position in range(1, 7):
        samples = synthetic_records(0)
        samples[0]["reached"][position] = 19
        assert "VERDICT: undetermined (long-context requests=5, min position-2..7 reaches=99/100)" in report_text(samples)

    samples = synthetic_records(20)
    samples[-1]["req"] = samples[0]["req"]
    assert "VERDICT: undetermined (long-context requests=4, min position-2..7 reaches=100/100)" in report_text(samples)

    five_epochs = "".join(
        "engine started: test\n"
        "strata serve: prompt 180001 tokens =\n"
        + hist_line()
        for _ in range(5)
    )
    records = read_log_text(five_epochs)
    assert "VERDICT: undetermined (long-context requests=5, min position-2..7 reaches=5/100)" in report_text(records)

    same_epoch = read_log_text(
        "engine started: test\n"
        "strata serve: prompt 180001 tokens =\n"
        + hist_line()
        + "strata serve: prompt 180001 tokens =\n"
        + hist_line()
    )
    assert "VERDICT: undetermined (long-context requests=1, min position-2..7 reaches=2/100)" in report_text(same_epoch)

    restarted_without_prompt = read_log_text(
        "engine started: test\n"
        "strata serve: prompt 180001 tokens =\n"
        + hist_line()
        + "engine started: test\n"
        + hist_line()
    )
    assert restarted_without_prompt[1]["prompt_tokens"] is None

    short = synthetic_records(0, reached=1000, prompt_tokens=150_000)
    unknown = synthetic_records(0, reached=1000, prompt_tokens=None)
    assert "VERDICT: flat (mean positions 2-7=100.0%, n=600)" in report_text(synthetic_records(20) + short + unknown)
    short = synthetic_records(1000, reached=1000, prompt_tokens=150_000)
    assert "VERDICT: steep (mean positions 2-7=75.0%, n=600)" in report_text(synthetic_records(15) + short)


if __name__ == "__main__":
    main()
