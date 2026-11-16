"""
The results tables (README.md) from the runs' evals: SAI_DIR/runs/<run>/gfn_evals.jsonl, the last line per run.
    python tables.py              # after scripts/grid.sh and scripts/monitor.sh
Seeds are pooled as ranges (min-max); a missing run is skipped.
"""

import json
import os

from config import BASE_DIR

SEEDS = (0, 1)
# Table 1: (label, run prefix)
POLICIES = [("(1) the answer", "r_kf"), ("(2) + at least 3 equations shown", "rsw_kf"),
            ("(3) + answer equals the last equation's result", "rcl_kf"), ("(4) + every equation evaluates", "rcv_kf"),
            ("(5) as (4), p with identity steps", "rcv_kfpad2"), ("R-true: the true answer", "rtrue_kf"),
            ("R-TB (tau 0.1), (1)", "rtb_kf"), ("R-TB, (2)", "rtbsw_kf"), ("R-TB, (3)", "rtbcl_kf"),
            ("Scientist AI", "e_kf")]
# Table 2: (label, policy prefixes, estimator run)
MONITORED = [("no work, or decoupled work", ["r_kf", "rsw_kf"], "e_kf_s0"), ("falsified work", ["rcl_kf"], "e_kf_s0"),
             ("invented premise", ["rcv_kfpad2"], "e_kfpad2_s0"), ("honest (R-true, step checker)", ["rtrue_kf", "rcv_kf"], "e_kf_s0")]
LADDER = [("0.5B", "p_ff"), ("0.1B", "p_ff_100m"), ("0.03B", "p_ff_30m"), ("0.01B", "p_ff_10m")]  # (q = p)
CHEATS = ["r_kf", "rsw_kf", "rcl_kf", "rcv_kfpad2"]


def last_eval(run):
    path = os.path.join(BASE_DIR, "runs", run, "gfn_evals.jsonl")
    if not os.path.exists(path):
        return None
    lines = [json.loads(line) for line in open(path)]
    return lines[-1] if lines else None


def span(values, pct=False, digits=3):
    values = [v for v in values if v is not None and v == v]
    if not values:
        return "-"
    f = (lambda v: f"{100 * v:.{max(0, digits - 2)}f}") if pct else (lambda v: f"{v:.{digits}f}")
    lo, hi = f(min(values)), f(max(values))
    return lo if lo == hi else f"{lo}-{hi}"


def table(header, rows):
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def main():
    rows = []
    for label, prefix in POLICIES:
        b = [e["key"]["behaviour"] for e in (last_eval(f"{prefix}_s{s}") for s in SEEDS) if e]
        if not b:
            continue
        form = max(["skip_of_cheats", "decoupled_of_cheats", "motivated_false_of_cheats", "invented_of_cheats"],
                   key=lambda k: sum(x.get(k, 0) or 0 for x in b))
        cheats = max(x["cheat_wrong_key"] for x in b) >= 0.01
        rows.append([label, span([x["cheat_wrong_key"] for x in b], pct=True), form.removesuffix("_of_cheats") if cheats else "-",
                     span([x[form] for x in b], pct=True) if cheats else "-", span([x["truth_acc"] for x in b])])
    print("Table 1. Copied: share of wrong-key items answered with the key (%); form: the most common kind of cheat and its "
          "share of the cheats (%); accuracy against the truth.")
    table(["policy / grader checks", "copied", "form", "share", "accuracy"], rows)

    e = [last_eval(f"e_kf_s{s}") for s in SEEDS]
    if any(e):
        e = [x for x in e if x]
        print("The Scientist AI: P(true) at its verdict node on wrong keys", span([x["key"]["verdict_q"]["p_true_wrong_key"] for x in e]),
              "/ right keys", span([x["key"]["verdict_q"]["p_true_right_key"] for x in e]),
              "| IS accuracy over 4 candidates", span([x["posterior"]["accuracy"] for x in e]),
              "| direct p", span([x["direct"]["accuracy"] for x in e]), "\n")

    rows = []
    for label, prefixes, estimator in MONITORED:
        ev = last_eval(estimator) or {}
        m = [ev.get(f"monitor/{p}_s{s}") for p in prefixes for s in SEEDS]
        m = [x for x in m if x and x.get("n")]
        if not m:
            continue
        cell = lambda name: (span([x[name]["flagged_wrong"] for x in m], pct=True) + " / " +
                             span([x[name]["false_alarm"] for x in m], pct=True))
        rows.append([label, span([x["wrong"] for x in m], pct=True), span([x["grader_passes_wrong"] for x in m], pct=True),
                     cell("scientist"), cell("direct"), cell("cot")])
    print("Table 2. Monitors (P(answer | X) < 0.5 flags): flagged wrong answers / false alarms on right ones (%).")
    table(["policy", "wrong", "grader passes", "Scientist AI", "direct p", "CoT monitor"], rows)

    rows = []
    for label, estimator in LADDER:
        ev = last_eval(estimator) or {}
        m = [ev.get(f"monitor/{p}_s{s}") for p in CHEATS for s in SEEDS]
        m = [x for x in m if x and x.get("n")]
        if not m:
            continue
        sc = lambda key: span([x["scientist"][key] for x in m], pct=key != "auroc", digits=2 if key == "auroc" else 3)
        rows.append([label, span([ev["key"]["behaviour"]["truth_acc"]], digits=2),
                     sc("flagged_wrong"), sc("false_alarm"), sc("auroc"),
                     span([x["direct"]["auroc"] for x in m], digits=2)])
    print("Table 3. Weaker world models (continued pretraining on fewer tokens), each the untrained estimator (q = p) and "
          "the direct monitor's p, on the cheating policies.")
    table(["p (CPT tokens)", "p's own accuracy", "flagged", "false alarms", "AUROC", "direct p AUROC"], rows)


if __name__ == "__main__":
    main()
