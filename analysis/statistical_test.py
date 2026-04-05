"""
Statistical test: Stealth vs Control ASR under hardened victim.

Reads metrics.jsonl from Turnstile experiments and runs Mann-Whitney U
and Wilcoxon signed-rank tests.

Usage:
    python statistical_test.py --turnstile_dir ../../turnstile/experiments
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def load_metrics(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compare_seed(seed, stealth, control):
    """Compare stealth vs control for a single seed."""
    min_rounds = min(len(stealth), len(control))
    st_asr = [stealth[i]["asr"] for i in range(min_rounds)]
    ct_asr = [control[i]["asr"] for i in range(min_rounds)]

    result = {
        "seed": seed,
        "n_rounds": min_rounds,
        "stealth_mean": float(np.mean(st_asr)),
        "control_mean": float(np.mean(ct_asr)),
        "difference": float(np.mean(st_asr) - np.mean(ct_asr)),
    }

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(st_asr, ct_asr, alternative="greater")
    result["mwu_U"] = float(u_stat)
    result["mwu_p"] = float(u_p)

    # Wilcoxon signed-rank (paired)
    if min_rounds >= 5:
        try:
            w_stat, w_p = stats.wilcoxon(st_asr, ct_asr, alternative="greater")
            result["wilcoxon_W"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        except ValueError:
            pass

    # Cohen's d
    pooled_std = np.sqrt((np.var(st_asr) + np.var(ct_asr)) / 2)
    if pooled_std > 0:
        result["cohens_d"] = float((np.mean(st_asr) - np.mean(ct_asr)) / pooled_std)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turnstile_dir", type=str, default="../../turnstile/experiments")
    parser.add_argument("--output", type=str, default="../figures/statistical_test.json")
    args = parser.parse_args()

    exp_dir = Path(args.turnstile_dir)
    results = {"seeds": [], "pooled": {}}

    all_stealth = []
    all_control = []

    for seed in ["s42", "s123", "s456"]:
        st_path = exp_dir / f"stealth_hard_{seed}" / "metrics.jsonl"
        ct_path = exp_dir / f"control_hard_{seed}" / "metrics.jsonl"

        if not st_path.exists() or not ct_path.exists():
            print(f"Skipping {seed}: missing data")
            continue

        stealth = load_metrics(st_path)
        control = load_metrics(ct_path)
        result = compare_seed(seed, stealth, control)
        results["seeds"].append(result)

        min_r = min(len(stealth), len(control))
        all_stealth.extend([stealth[i]["asr"] for i in range(min_r)])
        all_control.extend([control[i]["asr"] for i in range(min_r)])

        print(f"Seed {seed}: stealth {result['stealth_mean']:.3f} vs control {result['control_mean']:.3f} "
              f"(+{result['difference']:.3f}), MWU p={result['mwu_p']:.4f}, d={result.get('cohens_d', 'N/A')}")

    # Pooled
    if all_stealth:
        u_stat, u_p = stats.mannwhitneyu(all_stealth, all_control, alternative="greater")
        pooled_std = np.sqrt((np.var(all_stealth) + np.var(all_control)) / 2)
        d = (np.mean(all_stealth) - np.mean(all_control)) / pooled_std if pooled_std > 0 else 0

        results["pooled"] = {
            "n_stealth": len(all_stealth),
            "n_control": len(all_control),
            "stealth_mean": float(np.mean(all_stealth)),
            "control_mean": float(np.mean(all_control)),
            "difference": float(np.mean(all_stealth) - np.mean(all_control)),
            "mwu_U": float(u_stat),
            "mwu_p": float(u_p),
            "cohens_d": float(d),
            "kill_gate": "PASS" if u_p < 0.05 and d > 0.5 else "FAIL",
        }

        print(f"\nPooled: stealth {np.mean(all_stealth):.3f} vs control {np.mean(all_control):.3f} "
              f"(+{np.mean(all_stealth) - np.mean(all_control):.3f}), "
              f"MWU p={u_p:.4f}, d={d:.3f}")
        print(f"Kill gate: {results['pooled']['kill_gate']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
