from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # Maximize thrust, minimize pressureLoss.
    not_worse = (a["thrust"] >= b["thrust"]) and (a["pressureLoss"] <= b["pressureLoss"])
    strictly_better = (a["thrust"] > b["thrust"]) or (a["pressureLoss"] < b["pressureLoss"])
    return not_worse and strictly_better


def _pareto_front(records: List[Dict[str, Any]]) -> List[int]:
    front: List[int] = []
    for i, ri in enumerate(records):
        dominated = False
        for j, rj in enumerate(records):
            if i == j:
                continue
            if _dominates(rj, ri):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _build_scored(records: List[Dict[str, Any]], w_thrust: float, w_loss: float) -> List[Dict[str, Any]]:
    thrusts = [float(r["thrust"]) for r in records]
    losses = [float(r["pressureLoss"]) for r in records]
    nt = _normalize(thrusts)
    nl = _normalize(losses)

    scored: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        score = w_thrust * nt[i] + w_loss * (1.0 - nl[i])
        row = dict(r)
        row["candidate_index"] = i
        row["score"] = score
        row["thrust_norm"] = nt[i]
        row["pressure_loss_norm"] = nl[i]
        scored.append(row)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _knee_from_pareto(front: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not front:
        return {}
    thrusts = [float(r["thrust"]) for r in front]
    losses = [float(r["pressureLoss"]) for r in front]
    nt = _normalize(thrusts)  # maximize
    nl = _normalize(losses)   # minimize
    best_i = 0
    best_d = 1e18
    for i in range(len(front)):
        # Utopia in normalized space is (thrust=1, loss=0).
        d = ((1.0 - nt[i]) ** 2 + (nl[i] - 0.0) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_i = i
    out = dict(front[best_i])
    out["knee_distance"] = best_d
    return out


def _legend_unique() -> None:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2 = []
    l2 = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    if l2:
        ax.legend(h2, l2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-objective ranking from existing optimization history.")
    ap.add_argument("--history", required=True, help="Path to optimization_history.json")
    ap.add_argument("--out", required=True, help="Output folder for ranking artifacts")
    ap.add_argument("--w-thrust", type=float, default=0.7, help="Weight for thrust objective")
    ap.add_argument("--w-loss", type=float, default=0.3, help="Weight for pressure loss objective")
    ap.add_argument("--top-k", type=int, default=20, help="Top K records to save")
    ap.add_argument("--moc-result", default="", help="Optional path to moc_result.json for comparison overlays")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = json.loads(Path(args.history).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError("History file must contain a non-empty JSON list.")

    for r in records:
        if "thrust" not in r or "pressureLoss" not in r:
            raise RuntimeError("Each history record must contain 'thrust' and 'pressureLoss'.")

    w_sum = args.w_thrust + args.w_loss
    if w_sum <= 0.0:
        raise RuntimeError("w-thrust + w-loss must be > 0")
    w_thrust = args.w_thrust / w_sum
    w_loss = args.w_loss / w_sum

    scored = _build_scored(records, w_thrust=w_thrust, w_loss=w_loss)
    front_idx = _pareto_front(records)
    front = []
    for i in front_idx:
        rec = dict(records[i])
        rec["candidate_index"] = i
        front.append(rec)
    front.sort(key=lambda x: x["thrust"], reverse=True)

    top_k = max(1, min(args.top_k, len(scored)))
    top = scored[:top_k]

    moc_thrust = None
    moc_loss = None
    if args.moc_result:
        moc_data = json.loads(Path(args.moc_result).read_text(encoding="utf-8"))
        if "thrust" in moc_data:
            moc_thrust = float(moc_data["thrust"])
        if "pressureLoss" in moc_data:
            moc_loss = float(moc_data["pressureLoss"])

    best_by_thrust = max(enumerate(records), key=lambda x: float(x[1]["thrust"]))
    best_by_loss = min(enumerate(records), key=lambda x: float(x[1]["pressureLoss"]))
    knee = _knee_from_pareto(front)

    summary = {
        "history_path": str(Path(args.history)),
        "n_candidates": len(records),
        "weights": {"thrust": w_thrust, "pressure_loss": w_loss},
        "n_dominated": len(records) - len(front),
        "best_by_score": {
            "candidate_index": int(top[0]["candidate_index"]),
            "score": float(top[0]["score"]),
            "thrust": float(top[0]["thrust"]),
            "pressureLoss": float(top[0]["pressureLoss"]),
            "params": top[0].get("params", {}),
        },
        "best_by_thrust": {
            "candidate_index": int(best_by_thrust[0]),
            "thrust": float(best_by_thrust[1]["thrust"]),
            "pressureLoss": float(best_by_thrust[1]["pressureLoss"]),
            "params": best_by_thrust[1].get("params", {}),
        },
        "best_by_pressure_loss": {
            "candidate_index": int(best_by_loss[0]),
            "thrust": float(best_by_loss[1]["thrust"]),
            "pressureLoss": float(best_by_loss[1]["pressureLoss"]),
            "params": best_by_loss[1].get("params", {}),
        },
        "best_pareto_knee": {
            "candidate_index": int(knee.get("candidate_index", -1)),
            "thrust": float(knee.get("thrust", 0.0)),
            "pressureLoss": float(knee.get("pressureLoss", 0.0)),
            "knee_distance": float(knee.get("knee_distance", 0.0)),
            "params": knee.get("params", {}),
        },
        "pareto_front_size": len(front),
    }

    (out / "multiobjective_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "multiobjective_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    (out / "pareto_front.json").write_text(json.dumps(front, indent=2), encoding="utf-8")

    thrusts = [float(r["thrust"]) for r in records]
    losses = [float(r["pressureLoss"]) for r in records]
    plt.figure(figsize=(7, 5))
    plt.scatter(losses, thrusts, s=15, alpha=0.35, label="Candidates")
    if front:
        f_losses = [float(r["pressureLoss"]) for r in front]
        f_thrusts = [float(r["thrust"]) for r in front]
        plt.scatter(f_losses, f_thrusts, s=28, alpha=0.9, label="Pareto front")
    if moc_thrust is not None and moc_loss is not None:
        plt.scatter([moc_loss], [moc_thrust], s=65, marker="X", color="tab:green", label="MOC")
    plt.xlabel("Pressure loss [-] (lower is better)")
    plt.ylabel("Thrust [N] (higher is better)")
    plt.title("Pareto Scatter: Thrust vs Pressure Loss")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "pareto_scatter.png", dpi=180)
    plt.close()

    # Pareto curve (sorted by pressure loss)
    if front:
        sf = sorted(front, key=lambda r: float(r["pressureLoss"]))
        fx = [float(r["pressureLoss"]) for r in sf]
        fy = [float(r["thrust"]) for r in sf]
        plt.figure(figsize=(7, 5))
        plt.plot(fx, fy, marker="o", linewidth=1.4, color="tab:orange")
        if moc_thrust is not None and moc_loss is not None:
            plt.scatter([moc_loss], [moc_thrust], s=65, marker="X", color="tab:green", label="MOC")
        plt.xlabel("Pressure loss [-] (lower is better)")
        plt.ylabel("Thrust [N] (higher is better)")
        plt.title("Pareto Frontier Curve")
        plt.grid(True, alpha=0.25)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "pareto_frontier_curve.png", dpi=180)
        plt.close()

        # Zoomed Pareto with candidate labels for direct identification.
        plt.figure(figsize=(8, 5))
        plt.scatter(fx, fy, s=42, color="tab:orange", label="Pareto")
        for r in sf:
            x = float(r["pressureLoss"])
            y = float(r["thrust"])
            idx = int(r["candidate_index"])
            plt.annotate(
                f"{idx}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )
        if moc_thrust is not None and moc_loss is not None:
            plt.scatter([moc_loss], [moc_thrust], s=75, marker="X", color="tab:green", label="MOC")
            plt.annotate(
                "MOC",
                (moc_loss, moc_thrust),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color="tab:green",
            )
        if fx and fy:
            xpad = max((max(fx) - min(fx)) * 0.35, 1e-4)
            ypad = max((max(fy) - min(fy)) * 0.35, 0.1)
            plt.xlim(min(fx) - xpad, max(max(fx), moc_loss if moc_loss is not None else max(fx)) + xpad)
            plt.ylim(min(min(fy), moc_thrust if moc_thrust is not None else min(fy)) - ypad, max(max(fy), moc_thrust if moc_thrust is not None else max(fy)) + ypad)
        plt.xlabel("Pressure loss [-] (lower is better)")
        plt.ylabel("Thrust [N] (higher is better)")
        plt.title("Pareto Zoom (Labeled by candidate index)")
        plt.grid(True, alpha=0.25)
        _legend_unique()
        plt.tight_layout()
        plt.savefig(out / "pareto_zoom_labeled.png", dpi=200)
        plt.close()

    # Score by candidate index
    idx = [int(r["candidate_index"]) for r in scored]
    score_by_idx = [0.0 for _ in records]
    for r in scored:
        score_by_idx[int(r["candidate_index"])] = float(r["score"])
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(range(len(records)), score_by_idx, color="tab:blue", linewidth=1.2)
    plt.xlabel("Candidate index")
    plt.ylabel("Score [-]")
    plt.title("Multiobjective Score by Candidate")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "score_by_candidate.png", dpi=180)
    plt.close()

    # Candidate index diagnostics with Pareto marks.
    pset = set(front_idx)
    xs = list(range(len(records)))
    th = [float(r["thrust"]) for r in records]
    pl = [float(r["pressureLoss"]) for r in records]
    pareto_x = [i for i in xs if i in pset]
    pareto_th = [th[i] for i in pareto_x]
    pareto_pl = [pl[i] for i in pareto_x]

    plt.figure(figsize=(8.5, 4.5))
    plt.scatter(xs, th, s=12, alpha=0.35, color="tab:gray", label="All")
    plt.scatter(pareto_x, pareto_th, s=24, alpha=0.95, color="tab:red", label="Pareto")
    if moc_thrust is not None:
        plt.axhline(moc_thrust, color="tab:green", linestyle="--", linewidth=1.2, label="MOC thrust")
    plt.xlabel("Candidate index")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust by Candidate (Pareto Highlighted)")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "thrust_by_candidate_pareto.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 4.5))
    plt.scatter(xs, pl, s=12, alpha=0.35, color="tab:gray", label="All")
    plt.scatter(pareto_x, pareto_pl, s=24, alpha=0.95, color="tab:red", label="Pareto")
    if moc_loss is not None:
        plt.axhline(moc_loss, color="tab:green", linestyle="--", linewidth=1.2, label="MOC pressure loss")
    plt.xlabel("Candidate index")
    plt.ylabel("Pressure loss [-]")
    plt.title("Pressure Loss by Candidate (Pareto Highlighted)")
    plt.grid(True, alpha=0.25)
    _legend_unique()
    plt.tight_layout()
    plt.savefig(out / "pressure_loss_by_candidate_pareto.png", dpi=180)
    plt.close()

    # Compact summary figure for reporting.
    if moc_thrust is not None and moc_loss is not None:
        best_score = summary["best_by_score"]
        best_thrust = summary["best_by_thrust"]
        best_loss = summary["best_by_pressure_loss"]
        knee_pt = summary["best_pareto_knee"]

        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.axis("off")
        columns = ["Selection", "Candidate", "Thrust [N]", "Pressure loss [-]"]
        rows = [
            ["MOC", "-", f"{moc_thrust:.3f}", f"{moc_loss:.6f}"],
            ["Best by score", str(best_score["candidate_index"]), f"{best_score['thrust']:.3f}", f"{best_score['pressureLoss']:.6f}"],
            ["Best thrust", str(best_thrust["candidate_index"]), f"{best_thrust['thrust']:.3f}", f"{best_thrust['pressureLoss']:.6f}"],
            ["Best loss", str(best_loss["candidate_index"]), f"{best_loss['thrust']:.3f}", f"{best_loss['pressureLoss']:.6f}"],
            ["Pareto knee", str(knee_pt["candidate_index"]), f"{knee_pt['thrust']:.3f}", f"{knee_pt['pressureLoss']:.6f}"],
        ]
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.3)
        ax.set_title("MOC vs Multiobjective Selections", fontsize=12, pad=8)
        plt.tight_layout()
        plt.savefig(out / "moc_vs_best_table.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    main()
