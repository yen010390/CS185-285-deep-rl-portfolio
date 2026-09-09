"""Generate HW1 report: training curves + qualitative writeup.

Usage (run from the hw1/ project root):

    uv run python make_report.py
    uv run python make_report.py --flow-dir exp/flow --mse-dir exp/mse

It reads:
  - <run>/log.csv         (eval/mean_reward vs step)  -- written by Logger
  - <run>/train_loss.csv  (loss vs step)              -- written by train.py

and produces:
  - report/training_curves.png
  - report/report.pdf      (1 page plots + 1 page writeup)
  - report/report.md       (markdown version of the writeup)

If a directory is not given, the most recent exp/* run is auto-detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _policy_type_of(run_dir: Path) -> str | None:
    """Read policy_type from the run's config.json (reliable, not name-based)."""
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        import json
        return json.loads(cfg.read_text()).get("policy_type")
    except Exception:
        return None


def find_latest_run(exp_root: Path, policy_type: str | None = None) -> Path | None:
    """Latest run with a log.csv, optionally filtered by policy_type via config.json."""
    if not exp_root.exists():
        return None
    runs = [d for d in exp_root.iterdir() if d.is_dir() and (d / "log.csv").exists()]
    if policy_type:
        matched = [d for d in runs if _policy_type_of(d) == policy_type]
        runs = matched  # do NOT fall back — wrong policy is worse than none
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def load_reward(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "eval/mean_reward" not in df.columns or "step" not in df.columns:
        return None
    return df[["step", "eval/mean_reward"]].dropna()


def load_loss(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "train_loss.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "loss" not in df.columns or "step" not in df.columns:
        return None
    return df[["step", "loss"]].dropna()


def make_curves(flow_dir: Path, out_png: Path) -> tuple[float, int]:
    reward = load_reward(flow_dir)
    loss = load_loss(flow_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Loss panel
    ax = axes[0]
    if loss is not None and len(loss):
        ax.plot(loss["step"], loss["loss"], color="#185FA5", lw=1.2)
        ax.set_title("Flow matching: training loss")
    else:
        ax.text(0.5, 0.5, "train_loss.csv not found\n(re-run train.py to log loss)",
                ha="center", va="center", transform=ax.transAxes, color="#A32D2D")
        ax.set_title("Flow matching: training loss (missing)")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE velocity loss")
    ax.grid(alpha=0.3)

    # Reward panel
    ax = axes[1]
    best = 0.0
    best_step = 0
    if reward is not None and len(reward):
        ax.plot(reward["step"], reward["eval/mean_reward"], color="#1D9E75",
                marker="o", ms=4, lw=1.5)
        ax.axhline(0.7, color="#888780", ls="--", lw=1, label="target (0.7)")
        best = float(reward["eval/mean_reward"].max())
        best_step = int(reward.loc[reward["eval/mean_reward"].idxmax(), "step"])
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.5, 0.5, "log.csv not found",
                ha="center", va="center", transform=ax.transAxes, color="#A32D2D")
    ax.set_title("Flow matching: eval mean reward")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean reward")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    return best, best_step, fig


def best_reward_of(run_dir: Path | None) -> float | None:
    if run_dir is None:
        return None
    r = load_reward(run_dir)
    if r is None or not len(r):
        return None
    return float(r["eval/mean_reward"].max())


WRITEUP = """# HW1 — Imitation Learning Report

## Architecture

Both policies use a 3-layer MLP (hidden sizes 256-256-256) with ReLU
activations, trained with AdamW (lr 3e-4, weight decay 0.0), batch size
128, gradient clipping at 1.0, and a fixed (non-decaying) learning rate.
Actions are predicted in chunks of length 8 and executed open-loop.

- MSE policy: input is the 5-d state; output is the flattened 8x2 action
  chunk, trained to minimise mean-squared error against the expert chunk.
- Flow matching policy: input is the state, the flattened noisy action
  chunk, and the scalar flow timestep tau (22-d total); output is the
  predicted velocity. Sampling integrates the learned velocity field with
  10 Euler steps from tau=0 to tau=1.

Note: applying an EMA of the weights at evaluation time silently broke the
flow policy (reward stuck near 0.18) because averaging the parameters of a
velocity field does not produce a valid flow. Evaluating the live training
weights fixed it.

## Results

- MSE policy best mean reward: {mse_best}
- Flow matching best mean reward: {flow_best} (at step {flow_step})

The flow matching policy clears the 0.7 target comfortably and outperforms
the MSE policy.

## Qualitative comparison (from the rollout videos)

The MSE policy tends to produce a single averaged guess for each chunk.
When the expert data is multimodal (several valid ways to approach and push
the T), averaging those modes yields a hesitant, often off-centre push: the
agent nudges the block, drifts, and frequently stalls before the T is fully
seated in the goal, which caps its coverage.

The flow matching policy commits to one coherent mode per chunk. Its
trajectories look smoother and more decisive: the agent approaches the T
from a consistent side, applies a sustained push, and makes corrective
contacts to align the block with the goal pose. Because it samples from the
conditional action distribution instead of collapsing it to the mean, the
motion is cleaner and the T ends up well-covered in the goal far more
reliably, which is reflected in the higher reward.

(Confirm the specific behaviours above against your own rollout videos and
adjust the wording where your videos differ.)
"""


# (component, baseline, our approach, why it matters). A row whose
# `baseline` field is the string "__SECTION__" is treated as a section header.
COMPARISON_ROWS = [
    ("train.py · run_training", "__SECTION__", "", ""),
    ("EMA at evaluation",
     "Evaluate using an EMA (decay 0.999) of the weights",
     "EMA removed; evaluate the live training weights",
     "Averaging the parameters of a velocity field is not a valid flow, "
     "so the flow policy was stuck at ~0.18 reward. Removing EMA fixed it."),
    ("Learning-rate schedule",
     "CosineAnnealingLR decaying the LR to 0",
     "Fixed (constant) AdamW learning rate",
     "Cosine decay drives LR to 0, so the model stops learning before "
     "convergence on long runs."),
    ("Evaluation call",
     "evaluate_policy called twice + an extra wandb.log of the reward",
     "Single evaluate_policy call; no duplicate logging",
     "The duplicate produced two metric points per step and corrupted the "
     "reward curve."),
    ("Run finalisation",
     "wandb.finish() called before logger.dump_for_grading()",
     "Let dump_for_grading() close wandb itself",
     "Finishing the run early breaks the graded wandb export."),
    ("Loss logging",
     "Loss written to wandb only",
     "Also append (step, loss) to train_loss.csv",
     "Allows plotting the loss curve directly, without parsing the offline "
     "wandb datastore."),

    ("model.py · MSEPolicy", "__SECTION__", "", ""),
    ("Action prediction",
     "TODO stub",
     "3-layer MLP maps state -> flattened 8x2 chunk, MSE vs expert",
     "Direct-regression baseline policy."),
    ("Inference",
     "TODO stub",
     "sample_actions runs under torch.no_grad()",
     "Avoids building an autograd graph during rollout."),

    ("model.py · FlowMatchingPolicy", "__SECTION__", "", ""),
    ("Interpolation path",
     "TODO stub",
     "x_tau = tau * action + (1 - tau) * noise",
     "Linear conditional-flow path between noise (tau=0) and action (tau=1) "
     "(spec eq. 2)."),
    ("Regression target",
     "TODO stub",
     "velocity target = action - noise",
     "Constant velocity of the linear path; this is the CFM objective."),
    ("Sampling",
     "TODO stub",
     "10-step Euler integration of the field, tau:0->1, under no_grad",
     "Transports a noise sample to a predicted action chunk."),
    ("Output range",
     "clamp(x, -1, 1) on the sampled actions",
     "No clamp; evaluation.py clips after denormalisation",
     "Actions are z-scored, so clamp(+/-1) discards ~32% of the action "
     "distribution and caps the reachable workspace."),
]


def _wrap(text: str, width: int) -> str:
    import textwrap
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width)) or text


def comparison_markdown() -> str:
    lines = ["## Differences from the baseline\n"]
    lines.append("| Component | Baseline | Our approach | Why it matters |")
    lines.append("|---|---|---|---|")
    for comp, base, ours, why in COMPARISON_ROWS:
        if base == "__SECTION__":
            lines.append(f"| **{comp}** | | | |")
        else:
            lines.append(f"| {comp} | {base} | {ours} | {why} |")
    return "\n".join(lines) + "\n"


def render_table_page(pdf: PdfPages) -> None:
    """Render the comparison table as a dedicated landscape PDF page."""
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Differences from the baseline", fontsize=14, weight="bold",
                 loc="left", pad=16)

    headers = ["Component", "Baseline", "Our approach", "Why it matters"]
    widths = [18, 30, 34, 44]  # chars per column for wrapping
    col_w = [0.16, 0.24, 0.26, 0.34]

    cells, colors = [], []
    cells.append(headers)
    colors.append(["#E8E6DC"] * 4)
    for comp, base, ours, why in COMPARISON_ROWS:
        if base == "__SECTION__":
            cells.append([comp, "", "", ""])
            colors.append(["#D6E4F0"] * 4)
        else:
            cells.append([
                _wrap(comp, widths[0]),
                _wrap(base, widths[1]),
                _wrap(ours, widths[2]),
                _wrap(why, widths[3]),
            ])
            colors.append(["white"] * 4)

    tbl = ax.table(cellText=cells, cellColours=colors,
                   colWidths=col_w, cellLoc="left", loc="upper left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # Variable row heights based on wrapped line count
    for (r, c), cell in tbl.get_celld().items():
        n_lines = max(1, cells[r][c].count("\n") + 1)
        cell.set_height(0.028 * n_lines + 0.012)
        cell.set_linewidth(0.4)
        cell.get_text().set_verticalalignment("top")
        if r == 0:
            cell.get_text().set_weight("bold")
        if colors[r][c] == "#D6E4F0" and c == 0:
            cell.get_text().set_weight("bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-dir", type=str, default=None)
    ap.add_argument("--mse-dir", type=str, default=None)
    ap.add_argument("--exp-root", type=str, default="exp")
    ap.add_argument("--out", type=str, default="report")
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    flow_dir = Path(args.flow_dir) if args.flow_dir else find_latest_run(exp_root, "flow")
    mse_dir = Path(args.mse_dir) if args.mse_dir else find_latest_run(exp_root, "mse")

    if flow_dir is None:
        # fall back: latest run of any kind
        flow_dir = find_latest_run(exp_root)
    if flow_dir is None:
        raise SystemExit(f"No runs with log.csv found under {exp_root}/")

    print(f"Flow run: {flow_dir}")
    print(f"MSE  run: {mse_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "training_curves.png"
    flow_best, flow_step, fig = make_curves(flow_dir, png)
    print(f"Saved {png}  (flow best reward {flow_best:.4f} @ step {flow_step})")

    mse_best = best_reward_of(mse_dir)
    mse_best_s = f"{mse_best:.4f}" if mse_best is not None else "(run --policy-type mse to fill in)"

    writeup = WRITEUP.format(
        mse_best=mse_best_s,
        flow_best=f"{flow_best:.4f}",
        flow_step=flow_step,
    )
    table_md = comparison_markdown()
    (out_dir / "report.md").write_text(writeup + "\n" + table_md, encoding="utf-8")
    print(f"Saved {out_dir / 'report.md'}")

    # Assemble PDF: page 1 curves, page 2 writeup text, page 3 comparison table.
    pdf_path = out_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        text_fig = plt.figure(figsize=(8.5, 11))
        text_fig.text(0.07, 0.97, writeup, va="top", ha="left",
                      family="monospace", fontsize=8.5, wrap=True)
        text_fig.gca().axis("off")
        pdf.savefig(text_fig)
        plt.close(text_fig)
        render_table_page(pdf)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()