"""
Vẽ 2 plot cho Experiment 1 (CartPole) của HW2.

Cách dùng (chạy trong thư mục hw2, nơi có folder exp/):
    uv run python make_cartpole_plots.py
    # hoặc
    python make_cartpole_plots.py --exp-root exp --out .

Sinh ra:
    cartpole_small_batch.png   (4 đường, b=1000)
    cartpole_large_batch.png   (4 đường, b=4000)
Trục x = Train_EnvstepsSoFar, trục y = Eval_AverageReturn.
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


# prefix (phần sau "CartPole-v0_") -> nhãn hiển thị trên legend
SMALL_BATCH = {
    "cartpole":          "vanilla (no rtg, no na)",
    "cartpole_rtg":      "rtg",
    "cartpole_na":       "na",
    "cartpole_rtg_na":   "rtg + na",
}
LARGE_BATCH = {
    "cartpole_lb":         "vanilla (no rtg, no na)",
    "cartpole_lb_rtg":     "rtg",
    "cartpole_lb_na":      "na",
    "cartpole_lb_rtg_na":  "rtg + na",
}


def find_log_for_prefix(exp_root, prefix):
    """Tìm log.csv cho run có exp_name = prefix (khớp chính xác, tránh nhầm
    'cartpole' với 'cartpole_rtg' hay 'cartpole_lb')."""
    # Tên thư mục: CartPole-v0_<prefix>_sd<seed>_<timestamp>
    pattern = os.path.join(exp_root, f"CartPole-v0_{prefix}_sd*")
    matches = []
    for d in glob.glob(pattern):
        # đảm bảo phần exp_name khớp đúng prefix (không phải prefix dài hơn)
        base = os.path.basename(d)
        m = re.match(r"CartPole-v0_(.+)_sd\d+", base)
        if m and m.group(1) == prefix:
            log = os.path.join(d, "log.csv")
            if os.path.exists(log):
                matches.append((d, log))
    if not matches:
        return None
    # nếu có nhiều run cùng prefix, lấy cái mới nhất (timestamp lớn nhất)
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def make_plot(exp_root, group, title, out_path):
    plt.figure(figsize=(9, 5.5))
    plotted = 0
    for prefix, label in group.items():
        log = find_log_for_prefix(exp_root, prefix)
        if log is None:
            print(f"  [skip] khong tim thay run cho '{prefix}'")
            continue
        df = pd.read_csv(log)
        if "Train_EnvstepsSoFar" not in df or "Eval_AverageReturn" not in df:
            print(f"  [skip] '{prefix}': thieu cot can thiet")
            continue
        plt.plot(
            df["Train_EnvstepsSoFar"],
            df["Eval_AverageReturn"],
            marker="o", markersize=3, linewidth=1.5, label=label,
        )
        plotted += 1

    if plotted == 0:
        print(f"  [!] khong co run nao cho '{title}', bo qua.")
        plt.close()
        return

    plt.axhline(200, color="gray", linestyle="--", linewidth=1, label="max (200)")
    plt.xlabel("Environment steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval average return")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}  ({plotted} duong)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root", default="exp")
    p.add_argument("--out", default=".")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Small batch (b=1000):")
    make_plot(
        args.exp_root, SMALL_BATCH,
        "CartPole-v0: small batch (b=1000)",
        os.path.join(args.out, "cartpole_small_batch.png"),
    )
    print("Large batch (b=4000):")
    make_plot(
        args.exp_root, LARGE_BATCH,
        "CartPole-v0: large batch (b=4000)",
        os.path.join(args.out, "cartpole_large_batch.png"),
    )


if __name__ == "__main__":
    main()
