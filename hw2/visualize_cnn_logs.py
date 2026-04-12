#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "best_acc",
}


@dataclass
class Experiment:
    name: str
    model: str
    stage: str
    file_path: Path
    data: pd.DataFrame


def parse_name(file_path: Path) -> tuple[str, str, str]:
    stem = file_path.stem

    if stem.startswith("densenet"):
        model = "DenseNet"
    elif stem.startswith("resnext"):
        model = "ResNeXt"
    else:
        model = stem.split("_")[0].upper()

    if "cifar10_transfer" in stem:
        stage = "CIFAR-10 Transfer"
    elif "cifar100" in stem:
        stage = "CIFAR-100 Pretrain"
    elif "train" in stem:
        stage = "Base Train"
    else:
        stage = "Unknown Stage"

    name = f"{model} | {stage}"
    return name, model, stage


def load_experiments(input_dir: Path) -> list[Experiment]:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    experiments: list[Experiment] = []
    for file_path in files:
        data = pd.read_csv(file_path)
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"{file_path.name} is missing required columns: {sorted(missing)}"
            )

        name, model, stage = parse_name(file_path)
        experiments.append(
            Experiment(
                name=name,
                model=model,
                stage=stage,
                file_path=file_path,
                data=data.sort_values("epoch").reset_index(drop=True),
            )
        )

    return experiments


def apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 9),
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "axes.prop_cycle": plt.cycler(
                color=[
                    "#E64B35",  # vivid red
                    "#4DBBD5",  # cyan
                    "#00A087",  # green
                    "#3C5488",  # blue
                    "#F39B7F",  # orange
                    "#8491B4",  # indigo
                    "#91D1C2",  # mint
                    "#7E6148",  # brown
                ]
            ),
        }
    )


def plot_metric_grid(experiments: list[Experiment], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    metrics = [
        ("train_loss", "Train Loss", "Loss"),
        ("test_loss", "Test Loss", "Loss"),
        ("train_acc", "Train Accuracy", "Accuracy (%)"),
        ("test_acc", "Test Accuracy", "Accuracy (%)"),
    ]

    for idx, exp in enumerate(experiments):
        marker_step = max(1, len(exp.data) // 10)
        for ax, (column, title, ylabel) in zip(axes.flat, metrics):
            ax.plot(
                exp.data["epoch"],
                exp.data[column],
                marker="o",
                markersize=4,
                markevery=marker_step,
                alpha=0.95,
                label=exp.name,
            )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="best", frameon=True)
    fig.suptitle("CNN Training Curves Comparison", fontsize=18, weight="bold")
    fig.savefig(output_dir / "01_training_curves_overview.png", dpi=220)
    plt.close(fig)


def plot_best_acc_progress(experiments: list[Experiment], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    for exp in experiments:
        ax.plot(
            exp.data["epoch"],
            exp.data["best_acc"],
            marker="s",
            markersize=4,
            markevery=max(1, len(exp.data) // 12),
            alpha=0.95,
            label=exp.name,
        )
        final_epoch = exp.data["epoch"].iloc[-1]
        final_best = exp.data["best_acc"].iloc[-1]
        ax.scatter(final_epoch, final_best, s=60, edgecolor="black", linewidth=0.7, zorder=3)

    ax.set_title("Best Accuracy Progress by Experiment", fontsize=16, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2, frameon=True)
    fig.savefig(output_dir / "02_best_acc_progress.png", dpi=220)
    plt.close(fig)


def plot_final_score_bars(experiments: list[Experiment], output_dir: Path) -> None:
    rows = []
    for exp in experiments:
        rows.append(
            {
                "experiment": exp.name,
                "model": exp.model,
                "stage": exp.stage,
                "final_test_acc": exp.data["test_acc"].iloc[-1],
                "final_best_acc": exp.data["best_acc"].iloc[-1],
            }
        )
    score_df = pd.DataFrame(rows).sort_values(["model", "stage"]).reset_index(drop=True)

    x = range(len(score_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        score_df["final_test_acc"],
        width=width,
        color="#2E86AB",
        label="Final Test Acc",
        alpha=0.9,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        score_df["final_best_acc"],
        width=width,
        color="#F6C85F",
        label="Final Best Acc",
        alpha=0.95,
    )

    ax.set_title("Final Accuracy Comparison (High-Contrast Bars)", fontsize=16, weight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(list(x), score_df["experiment"], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig.savefig(output_dir / "03_final_accuracy_bars.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CNN training logs from CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory that contains CSV logs (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save generated plots (default: plots)",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    plot_metric_grid(experiments, args.output_dir)
    plot_best_acc_progress(experiments, args.output_dir)
    plot_final_score_bars(experiments, args.output_dir)

    print(f"Loaded {len(experiments)} CSV files from: {args.input_dir.resolve()}")
    print(f"Plots saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()