import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (run name, legend label, color)
RUNS = [
    ("fp32", "fp32 (baseline)", "#9a9a9a"),
    ("lbrq_6bit", "lbrq_6bit", "#4fc3f7"),
    ("lbrq_4bit", "lbrq_4bit", "#ffb74d"),
    ("lbrq_3bit", "lbrq_3bit", "#ef5350"),
]


def load_scalar(run_dir: Path, tag: str):
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plt.style.use("dark_background")
    fig, (ax_loss, ax_fid) = plt.subplots(1, 2, figsize=(14, 5))

    for name, label, color in RUNS:
        run_dir = args.runs_dir / name

        epochs, loss = load_scalar(run_dir, "train/loss")
        # lbrq_6bit train loss coincides with the baseline, so it is dashed to stay visible
        linestyle = "--" if name == "lbrq_6bit" else "-"
        ax_loss.plot(
            epochs, loss, color=color, linestyle=linestyle, linewidth=2, label=label
        )

        epochs, fid = load_scalar(run_dir, "eval/fid")
        ax_fid.plot(epochs, fid, color=color, linestyle="-", linewidth=2, label=label)

    ax_loss.set_title("Training loss")
    ax_loss.set_ylabel("loss")
    ax_fid.set_title("FID")
    ax_fid.set_ylabel("FID (lower is better)")

    for ax in (ax_loss, ax_fid):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
