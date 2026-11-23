import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot_requests(
    input_csv: str, output_path: Optional[str] = None, show: bool = False
) -> None:
    df = pd.read_csv(input_csv)

    # Determine mod columns dynamically (columns starting with 'mod')
    mod_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("mod")]
    if not mod_cols:
        raise ValueError("No columns starting with 'mod' found in CSV.")

    # Sort mod columns by numeric suffix when available (e.g., mod1, mod2, ...)
    def _mod_sort_key(col: str):
        suffix = col[3:]
        try:
            return (0, int(suffix))
        except Exception:
            return (1, col)

    mod_cols.sort(key=_mod_sort_key)
    num_mod = len(mod_cols)

    # Extract timesteps 41..64 (1-based) => indices 40..63 (0-based)
    start_idx = 41 - 1
    end_idx_exclusive = (
        64  # since pandas iloc end is exclusive, use 64 to include 64th (0-based 63)
    )
    time_indices = np.arange(41, 65)  # for x-axis (1-based labels)

    # Create grid with 10 columns and enough rows, turn off unused axes
    if num_mod <= 20:
        num_cols = 5
        fig_weight = 14
    else:
        num_cols = 10
        fig_weight = 24
    num_rows = int(np.ceil(num_mod / num_cols))
    # Keep visual scale similar to original: ~2.4 height per row when width is 24
    fig_height = 2.4 * max(num_rows, 1)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_weight, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle("Request Curves (Timesteps 41-64)", fontsize=18)

    cmap = plt.get_cmap("tab20")

    for i, col in enumerate(mod_cols):
        r = i // num_cols
        c = i % num_cols
        ax = axes[r, c]
        series = df[col].iloc[start_idx:end_idx_exclusive].to_numpy(dtype=float)

        # Replace -1 with NaN for plotting clarity in non-response intervals
        plot_values = series.copy()
        plot_values[plot_values == -1] = np.nan

        color = cmap(i % cmap.N)
        ax.plot(time_indices, plot_values, color="#1f77b4", linewidth=2)
        ax.set_title(col, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

        # y limits include [0,1]; keep a little margin
        ax.set_ylim(-0.8, 1.1)
        ax.set_xlim(41, 64)

        # Clean ticks density for readability
        # if r < num_rows - 1:
        #     ax.set_xticklabels([])
        # if c > 0:
        #     ax.set_yticklabels([])

        # If a curve has no response in the window (all NaN), annotate
        if np.all(np.isnan(plot_values)):
            ax.text(
                0.5,
                0.5,
                "no request",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
                ha="center",
                va="center",
            )

    # Turn off any unused axes in the grid
    total_axes = num_rows * num_cols
    for i in range(num_mod, total_axes):
        r = i // num_cols
        c = i % num_cols
        axes[r, c].axis("off")

    # Common labels on outer axes
    fig.text(0.5, 0.04, "Timestep", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Power (MW)", va="center", rotation="vertical", fontsize=14)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot request curves for timesteps 41-64 as a 10x10 grid."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).with_name("request.csv")),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("request_plot_41_64.png")),
        help="Output image path",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the plot interactively"
    )
    args = parser.parse_args()

    plot_requests(input_csv=args.input, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
