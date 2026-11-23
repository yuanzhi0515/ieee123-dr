import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def find_latest_seed_dirs(base_path: str) -> list:
    """Scan base_path for directories named like 'seed{num}_*' and
    return the latest directory per seed number.
    """
    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(
            f"Base path does not exist or is not a directory: {base}"
        )
    seed_pattern = re.compile(r"^seed(\d+)_")
    grouped = {}
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = seed_pattern.match(d.name)
        if not m:
            continue
        seed_id = int(m.group(1))
        prev = grouped.get(seed_id)
        if prev is None or d.stat().st_mtime > prev.stat().st_mtime:
            grouped[seed_id] = d
    # sort by seed id
    latest = [grouped[k] for k in sorted(grouped.keys())]
    if not latest:
        raise FileNotFoundError(f"No seed*_* directories found under {base}")
    return latest


def load_all_results(base_path: str) -> list:
    """Load test_results.pkl from the latest run directory of each seed under base_path.
    Return a list where each element corresponds to one seed's results list.
    """
    seed_dirs = find_latest_seed_dirs(base_path)
    results_per_seed = []
    for d in seed_dirs:
        pkl_path = d / "test" / "test_results.pkl"
        if not pkl_path.exists():
            # skip seeds without results file
            continue
        with open(pkl_path, "rb") as f:
            results_per_seed.append(pickle.load(f))
    if not results_per_seed:
        raise FileNotFoundError(
            f"No test_results.pkl found under any seed directories in {base_path}"
        )
    return results_per_seed


def z_from_confidence(conf: float) -> float:
    """Map common confidence levels to z-scores. If not matched, treat conf as direct multiplier."""
    mapping = {
        0.80: 1.2816,
        0.85: 1.4395,
        0.90: 1.6449,
        0.95: 1.9600,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    # round to 2 decimals to better match user inputs like 0.95
    key = round(conf, 2)
    return mapping.get(key, conf)


def nanmean_and_ci(data_2d: np.ndarray, z: float) -> tuple:
    """Compute mean and CI bands across axis=0 for a 2D array with NaNs.
    Returns (mean, low, high, n_eff) where CI = mean ± z * std/sqrt(n_eff) with NaN-aware counts.
    """
    mean = np.nanmean(data_2d, axis=0)
    std = np.nanstd(data_2d, axis=0, ddof=1)
    n_eff = np.sum(~np.isnan(data_2d), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(np.maximum(n_eff, 1))
        half_width = z * se
        low = mean - half_width
        high = mean + half_width
    # places with n_eff==0 should be NaN
    mean[n_eff == 0] = np.nan
    low[n_eff == 0] = np.nan
    high[n_eff == 0] = np.nan
    return mean, low, high, n_eff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot request and response mean±CI across seeds for multiple paths (labels)"
    )
    parser.add_argument(
        "--path",
        nargs="+",
        required=True,
        help="One or more base folders containing seed*_*/test/test_results.pkl (same order as --label)",
    )
    parser.add_argument("--task", type=int, required=True, help="Task id (0-19)")
    parser.add_argument(
        "--label", nargs="+", required=True, help="Labels corresponding to --path"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Confidence level (e.g., 0.95). For uncommon values it is treated as a direct multiplier for std error.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oreq_rep_mean_ci.png",
        help="Output figure file (png/pdf etc.)",
    )
    args = parser.parse_args()

    if len(args.path) != len(args.label):
        raise ValueError("--path and --label must have the same length")

    # Load all seeds' results per path
    datasets_per_path = []  # list of list[results_per_seed]
    for p in args.path:
        datasets_per_path.append(load_all_results(p))

    # Time axis
    t = np.arange(96)

    # Prepare style cycles (distinct and readable)
    color_cycle = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    linestyle_cycle = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    marker_cycle = ["o", "^", "x", "D", "v", "*", "P", "h"]

    # Subplot grid: 4x5 = 20 (first 10 for gd=0, next 10 for gd=1)
    num_req_mods = 15
    ncols = 5
    nrows = (num_req_mods // ncols) * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows*3), sharex=False)
    axes_flat = axes.flatten()

    z = z_from_confidence(args.confidence)

    # For title annotation and iteration
    gd_list = [0] * num_req_mods + [1] * num_req_mods

    # Build per subplot (req_mod index per gd)
    for sub_idx in range(nrows*ncols):
        gd = gd_list[sub_idx]
        ax = axes_flat[sub_idx]

        # For each path/label, collect across seeds and group by req_mod
        # We also need a canonical req_mod order (use the first path's first available seed as reference)
        # Determine req_mods for this gd and task from the first available dataset
        req_mod_order = None
        for results_per_seed in datasets_per_path[0]:
            items = [
                d
                for d in results_per_seed
                if d.get("task_id") == args.task and d.get("gendown_mod") == gd
            ]
            if items:
                req_mod_order = sorted({d.get("req_mod") for d in items})
                break
        if not req_mod_order:
            ax.set_axis_off()
            continue
        if sub_idx % num_req_mods >= len(req_mod_order):
            ax.set_axis_off()
            continue
        req_mod = req_mod_order[sub_idx % num_req_mods]

        # Plot request (only for the first label) with mean±CI across seeds
        first_path_results = datasets_per_path[0]
        request_stack = []
        for seed_results in first_path_results:
            matches = [
                d
                for d in seed_results
                if d.get("task_id") == args.task
                and d.get("gendown_mod") == gd
                and d.get("req_mod") == req_mod
            ]
            if not matches:
                continue
            # Expect exactly one item per (task, gd, req_mod) per seed
            request = np.array(matches[0]["request"], dtype=np.float32)
            # request2 = request.copy()
            # request2[request == -1] = np.nan
            request_stack.append(request)
        if request_stack:
            request_arr = np.vstack(request_stack)
            m, lo, hi, _ = nanmean_and_ci(request_arr, z)
            m2 = m.copy()
            m2[m == -1] = np.nan
            ax.plot(t, m2, color="#1f77b4", linewidth=2.0, marker="s", label="request")
            # ax.fill_between(t, lo, hi, color="#000000", alpha=0.12, label=f"request CI")

        # Plot response mean±CI per label
        for idx, (label, results_list) in enumerate(zip(args.label, datasets_per_path)):
            resp_stack = []
            for seed_results in results_list:
                matches = [
                    d
                    for d in seed_results
                    if d.get("task_id") == args.task
                    and d.get("gendown_mod") == gd
                    and d.get("req_mod") == req_mod
                ]
                if not matches:
                    continue
                p_ext = -np.array(matches[0]["p_ext"], dtype=np.float32)
                resp_stack.append(p_ext)
            if not resp_stack:
                continue
            resp_arr = np.vstack(resp_stack)
            m, lo, hi, _ = nanmean_and_ci(resp_arr, z)
            # 非响应曲线段不需要画出来
            m[np.isnan(m2)] = np.nan
            lo[np.isnan(m2)] = np.nan
            hi[np.isnan(m2)] = np.nan
            color = color_cycle[idx % len(color_cycle)]
            linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
            marker = marker_cycle[idx % len(marker_cycle)]
            ax.plot(
                t,
                m,
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
            ax.fill_between(t, lo, hi, color=color, alpha=0.18)

        ax.set_xlim(40, 63)
        # ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("timestep")
        ax.set_ylabel("power (MW)")
        ax.set_title(f"req_mod={req_mod} | gd={gd}")
        ax.grid(True, linestyle=":")

        # Show legend in the first subplot of each row to reduce clutter
        # if (sub_idx % ncols) == 0:
        ax.legend(ncol=1, fontsize=9)

    # Hide axes that might be empty (if any)
    for ax in axes_flat:
        if not ax.has_data():
            ax.set_axis_off()

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved figure to: {out_path}")
