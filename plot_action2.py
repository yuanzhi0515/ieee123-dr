import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _extract_seed_id(dir_name: str) -> Optional[int]:
    """Extract integer seed id from directory name like 'seed7_...'."""
    if not dir_name.startswith("seed"):
        return None
    try:
        rest = dir_name[4:]
        number_part = rest.split("_", 1)[0]
        return int(number_part)
    except Exception:
        return None


def collect_newest_run_per_seed(base_path: str) -> Dict[int, Path]:
    """Collect the newest run directory for each seed under base_path.

    Returns a mapping: seed_id -> newest Path
    """
    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(
            f"Base path does not exist or is not a directory: {base}"
        )
    newest: Dict[int, Tuple[float, Path]] = {}
    for d in base.iterdir():
        if not d.is_dir():
            continue
        sid = _extract_seed_id(d.name)
        if sid is None:
            continue
        mtime = d.stat().st_mtime
        if sid not in newest or mtime > newest[sid][0]:
            newest[sid] = (mtime, d)
    return {sid: path for sid, (mt, path) in newest.items()}


def load_test_results(seed_dir: Path) -> List[Dict]:
    """Load list of dicts from test/test_results.pkl under seed_dir."""
    pkl_path = seed_dir / "test" / "test_results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"test_results.pkl not found at: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(
            "Unexpected format in test_results.pkl; expected a list of dicts."
        )
    return data


def select_record(records: List[Dict], task: int, req_mod: int, gen_down: int) -> Dict:
    """Select the record matching task, req_mod and gendown_mod."""
    matched = [
        r
        for r in records
        if r.get("task_id") == task
        and r.get("req_mod") == req_mod
        and r.get("gendown_mod") == gen_down
    ]
    if not matched:
        # Provide helpful context in error
        available = sorted({(r.get("req_mod"), r.get("gendown_mod")) for r in records})
        raise ValueError(
            f"No matching record for req_mod={req_mod}, gen_down={gen_down}. "
            f"Available pairs: {available}"
        )
    print(f"Found {len(matched)} matching records")
    return matched[0]


def _normal_ppf(p: float) -> float:
    """Approximate inverse CDF (quantile) of standard normal using Acklam's method.

    Accurate to ~1e-9 for p in (0,1)."""
    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    # Define break-points.
    plow = 0.02425
    phigh = 1 - plow

    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        x = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

    return float(x)


def _compute_mean_and_halfwidth(
    data: np.ndarray, confidence: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and half-width (z * std/sqrt(n)) along axis=0 for data of shape (N, T, D)."""
    if data.ndim != 3:
        raise ValueError("data must have shape (N, T, D)")
    n = data.shape[0]
    mean = data.mean(axis=0)
    # ddof=1 for sample std; guard n>1
    if n > 1:
        std = data.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    z = _normal_ppf(0.5 * (1.0 + confidence))
    half_width = z * (std / np.sqrt(max(n, 1)))
    return mean, half_width


def plot_mean_with_ci(
    actions_mean: np.ndarray,
    actions_hw: np.ndarray,
    socs_mean: np.ndarray,
    socs_hw: np.ndarray,
    pext_mean: np.ndarray,
    pext_hw: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """Plot mean curves with confidence bands for actions (22), SOCs (3), and external grid (1)."""
    T = actions_mean.shape[0]
    time_axis = np.arange(1, T + 1)

    storage_m = actions_mean[:, 0:3]
    storage_h = actions_hw[:, 0:3]
    gen_m = actions_mean[:, 3:7]
    gen_h = actions_hw[:, 3:7]
    ren_m = actions_mean[:, 7:17]
    ren_h = actions_hw[:, 7:17]
    vp_m = actions_mean[:, 17:22]
    vp_h = actions_hw[:, 17:22]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    axes = axes.ravel()

    def plot_block(ax, mean_block, hw_block, title_str, ylabel):
        num_series = mean_block.shape[1]
        # Choose a large qualitative palette for many series
        cmap = plt.get_cmap("tab20")
        if "grid" in title_str:
            request = mean_block[:, 0].copy()
            # 对需求值为 -1 的位置不绘制（用 NaN 屏蔽）
            request[mean_block[:, 0] == -1] = np.nan
            ax.plot(
                time_axis,
                request,
                label=f"request",
                lw=3,
                color=cmap(0),
            )
            ax.plot(
                time_axis,
                mean_block[:, 1],
                label=f"ext grid",
                lw=2,
                color=cmap(2),
            )
            lower = mean_block[:, 1] - hw_block[:, 1]
            upper = mean_block[:, 1] + hw_block[:, 1]
            ax.fill_between(
                time_axis, lower, upper, color=cmap(2), alpha=0.18, linewidth=0
            )
        else:
            for i in range(num_series):
                color = cmap((2 * i) % 20)
                ax.plot(
                    time_axis,
                    mean_block[:, i],
                    label=f"{title_str} {i}",
                    lw=2,
                    color=color,
                )
                lower = mean_block[:, i] - hw_block[:, i]
                upper = mean_block[:, i] + hw_block[:, i]
                ax.fill_between(
                    time_axis, lower, upper, color=color, alpha=0.18, linewidth=0
                )
        ax.set_title(title_str)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time step")
        ax.grid(True, linestyle="--", alpha=0.4)
        ncol = 2 if num_series > 6 else 1
        ax.legend(fontsize=8, ncol=ncol, frameon=False)

    plot_block(axes[0], storage_m, storage_h, "Storage", "Power (MW)")
    plot_block(axes[1], gen_m, gen_h, "Gen", "Power (MW)")
    plot_block(axes[2], ren_m, ren_h, "Ren", "Power (MW)")
    plot_block(axes[3], vp_m, vp_h, "VP", "Power (MW)")
    plot_block(axes[4], socs_mean, socs_hw, "SOC", "SOC (%)")
    plot_block(axes[5], pext_mean, pext_hw, "Ext grid", "Power (MW)")

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean and confidence bands across seeds from test_results.pkl"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Base path containing seed runs"
    )
    parser.add_argument("--task", type=int, required=True, help="Task id (0-19)")
    parser.add_argument(
        "--gen_down", type=int, required=True, help="Generator down mode (0 or 1)"
    )
    parser.add_argument(
        "--req_mod", type=int, required=True, help="Request mode (e.g., 1, 11, 21, ...)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.68,
        help="Confidence level for bands (e.g., 0.68, 0.90, 0.95, 0.99)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path",
    )
    args = parser.parse_args()

    seed_map = collect_newest_run_per_seed(args.path)
    if not seed_map:
        raise FileNotFoundError(f"No seed* directories found under {args.path}")

    actions_list: List[np.ndarray] = []
    socs_list: List[np.ndarray] = []
    pext_list: List[np.ndarray] = []
    requests_list: List[np.ndarray] = []
    used_seeds: List[int] = []

    for sid, seed_dir in sorted(seed_map.items()):
        try:
            records = load_test_results(seed_dir)
            rec = select_record(
                records, task=args.task, req_mod=args.req_mod, gen_down=args.gen_down
            )

            actions = rec.get("real_actions")
            socs = rec.get("socs")
            p_ext = rec.get("p_ext")
            requests = rec.get("request")
            if actions is None or socs is None or p_ext is None or requests is None:
                print(f"Skip seed {sid}: missing required fields")
                continue
            # shapes and transforms
            if actions.shape != (96, 22):
                print(f"Skip seed {sid}: unexpected actions shape {actions.shape}")
                continue
            if socs.shape != (96, 3):
                print(f"Skip seed {sid}: unexpected socs shape {socs.shape}")
                continue
            if p_ext.shape != (96,):
                print(f"Skip seed {sid}: unexpected p_ext shape {p_ext.shape}")
                continue
            if requests.shape != (96,):
                print(f"Skip seed {sid}: unexpected requests shape {requests.shape}")
                continue

            actions_list.append(actions.astype(float))
            socs_list.append(socs.astype(float))
            pext_list.append((-p_ext.astype(float)).reshape(96, 1))
            requests_list.append(requests.astype(float).reshape(96, 1))
            used_seeds.append(sid)
        except Exception as e:
            print(f"Skip seed {sid}: {e}")
            continue

    if not actions_list:
        raise RuntimeError(
            "No valid records found across seeds for the specified filters"
        )

    actions_stack = np.stack(actions_list, axis=0)  # (N, 96, 22)
    socs_stack = np.stack(socs_list, axis=0)  # (N, 96, 3)
    pext_stack = np.stack(pext_list, axis=0)  # (N, 96, 1)
    requests_stack = np.stack(requests_list, axis=0)  # (N, 96, 1)
    pext_stack = np.concatenate([requests_stack, pext_stack], axis=-1)  # (N, 96, 2)

    actions_mean, actions_hw = _compute_mean_and_halfwidth(
        actions_stack, args.confidence
    )
    socs_mean, socs_hw = _compute_mean_and_halfwidth(socs_stack, args.confidence)
    pext_mean, pext_hw = _compute_mean_and_halfwidth(pext_stack, args.confidence)

    if args.output is not None:
        save_path = Path(args.output)
    else:
        raise ValueError("Output path is required")

    title = f"Task {args.task} (gd={args.gen_down}, req={args.req_mod}) | N={len(used_seeds)}, CI={args.confidence:.2f}"
    plot_mean_with_ci(
        actions_mean,
        actions_hw,
        socs_mean,
        socs_hw,
        pext_mean,
        pext_hw,
        save_path=save_path,
        title=title,
    )


if __name__ == "__main__":
    main()
