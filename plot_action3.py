import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple
import pandas as pd
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


def load_opf_test_results(seed_dir: Path) -> List[Dict]:
    """Load list of dicts from test/opf_results.pkl under seed_dir."""
    pkl_path = seed_dir / "test" / "opf_results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"opf_results.pkl not found at: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(
            "Unexpected format in opf_results.pkl; expected a list of dicts."
        )
    return data


def load_opf_results(file_path: Path):
    if not file_path.exists():
        print("不存在对应的最优潮流结果文件！")
        return None
    pd_data = pd.read_csv(file_path, encoding="utf-8")
    data = {}
    for col in pd_data.columns:
        data[col] = pd_data[col].values
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


def plot_row_on_axes(
    axes_row,
    actions: dict,
    socs: dict,
    grid: dict,
    lines: Dict[str, dict],
    rewards: Dict[str, dict],
    opf_data: Dict[str, np.ndarray],
) -> None:
    """在给定的一行6个子图轴上绘制一行数据。"""
    T = 96
    time_axis = np.arange(1, T + 1)

    storage = {}
    storage["mean"] = actions["mean"][:, 0:3]
    storage["hw"] = actions["hw"][:, 0:3]
    gen = {}
    gen["mean"] = actions["mean"][:, 3:7]
    gen["hw"] = actions["hw"][:, 3:7]
    ren = {}
    ren["mean"] = actions["mean"][:, 7:17]
    ren["hw"] = actions["hw"][:, 7:17]
    vp = {}
    vp["mean"] = actions["mean"][:, 17:22]
    vp["hw"] = actions["hw"][:, 17:22]
    soc = socs
    grid = grid
    lines = lines
    rewards = rewards

    def plot_block(ax, dict1, title_str, ylabel):
        # num_series = mean_block.shape[1]
        num_series = 0
        cmap = plt.get_cmap("tab20")
        if "line" in title_str:
            num_series = 3
            ax.plot(
                time_axis,
                dict1["max_p_line"],
                label="max_p_line",
                lw=3,
                linestyle="--",
                color=cmap(0),
            )
            ax.plot(
                time_axis,
                -dict1["max_p_line"],
                label="min_p_line",
                lw=3,
                linestyle="--",
                color=cmap(0),
            )
            ax.plot(
                time_axis, dict1["p_line_mean"], label="p_line", lw=2, color=cmap(2)
            )
            lower = dict1["p_line_mean"] - dict1["p_line_hw"]
            upper = dict1["p_line_mean"] + dict1["p_line_hw"]
            ax.fill_between(
                time_axis, lower, upper, color=cmap(2), alpha=0.18, linewidth=0
            )
            if opf_data is not None:
                opf_key = "line" + f"{title_str[5:].replace('-', '_')}"
                if opf_key in opf_data.keys():
                    ax.plot(
                        time_axis,
                        opf_data[opf_key],
                        label="opf_" + opf_key,
                        lw=2,
                        linestyle="--",
                        marker="x",
                        color="black",
                    )
                else:
                    print(f"No opf data for {opf_key}")
        elif "grid" in title_str:
            num_series = 3
            request = dict1["request"].copy()
            request[dict1["request"] == -1] = np.nan
            ax.plot(time_axis, request, label="request", lw=3, color=cmap(0))
            ax.plot(
                time_axis, dict1["pext_mean"], label="ext grid", lw=2, color=cmap(2)
            )
            lower = dict1["pext_mean"] - dict1["pext_hw"]
            upper = dict1["pext_mean"] + dict1["pext_hw"]
            ax.fill_between(
                time_axis, lower, upper, color=cmap(2), alpha=0.18, linewidth=0
            )
            if opf_data is not None:
                opf_key = "ext116"
                ax.plot(
                    time_axis,
                    -opf_data[opf_key],
                    label="opf_" + opf_key,
                    lw=2,
                    linestyle="--",
                    marker="x",
                    color="black",
                )
                # opf_key = "ext116buy"
                # ax.plot(
                #     time_axis,
                #     -opf_data[opf_key],
                #     label="opf_" + opf_key,
                #     lw=2,
                #     linestyle="--",
                #     # marker="x",
                #     color=cmap(4),
                # )
                # opf_key = "ext116sell"
                # ax.plot(
                #     time_axis,
                #     opf_data[opf_key],
                #     label="opf_" + opf_key,
                #     lw=2,
                #     linestyle="--",
                #     # marker="x",
                #     color=cmap(6),
                # )
        elif "Cost" in title_str:
            num_series = 2
            ax.plot(time_axis, dict1["mean"], label="cost", lw=2, color=cmap(2))
            lower = dict1["mean"] - dict1["hw"]
            upper = dict1["mean"] + dict1["hw"]
            ax.fill_between(
                time_axis, lower, upper, color=cmap(2), alpha=0.18, linewidth=0
            )
            if opf_data is not None:
                opf_key = "cost_vg"
                if "total" in title_str:
                    opf_key = "cost_total"
                elif "gen" in title_str:
                    opf_key = "cost_dg"
                elif "ext" in title_str:
                    opf_key = "cost_grid"
                elif "pl" in title_str:
                    opf_key = "cost_line"
                elif "cur" in title_str:
                    opf_key = "cost_curtail"
                ax.plot(
                    time_axis,
                    -4 * opf_data[opf_key],
                    label="opf_" + opf_key,
                    lw=2,
                    linestyle="--",
                    marker="x",
                    color="black",
                )
        else:
            num_series = dict1["mean"].shape[1]
            for i in range(num_series):
                color = cmap((2 * i) % 20)
                ax.plot(
                    time_axis,
                    dict1["mean"][:, i],
                    label=f"{title_str} {i}",
                    lw=2,
                    color=color,
                )
                lower = dict1["mean"][:, i] - dict1["hw"][:, i]
                upper = dict1["mean"][:, i] + dict1["hw"][:, i]
                ax.fill_between(
                    time_axis, lower, upper, color=color, alpha=0.18, linewidth=0
                )
            i = 0
            if opf_data is not None:
                if title_str.lower() == "storage":
                    for key in opf_data.keys():
                        if key.startswith("bs"):
                            color = cmap((2 * i) % 20)
                            ax.plot(
                                time_axis,
                                -opf_data[key],
                                label="opf_" + key,
                                lw=2,
                                linestyle="--",
                                marker="x",
                                color=color,
                            )
                            i += 1
                            num_series += 1
                elif title_str.lower() == "gen":
                    for key in opf_data.keys():
                        if key.startswith("gen"):
                            color = cmap((2 * i) % 20)
                            ax.plot(
                                time_axis,
                                opf_data[key],
                                label="opf_" + key,
                                lw=2,
                                linestyle="--",
                                marker="x",
                                color=color,
                            )
                            i += 1
                            num_series += 1
                elif title_str.lower() == "ren":
                    for key in opf_data.keys():
                        if key.startswith("ren"):
                            color = cmap((2 * i) % 20)
                            ax.plot(
                                time_axis,
                                opf_data[key],
                                label="opf_" + key,
                                lw=2,
                                linestyle="--",
                                marker="x",
                                color=color,
                            )
                            i += 1
                            num_series += 1
                elif title_str.lower() == "vp":
                    for key in opf_data.keys():
                        if key.startswith("vg"):
                            color = cmap((2 * i) % 20)
                            ax.plot(
                                time_axis,
                                opf_data[key],
                                label="opf_" + key,
                                lw=2,
                                linestyle="--",
                                marker="x",
                                color=color,
                            )
                            i += 1
                            num_series += 1
                elif title_str.lower() == "soc":
                    for key in opf_data.keys():
                        if key.startswith("soc"):
                            color = cmap((2 * i) % 20)
                            ax.plot(
                                time_axis,
                                opf_data[key],
                                label="opf_" + key,
                                lw=2,
                                linestyle="--",
                                marker="x",
                                color=color,
                            )
                            i += 1
                            num_series += 1
                else:
                    print(f"No opf data for {title_str}")
        ax.set_title(title_str)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time step")
        ax.grid(True, linestyle="--", alpha=0.4)
        ncol = 2 if num_series >= 6 else 1
        ax.legend(fontsize=8, ncol=ncol, frameon=False)

    plot_block(axes_row[0][0], storage, "Storage", "Power (MW)")
    plot_block(axes_row[0][1], gen, "Gen", "Power (MW)")
    plot_block(axes_row[0][2], ren, "Ren", "Power (MW)")
    plot_block(axes_row[0][3], vp, "VP", "Power (MW)")
    plot_block(axes_row[0][4], soc, "SOC", "SOC (%)")
    plot_block(axes_row[0][5], grid, "Ext grid", "Power (MW)")
    # 第二行绘制曲线功率
    i = 0
    for key in lines.keys():
        if i > 5:
            print(f"Too many lines: {len(lines.keys())}, skipping.")
            break
        plot_block(
            axes_row[1][i],
            lines[key],
            "line " + key,
            "Power (MW)",
        )
        i += 1
    # 第三行绘制成本曲线
    # plot_block(axes_row[2][0], rewards["reward"], "Cost total", "RMB")
    # plot_block(axes_row[2][1], rewards["cost_gen"], "Cost gen", "RMB")
    # plot_block(axes_row[2][2], rewards["cost_ext"], "Cost ext", "RMB")
    # plot_block(axes_row[2][3], rewards["cost_pl"], "Cost pl", "RMB")
    # plot_block(axes_row[2][4], rewards["cost_cur"], "Cost cur", "RMB")
    # plot_block(axes_row[2][5], rewards["cost_vp"], "Cost vp", "RMB")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean and confidence bands across seeds from test_results.pkl"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Base path containing seed runs"
    )
    parser.add_argument(
        "--opf_path", type=str, required=True, help="Base path containing opf results"
    )
    parser.add_argument("--task", type=int, required=True, help="Task id (0-19)")
    parser.add_argument(
        "--req_mod",
        type=int,
        nargs="+",
        required=True,
        help="多个请求模式 (例如: 1 2 5 8)",
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

    req_list: List[int] = list(args.req_mod)
    n_rows = len(req_list) * 2 * 2  # 每个 req_mod 两行：gd=0 和 gd=1
    n_cols = 6

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(42, 3.8 * n_rows + 2), sharex=False
    )
    # 保证 axes 为二维 (n_rows, n_cols)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    row_idx = 0
    any_row_plotted = False

    opf_test_data = load_opf_test_results(
        Path("/aiarena/gpfs/ieee123_results/opf_results/hard3_opf_1128/")
    )

    for req in req_list:
        for gd in (0, 1):
            actions_list: List[np.ndarray] = []
            socs_list: List[np.ndarray] = []
            pext_list: List[np.ndarray] = []
            requests_list: List[np.ndarray] = []
            p_lines_list: List[np.ndarray] = {}
            max_p_lines_list: List[np.ndarray] = {}
            reward_dicts_list: List[np.ndarray] = {}
            used_seeds: List[int] = []

            opf_data = load_opf_results(
                Path(args.opf_path)
                / f"task{args.task}"
                / f"task{args.task}_req{req}_gd{gd}.csv"
            )

            for sid, seed_dir in sorted(seed_map.items()):
                try:
                    records = load_test_results(seed_dir)
                    rec = select_record(
                        records, task=args.task, req_mod=req, gen_down=gd
                    )

                    actions = rec.get("real_actions")
                    socs = rec.get("socs")
                    p_ext = rec.get("p_ext")
                    requests = rec.get("request")
                    p_lines = rec.get("p_lines")  # 字典
                    max_p_lines = rec.get("max_p_lines")  # 字典
                    reward_dicts = rec.get("reward_dicts")  # 字典
                    if (
                        actions is None
                        or socs is None
                        or p_ext is None
                        or requests is None
                        or p_lines is None
                        or max_p_lines is None
                        or reward_dicts is None
                    ):
                        print(
                            f"Skip seed {sid} (req={req}, gd={gd}): missing required fields"
                        )
                        continue
                    if actions.shape != (96, 22):
                        print(
                            f"Skip seed {sid} (req={req}, gd={gd}): unexpected actions shape {actions.shape}"
                        )
                        continue
                    if socs.shape != (96, 3):
                        print(
                            f"Skip seed {sid} (req={req}, gd={gd}): unexpected socs shape {socs.shape}"
                        )
                        continue
                    if p_ext.shape != (96,):
                        print(
                            f"Skip seed {sid} (req={req}, gd={gd}): unexpected p_ext shape {p_ext.shape}"
                        )
                        continue
                    if requests.shape != (96,):
                        print(
                            f"Skip seed {sid} (req={req}, gd={gd}): unexpected requests shape {requests.shape}"
                        )
                        continue
                    for key, value in p_lines.items():
                        if value.shape != (96,):
                            print(
                                f"Skip seed {sid} (req={req}, gd={gd}): unexpected p_lines shape {value.shape}"
                            )
                            continue
                    for key, value in max_p_lines.items():
                        if value.shape != (96,):
                            print(
                                f"Skip seed {sid} (req={req}, gd={gd}): unexpected max_p_lines shape {value.shape}"
                            )
                            continue
                    for key, value in reward_dicts.items():
                        if value.shape != (96,):
                            print(
                                f"Skip seed {sid} (req={req}, gd={gd}): unexpected reward_dicts shape {value.shape}"
                            )
                            continue

                    actions_list.append(actions.astype(float))
                    socs_list.append(socs.astype(float))
                    pext_list.append((-p_ext.astype(float)).reshape(96, 1))
                    requests_list.append(requests.astype(float).reshape(96, 1))
                    for key, value in p_lines.items():
                        if key not in p_lines_list:
                            p_lines_list[key] = []
                        p_lines_list[key].append(value.astype(float).reshape(96, 1))
                    for key, value in max_p_lines.items():
                        if key not in max_p_lines_list:
                            max_p_lines_list[key] = []
                        max_p_lines_list[key].append(value.astype(float).reshape(96, 1))
                    for key, value in reward_dicts.items():
                        if key not in reward_dicts_list:
                            reward_dicts_list[key] = []
                        reward_dicts_list[key].append(
                            value.astype(float).reshape(96, 1)
                        )
                    used_seeds.append(sid)
                except Exception as e:
                    print(f"Skip seed {sid} (req={req}, gd={gd}): {e}")
                    continue

            if not actions_list:
                print(
                    f"No valid records for req_mod={req}, gen_down={gd}; this row will be empty."
                )
                row_idx += 1
                continue

            actions_stack = np.stack(actions_list, axis=0)  # (N, 96, 22)
            socs_stack = np.stack(socs_list, axis=0)  # (N, 96, 3)
            pext_stack = np.stack(pext_list, axis=0)  # (N, 96, 1)
            requests_stack = np.stack(requests_list, axis=0)  # (N, 96, 1)
            # pext_stack = np.concatenate(
            #     [requests_stack, pext_stack], axis=-1
            # )  # (N, 96, 2)
            for key in p_lines_list.keys():
                p_lines_list[key] = np.stack(p_lines_list[key], axis=0)  # (N, 96, 1)
                max_p_lines_list[key] = np.stack(
                    max_p_lines_list[key], axis=0
                )  # (N, 96, 1)
                # p_lines_list[key] = np.concatenate(
                #     [max_p_lines_list[key], p_lines_list[key]], axis=-1
                # )  # (N, 96, 2)
            for key in reward_dicts_list.keys():
                reward_dicts_list[key] = np.stack(
                    reward_dicts_list[key], axis=0
                )  # (N, 96, 1)

            actions = {}
            actions["mean"], actions["hw"] = _compute_mean_and_halfwidth(
                actions_stack, args.confidence
            )
            socs = {}
            socs["mean"], socs["hw"] = _compute_mean_and_halfwidth(
                socs_stack, args.confidence
            )
            grid = {}
            grid["pext_mean"], grid["pext_hw"] = _compute_mean_and_halfwidth(
                pext_stack, args.confidence
            )
            grid["pext_mean"] = grid["pext_mean"][:, 0]
            grid["pext_hw"] = grid["pext_hw"][:, 0]
            grid["request"], _ = _compute_mean_and_halfwidth(
                requests_stack, args.confidence
            )
            grid["request"] = grid["request"][:, 0]
            lines = {}
            for key in p_lines_list.keys():
                lines[key] = {}
                lines[key]["p_line_mean"], lines[key]["p_line_hw"] = (
                    _compute_mean_and_halfwidth(p_lines_list[key], args.confidence)
                )
                lines[key]["max_p_line"], _ = _compute_mean_and_halfwidth(
                    max_p_lines_list[key], args.confidence
                )
                lines[key]["p_line_mean"] = lines[key]["p_line_mean"][:, 0]
                lines[key]["p_line_hw"] = lines[key]["p_line_hw"][:, 0]
                lines[key]["max_p_line"] = lines[key]["max_p_line"][:, 0]

            rewards = {}
            for key in reward_dicts_list.keys():
                rewards[key] = {}
                rewards[key]["mean"], rewards[key]["hw"] = _compute_mean_and_halfwidth(
                    reward_dicts_list[key], args.confidence
                )
                rewards[key]["mean"] = rewards[key]["mean"][:, 0]
                rewards[key]["hw"] = rewards[key]["hw"][:, 0]

            plot_row_on_axes(
                axes[row_idx : row_idx + 3],
                actions,
                socs,
                grid,
                lines,
                rewards,
                opf_data,
            )
            # 在该行第一个子图顶部标注行信息
            axes[row_idx][0].text(
                0.01,
                1.08,
                f"req={req}, gd={gd} | N={len(used_seeds)}, CI={args.confidence:.2f}",
                transform=axes[row_idx][0].transAxes,
                fontsize=12,
                va="bottom",
            )
            any_row_plotted = True
            row_idx += 2

    if not any_row_plotted:
        raise RuntimeError(
            "No valid records found for any (req_mod, gen_down) combination."
        )

    if args.output is not None:
        save_path = Path(args.output)
    else:
        raise ValueError("Output path is required")

    fig.suptitle(
        f"Task {args.task} | req_mods={req_list} | rows: gd=0, 1",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()
