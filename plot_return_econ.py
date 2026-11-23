import pickle
import os
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="按任务与算法绘制 test_return(均值±方差) 曲线"
    )
    parser.add_argument(
        "--path",
        nargs="+",
        type=str,
        help="包含若干 seed*/test/return.jsonl 的根目录（可多次提供）",
        required=True,
    )
    parser.add_argument(
        "--label",
        nargs="+",
        type=str,
        help="曲线图例标签（可多次提供；缺省则用 --algo 或路径名）",
        required=True,
    )
    parser.add_argument(
        "--task",
        nargs="+",
        type=int,
        required=True,
        help="需要绘制的 task_id 列表，例如: --task 0 1 2 3",
    )
    parser.add_argument(
        "--opf_path",
        type=str,
        help="最优潮流结果路径",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plot_18.png",
        help="输出图文件路径（png/pdf 等）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="保存图像的 DPI",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="是否对每个 seed 的序列使用滑动窗口进行平滑，再计算均值和方差",
    )
    args = parser.parse_args()

    if len(args.label) != len(args.path):
        raise ValueError("提供了 --label 时，其数量需与 --path 一致")

    return args


def find_seed_runs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    seed_dirs = []
    for name in sorted(os.listdir(root)):
        p = root / name
        if p.is_dir() and name.startswith("seed"):
            seed_dirs.append(p)
    return seed_dirs


def load_opf_results_from_jsonl(jsonl_path: Path) -> Dict[int, float]:
    opf = {}
    if not jsonl_path.exists():
        return opf
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("avg_return", None) != None and rec.get("task_id", None) != None:
                opf[rec.get("task_id")] = float(rec.get("avg_return"))
    return opf


def load_task_series_from_jsonl(jsonl_path: Path, task_id: int) -> Dict[int, float]:
    series: Dict[int, float] = {}
    if not jsonl_path.exists():
        return series
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("avg_return_econ", None) == None and rec.get("test_return_econ", None) == None:
                continue
            if rec.get("task_id") != task_id:
                continue
            step = rec.get("step")
            if step is None:
                continue
            # 兼容不同键名
            val = None
            if "avg_return_econ" in rec:
                val = rec["avg_return_econ"]
            elif "test_return_econ" in rec:
                val = rec["test_return_econ"]
            if val is None:
                continue
            if int(step) not in series:
                series[int(step)] = []
            series[int(step)].append(float(val))
    for key, val in series.items():
        series[key] = np.mean(val)
    return series


def align_and_aggregate_seed_series(seed_step2val_list: List[Dict[int, float]]):
    # 使用所有种子 step 的并集，并以 NaN 填充缺失，随后计算 nanmean/nanstd
    all_steps = sorted({s for d in seed_step2val_list for s in d.keys()})
    if len(all_steps) == 0:
        return np.array([]), np.array([]), np.array([])
    values = []
    for d in seed_step2val_list:
        arr = [d.get(s, np.nan) for s in all_steps]
        values.append(arr)
    values_np = np.array(values, dtype=float)  # [num_seeds, num_steps]
    mean = np.nanmean(values_np, axis=0)
    std = np.nanstd(values_np, axis=0)
    return np.array(all_steps, dtype=int), mean, std


def smooth_series_with_window(
    step2val: Dict[int, float], window: int = 5
) -> Dict[int, float]:
    """
    对单个 seed 的序列做滑动窗口平滑：
    - 先按 step 升序排列
    - 每个位置 i 的平滑值 = 该位置及之前最多 window-1 个位置的均值
      即 y[i] = mean(x[max(0, i-window+1) : i+1])
    """
    if not step2val:
        return step2val
    steps_sorted = sorted(step2val.keys())
    vals = np.array([step2val[s] for s in steps_sorted], dtype=float)
    n = len(vals)
    smoothed = np.zeros_like(vals)
    # 直接做前缀和，然后用窗口差分高效计算
    prefix = np.zeros(n + 1, dtype=float)
    prefix[1:] = np.cumsum(vals)
    for i in range(n):
        start = max(0, i - window + 1)
        length = i - start + 1
        window_sum = prefix[i + 1] - prefix[start]
        smoothed[i] = window_sum / length
    return {s: float(v) for s, v in zip(steps_sorted, smoothed)}


def aggregate_over_tasks(task_step_mean_list: List[Tuple[np.ndarray, np.ndarray]]):
    # 输入为[(steps_i, mean_i), ...]，对齐 step（并集），然后对任务做均值/方差
    all_steps = sorted({int(s) for steps, _ in task_step_mean_list for s in steps})
    if len(all_steps) == 0:
        return np.array([]), np.array([]), np.array([])
    mat = []
    for steps, mean in task_step_mean_list:
        mp = {int(s): float(v) for s, v in zip(steps, mean)}
        row = [mp.get(s, np.nan) for s in all_steps]
        mat.append(row)
    mat_np = np.array(mat, dtype=float)  # [num_tasks, num_steps]
    mean = np.nanmean(mat_np, axis=0)
    std = np.nanstd(mat_np, axis=0)
    return np.array(all_steps, dtype=int), mean, std


def main():
    args = parse_args()

    paths = args.path
    labels = args.label
    tasks = args.task
    opf_path = args.opf_path
    use_smooth = args.smooth
    smooth_window = 5  # N 默认值为 5

    # 收集原始结果：
    # per_algo -> per_task -> per_seed(list[dict[int,float]])
    raw_series: dict[str, dict[int, list[dict[int, float]]]] = {}
    # 同时保留用于每个 task 子图展示的均值±方差
    # per_algo -> per_task -> (steps, mean, std)
    results: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for root_path, label in zip(paths, labels):
        seed_dirs = find_seed_runs(root_path)
        if len(seed_dirs) == 0:
            print(f"[WARN] 未找到 seed* 目录: {root_path}")
        per_task = {}
        per_task_raw: dict[int, list[dict[int, float]]] = {}
        for task_id in tasks:
            seed_series = []
            for sd in seed_dirs:
                jsonl = sd / "test" / "return.jsonl"
                step2val = load_task_series_from_jsonl(jsonl, task_id)
                if len(step2val) == 0:
                    # 允许缺失
                    pass
                if use_smooth:
                    step2val = smooth_series_with_window(step2val, window=smooth_window)
                seed_series.append(step2val)
            steps, mean, std = align_and_aggregate_seed_series(seed_series)
            per_task[task_id] = (steps, mean, std)
            per_task_raw[task_id] = seed_series
        results[label] = per_task
        raw_series[label] = per_task_raw

    # 基于 SAC 的基准进行归一化，并按“种子维度”跨任务求综合表现
    # 1) 计算每个 task 的基准：SAC 在该 task 的“最终平均性能”。
    #    具体做法：取 SAC 在该 task 上所有种子的对齐均值曲线的最后一个非 NaN 点。
    baselines: dict[int, float] = {}
    # if "sac" not in raw_series:
    #     print(
    #         "[WARN] 未找到标签为 'sac' 的算法，无法执行归一化，综合表现将使用未归一化的旧逻辑。"
    #     )
    # else:
    #     sac_per_task = results.get("sac", {})
    #     for task_id in tasks:
    #         steps, mean, _std = sac_per_task.get(
    #             task_id, (np.array([]), np.array([]), np.array([]))
    #         )
    #         base = np.nan
    #         if steps.size > 0 and mean.size > 0:
    #             # 最后一个非 NaN
    #             valid_idx = np.where(~np.isnan(mean))[0]
    #             if valid_idx.size > 0:
    #                 base = float(mean[valid_idx[-1]])
    #         if np.isnan(base) or base == 0.0:
    #             print(
    #                 f"[WARN] task {task_id} 的 'sac' 基准无效({base})，该任务将被忽略在综合归一化中。"
    #             )
    #         else:
    #             baselines[task_id] = base

    # 加载最优潮流结果作为 baseline
    if opf_path is not None:
        opf_file = Path(f"{opf_path}/test/opf_returns.jsonl")
        baselines = load_opf_results_from_jsonl(opf_file)
    else:
        raise "No OPF results!"

    # 2) 计算综合表现：对每个算法，先构造“综合 seed 序列”
    # 对于每个算法，首先需要找出其所有 task 的共同 step 网格（并集），并在该步长上按种子聚合跨任务的归一化均值。
    # 做法：
    #   - 为每个 task、每个种子，先对该 seed 的序列按该 task 的 baseline 做除法得到归一化值；
    #   - 对于每个算法的第 k 个种子，按任务把该 seed 的归一化序列在同一 step 上平均（仅使用有 baseline 的任务）；
    #   - 最终得到“综合 seed 序列”列表，然后在种子维度上计算均值±方差。
    def build_seed_aggregate_over_tasks(
        per_task_seed_series: dict[int, list[dict[int, float]]],
        valid_baselines: dict[int, float],
    ):
        # 先收集所有任务的 step 并集
        all_steps = sorted(
            {
                s
                for task_id, series_list in per_task_seed_series.items()
                if task_id in valid_baselines
                for d in series_list
                for s in d.keys()
            }
        )
        if len(all_steps) == 0:
            return np.array([]), np.array([]), np.array([])
        # 推断最大种子数
        max_seeds = 0
        for task_id, series_list in per_task_seed_series.items():
            if task_id in valid_baselines:
                max_seeds = max(max_seeds, len(series_list))
        if max_seeds == 0:
            return np.array([]), np.array([]), np.array([])
        step_array = np.array(all_steps, dtype=int)
        # 收集每个种子的跨任务归一化均值曲线
        seed_curves = []  # list[[len_steps] floats]
        for seed_idx in range(max_seeds):
            rows = []
            for task_id, series_list in per_task_seed_series.items():
                if task_id not in valid_baselines:
                    continue
                baseline = abs(valid_baselines[task_id])
                if seed_idx >= len(series_list):
                    continue
                step2val = series_list[seed_idx]
                # 归一化并映射到并集 steps
                row = []
                for s in all_steps:
                    v = step2val.get(s, np.nan)
                    if np.isnan(v):
                        row.append(np.nan)
                    else:
                        row.append(v / baseline)
                rows.append(row)
            if len(rows) == 0:
                continue
            rows_np = np.array(rows, dtype=float)
            seed_curve = np.nanmean(rows_np, axis=0)  # 跨任务平均
            seed_curves.append(seed_curve)
        if len(seed_curves) == 0:
            return np.array([]), np.array([]), np.array([])
        seed_mat = np.vstack(seed_curves)  # [num_seeds, len_steps]
        mean = np.nanmean(seed_mat, axis=0)
        std = np.nanstd(seed_mat, axis=0)
        return step_array, mean, std

    overall_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label in results.keys():
        if len(baselines) == 0:
            # 无法归一化，回退到旧逻辑（基于每任务均值的再平均）
            per_task = results[label]
            task_arrays = []
            for task_id in tasks:
                steps, mean, _std = per_task.get(
                    task_id, (np.array([]), np.array([]), np.array([]))
                )
                if steps.size > 0:
                    task_arrays.append((steps, mean))
            steps_overall, mean_overall, std_overall = aggregate_over_tasks(task_arrays)
            overall_results[label] = (steps_overall, mean_overall, std_overall)
        else:
            per_task_seed = raw_series[label]
            steps_overall, mean_overall, std_overall = build_seed_aggregate_over_tasks(
                per_task_seed, baselines
            )
            overall_results[label] = (steps_overall, mean_overall, std_overall)

    # 绘图：每个 task 一个子图，最后追加一个“跨任务均值”子图
    num_subplots = len(tasks) + 1
    ncols = 4
    nrows = int(np.ceil(num_subplots / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False
    )

    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get(
            "color",
            [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ],
        )
    )

    def plot_one(ax, title, data_dict, show_legend=False):
        for idx, (label, per_task) in enumerate(data_dict.items()):
            color = color_cycle[idx % len(color_cycle)]
            if title == "mean":
                steps, mean, std = overall_results[label]
                baseline = -1.0
            else:
                task_id = title
                steps, mean, std = per_task.get(
                    task_id, (np.array([]), np.array([]), np.array([]))
                )
                baseline = baselines[task_id]
            if steps.size == 0:
                continue
            ax.plot(steps, mean, label=label, color=color, linewidth=2.0)
            ax.fill_between(
                steps, mean - std, mean + std, color=color, alpha=0.2, linewidth=0
            )
        ax.axhline(
            baseline,
            label="opf",
            color="black",
            linestyle="--",
            linewidth=2.0,
        )
        ax.annotate(
            f"{int(baseline)}",  # 文本内容，可以改成 "阈值=0.3" 等
            xy=(0, baseline),  # 目标点：x=1 在坐标轴最右侧，y=y_const 在数据坐标中
            xycoords=("axes fraction", "data"),  # x 用坐标轴比例，y 用数据坐标
            xytext=(4, -4),  # 在目标点基础上，向右/向下略微偏移几个像素
            textcoords="offset points",
            ha="left",  # 与左边框对齐
            va="top",  # 文本的顶部贴着那条线 → 看起来像在曲线下方
        )
        ax.set_title(
            f"Task {title}" if title != "mean" else "Task mean (normalized by OPF)"
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Test Return" if title != "mean" else "Normalized Return")
        ax.grid(True, linestyle=":", alpha=0.6, color=color)
        # 让横坐标总是用科学计数法，并去掉offset
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
        if title != "mean":
            id = title
            if id % 4 == 3:
                ax.set_ylim(-8000, -6000)
            else:
                ax.set_ylim(-5000, -2900)
        if title == "mean":
            ax.set_facecolor((0.9, 0.9, 0.9))
            ax.set_ylim(-1.55, -0.95)
        if show_legend:
            ax.legend(frameon=True, fontsize=9, loc="lower right")

    # 绘制每个任务
    for i, task_id in enumerate(tasks):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plot_one(ax, task_id, results, show_legend=True)

    # 最后一个总体子图
    overall_idx = len(tasks)
    r, c = divmod(overall_idx, ncols)
    ax_overall = axes[r][c]
    plot_one(ax_overall, "mean", results, show_legend=True)

    # 清理多余空轴
    for j in range(overall_idx + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] 已保存图像至: {out_path}")


if __name__ == "__main__":
    main()
