import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在给定若干结果目录下（每个目录包含若干 seed*/test/return.jsonl），\n"
            "按需求模式 req_mod=1..N、固定 step（默认 1_000_000）统计 test_error，\n"
            "先在单个 seed 内对同一模式的多条记录求均值，再在 seed 维度上求均值，\n"
            "最终对比不同算法在各个需求模式下的平均误差（分组柱状图）。"
        )
    )
    parser.add_argument(
        "--path",
        nargs="+",
        type=str,
        required=True,
        help="一个或多个算法结果根目录，例如：/aiarena/gpfs/ieee123_results/.../td3/ sac/",
    )
    parser.add_argument(
        "--label",
        nargs="+",
        type=str,
        default=None,
        help="与 --path 一一对应的算法名称（用于图例）；未提供则使用路径名",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="1000000",
        help="筛选的 step（默认 1000000）；若设为 None 或 none 则不按 step 过滤",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=15,
        help="需求模式数量（默认 15，对应 req_mod=1..15）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出柱状图文件路径（默认保存在 --path 下，文件名为 test_error_bar.png）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="保存图像的 DPI（默认 200）",
    )
    return parser.parse_args()


def find_seed_dirs(root_dir: str) -> List[Path]:
    """
    在 root_dir 下查找所有以 'seed' 开头的子目录，并按目录名排序返回。
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"路径不存在或不是目录: {root}")

    seed_dirs: List[Path] = []
    for name in sorted(os.listdir(root)):
        p = root / name
        if p.is_dir() and name.startswith("seed"):
            seed_dirs.append(p)

    if not seed_dirs:
        raise FileNotFoundError(f"在目录下未找到任何以 'seed' 开头的子目录: {root}")
    return seed_dirs


def collect_mode_errors_for_seed(
    jsonl_path: Path, target_step: Optional[int], num_modes: int
) -> Dict[int, List[float]]:
    """
    从单个 seed 的 return.jsonl 中读取数据：
    - 仅保留 step == target_step 且 1 <= req_mod <= num_modes 的行；
    - 返回: req_mod -> [test_error1, test_error2, ...]。
    """
    mode_to_errors: Dict[int, List[float]] = {i: [] for i in range(1, num_modes + 1)}

    if not jsonl_path.exists():
        print(f"[WARN] file not found, skip: {jsonl_path}")
        return mode_to_errors

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            step_val = rec.get("step", None)
            req_mod = rec.get("req_mod", None)
            # req_mod 必须存在
            if req_mod is None:
                continue

            # 若指定了 target_step，则按 step 过滤；否则不使用 step 作为过滤条件
            if target_step is not None:
                if step_val is None:
                    continue
                try:
                    step_int = int(step_val)
                except Exception:
                    continue
                if step_int != target_step:
                    continue

            try:
                req_int = int(req_mod)
            except Exception:
                continue

            if req_int < 1 or req_int > num_modes:
                continue

            if "test_error" not in rec:
                # 若键名不匹配，可在此根据实际数据结构调整
                continue

            try:
                err = float(rec["test_error"])
            except Exception:
                continue

            mode_to_errors[req_int].append(err)

    return mode_to_errors


def aggregate_across_seeds(
    seed_dirs: List[Path], target_step: Optional[int], num_modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对所有 seed 聚合：
    1. 对于单个 seed、某个 req_mod i，先对该 seed 下所有满足条件的记录求均值；
    2. 再在 seed 维度对这些均值做平均，得到该模式下的总平均 test_error。

    返回:
        modes: 形状 [num_modes] 的数组 [1, 2, ..., num_modes]
        mean_errors: 每个模式的平均误差（float），若该模式无任何有效数据则为 np.nan。
    """
    # req_mod -> list[ per-seed mean_error ]
    mode_to_seed_means: Dict[int, List[float]] = {i: [] for i in range(1, num_modes + 1)}

    for sd in seed_dirs:
        jsonl = sd / "test" / "final_returns.jsonl"
        per_mode_errors = collect_mode_errors_for_seed(jsonl, target_step, num_modes)

        for mode_id in range(1, num_modes + 1):
            vals = per_mode_errors.get(mode_id, [])
            if not vals:
                continue
            mean_for_seed = float(np.mean(vals))
            mode_to_seed_means[mode_id].append(mean_for_seed)

    modes = np.arange(1, num_modes + 1, dtype=int)
    mean_errors = np.full_like(modes, np.nan, dtype=float)
    for mode_id in range(1, num_modes + 1):
        seed_means = mode_to_seed_means.get(mode_id, [])
        if seed_means:
            mean_errors[mode_id - 1] = float(np.mean(seed_means))

    return modes, mean_errors


def plot_bar(
    modes: np.ndarray,
    algo_to_errors: Dict[str, np.ndarray],
    output_path: Path,
    title: str,
    dpi: int = 200,
) -> None:
    """
    Plot grouped bar chart of mean test_error for different request modes and algorithms.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    algo_names = list(algo_to_errors.keys())
    num_algos = len(algo_names)
    x = np.arange(len(modes), dtype=float)  # group positions

    total_width = 0.8
    if num_algos > 0:
        bar_width = total_width / num_algos
    else:
        bar_width = total_width

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
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

    for idx, name in enumerate(algo_names):
        errors = np.asarray(algo_to_errors[name], dtype=float)
        if errors.shape[0] != modes.shape[0]:
            raise ValueError(
                f"Length mismatch between modes and errors for algo '{name}': "
                f"{modes.shape[0]} vs {errors.shape[0]}"
            )
        # center groups at x, shift each algorithm within the group
        offsets = -total_width / 2 + bar_width / 2 + idx * bar_width
        x_pos = x + offsets
        color = color_cycle[idx % len(color_cycle)]
        ax.bar(
            x_pos,
            errors,
            width=bar_width,
            label=name,
            color=color,
            edgecolor="black",
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_xlabel("Demand mode (req_mod)")
    ax.set_ylabel("Mean test error")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    if num_algos > 0:
        ax.legend(frameon=True, fontsize=9, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] 已保存柱状图至: {output_path}")


def main() -> None:
    args = parse_args()

    base_paths = args.path
    labels = args.label
    raw_step = args.step
    num_modes = args.num_modes

    # 解析 step：允许为整数或字符串 "None"/"none" 表示不按 step 过滤
    step: Optional[int]
    if raw_step is None:
        step = None
    else:
        if isinstance(raw_step, str) and raw_step.lower() == "none":
            step = None
        else:
            step = int(raw_step)

    if labels is not None and len(labels) != len(base_paths):
        raise ValueError("当提供 --label 时，其数量必须与 --path 相同。")
    if labels is None:
        labels = [Path(p).name for p in base_paths]

    algo_to_errors: Dict[str, np.ndarray] = {}
    modes_ref: np.ndarray | None = None

    for path, label in zip(base_paths, labels):
        seed_dirs = find_seed_dirs(path)
        print(f"[INFO] 在 {path} 下为算法 '{label}' 找到 {len(seed_dirs)} 个 seed 目录")

        modes, mean_errors = aggregate_across_seeds(seed_dirs, step, num_modes)

        if modes_ref is None:
            modes_ref = modes
        else:
            if not np.array_equal(modes_ref, modes):
                raise ValueError("不同路径返回的 modes 不一致，请检查数据。")

        algo_to_errors[label] = mean_errors

        # 打印当前算法在各模式下的平均误差（包括可能为 NaN 的情况）
        print(
            f"\n=== Algorithm: {label} | Mean test_error per demand mode "
            f"(per-seed mean, then mean over seeds) ==="
        )
        for mode_id, err in zip(modes, mean_errors):
            if np.isnan(err):
                print(f"req_mod={mode_id:2d}: 无有效数据")
            else:
                print(f"req_mod={mode_id:2d}: mean_test_error={err:.6f}")

    if modes_ref is None:
        raise RuntimeError("未能从任何路径中读取到有效的 modes。")

    # 绘制柱状图
    if args.output is not None:
        out_path = Path(args.output)
    else:
        # 默认保存到第一个路径下
        out_path = Path(base_paths[0]) / "test_error_bar.png"

    step_str = str(step) if step is not None else "all"
    title = f"Mean test_error over seeds @ step={step_str}"
    plot_bar(modes_ref, algo_to_errors, out_path, title, dpi=args.dpi)


if __name__ == "__main__":
    main()
