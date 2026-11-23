import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def find_seed_run_dir(base_path: str, seed: int) -> Path:
    """Find the newest directory under base_path that starts with f"seed{seed}_"."""
    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(
            f"Base path does not exist or is not a directory: {base}"
        )
    prefix = f"seed{seed}_"
    candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"No directory starting with '{prefix}' found in {base}"
        )
    # Pick the most recently modified run
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制请求曲线与多个模型响应曲线对比图")
    parser.add_argument(
        "--path",
        nargs="+",
        required=True,
        help="一个或多个 .pkl 结果文件路径（顺序与 --label 对应）",
    )
    parser.add_argument("--task", type=int, required=True, help="Task id (0-19)")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to locate run directory (seed{seed}_*)",
    )
    parser.add_argument(
        "--gen_down", type=int, required=True, help="Generator down mode (0 or 1)"
    )
    parser.add_argument(
        "--label", nargs="+", required=True, help="与 --path 一一对应的标签"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="oreq_rep.png",
        help="输出图文件路径（png/pdf 等）",
    )
    args = parser.parse_args()

    paths = []
    for i in range(len(args.path)):
        seed_dir = find_seed_run_dir(args.path[i], args.seed)
        paths.append(seed_dir)

    if len(paths) != len(args.label):
        raise ValueError("--path 与 --label 数量需一致")

    # 读取多个结果文件
    datasets = []
    for p in paths:
        try:
            with open(p / "test" / "test_results.pkl", "rb") as f:
                datasets.append(pickle.load(f))
        except FileNotFoundError:
            raise (f"File not found: {p / 'test' / 'test_results.pkl'}")

    # 按 task 与 gendown_mod=0 过滤
    filtered_lists = []
    for data in datasets:
        flt = [
            d
            for d in data
            if d.get("task_id") == args.task and d.get("gendown_mod") == args.gen_down
        ]
        filtered_lists.append(flt)

    counts = [len(flt) for flt in filtered_lists]
    print(f"各路径筛选数量: {counts}（期望每条为 10）")

    # 时间步为 0-95
    t = np.arange(96)

    # 在一个 figure 中创建 10 个子图（2x5）
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), sharex=True)
    axes_flat = axes.flatten()

    total_slots = nrows * ncols
    num_items = min([total_slots] + [len(flt) for flt in filtered_lists])

    # 样式循环，支持任意数量模型
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
    linestyle_cycle = ["--", "-.", ":", "-", "--", ":", "-.", "-"]
    marker_cycle = ["o", "^", "x", "D", "v", "*", "P", "h"]

    for i in range(num_items):
        base_item = filtered_lists[0][i]

        request = np.array(base_item["request"], dtype=np.float32)
        # 对需求值为 -1 的位置不绘制（用 NaN 屏蔽）
        request_plot = request.copy()
        request_plot[request == -1] = np.nan

        ax = axes_flat[i]
        # 需求曲线（统一使用蓝色实线圆点）
        ax.plot(
            t,
            request_plot,
            color="#1f77b4",
            linewidth=2.0,
            marker="s",
            markersize=4,
            label="request",
        )

        # 多条响应曲线：来源于不同路径
        for idx, flt in enumerate(filtered_lists):
            item = flt[i]
            # 响应曲线取负值
            p_ext = -np.array(item["p_ext"], dtype=np.float32)
            color = color_cycle[idx % len(color_cycle)]
            linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
            marker = marker_cycle[idx % len(marker_cycle)]
            ax.plot(
                t,
                p_ext,
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=args.label[idx],
            )

        ax.set_xlim(40, 63)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("timestep")
        ax.set_ylabel("power(MW)")
        ax.set_title(f"req_mod={base_item.get('req_mod')}")
        ax.legend()
        ax.grid(True, linestyle=":")

    # 若不足 10 条，隐藏多余子图
    for j in range(num_items, total_slots):
        axes_flat[j].axis("off")

    # 布局与显示
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
