import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


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
    """Select the record matching req_mod and gendown_mod. If multiple, choose the largest step."""
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
    # matched.sort(key=lambda r: r.get("step", -1))
    print(f"找到 {len(matched)} 条符合条件的记录")
    return matched[0]


def plot_actions(
    actions: np.ndarray,
    socs: np.ndarray,
    requests: np.ndarray,
    p_exts: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """Plot actions into 2x2 subplots: storage(3), gen(4), ren(10), vp(5)."""
    T = actions.shape[0]
    time_axis = np.arange(1, T + 1)

    storage = actions[:, 0:3]
    gen = actions[:, 3:7]
    ren = actions[:, 7:17]
    vp = actions[:, 17:22]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    axes = axes.ravel()

    def plot_block(ax, data, title, n_labels):
        cmap = plt.get_cmap("tab10")
        for i in range(data.shape[1]):
            color = cmap(i % 10)
            ax.plot(time_axis, data[:, i], label=f"{title} {i}", lw=2, color=color)
        ax.set_title(f"{title}")
        ax.set_ylabel("SOC (%)" if title == "Soc" else "Power (MW)")
        ax.set_xlabel("Time step")
        ax.grid(True, linestyle="--", alpha=0.4)
        # Compact legend
        ncol = 2 if data.shape[1] > 6 else 1
        ax.legend(fontsize=8, ncol=ncol, frameon=False)

    plot_block(axes[0], storage, "Storage", 3)
    plot_block(axes[1], gen, "Gen", 4)
    plot_block(axes[2], ren, "Ren", 10)
    plot_block(axes[3], vp, "VP", 5)
    plot_block(axes[4], socs, "Soc", 3)
    plot_block(axes[5], p_exts, "ext grid", 1)
    # axes[-1].axis("off")

    # axes[2].set_xlabel("Time step")
    # axes[3].set_xlabel("Time step")
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
        description="Plot action curves from test_results.pkl"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Base path containing seed runs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to locate run directory (seed{seed}_*)",
    )
    parser.add_argument("--task", type=int, required=True, help="Task id (0-19)")
    parser.add_argument(
        "--gen_down", type=int, required=True, help="Generator down mode (0 or 1)"
    )
    parser.add_argument(
        "--req_mod", type=int, required=True, help="Request mode (e.g., 1, 11, 21, ...)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path; default saves under run/test",
    )
    args = parser.parse_args()

    seed_dir = find_seed_run_dir(args.path, args.seed)
    records = load_test_results(seed_dir)
    rec = select_record(
        records, task=args.task, req_mod=args.req_mod, gen_down=args.gen_down
    )

    action_type = "real_actions"
    actions = rec.get(action_type)
    if actions is None:
        raise KeyError("Selected record does not contain 'actions'")
    if action_type == "actions":
        assert actions.shape == (96, 17)
        actions = np.concatenate([actions, np.zeros((96, 5))], axis=-1)
    elif action_type == "real_actions":
        assert actions.shape == (96, 22)

    socs = rec.get("socs")
    if socs is None:
        raise KeyError("Selected record does not contain 'socs'")
    assert socs.shape == (96, 3)

    requests = rec.get("request")
    if requests is None:
        raise KeyError("Selected record does not contain 'request'")
    assert requests.shape == (96,)
    requests = requests.reshape(-1, 1)

    p_exts = -rec.get("p_ext")
    if p_exts is None:
        raise KeyError("Selected record does not contain 'p_ext'")
    assert p_exts.shape == (96,)
    p_exts = p_exts.reshape(-1, 1)

    if args.output is not None:
        save_path = Path(args.output)
    else:
        save_name = (
            f"actions_plot_seed{args.seed}_gd{args.gen_down}_rm{args.req_mod}.png"
        )
        save_path = seed_dir / "test" / save_name

    title = (
        f"Task {args.task} (seed={args.seed}, gd={args.gen_down}, req={args.req_mod})"
    )
    plot_actions(actions, socs, requests, p_exts, save_path=save_path, title=title)


if __name__ == "__main__":
    main()
