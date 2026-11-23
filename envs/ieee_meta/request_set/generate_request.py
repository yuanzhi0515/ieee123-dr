import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random


def generate_request(
    seed: int,
    output_csv: str,
    req_min: float = -0.4,
    req_max: float = 1.0,
    time_start: int = 41,
    time_end: int = 64,
) -> None:
    """
    生成符合要求的 request.csv：
    - 列：mod1 + 常值曲线（覆盖所有可选需求值）+ 200 条随机曲线
    - 每列 120 个时间步；默认值为 -1（非需求时间段）
    - 需求时间段：固定为 [41, 64]（含）；长度为 24（亦满足 4 的整数倍）
    - 变化步长可选集：{-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2}
    - 允许值域：[req_min, req_max]（默认 [-0.4, 1.0]），写出保留 2 位小数
    - 起始值从 {req_min, req_min+0.01, ..., req_max} 中选择
    - mod1 无需求时间段（整列均为 -1）
    - 为每个可选需求值生成一条“常值曲线”：其 41..64 全为该值
    - 额外生成 200 条随机曲线：41..64 内每 4 步可按上表步长变化，且始终在 [req_min, req_max]
    - seed 保证结果可复现
    """

    # 保证可复现
    np.random.seed(seed)
    random.seed(seed)

    num_steps = 120
    # 1-based 时间步 [41, 64]，转为 0-based 索引为 [40, 63]
    req_start_min = time_start
    req_end_max = time_end
    req_index_min = req_start_min - 1
    req_index_max = req_end_max - 1

    # 允许的需求长度（必须为 4 的整数倍，且不超过 24）
    possible_lengths = np.array([4, 8, 12, 16, 20, 24], dtype=int)
    # delta_candidates = np.array([-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2])
    delta_limit = 0.26
    delta_step = 0.02
    delta_candidates = (
        np.arange(
            -delta_limit * 100, delta_limit * 100 + delta_step * 100, delta_step * 100
        )
        / 100.0
    )

    data = {}

    # mod1：无需求时间段
    data["mod1"] = [-1.0] * num_steps

    # 构建可选起始值（以及常值曲线的值）集合，步长 0.01，包含端点
    imin = int(round(req_min * 100))
    imax = int(round(req_max * 100))
    if imin > imax:
        imin, imax = imax, imin
    candidate_values = [v / 100.0 for v in range(imin, imax + 1)]

    # 常值曲线：41..64 全为该值，其余为 -1
    next_mod_idx = 2
    for v in candidate_values:
        series_const = np.full((num_steps,), -1.0, dtype=float)
        series_const[req_index_min : req_index_max + 1] = v
        data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series_const.tolist()]
        next_mod_idx += 1

    # 随机曲线：200 条，41..64 内每 4 步变化，值域限制在 [req_min, req_max]
    num_random = 200
    for _ in range(num_random):
        series = np.full((num_steps,), -1.0, dtype=float)

        # 随机选择需求长度和起始位置（长度为 4 的倍数；窗口位于 [41,64] 内）
        req_len = int(np.random.choice(possible_lengths))
        max_start_1based = req_end_max - req_len + 1
        start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
        start_idx = start_1based - 1  # 转为 0-based

        # 生成分块数
        num_blocks = req_len // 4
        # 初始化起始值：从 candidate_values 中选择
        current_value = float(np.random.choice(candidate_values))

        # 为每个 4-step block 生成值
        for b in range(num_blocks):
            if b > 0:
                # 在保证 [req_min, req_max] 的前提下选择增量
                valid_deltas = []
                for d in delta_candidates:
                    nv = current_value + float(d)
                    if req_min <= nv <= req_max:
                        valid_deltas.append(float(d))
                # 至少包含 0.0
                d = float(np.random.choice(valid_deltas))
                current_value = round(current_value + d, 2)

            # 当前块的 4 个时间步全部赋同一值
            block_start = start_idx + b * 4
            block_end = block_start + 4  # 非含尾索引
            series[block_start:block_end] = current_value

            # 有概率需求时间段不连续
            if b > 0 and num_blocks >= 4:
                rnd = np.random.random()
                if rnd < 0.1:
                    series[block_start:block_end] = -1.0

        # 赋值到列
        data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
        next_mod_idx += 1

    # 组织为 DataFrame，列名按顺序 mod1..modN
    total_mods = next_mod_idx - 1
    columns = [f"mod{i}" for i in range(1, total_mods + 1)]
    df = pd.DataFrame({col: data[col] for col in columns})

    # 写出 CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # 额外生成测试集 CSV（仅 10 列）：request_test.csv
    # 规则：
    # - mod1 全为 -1
    # - mod2-5：常值需求；mod6-10：随机需求
    # - mod2-10 需求时间段长度至少为 16（且为 4 的倍数）
    # - 起始值在 [req_min, req_max] 之间交叉均匀分布

    test_data = {}
    test_data["mod1"] = [-1.0] * num_steps
    next_mod_idx = 2

    # 常值曲线：5 条，覆盖 [req_min, req_max]
    possible_lengths_test = np.array([24], dtype=int)
    test_const_num = 5
    candidate_values = [req_min, -0.25, 0.40, 0.51, req_max]
    # delta_step = (req_max - req_min) / (test_const_num - 1)
    # delta_step = round(delta_step, 2)
    # for i in range(1, test_const_num):
    #     v = round(req_min + i * delta_step, 2)
    #     candidate_values.append(v)
    # next_mod_idx = 2

    # 5 条常值曲线（1）mod2
    start_val = req_min
    req_len = 24
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1  # 转为 0-based
    series_const = np.full((num_steps,), -1.0, dtype=float)
    series_const[start_idx : start_idx + req_len] = start_val
    # 随机缺失值
    series_const[48:52] = -1.0
    series_const[56:60] = req_min
    series_const[60:64] = req_min
    # num_blocks = req_len // 4
    # for b in range(num_blocks):
    #     if b > 0:
    #         if np.random.random() < 0.15:
    #             series_const[start_idx + b * 4 : start_idx + b * 4 + 4] = -1.0
    test_data[f"mod{next_mod_idx}"] = [
        round(float(x), 2) for x in series_const.tolist()
    ]
    next_mod_idx += 1

    # 5 条常值曲线（2）mod3
    start_val = -0.30
    req_len = 24
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1  # 转为 0-based
    series_const = np.full((num_steps,), -1.0, dtype=float)
    series_const[start_idx : start_idx + req_len] = start_val
    # 随机缺失值
    series_const[44:48] = -1.0
    series_const[48:52] = -0.16
    series_const[52:56] = -0.16
    series_const[56:60] = -0.36
    series_const[60:64] = -0.36
    # num_blocks = req_len // 4
    # for b in range(num_blocks):
    #     if b > 0:
    #         if np.random.random() < 0.15:
    #             series_const[start_idx + b * 4 : start_idx + b * 4 + 4] = -1.0
    test_data[f"mod{next_mod_idx}"] = [
        round(float(x), 2) for x in series_const.tolist()
    ]
    next_mod_idx += 1

    # 5 条常值曲线（3）mod4
    start_val = -0.31
    req_len = 24
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1  # 转为 0-based
    series_const = np.full((num_steps,), -1.0, dtype=float)
    series_const[start_idx : start_idx + req_len] = start_val
    # 随机缺失值
    series_const[48:56] = 0.12
    # series_const[44:48] = -1
    series_const[56:64] = 0.52
    # num_blocks = req_len // 4
    # for b in range(num_blocks):
    #     if b > 0:
    #         if np.random.random() < 0.15:
    #             series_const[start_idx + b * 4 : start_idx + b * 4 + 4] = -1.0
    test_data[f"mod{next_mod_idx}"] = [
        round(float(x), 2) for x in series_const.tolist()
    ]
    next_mod_idx += 1

    # 5 条常值曲线（4）mod5
    start_val = 0.51
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1  # 转为 0-based
    series_const = np.full((num_steps,), -1.0, dtype=float)
    series_const[start_idx : start_idx + req_len] = start_val
    # 随机缺失值
    series_const[48:52] = -1.0
    series_const[52:56] = 0.11
    series_const[56:64] = -0.31
    # num_blocks = req_len // 4
    # for b in range(num_blocks):
    #     if b > 0:
    #         if np.random.random() < 0.15:
    #             series_const[start_idx + b * 4 : start_idx + b * 4 + 4] = -1.0
    test_data[f"mod{next_mod_idx}"] = [
        round(float(x), 2) for x in series_const.tolist()
    ]
    next_mod_idx += 1

    # 5 条常值曲线（5）mod6
    start_val = 0.6
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1  # 转为 0-based
    series_const = np.full((num_steps,), -1.0, dtype=float)
    series_const[start_idx : start_idx + req_len] = start_val
    # 随机缺失值
    # series_const[44:48] = -1.0
    series_const[52:56] = -1.0
    # num_blocks = req_len // 4
    # for b in range(num_blocks):
    #     if b > 0:
    #         if np.random.random() < 0.15:
    #             series_const[start_idx + b * 4 : start_idx + b * 4 + 4] = -1.0
    test_data[f"mod{next_mod_idx}"] = [
        round(float(x), 2) for x in series_const.tolist()
    ]
    next_mod_idx += 1

    # 4 条困难曲线（1） ↗ mod7
    start_val = req_min
    possible_lengths_test = np.array([24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            valid_deltas = []
            ok = False
            for d in delta_candidates:
                if d >= 0.15:
                    nv = current_value + float(d)
                    if req_min <= nv <= req_max:
                        ok = True
                        valid_deltas.append(float(d))
            if not ok:
                valid_deltas.append(req_max - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（2） ↘ mod8
    start_val = req_max
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            valid_deltas = []
            ok = False
            for d in delta_candidates:
                if d <= -0.15:
                    nv = current_value + float(d)
                    if req_min <= nv <= req_max:
                        ok = True
                        valid_deltas.append(float(d))
            if not ok:
                valid_deltas.append(req_min - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（3） ↘↗ mod9
    start_val = 0.32
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            if b <= num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d <= -0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(current_value - req_min)
            elif b > num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d >= 0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_max - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[44:48] = -1.0
    series[56:60] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（4） ↗↘ mod10
    start_val = 0.0
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            if b <= num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d >= 0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_max - current_value)
            elif b > num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d <= -0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_min - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[44:48] = -1.0
    series[56:60] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（4） ↗↘ mod11
    start_val = req_min
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            if b <= num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d >= 0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_max - current_value)
            elif b > num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d <= -0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_min - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[44:48] = -1.0
    # series[52:56] = -0.23
    # series[56:60] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（3） ↘↗ mod12
    start_val = 0.54
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            if b <= num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d <= -0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(current_value - req_min)
            elif b > num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d >= 0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_max - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[44:48] = -1.0
    series[48:52] = 0.43
    series[52:56] = 0.58
    series[56:60] = 0.32
    series[60:64] = 0.51
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（4） ↗↘ mod13
    start_val = 0.12
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            if b <= num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d >= 0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_max - current_value)
            elif b > num_blocks // 2:
                valid_deltas = []
                ok = False
                for d in delta_candidates:
                    if d <= -0.15:
                        nv = current_value + float(d)
                        if req_min <= nv <= req_max:
                            ok = True
                            valid_deltas.append(float(d))
                if not ok:
                    valid_deltas.append(req_min - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[48:56] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（1） ↗ mod14
    start_val = -0.21
    possible_lengths_test = np.array([24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            valid_deltas = []
            ok = False
            for d in delta_candidates:
                if 0.20 >= d >= 0.10:
                    nv = current_value + float(d)
                    if req_min <= nv <= req_max-0.05:
                        ok = True
                        valid_deltas.append(float(d))
            if not ok:
                valid_deltas.append(req_max - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[48:52] = -1.0
    series[56:60] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 4 条困难曲线（2） ↘ mod15
    start_val = 0.42
    # possible_lengths_test = np.array([20, 24], dtype=int)
    series = np.full((num_steps,), -1.0, dtype=float)
    # 随机选择需求长度和起始位置
    req_len = int(np.random.choice(possible_lengths_test))
    max_start_1based = req_end_max - req_len + 1
    start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    start_idx = start_1based - 1
    # 随机曲线，以 start_val 为初值，每 4 步可能变化
    num_blocks = req_len // 4
    current_value = float(start_val)
    for b in range(num_blocks):
        if b > 0:
            valid_deltas = []
            ok = False
            for d in delta_candidates:
                if -0.2 <= d <= -0.10:
                    nv = current_value + float(d)
                    if req_min+0.05 <= nv <= req_max:
                        ok = True
                        valid_deltas.append(float(d))
            if not ok:
                valid_deltas.append(req_min - current_value)
            d = float(np.random.choice(valid_deltas))
            current_value = round(current_value + d, 2)

        block_start = start_idx + b * 4
        block_end = block_start + 4
        series[block_start:block_end] = current_value
        # 有概率需求时间段不连续
        # if b > 0 and num_blocks >= 4:
        #     rnd = np.random.random()
        #     if rnd < 0.1:
        #         series[block_start:block_end] = -1.0
    series[48:52] = -1.0
    series[56:60] = -1.0
    test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    next_mod_idx += 1

    # 9 个随机曲线，均匀覆盖 [req_min, req_max]，含端点
    # test_rand_num = 5
    # candidate_values = [req_min,]
    # delta_step = (req_max - req_min) / (test_rand_num - 1)
    # delta_step = round(delta_step, 2)
    # for i in range(1, test_rand_num):
    #     v = round(req_min + i * delta_step, 2)
    #     candidate_values.append(v)

    # possible_lengths_test = np.array([24], dtype=int)

    # for start_val in candidate_values:
    #     series = np.full((num_steps,), -1.0, dtype=float)

    #     # 随机选择需求长度和起始位置
    #     req_len = int(np.random.choice(possible_lengths_test))
    #     max_start_1based = req_end_max - req_len + 1
    #     start_1based = int(np.random.randint(req_start_min, max_start_1based + 1))
    #     start_idx = start_1based - 1

    #     # 随机曲线，以 start_val 为初值，每 4 步可能变化
    #     num_blocks = req_len // 4
    #     current_value = float(start_val)
    #     for b in range(num_blocks):
    #         if b > 0:
    #             valid_deltas = []
    #             for d in delta_candidates:
    #                 if abs(d) >= 0.15:
    #                     nv = current_value + float(d)
    #                     if req_min <= nv <= req_max:
    #                         valid_deltas.append(float(d))
    #             # 至少包含 0.0，随机选择一个
    #             d = float(np.random.choice(valid_deltas))
    #             current_value = round(current_value + d, 2)

    #         block_start = start_idx + b * 4
    #         block_end = block_start + 4
    #         series[block_start:block_end] = current_value

    #         # 有概率需求时间段不连续
    #         if b > 0 and num_blocks >= 5 and b < 5:
    #             rnd = np.random.random()
    #             if rnd < 0.18:
    #                 series[block_start:block_end] = -1.0

    #     test_data[f"mod{next_mod_idx}"] = [round(float(x), 2) for x in series.tolist()]
    #     next_mod_idx += 1

    # 组织测试集列顺序 mod1..mod10
    test_columns = [f"mod{i}" for i in range(1, next_mod_idx)]
    test_df = pd.DataFrame({col: test_data[col] for col in test_columns})

    test_out_path = out_path.with_name("request_test.csv")
    test_df.to_csv(test_out_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate request.csv with deterministic rules."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("request_train.csv")),
        help="Output CSV path (default: request.csv in this directory)",
    )
    parser.add_argument(
        "--req_min",
        type=float,
        default=-0.4,
        help="Minimum allowed demand value (default: -0.4)",
    )
    parser.add_argument(
        "--req_max",
        type=float,
        default=0.6,
        help="Maximum allowed demand value (default: 1.0)",
    )
    parser.add_argument(
        "--time_start",
        type=int,
        default=41,
        help="Minimum allowed demand start time (default: 41)",
    )
    parser.add_argument(
        "--time_end",
        type=int,
        default=64,
        help="Maximum allowed demand start time (default: 64)",
    )
    args = parser.parse_args()
    generate_request(
        seed=args.seed,
        output_csv=args.output,
        req_min=args.req_min,
        req_max=args.req_max,
        time_start=args.time_start,
        time_end=args.time_end,
    )


if __name__ == "__main__":
    main()
