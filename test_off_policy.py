import json
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
import pickle
import numpy as np
import torch
import tyro
from pathlib import Path
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_RL, IEEE123_RL_Vec
from envs.ieee_meta.shmem_vec_env import IEEE123_RL_ShmVec
from algo.sac import SAC
from algo.ddpg import DDPG
from algo.td3 import TD3


def transform(data: list[dict]) -> dict:
    return {key: np.array([d[key] for d in data]) for key in data[0]}


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "IEEE123Env-V0"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "IEEE123"
    """the environment id of the task"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2000
    """timestep to start learning"""
    actor_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    critic_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the size of the hidden layer in the policy network"""
    hidden_dim: int = 256
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    test_freq: int = 500
    log_freq: int = 10000
    save_freq: int = 10000
    save_model: bool = False
    """model save frequency"""
    log_dir: str = "exps"
    train_task_nums: tuple[int, ...] = (15,)
    test_task_nums: tuple[int, ...] = (15,)
    add_edge: bool = False
    penalty_factor: int = 1000
    algo: str = "sac"
    use_emb: int = 1
    # RNN settings
    use_actor_encoder: int = 1
    use_actor_rnn: int = 1
    use_critic_encoder: int = 1
    use_critic_rnn: int = 1
    use_norm: int = 1
    rnn_num_layers: int = 1
    req_file: str = "easy"
    use_weight: int = 1
    sampling_weight: float = 1.0
    reward_norm: float = 0
    time_enc_k: int = 4
    use_power_pred: int = 1
    pred_loss_weight: float = 1.0
    use_gen_down: int = 0
    use_action_mask: int = 1
    # 并行化环境需要补偿 off policy 算法的更新频次
    compensate_update_freq: int = 1
    use_vp_rl: int = 1
    use_multi_goal: int = 0
    multi_goal_weight: float = 0
    n_step: int = 1
    use_opf: int = 1
    opf_dir: str = "opf_results"


def compute_rep_err(request, p_ext):
    diffs = []
    for i in range(40, 64):
        if i < len(p_ext) and request[i] > -1:
            diffs.append(abs(request[i] + p_ext[i]))
    return np.mean(diffs) if len(diffs) > 0 else 0


def load_opf_action(opf_csv_file: Path):
    opf_data = pd.read_csv(opf_csv_file)
    # opf_data = opf_data.to_dict(orient="records")
    opf_stor = np.stack([opf_data["bs37"], opf_data["bs56"], opf_data["bs85"]], axis=-1)
    opf_gen = np.stack(
        [
            opf_data["gen24"],
            opf_data["gen114"],
            opf_data["gen61"],
            opf_data["gen96"],
        ],
        axis=-1,
    )
    opf_ren = np.stack(
        [
            opf_data["ren22"],
            opf_data["ren121"],
            opf_data["ren41"],
            opf_data["ren123"],
            opf_data["ren39"],
            opf_data["ren4"],
            opf_data["ren59"],
            opf_data["ren46"],
            opf_data["ren75"],
            opf_data["ren83"],
        ],
        axis=-1,
    )
    opf_vp = np.stack(
        [
            opf_data["vg66"],
            opf_data["vg90"],
            opf_data["vg107"],
            opf_data["vg79"],
            opf_data["vg33"],
        ],
        axis=-1,
    )
    opf_action = {
        "storage": -opf_stor,  # 这里要取相反数
        "gen": opf_gen,
        "ren": opf_ren,
        "vp": opf_vp,
    }
    opf_action = np.concatenate(
        [
            opf_action["storage"],
            opf_action["gen"],
            opf_action["ren"],
            opf_action["vp"],
        ],
        axis=-1,
    )
    # !!! stor_max_p 随着时间是一直在变化的 !!!
    # stor_max_p = np.array([0.3, 0.2, 0.3])
    # gen_rl_max = np.array([0.4, 0.6, 0.4, 0.8])
    # gen_rl_cr = np.array([0.3, 0.2, 0.2, 0.3])
    # ren_max_p_wm = np.array(
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.07635, 0.3702, 0.1018, 0.1026, 0.1234]
    # )
    # vp_max_p_wm = np.array([0.1308, 0.1026, 0.10296, 0.0952, 0.1287])
    # opf_action2rl = copy.deepcopy(opf_action)
    # opf_action2rl["storage"] = opf_action["storage"] / stor_max_p
    # for i in range(len(opf_action["gen"]) - 1, 1, -1):
    #     opf_action2rl["gen"][i] = (
    #         (opf_action["gen"][i] - opf_action["gen"][i - 1]) / gen_rl_max / gen_rl_cr
    #     )
    # opf_action2rl["gen"][0] = (opf_action["gen"][0]) / gen_rl_max / gen_rl_cr
    # opf_action2rl["ren"] = (0.5 - opf_action["ren"] / ren_max_p_wm) * 2
    # opf_action2rl["vp"] = (opf_action["vp"] / vp_max_p_wm - 0.5) * 2
    return opf_action


if __name__ == "__main__":

    args = tyro.cli(Args)

    # 找到以 seed${args.seed} 开头的文件夹
    seed_prefix = Path(args.log_dir)
    seed_path = []
    for p in seed_prefix.glob(f"seed{args.seed}_*"):
        if p.is_dir():
            seed_path.append(p)
    print(f"找到 {len(seed_path)} 个 seed{args.seed}_* 文件夹")
    seed_path = seed_path[0]
    print(f"使用 seed_path: {seed_path}")
    model_path = seed_path / "model"

    if args.use_opf:
        opf_path = Path(args.opf_dir)
        test_results_file = opf_path / "test" / "opf_returns.jsonl"
        if not test_results_file.parent.exists():
            test_results_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        test_results_file = seed_path / "test" / "final_returns.jsonl"
    # 创建文件
    if test_results_file.exists():
        os.remove(test_results_file)
        # raise FileExistsError(f"file already exists: {test_results_file}")
    test_results_file.touch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_gen_down = bool(args.use_gen_down)
    test_envs = IEEE123_RL_ShmVec(
        num_envs=len(args.test_task_nums),
        N_topo=5,
        N_scenerio=4,
        is_train=False,
        req_file=args.req_file,
        use_gen_down=use_gen_down,
        penalty_factor=args.penalty_factor,
        use_vp_rl=args.use_vp_rl,
        use_opf=args.use_opf,
    )
    obs_space = test_envs.single_observation_space
    act_space = test_envs.single_action_space

    # 固定 seed，保证可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_envs.seed(args.seed + 100)

    kwargs = {
        "use_actor_encoder": args.use_actor_encoder,
        "use_critic_encoder": args.use_critic_encoder,
        "state_dim": obs_space.shape[0],
        "action_dim": act_space.shape[0],
        "max_action": act_space.high[0],
        "min_action": act_space.low[0],
        "use_norm": args.use_norm,
        "use_emb": args.use_emb,
        "use_actor_rnn": args.use_actor_rnn,
        "use_critic_rnn": args.use_critic_rnn,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "reward_norm": args.reward_norm,
        "time_enc_k": args.time_enc_k,
        "use_power_pred": args.use_power_pred,
        "pred_loss_weight": args.pred_loss_weight,
        "use_action_mask": args.use_action_mask,
        "use_multi_goal": args.use_multi_goal,
        "multi_goal_weight": args.multi_goal_weight,
        "n_step": args.n_step,
    }
    if args.algo == "ddpg":
        policy = DDPG(**kwargs)
    elif args.algo == "td3":
        policy = TD3(**kwargs)
    elif args.algo == "sac":
        kwargs["autotune"] = args.autotune
        policy = SAC(**kwargs)
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")
    policy.load(model_path, only_actor_critic=True)

    start_time = time.time()

    test_returns = []
    if args.req_file == "easy":
        test_req_mods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif args.req_file == "hard":
        test_req_mods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif args.req_file == "hard2":
        test_req_mods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif args.req_file == "hard3":
        test_req_mods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif args.req_file == "hard4":
        test_req_mods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        raise ValueError(f"Invalid request file: {args.req_file}")
    if use_gen_down:
        # 0 表示发电厂正常运行，1 表示发电厂维护
        test_gendown_mods = [0, 1]
    else:
        test_gendown_mods = [0]
    all_test_returns = []
    all_test_p_exts = []
    for req_mod in test_req_mods:
        for gendown_mod in test_gendown_mods:
            if args.use_opf:
                opf_actions = []
                for id in args.test_task_nums:
                    opf_action = load_opf_action(
                        Path(
                            f"{args.opf_dir}/task{id}/task{id}_req{req_mod}_gd{gendown_mod}.csv"
                        )
                    )
                    opf_actions.append(opf_action)
                opf_actions = np.stack(opf_actions)

            test_obs, _ = test_envs.reset(
                task_nums=args.test_task_nums,
                req_mod=req_mod,
                gen_down=gendown_mod,
            )
            test_returns = np.zeros(test_envs.num_envs)
            test_returns_econ = np.zeros(test_envs.num_envs)
            test_lengths = np.zeros(test_envs.num_envs)
            test_req = [[] for _ in range(test_envs.num_envs)]
            test_p_ext = [[] for _ in range(test_envs.num_envs)]
            episode_q_values = [[] for _ in range(test_envs.num_envs)]
            episode_actions = [[] for _ in range(test_envs.num_envs)]
            episode_real_actions = [[] for _ in range(test_envs.num_envs)]
            episode_socs = [[] for _ in range(test_envs.num_envs)]
            test_episode_p_lines = [[] for _ in range(test_envs.num_envs)]
            test_reward_dicts = [[] for _ in range(test_envs.num_envs)]
            test_episode_max_p_lines = [[] for _ in range(test_envs.num_envs)]
            test_episode_outlines = [[] for _ in range(test_envs.num_envs)]
            all_envs_tested = np.zeros(test_envs.num_envs, dtype=bool)
            if args.use_actor_rnn:
                test_rnn_states = torch.zeros(
                    test_envs.num_envs,
                    args.rnn_num_layers,
                    args.hidden_dim,
                    device=device,
                )
            else:
                test_rnn_states = None
            while True:
                with torch.no_grad():
                    test_actions, test_rnn_states, _ = policy.select_action(
                        test_obs,
                        test_lengths,
                        test_rnn_states,
                        None,
                        deterministic=True,
                    )
                    test_q_values = policy.compute_q_value(test_obs, test_actions, None)
                    test_actions = test_actions.detach().cpu().numpy()
                    test_q_values = test_q_values.detach().cpu().numpy()
                if args.use_opf:
                    test_actions = opf_actions[:, int(test_lengths[0])]
                (
                    test_obs,
                    test_rewards,
                    test_terminations,
                    test_truncations,
                    test_infos,
                ) = test_envs.step(test_actions)
                # if req_mod == 1 and gendown_mod == 0:
                #     print(f"seed: {args.seed}, test_actions: {test_actions}")
                for i in range(test_envs.num_envs):
                    test_returns_econ[i] += (
                        test_rewards[i] - test_infos[i]["reward_dict"]["resp_penalty"]
                    )
                    test_req[i].append(test_infos[i]["last_request"])
                    test_p_ext[i].append(test_infos[i]["p_ext"])
                    episode_real_actions[i].append(test_infos[i]["real_action"])
                    episode_socs[i].append(test_infos[i]["soc"])
                    test_episode_p_lines[i].append(test_infos[i]["res_p_line"])  # 字典
                    test_episode_max_p_lines[i].append(
                        test_infos[i]["max_p_line"]
                    )  # 字典
                    test_reward_dicts[i].append(test_infos[i]["reward_dict"])  # 字典
                    test_episode_outlines[i].append(
                        test_infos[i]["res_p_outline"]
                    )  # 浮点值
                test_returns += test_rewards
                test_lengths += 1
                test_dones = test_terminations | test_truncations
                for i in range(test_envs.num_envs):
                    episode_q_values[i].append(test_q_values[i])
                    episode_actions[i].append(test_actions[i])
                if test_dones.any():
                    for i, task_id in enumerate(test_envs.task_nums):
                        if test_dones[i]:
                            # 保证每个任务只记录一次测试结果
                            if not all_envs_tested[i]:
                                all_envs_tested[i] = True
                            else:
                                continue

                            # 20种情况分别对应不同的 key
                            key = {
                                "task_id": task_id,
                                "req_mod": req_mod,
                                "gendown_mod": gendown_mod,
                            }

                            # 记录每局奖励和响应准确率
                            # request = test_envs.envs[task_id].request_mode["request"][
                            #     : len(test_p_ext[task_id])
                            # ]
                            all_test_returns.append(
                                {
                                    **key,
                                    "test_return": test_returns[i],
                                    "test_return_econ": test_returns_econ[i],
                                    "test_length": test_lengths[i],
                                    "test_error": compute_rep_err(
                                        test_req[i], test_p_ext[i]
                                    ),
                                    "test_outline": np.sum(test_episode_outlines[i]),
                                }
                            )
                            print(
                                f"task_id: {task_id}, req_mod: {req_mod}, gendown_mod: {gendown_mod}, "
                                f"test_length: {test_lengths[i]}, test_returns: {test_returns[i]}"
                            )
                            # 记录一些重要信息，比如响应值，动作等
                            all_test_p_exts.append(
                                {
                                    **key,
                                    "request": np.array(test_req[i]),
                                    "p_ext": np.array(test_p_ext[i]),
                                    "q_values": np.array(episode_q_values[i]),
                                    "actions": np.array(episode_actions[i]),
                                    "real_actions": np.array(episode_real_actions[i]),
                                    "socs": np.array(episode_socs[i]),
                                    "p_lines": transform(test_episode_p_lines[i]),
                                    "max_p_lines": transform(
                                        test_episode_max_p_lines[i]
                                    ),
                                    "reward_dicts": transform(test_reward_dicts[i]),
                                    "outlines": np.array(test_episode_outlines[i]),
                                }
                            )

                            test_returns[i] = 0
                            test_returns_econ[i] = 0
                            test_lengths[i] = 0
                            test_req[i] = []
                            test_p_ext[i] = []
                            episode_q_values[i] = []
                            episode_actions[i] = []
                            episode_real_actions[i] = []
                            episode_socs[i] = []
                            test_episode_p_lines[i] = []
                            test_episode_max_p_lines[i] = []
                            test_reward_dicts[i] = []
                            test_episode_outlines[i] = []
                            if args.use_actor_rnn:
                                test_rnn_states[i, :, :] = 0.0

                    if all_envs_tested.all():
                        # 所有 task 均完成测试
                        break
    with open(test_results_file, "a") as f:
        for r in all_test_returns:
            json.dump(r, f)
            f.write("\n")
        for _, task_id in enumerate(test_envs.task_nums):
            json.dump(
                {
                    "task_id": task_id,
                    "avg_return": np.mean(
                        [
                            x["test_return"]
                            for x in all_test_returns
                            if x.get("task_id") == task_id
                        ]
                    ),
                    "avg_return_econ": np.mean(
                        [
                            x["test_return_econ"]
                            for x in all_test_returns
                            if x.get("task_id") == task_id
                        ]
                    ),
                    "avg_length": np.mean(
                        [
                            x["test_length"]
                            for x in all_test_returns
                            if x.get("task_id") == task_id
                        ]
                    ),
                    "avg_error": np.mean(
                        [
                            x["test_error"]
                            for x in all_test_returns
                            if x.get("task_id") == task_id and x["req_mod"] != 1
                        ]
                    ),
                    "avg_outline": np.mean(
                        [
                            x["test_outline"]
                            for x in all_test_returns
                            if x.get("task_id") == task_id
                        ]
                    ),
                },
                f,
            )
            f.write("\n")
        # 记录所有场景的平均测试结果
        overall_return = np.mean([x["test_return"] for x in all_test_returns])
        overall_length = np.mean([x["test_length"] for x in all_test_returns])
        overall_error = np.mean(
            [x["test_error"] for x in all_test_returns if x["req_mod"] != 1]
        )
        json.dump(
            {
                "overall_return": overall_return,
                "overall_length": overall_length,
                "overall_error": overall_error,
            },
            f,
        )
        f.write("\n")
    # 覆盖保存，只记录最后一次的结果，主要是为了最终的 p_ext 和 actions
    if args.use_opf:
        final_results_file = opf_path / "test" / "opf_results.pkl"
    else:
        final_results_file = seed_path / "test" / "final_results.pkl"
    with open(final_results_file, "wb") as f:
        pickle.dump(all_test_p_exts, f)
    print(f"test results have been saved to {final_results_file}")
