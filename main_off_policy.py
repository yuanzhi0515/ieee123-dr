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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from copy import deepcopy
from collections import deque
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

from envs.ieee_meta.ieee123_rl_env_v1 import IEEE123_RL, IEEE123_RL_Vec
from envs.ieee_meta.shmem_vec_env import IEEE123_RL_ShmVec
from utils import write2json
from algo.ddpg import DDPG
from algo.td3 import TD3
from algo.sac import SAC
from algo.vqvae import VQVAE
from algo.utils.replay_buffer import ReplayBuffer


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
    pred_lr: float = 1e-3
    use_gen_down: int = 0
    use_action_mask: int = 1
    # 并行化环境需要补偿 off policy 算法的更新频次
    compensate_update_freq: int = 1
    use_vp_rl: int = 1
    use_multi_goal: int = 0
    multi_goal_weight: float = 0
    n_step: int = 1


def compute_rep_err(request, p_ext):
    diffs = []
    for i in range(40, 64):
        if i < len(p_ext) and request[i] > -1:
            diffs.append(abs(request[i] + p_ext[i]))
    return np.mean(diffs) if len(diffs) > 0 else 0


def transform(data: list[dict]) -> dict:
    return {key: np.array([d[key] for d in data]) for key in data[0]}


if __name__ == "__main__":

    args = tyro.cli(Args)

    run_name = f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.track:
        import wandb

        wandb_api_key = os.environ["WANDB_API_KEY"]
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    write2json(f"{args.log_dir}/{run_name}", vars(args))

    # model save
    save_path = Path(f"{args.log_dir}/{run_name}/model")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # test info save
    test_path = Path(f"{args.log_dir}/{run_name}/test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    test_results_file = Path(f"{test_path}/return.jsonl")
    # 创建文件
    if test_results_file.exists():
        raise FileExistsError(f"file already exists: {test_results_file}")
    test_results_file.touch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_gen_down = bool(args.use_gen_down)
    envs = IEEE123_RL_ShmVec(
        num_envs=len(args.train_task_nums),
        N_topo=5,
        N_scenerio=4,
        is_train=True,
        req_file=args.req_file,
        use_gen_down=use_gen_down,
        penalty_factor=args.penalty_factor,
        use_vp_rl=args.use_vp_rl,
        use_opf=False,
    )
    test_envs = IEEE123_RL_ShmVec(
        num_envs=len(args.test_task_nums),
        N_topo=5,
        N_scenerio=4,
        is_train=False,
        req_file=args.req_file,
        use_gen_down=use_gen_down,
        penalty_factor=args.penalty_factor,
        use_vp_rl=args.use_vp_rl,
        use_opf=False,
    )
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    print(f"环境观测空间: {envs.observation_space}, 环境动作空间: {envs.action_space}")

    # 固定 seed，保证可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    envs.seed(args.seed)
    test_envs.seed(args.seed + 100)

    if args.use_action_mask:
        print("智能体启动动作掩码应对发电厂无效动作")
    else:
        print("智能体不使用动作掩码")

    kwargs = {
        "use_actor_encoder": args.use_actor_encoder,
        "use_critic_encoder": args.use_critic_encoder,
        "state_dim": obs_space.shape[0],
        "action_dim": act_space.shape[0],
        "max_action": act_space.high[0],
        "min_action": act_space.low[0],
        "discount": args.gamma,
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
        "pred_lr": args.pred_lr,
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

    args.total_timesteps = args.total_timesteps // envs.num_envs
    args.test_freq = args.test_freq // envs.num_envs
    args.save_freq = args.save_freq // envs.num_envs
    args.log_freq = args.log_freq // envs.num_envs
    args.learning_starts = args.learning_starts // envs.num_envs

    use_actor_rnn = bool(args.use_actor_rnn)
    if use_actor_rnn:
        rnn_states = torch.zeros(
            envs.num_envs, args.rnn_num_layers, args.hidden_dim, device=device
        )
        prev_rnn_states = rnn_states.clone()
    else:
        rnn_states = None
        prev_rnn_states = None

    use_critic_rnn = bool(args.use_critic_rnn)
    if use_critic_rnn:
        critic_rnn_states = torch.zeros(
            envs.num_envs, args.rnn_num_layers, args.hidden_dim, device=device
        )
        prev_critic_rnn_states = critic_rnn_states.clone()
    else:
        critic_rnn_states = None
        prev_critic_rnn_states = None

    # weight 是根据 request 数据出现的频次计算得到的，保证经济运行和即时响应的均衡
    use_weight = bool(args.use_weight)
    if use_weight:
        weight = args.sampling_weight
    else:
        weight = 1.0
    rb = ReplayBuffer(
        obs_shape=obs_space.shape,
        act_shape=act_space.shape,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
        n_envs=envs.num_envs,
        n_step=args.n_step,
        weight=weight,
        use_actor_rnn=use_actor_rnn,
        use_critic_rnn=use_critic_rnn,
        rnn_state_shape=rnn_states.shape if use_actor_rnn else (),
        critic_rnn_state_shape=critic_rnn_states.shape if use_critic_rnn else (),
    )
    start_time = time.time()

    if args.compensate_update_freq > 1:
        print(f"使用补偿更新，补偿更新频次: {args.compensate_update_freq}")
    else:
        args.compensate_update_freq = 1
        print("不使用补偿更新, 更新频次设置为 1")

    episode_returns = np.zeros(envs.num_envs)
    episode_lengths = np.zeros(envs.num_envs)
    episode_p_outlines = [[] for _ in range(envs.num_envs)]
    episode_reward_dicts = {}

    obs, _ = envs.reset(task_nums=args.train_task_nums)
    loss_log = {}

    for step in range(args.total_timesteps + 1):
        global_step = step * envs.num_envs
        # ALGO LOGIC: put action logic here
        if use_actor_rnn:
            prev_rnn_states = rnn_states.clone()
        if use_critic_rnn:
            prev_critic_rnn_states = critic_rnn_states.clone()
        if step < args.learning_starts:
            actions = envs.action_space.sample()
            if use_actor_rnn or use_critic_rnn:
                with torch.no_grad():
                    _, rnn_states, critic_rnn_states = policy.select_action(
                        obs, episode_lengths, prev_rnn_states, prev_critic_rnn_states
                    )
        else:
            with torch.no_grad():
                actions, rnn_states, critic_rnn_states = policy.select_action(
                    obs, episode_lengths, prev_rnn_states, prev_critic_rnn_states
                )
                actions = actions.detach().cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        # 上级电网需求值
        requests = np.array([x["last_request"] for x in infos], dtype=np.float32)
        # 下级电网响应值
        pexts = np.array([x["p_ext"] for x in infos], dtype=np.float32)
        # 需求惩罚值，多任务强化学习需要
        req_rewards = np.array(
            [x["reward_dict"]["resp_penalty"] for x in infos], dtype=np.float32
        )
        reward_dict = transform([x["reward_dict"] for x in infos])
        for i in range(envs.num_envs):
            episode_p_outlines[i].append(infos[i]["res_p_outline"])
        # 因为并行环境遇到 termination 或 truncation 会自动 reset，所以需要记录真实的 next_obs
        real_next_obs = next_obs
        if dones.any():
            real_next_obs = next_obs.copy()
            for i in range(envs.num_envs):
                if dones[i]:
                    real_next_obs[i] = infos[i]["final_obs"]

        rb.add(
            obs=obs,
            next_obs=real_next_obs,
            action=actions,
            reward=rewards,
            # 需求惩罚值，多任务强化学习需要
            req_reward=req_rewards,
            # 只需要记录真正的结束，无需记录 truncation，因为是单步训练
            done=terminations,
            request=requests,
            rep=pexts,
            time=episode_lengths,
            rnn_state=prev_rnn_states.detach().cpu().numpy() if use_actor_rnn else None,
            next_rnn_state=rnn_states.detach().cpu().numpy() if use_actor_rnn else None,
            critic_rnn_state=(
                prev_critic_rnn_states.detach().cpu().numpy()
                if use_critic_rnn
                else None
            ),
            next_critic_rnn_state=(
                critic_rnn_states.detach().cpu().numpy() if use_critic_rnn else None
            ),
        )

        obs = next_obs
        episode_returns += rewards
        episode_lengths += 1
        for key, value in reward_dict.items():
            if key not in episode_reward_dicts:
                episode_reward_dicts[key] = np.zeros(envs.num_envs)
            episode_reward_dicts[key] += value
        if dones.any():
            for i, task_id in enumerate(envs.task_nums):
                if dones[i]:
                    writer.add_scalar(
                        f"train_return/return_{task_id}",
                        episode_returns[i],
                        global_step,
                    )
                    writer.add_scalar(
                        f"train_length/length_{task_id}",
                        episode_lengths[i],
                        global_step,
                    )
                    writer.add_scalar(
                        f"train_outline/outline_{task_id}",
                        np.mean(episode_p_outlines[i]),
                        global_step,
                    )
                    # print(f"avg_reward: {episode_returns[i] / episode_lengths[i]}")
                    for key, value in episode_reward_dicts.items():
                        writer.add_scalar(
                            f"reward_{key}/{key}_{task_id}",
                            value[i] / episode_lengths[i],
                            global_step,
                        )
                    for key in episode_reward_dicts.keys():
                        episode_reward_dicts[key][i] = 0
                    episode_returns[i] = 0
                    episode_lengths[i] = 0
                    episode_p_outlines[i] = []
                    if use_actor_rnn:
                        rnn_states[i, :, :] = 0.0
                    if use_critic_rnn:
                        critic_rnn_states[i, :, :] = 0.0
                    # 向量化环境不需要显式 reset

        # ALGO LOGIC: training.
        if step > args.learning_starts:

            for _ in range(args.compensate_update_freq):
                losses = policy.train(replay_buffer=rb)

            for key, value in losses.items():
                if key not in loss_log:
                    loss_log[key] = deque(maxlen=100)
                loss_log[key].append(value)

            if step % args.log_freq == 0:
                for key, value in loss_log.items():
                    if key.startswith("grad_"):
                        writer.add_scalar(f"grads/{key}", np.mean(value), global_step)
                    elif key.startswith("param_"):
                        writer.add_scalar(f"params/{key}", np.mean(value), global_step)
                    else:
                        writer.add_scalar(f"losses/{key}", np.mean(value), global_step)
                loss_log.clear()
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            if step % args.test_freq == 0:
                # if args.save_model and step % args.save_freq == 0:
                #     torch.save(model.state_dict(), save_path / f"model_{global_step}.pt")

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
                        test_episode_actions = [[] for _ in range(test_envs.num_envs)]
                        test_episode_real_actions = [
                            [] for _ in range(test_envs.num_envs)
                        ]
                        test_episode_socs = [[] for _ in range(test_envs.num_envs)]
                        test_episode_p_lines = [[] for _ in range(test_envs.num_envs)]
                        test_reward_dicts = [[] for _ in range(test_envs.num_envs)]
                        test_episode_max_p_lines = [
                            [] for _ in range(test_envs.num_envs)
                        ]
                        test_episode_outlines = [[] for _ in range(test_envs.num_envs)]
                        all_envs_tested = np.zeros(test_envs.num_envs, dtype=bool)
                        if use_actor_rnn:
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
                                test_q_values = policy.compute_q_value(
                                    test_obs, test_actions, None
                                )
                                test_actions = test_actions.detach().cpu().numpy()
                                test_q_values = test_q_values.detach().cpu().numpy()
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
                                    test_rewards[i]
                                    - test_infos[i]["reward_dict"]["resp_penalty"]
                                )
                                test_req[i].append(test_infos[i]["last_request"])
                                test_p_ext[i].append(test_infos[i]["p_ext"])
                                test_episode_real_actions[i].append(
                                    test_infos[i]["real_action"]
                                )
                                test_episode_socs[i].append(test_infos[i]["soc"])
                                test_episode_p_lines[i].append(
                                    test_infos[i]["res_p_line"]
                                )  # 字典
                                test_episode_max_p_lines[i].append(
                                    test_infos[i]["max_p_line"]
                                )  # 字典
                                test_reward_dicts[i].append(
                                    test_infos[i]["reward_dict"]
                                )  # 字典
                                test_episode_outlines[i].append(
                                    test_infos[i]["res_p_outline"]
                                )  # 浮点值
                            test_returns += test_rewards
                            test_lengths += 1
                            test_dones = test_terminations | test_truncations
                            for i in range(test_envs.num_envs):
                                episode_q_values[i].append(test_q_values[i])
                                test_episode_actions[i].append(test_actions[i])
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
                                            "step": global_step,
                                            "task_id": task_id,
                                            "req_mod": req_mod,
                                            "gendown_mod": gendown_mod,
                                        }

                                        # 记录每局奖励和响应准确率
                                        # 这里的环境是测试模式，自动 reset 后，request_mode 不会变化
                                        # request = test_envs.envs[task_id].request_mode[
                                        #     "request"
                                        # ][: len(test_p_ext[task_id])]
                                        all_test_returns.append(
                                            {
                                                **key,
                                                "test_return": test_returns[i],
                                                "test_return_econ": test_returns_econ[
                                                    i
                                                ],
                                                "test_length": test_lengths[i],
                                                "test_error": compute_rep_err(
                                                    test_req[i],
                                                    test_p_ext[i],
                                                ),
                                                "test_outline": np.sum(
                                                    test_episode_outlines[i]
                                                ),
                                            }
                                        )
                                        # 记录一些重要信息，比如响应值，动作等
                                        all_test_p_exts.append(
                                            {
                                                **key,
                                                "request": np.array(test_req[i]),
                                                "p_ext": np.array(test_p_ext[i]),
                                                "q_values": np.array(
                                                    episode_q_values[i]
                                                ),
                                                "actions": np.array(
                                                    test_episode_actions[i]
                                                ),
                                                "real_actions": np.array(
                                                    test_episode_real_actions[i]
                                                ),
                                                "socs": np.array(test_episode_socs[i]),
                                                "p_lines": transform(
                                                    test_episode_p_lines[i]
                                                ),
                                                "max_p_lines": transform(
                                                    test_episode_max_p_lines[i]
                                                ),
                                                "reward_dicts": transform(
                                                    test_reward_dicts[i]
                                                ),
                                                "outlines": np.array(
                                                    test_episode_outlines[i]
                                                ),
                                            }
                                        )

                                        test_returns[i] = 0
                                        test_returns_econ[i] = 0
                                        test_lengths[i] = 0
                                        test_req[i] = []
                                        test_p_ext[i] = []
                                        episode_q_values[i] = []
                                        test_episode_actions[i] = []
                                        test_episode_real_actions[i] = []
                                        test_episode_socs[i] = []
                                        test_episode_p_lines[i] = []
                                        test_episode_max_p_lines[i] = []
                                        test_reward_dicts[i] = []
                                        test_episode_outlines[i] = []
                                        if use_actor_rnn:
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
                                "step": global_step,
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
                                        if x.get("task_id") == task_id
                                        and x["req_mod"] != 1
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
                    overall_return = np.mean(
                        [x["test_return"] for x in all_test_returns]
                    )
                    overall_length = np.mean(
                        [x["test_length"] for x in all_test_returns]
                    )
                    overall_error = np.mean(
                        [x["test_error"] for x in all_test_returns if x["req_mod"] != 1]
                    )
                    json.dump(
                        {
                            "step": global_step,
                            "overall_return": overall_return,
                            "overall_length": overall_length,
                            "overall_error": overall_error,
                        },
                        f,
                    )
                    f.write("\n")
                    # 覆盖保存，只记录最后一次的结果，主要是为了最终的 p_ext 和 actions
                with open(f"{test_path}/test_results.pkl", "wb") as f:
                    pickle.dump(all_test_p_exts, f)
                writer.add_scalar("test/overall_return", overall_return, global_step)
                writer.add_scalar("test/overall_length", overall_length, global_step)
                writer.add_scalar("test/overall_error", overall_error, global_step)

    policy.save(save_path)
    print(f"model has been saved in {save_path}")
    writer.close()
    envs.close()
    test_envs.close()
    if args.track:
        wandb.finish()
