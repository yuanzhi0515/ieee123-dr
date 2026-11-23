import numpy as np
import torch
from types import SimpleNamespace


class ReplayBuffer:
    def __init__(
        self,
        obs_shape,
        act_shape,
        buffer_size: int = 1000000,
        batch_size: int = 1024,
        device: str = "cpu",
        n_envs: int = 1,
        weight: float = 1.0,
        use_actor_rnn: bool = True,
        use_critic_rnn: bool = True,
        rnn_state_shape: tuple[int, ...] = (),
        critic_rnn_state_shape: tuple[int, ...] = (),
        n_step: int = 1,
    ):
        # 重新设置 buffer_size 为 n_envs 的倍数
        self.buffer_size = int(np.ceil(buffer_size / n_envs) * n_envs)
        print(f"buffer_size: {self.buffer_size}, n_envs: {n_envs}")
        self.device = device
        self.n_envs = n_envs
        self.weight = weight
        self.use_actor_rnn = use_actor_rnn
        self.use_critic_rnn = use_critic_rnn
        self.n_step = n_step
        self.batch_size = batch_size
        if use_actor_rnn:
            assert len(rnn_state_shape) == 3
            assert rnn_state_shape[0] == n_envs
            self.rnn_num_layers = rnn_state_shape[1]
            self.rnn_hidden_dim = rnn_state_shape[2]
        if use_critic_rnn:
            assert len(critic_rnn_state_shape) == 3
            assert critic_rnn_state_shape[0] == n_envs
            self.critic_rnn_num_layers = critic_rnn_state_shape[1]
            self.critic_rnn_hidden_dim = critic_rnn_state_shape[2]
        self.pos = 0
        self.full = False

        # np.arange 是左闭右开，右边端点不会被采样到
        self.batch_offset = np.arange(0, self.n_step * self.n_envs, self.n_envs)
        self.batch_offset = self.batch_offset.reshape(1, -1)  # (1, n_step)
        self.batch_offset = self.batch_offset.repeat(self.batch_size, 0)  # (B, n_step)

        # core transition storage
        self.obs = np.zeros((self.buffer_size,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,) + act_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.req_rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.times = np.zeros((self.buffer_size,), dtype=np.int32)
        self.requests = np.zeros((self.buffer_size,), dtype=np.float32)
        self.reps = np.zeros((self.buffer_size,), dtype=np.float32)
        # 因为确定环境中是 4 个火力发电厂，所以 action_masks 的形状为 (B, 4)
        self.action_masks = np.zeros((self.buffer_size, 4), dtype=np.float32)
        if self.weight > 1:
            self.sampling_weights = np.zeros((self.buffer_size,), dtype=np.float32)

        if self.use_actor_rnn:
            # RNN hidden state storage aligned with transitions
            self.rnn_states = np.zeros(
                (
                    self.buffer_size,
                    self.rnn_num_layers,
                    self.rnn_hidden_dim,
                ),
                dtype=np.float32,
            )
            self.next_rnn_states = np.zeros(
                (
                    self.buffer_size,
                    self.rnn_num_layers,
                    self.rnn_hidden_dim,
                ),
                dtype=np.float32,
            )

        if self.use_critic_rnn:
            self.critic_rnn_states = np.zeros(
                (
                    self.buffer_size,
                    self.critic_rnn_num_layers,
                    self.critic_rnn_hidden_dim,
                ),
                dtype=np.float32,
            )
            self.next_critic_rnn_states = np.zeros(
                (
                    self.buffer_size,
                    self.critic_rnn_num_layers,
                    self.critic_rnn_hidden_dim,
                ),
                dtype=np.float32,
            )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        req_reward: np.ndarray,
        done: np.ndarray,
        request: np.ndarray,
        rep: np.ndarray,
        time: np.ndarray = None,
        rnn_state: np.ndarray = None,
        next_rnn_state: np.ndarray = None,
        critic_rnn_state: np.ndarray = None,
        next_critic_rnn_state: np.ndarray = None,
    ) -> None:
        cur = self.pos

        obs = np.asarray(obs, dtype=np.float32).reshape(self.n_envs, -1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.n_envs, -1)
        action = np.asarray(action, dtype=np.float32).reshape(self.n_envs, -1)
        reward = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        done = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        time = np.asarray(time, dtype=np.int32).reshape(self.n_envs)

        self.obs[cur : cur + self.n_envs] = obs
        self.next_obs[cur : cur + self.n_envs] = next_obs
        self.actions[cur : cur + self.n_envs] = action
        self.rewards[cur : cur + self.n_envs] = reward
        self.req_rewards[cur : cur + self.n_envs] = req_reward
        self.dones[cur : cur + self.n_envs] = done
        self.times[cur : cur + self.n_envs] = time
        # 这里确保不要取错了索引
        assert (request >= -1).all() and (request <= 1.0).all(), "request is wrong!"
        self.requests[cur : cur + self.n_envs] = request
        self.reps[cur : cur + self.n_envs] = rep
        self.action_masks[cur : cur + self.n_envs] = obs[:, -4:]
        if self.weight > 1:
            # 对于有需求的数据，权重为 weight，否则为 1
            self.sampling_weights[cur : cur + self.n_envs] = np.where(
                request == -1.0, 1, self.weight
            )

        if self.use_actor_rnn:
            self.rnn_states[cur : cur + self.n_envs] = rnn_state.astype(np.float32)
            self.next_rnn_states[cur : cur + self.n_envs] = next_rnn_state.astype(
                np.float32
            )

        if self.use_critic_rnn:
            self.critic_rnn_states[cur : cur + self.n_envs] = critic_rnn_state.astype(
                np.float32
            )
            self.next_critic_rnn_states[cur : cur + self.n_envs] = (
                next_critic_rnn_state.astype(np.float32)
            )

        # buffer_size 是 n_envs 的倍数，所以 pos 会循环回到 0
        self.pos = (self.pos + self.n_envs) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, discount: float = 0.99):
        # 1. 确定逻辑起点和有效长度
        if self.full:
            # 满了：pos 是最老的数据，整个 buffer 都是有效区
            start = self.pos
            length = self.buffer_size
        else:
            # 没满：0 是最老的数据，pos 是数据的终点
            start = 0
            length = self.pos

        length -= self.n_envs * (self.n_step - 1)
        if length <= 0:
            raise ValueError("数据量不足以进行 N-step 采样")
        end = start + length

        if self.weight > 1:
            # 处理 sampling_weights
            # 不能用 full 作为条件，因为当 full = True 同时 end <= self.buffer_size 时会有问题。
            if end <= self.buffer_size:
                weights = self.sampling_weights[start:end]
            else:
                weights = np.concatenate(
                    [
                        self.sampling_weights[start:],
                        self.sampling_weights[: end % self.buffer_size],
                    ]
                )
            batch_inds = np.random.choice(
                length,
                size=self.batch_size,
                p=weights / np.sum(weights),
            )
            batch_inds += start
        else:
            batch_inds = np.random.randint(start, end, size=self.batch_size)

        batch_inds = batch_inds % self.buffer_size

        batch_nstep_inds = batch_inds.reshape(-1, 1)  # (B, 1)
        batch_nstep_inds = batch_nstep_inds.repeat(self.n_step, 1)  # (B, n_step)
        batch_nstep_inds = (
            batch_nstep_inds + self.batch_offset
        ) % self.buffer_size  # (B, n_step)
        # 处理 n_step 条件下的 dones、rewards、next_obs
        not_dones_nstep = 1 - self.dones[batch_nstep_inds]  # (B, n_step)
        rewards_nstep = self.rewards[batch_nstep_inds]  # (B, n_step)
        # next_obs_nstep = self.next_obs[batch_nstep_inds]  # (B, n_step)
        not_dones = np.ones((self.batch_size,))  # (B,)
        rewards = np.zeros((self.batch_size,))  # (B, obs_dim)
        # next_obs = np.zeros_like(next_obs_nstep[:, 0])  # (B,)
        dones_index = -1 * np.ones((self.batch_size,), dtype=np.int32)  # (B,)
        for i in range(self.n_step):
            rewards += (discount**i) * rewards_nstep[:, i] * not_dones
            # [:, None] 和 reshape(-1, 1) 实现相同的效果
            # next_obs = np.where(not_dones[:, None], next_obs_nstep[:, i], next_obs)
            dones_index = np.where(not_dones, i, dones_index)
            not_dones *= not_dones_nstep[:, i]
        dones = 1 - not_dones
        # 这里必须用布尔索引而不是 int 索引！否则只能索引到 0、1
        dones_index[not_dones.astype(np.bool_)] = -1

        obs = self.obs[batch_inds]
        # next_obs 是 n_step 的最后一步！
        next_obs = self.next_obs[batch_nstep_inds[:, -1]]
        actions = self.actions[batch_inds]
        # dones = self.dones[batch_inds]
        # rewards = self.rewards[batch_inds]
        req_rewards = self.req_rewards[batch_inds]
        times = self.times[batch_inds]
        requests = self.requests[batch_inds]
        reps = self.reps[batch_inds]
        action_masks = self.action_masks[batch_inds]

        rnn_states = None
        next_rnn_states = None
        if self.use_actor_rnn:
            # rnn_state 的形状为 (B, num_layers, H)
            rnn_states = self.rnn_states[batch_inds]
            next_rnn_states = self.next_rnn_states[batch_inds]

        critic_rnn_states = None
        next_critic_rnn_states = None
        if self.use_critic_rnn:
            # rnn_state 的形状为 (B, num_layers, H)
            critic_rnn_states = self.critic_rnn_states[batch_inds]
            next_critic_rnn_states = self.next_critic_rnn_states[batch_inds]

        return SimpleNamespace(
            obs=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_obs=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
            req_rewards=self.to_torch(req_rewards),
            requests=self.to_torch(requests),
            reps=self.to_torch(reps),
            action_masks=self.to_torch(action_masks),
            times=self.to_torch(times),
            rnn_states=self.to_torch(rnn_states),
            next_rnn_states=self.to_torch(next_rnn_states),
            critic_rnn_states=self.to_torch(critic_rnn_states),
            next_critic_rnn_states=self.to_torch(next_critic_rnn_states),
        )

    def to_torch(self, array):
        if array is None:
            return None
        elif isinstance(array, torch.Tensor):
            return array.to(self.device, dtype=torch.float32)
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)
