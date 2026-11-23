from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from algo.encoder import Encoder, PowerPredictor, weight_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Critic(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_norm = use_norm
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim
        if encoder is not None:
            input_dim = encoder.hidden_dim + action_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()

        self.apply(weight_init)

    def forward(self, x, a, rnn_state=None):
        feat, _ = self.encode(x, rnn_state)
        feat = torch.cat([feat, a], dim=1)
        q = F.relu(self.norm1(self.fc1(feat)))
        q = F.relu(self.norm2(self.fc2(q)))
        q = self.fc3(q)
        return q.squeeze(1)

    def encode(self, x, rnn_state=None):
        if self.encoder is not None:
            feat, rnn_state = self.encoder(x, rnn_state)
        else:
            feat = x
        return feat, rnn_state


class Actor(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        min_action: float = -1.0,
        hidden_dim: int = 256,
        use_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_norm = use_norm

        input_dim = state_dim
        if encoder is not None:
            input_dim = encoder.hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.fc_out = nn.Linear(hidden_dim, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((max_action - min_action) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((max_action + min_action) / 2.0, dtype=torch.float32),
        )
        self.apply(weight_init)

    def forward(self, x, rnn_state=None):
        feat, rnn_state = self.encode(x, rnn_state)
        feat = self.fc1(feat)
        feat = self.norm1(feat)
        feat = F.relu(feat)
        feat = self.fc2(feat)
        feat = self.norm2(feat)
        feat = F.relu(feat)
        out = self.fc_out(feat)
        out = torch.tanh(out)
        action = self.action_scale * out + self.action_bias
        return action, rnn_state

    def encode(self, x, rnn_state=None):
        if self.encoder is not None:
            feat, rnn_state = self.encoder(x, rnn_state)
        else:
            feat = x
        return feat, rnn_state


class DDPG(object):
    def __init__(
        self,
        use_actor_encoder: int,
        use_critic_encoder: int,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        min_action: float = -1.0,
        use_emb: bool = False,
        use_actor_rnn: bool = True,
        use_critic_rnn: bool = True,
        rnn_num_layers: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        hidden_dim: int = 256,
        use_norm: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        actor_update_step: int = 1,
        reward_norm: float = 100.0,
        time_enc_k: int = 4,
        use_power_pred: bool = False,
        pred_lr: float = 1e-3,
        pred_loss_weight: float = 1.0,
        use_action_mask: bool = True,
        use_multi_goal: bool = False,
        n_step: int = 1,
        multi_goal_weight: float = 1.0,
        exploration_noise: float = 0.1,
    ):
        self.encoder = None
        if time_enc_k > 0:
            state_dim += 2 * time_enc_k

        if use_actor_encoder == 1:
            self.actor_encoder = Encoder(
                state_dim=state_dim,
                use_rnn=use_actor_rnn,
                rnn_num_layers=rnn_num_layers,
                hidden_dim=hidden_dim,
                use_norm=use_norm,
            ).to(device)
        else:
            self.actor_encoder = None

        if use_critic_encoder == 1:
            self.critic_encoder = Encoder(
                state_dim=state_dim,
                use_rnn=use_critic_rnn,
                rnn_num_layers=rnn_num_layers,
                hidden_dim=hidden_dim,
                use_norm=use_norm,
            ).to(device)
        else:
            self.critic_encoder = None

        self.actor = Actor(
            encoder=self.actor_encoder,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action,
            hidden_dim=hidden_dim,
            use_norm=use_norm,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(
            encoder=self.critic_encoder,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_norm=use_norm,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if use_multi_goal:
            self.critic2 = Critic(
                encoder=self.critic_encoder,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_norm=use_norm,
            ).to(device)
            self.critic2_target = copy.deepcopy(self.critic2)
            self.critic2_optimizer = torch.optim.Adam(
                self.critic2.parameters(), lr=critic_lr
            )

        if use_power_pred == 1:
            self.predictor = PowerPredictor(
                encoder=None,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_norm=use_norm,
            ).to(device)
            self.predictor_optimizer = torch.optim.Adam(
                self.predictor.parameters(), lr=pred_lr
            )

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.use_actor_rnn = use_actor_rnn
        self.use_critic_rnn = use_critic_rnn
        self.actor_update_step = actor_update_step
        self.reward_norm = reward_norm
        self.use_power_pred = use_power_pred
        self.pred_loss_weight = pred_loss_weight
        self.time_enc_k = time_enc_k
        self.use_action_mask = use_action_mask
        self.use_multi_goal = use_multi_goal
        self.multi_goal_weight = multi_goal_weight
        self.n_step = n_step
        self.exploration_noise = exploration_noise
        self.train_step = 0

    def select_action(
        self,
        state,
        time_idx=None,
        rnn_state=None,
        critic_rnn_state=None,
        deterministic=False,
    ):
        self.prep_rollout()
        with torch.no_grad():
            state = torch.as_tensor(state, device=device, dtype=torch.float32)
            if self.time_enc_k > 0:
                time_idx = torch.as_tensor(time_idx, device=device, dtype=torch.int32)
                state = self.add_time_features(state, time_idx)
            if rnn_state is not None:
                rnn_state = torch.as_tensor(
                    rnn_state, device=device, dtype=torch.float32
                )
            action, rnn_state = self.actor(state, rnn_state)
            if not deterministic:
                action += torch.normal(
                    0,
                    self.actor.action_scale * self.exploration_noise,
                    size=action.shape,
                    device=device,
                )
                action = action.clamp(self.min_action, self.max_action)
            if critic_rnn_state is not None:
                critic_rnn_state = torch.as_tensor(
                    critic_rnn_state, device=device, dtype=torch.float32
                )
                _, critic_rnn_state = self.critic.encode(state, critic_rnn_state)
            return action, rnn_state, critic_rnn_state

    def compute_q_value(self, state, action, critic_rnn_state=None):
        self.prep_rollout()
        with torch.no_grad():
            state = torch.as_tensor(state, device=device, dtype=torch.float32)
            action = torch.as_tensor(action, device=device, dtype=torch.float32)
            if self.time_enc_k > 0:
                time_idx = torch.as_tensor(time_idx, device=device, dtype=torch.int32)
                state = self.add_time_features(state, time_idx)
            if critic_rnn_state is not None:
                critic_rnn_state = torch.as_tensor(
                    critic_rnn_state, device=device, dtype=torch.float32
                )
            q = self.critic(state, action, critic_rnn_state)
            return q

    def train(self, replay_buffer):
        # Sample batches from replay buffer
        data = replay_buffer.sample(discount=self.discount)
        self.prep_training()
        losses = {}

        critic_losses = self.update_critic(data)
        losses.update(critic_losses)

        if self.use_multi_goal:
            critic2_losses = self.update_critic2(data)
            losses.update(critic2_losses)

        if self.use_power_pred == 1:
            pred_losses = self.update_predictor(data)
            losses.update(pred_losses)

        if self.train_step % self.actor_update_step == 0:
            actor_losses = self.update_actor(data)
            losses.update(actor_losses)

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.train_step += 1

        return losses

    def update_critic(self, data):
        state = data.obs
        action = data.actions
        next_state = data.next_obs
        if self.reward_norm > 0:
            reward = data.rewards / self.reward_norm
        else:
            reward = data.rewards
        done = data.dones
        action_masks = data.action_masks
        time_idx = data.times
        rnn_state = data.rnn_states
        next_rnn_state = data.next_rnn_states
        critic_rnn_state = data.critic_rnn_states
        next_critic_rnn_state = data.next_critic_rnn_states

        losses = {}

        if self.time_enc_k > 0:
            state = self.add_time_features(state, time_idx)
            next_state = self.add_time_features(next_state, time_idx)

        with torch.no_grad():
            next_action, _ = self.actor_target(next_state, next_rnn_state)
            if self.use_action_mask:
                next_action[:, 3:7] *= 1.0 - action_masks
            target_Q = self.critic_target(
                next_state, next_action, next_critic_rnn_state
            )
            target_Q = reward + (1.0 - done) * (self.discount**self.n_step) * target_Q

        if self.use_action_mask:
            action[:, 3:7] *= 1.0 - action_masks
        current_Q = self.critic(state, action, critic_rnn_state)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                # print(f"actor: {name} 梯度: {param.grad.abs().mean()}")
                losses[f"grad_critic_{name}"] = param.grad.abs().mean().item()

        self.critic_optimizer.step()

        # 打印更新后的参数值
        for name, param in self.actor.named_parameters():
            # print(f"actor: {name} 更新后的值: {param.data.abs().mean()}")
            losses[f"param_actor_{name}"] = param.data.abs().mean().item()

        losses["use_action_mask"] = self.use_action_mask
        losses["action_mask_ratio"] = (
            action_masks.sum(axis=-1) > 0
        ).sum().item() / action_masks.shape[0]
        losses["reward_scale"] = reward.abs().mean().item()
        losses["q_values"] = current_Q.mean().item()
        losses["q_qtarg_ratio"] = current_Q.mean().item() / target_Q.mean().item()
        losses["critic_loss"] = critic_loss.item()
        return losses

    def update_actor(self, data):
        state = data.obs
        request = data.requests
        action_masks = data.action_masks
        time_idx = data.times
        rnn_state = data.rnn_states
        critic_rnn_state = data.critic_rnn_states

        losses = {}

        if self.time_enc_k > 0:
            state = self.add_time_features(state, time_idx)

        action2, _ = self.actor(state, rnn_state)
        if self.use_action_mask:
            action2[:, 3:7] *= 1.0 - action_masks
        Q_pi = self.critic(state, action2, critic_rnn_state)
        actor_loss = -(Q_pi).mean()
        if self.use_multi_goal:
            mask = request > -1
            if mask.any():
                valid_state = state[mask]
                valid_action2 = action2[mask]
                if self.use_critic_rnn:
                    valid_critic_rnn_state = critic_rnn_state[mask]
                else:
                    valid_critic_rnn_state = None
                Q_pi2 = self.critic2(valid_state, valid_action2, valid_critic_rnn_state)
            else:
                Q_pi2 = torch.tensor(0.0, device=device, requires_grad=True)
            actor_loss += self.multi_goal_weight * -(Q_pi2).mean()

        if self.use_power_pred == 1:
            mask = request > -1
            if mask.any():
                valid_state = state[mask]
                valid_action2 = action2[mask]
                valid_request = request[mask]
                pred = self.predictor(valid_state, valid_action2, None)
                rep_loss = ((pred - valid_request) ** 2).mean()
            else:
                rep_loss = torch.tensor(0.0, device=device, requires_grad=True)
            actor_loss += self.pred_loss_weight * rep_loss
            losses["rep_loss"] = rep_loss.item()
            losses["rep_mask_ratio"] = mask.sum().item() / mask.shape[0]

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                losses[f"grad_actor_{name}"] = param.grad.abs().mean().item()

        self.actor_optimizer.step()

        for name, param in self.actor.named_parameters():
            losses[f"param_actor_{name}"] = param.data.abs().mean().item()

        losses["actor_loss"] = actor_loss.item()
        return losses

    def update_predictor(self, data):
        state = data.obs
        action = data.actions
        action_masks = data.action_masks
        next_state = data.next_obs
        request = data.requests
        response = data.reps
        time_idx = data.times

        losses = {}

        if self.time_enc_k > 0:
            state = self.add_time_features(state, time_idx)
            next_state = self.add_time_features(next_state, time_idx)

        if self.use_action_mask:
            action[:, 3:7] *= 1.0 - action_masks
        pred_power = self.predictor(state, action, None)
        # response 是实际功率的相反数，因此 -response 才是实际的功率
        pred_loss = F.mse_loss(pred_power, -response)
        pred_error = (pred_power + response).abs().mean()

        # Optimize the predictor
        self.predictor_optimizer.zero_grad()
        pred_loss.backward()

        # 打印梯度
        for name, param in self.predictor.named_parameters():
            if param.grad is not None:
                # print(f"pred: {name} 梯度: {param.grad.abs().mean()}")
                losses[f"grad_pred_{name}"] = param.grad.abs().mean().item()

        self.predictor_optimizer.step()

        # 打印更新后的参数值
        for name, param in self.predictor.named_parameters():
            # print(f"pred: {name} 更新后的值: {param.data.abs().mean()}")
            losses[f"param_pred_{name}"] = param.data.abs().mean().item()

        losses["pred_loss"] = pred_loss.item()
        losses["pred_error"] = pred_error.item()

        return losses

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        if self.use_power_pred == 1:
            self.predictor.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        if self.use_power_pred == 1:
            self.predictor.eval()

    def save(self, path):
        torch.save(self.actor.state_dict(), Path.joinpath(Path(path), "actor.pt"))
        torch.save(self.critic.state_dict(), Path.joinpath(Path(path), "critic.pt"))
        torch.save(
            self.actor_target.state_dict(), Path.joinpath(Path(path), "actor_target.pt")
        )
        torch.save(
            self.critic_target.state_dict(),
            Path.joinpath(Path(path), "critic_target.pt"),
        )
        if self.use_power_pred == 1:
            torch.save(
                self.predictor.state_dict(), Path.joinpath(Path(path), "predictor.pt")
            )

    def load(self, path, only_actor_critic=True):
        self.actor.load_state_dict(
            torch.load(Path.joinpath(Path(path), "actor.pt"), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load(Path.joinpath(Path(path), "critic.pt"), map_location=device)
        )
        if only_actor_critic:
            return
        self.actor_target.load_state_dict(
            torch.load(
                Path.joinpath(Path(path), "actor_target.pt"), map_location=device
            )
        )
        self.critic_target.load_state_dict(
            torch.load(
                Path.joinpath(Path(path), "critic_target.pt"), map_location=device
            )
        )
        if self.use_power_pred == 1:
            self.predictor.load_state_dict(
                torch.load(
                    Path.joinpath(Path(path), "predictor.pt"), map_location=device
                )
            )

    def add_time_features(self, state: torch.Tensor, t_idx: torch.Tensor):
        theta = 2 * math.pi * t_idx.float().unsqueeze(-1) / 96.0
        feats = [torch.sin(theta), torch.cos(theta)]
        for k in range(2, self.time_enc_k + 1):
            ktheta = k * theta
            feats += [torch.sin(ktheta), torch.cos(ktheta)]
        time_feats = torch.cat(feats, dim=-1)
        return torch.cat([state, time_feats], dim=-1)

    def update_critic2(self, data):
        state = data.obs
        request = data.requests
        action = data.actions
        next_state = data.next_obs
        if self.reward_norm > 0:
            reward = data.req_rewards / self.reward_norm
        else:
            reward = data.req_rewards
        done = data.dones
        action_masks = data.action_masks
        time_idx = data.times
        rnn_state = data.rnn_states
        next_rnn_state = data.next_rnn_states
        critic_rnn_state = data.critic_rnn_states
        next_critic_rnn_state = data.next_critic_rnn_states

        losses = {}

        if self.time_enc_k > 0:
            state = self.add_time_features(state, time_idx)
            next_state = self.add_time_features(next_state, time_idx)

        mask = request > -1
        if not mask.any():
            return losses
        valid_state = state[mask]
        valid_action = action[mask]
        valid_action_masks = action_masks[mask]
        valid_reward = reward[mask]
        valid_done = done[mask]
        valid_next_state = next_state[mask]
        if self.use_actor_rnn:
            valid_next_rnn_state = next_rnn_state[mask]
        else:
            valid_next_rnn_state = None
        if self.use_critic_rnn:
            valid_critic_rnn_state = critic_rnn_state[mask]
            valid_next_critic_rnn_state = next_critic_rnn_state[mask]
        else:
            valid_critic_rnn_state = None
            valid_next_critic_rnn_state = None
        with torch.no_grad():
            # Select action according to policy
            next_action, _ = self.actor_target(valid_next_state, valid_next_rnn_state)
            if self.use_action_mask:
                next_action[:, 3:7] *= 1.0 - valid_action_masks
            # Compute the target Q value
            target_Q = self.critic2_target(
                valid_next_state, next_action, valid_next_critic_rnn_state
            )

        # Get current Q estimates
        if self.use_action_mask:
            valid_action[:, 3:7] *= 1.0 - valid_action_masks
        current_Q = self.critic2(valid_state, valid_action, valid_critic_rnn_state)
        # Compute critic2 loss

        critic2_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()

        # 打印梯度
        for name, param in self.critic2.named_parameters():
            if param.grad is not None:
                # print(f"critic2: {name} 梯度: {param.grad.abs().mean()}")
                losses[f"grad_critic2_{name}"] = param.grad.abs().mean().item()

        self.critic2_optimizer.step()

        losses["critic2_use_action_mask"] = self.use_action_mask
        losses["critic2_action_mask_ratio"] = (
            valid_action_masks.sum(axis=-1) > 0
        ).sum().item() / valid_action_masks.shape[0]
        losses["critic2_reward_scale"] = valid_reward.abs().mean().item()
        losses["critic2_q_values"] = current_Q.mean().item()
        losses["critic2_q_qtarg_ratio"] = (
            current_Q.mean().item() / target_Q.mean().item()
        )
        losses["critic2_loss"] = critic2_loss.item()

        return losses
