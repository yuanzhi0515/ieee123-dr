import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from copy import deepcopy
import random

from envs.ieee_meta.ieee123_meta_env_v1 import IEEE123_Meta


class IEEE123_RL(IEEE123_Meta):
    def __init__(
        self,
        N_topo=5,
        N_scenerio=4,
        is_train=True,
        penalty_factor=1000,
        req_file="easy",
        use_gen_down=False,
        use_vp_rl=True,
        use_opf=False,
        **args
    ):
        super().__init__(
            N_topo=N_topo,
            N_scenerio=N_scenerio,
            is_train=is_train,
            penalty_factor=penalty_factor,
            req_file=req_file,
            use_gen_down=use_gen_down,
            use_vp_rl=use_vp_rl,
            use_opf=use_opf,
            **args
        )
        self.is_train = is_train
        self.penalty_factor = penalty_factor
        self.req_file = req_file
        self.use_gen_down = use_gen_down
        self.use_vp_rl = use_vp_rl
        self.use_opf = use_opf
        state, info = super().reset(task_num=1)
        state = self.preprocess_obs(state)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=state.shape)

    def preprocess_obs(self, obs, add_edge=True, add_request=True):
        tmp = []
        tmp.append(obs["rt"]["node_real_time"].reshape(-1))
        tmp.append(obs["rt"]["price_real_time"])
        tmp.append(np.array([obs["rt"]["time"]], dtype=float))
        for e in obs["pre"]["node"]:
            tmp.append(e.reshape(-1))
        tmp.extend(obs["pre"]["price"])
        # if len(obs["pre"]["node"]) == 3:
        #     tmp.append(np.zeros(15))
        if add_edge:
            tmp.append(np.array(obs["rt"]["edge"].reshape(-1), dtype=float))
        if add_request:
            tmp.append(np.array(obs["rt"]["request"].reshape(-1), dtype=float))
        tmp.append(obs["rt"]["request_real_time"])
        tmp.append(obs["rt"]["gen_down_mask_real_time"])
        return np.concatenate(tmp)

    def reset(self, task_num, req_mod=1, gen_down=False):
        state, info = super().reset(
            task_num,
            req_mod=req_mod,
            gen_down=gen_down,
        )
        state = self.preprocess_obs(state)

        return state, info

    def step(self, action):
        state, reward, done, tu, info = super().step(action)
        state = self.preprocess_obs(state)

        return state, reward, done, tu, info

    def seed(self, seed):
        # 这里的随机数很重要，主要针对的是多进程环境，因为主进程的 seed 不会影响子进程。
        np.random.seed(seed)
        random.seed(seed)
        # 这个实现参考的是 gymnasium 的源码：
        # https://gymnasium.farama.org/_modules/gymnasium/core/#Env.reset
        # 这个函数是用来设置随机种子，保证可复现性。但其实没有用，因为这个自定义环境没有使用过
        # gymnasium 内部的随机数，全是自己用 numpy 生成的。
        self._np_random, self._np_random_seed = seeding.np_random(seed)


class IEEE123_RL_Vec:
    def __init__(
        self,
        num_envs=1,
        N_topo=5,
        N_scenerio=4,
        is_train=True,
        penalty_factor=1000,
        req_file="easy",
        use_gen_down=False,
        use_vp_rl=True,
        use_opf=False,
        **args
    ):
        self.envs = [
            IEEE123_RL(
                N_topo,
                N_scenerio,
                is_train,
                penalty_factor,
                req_file=req_file,
                use_gen_down=use_gen_down,
                use_vp_rl=use_vp_rl,
                use_opf=use_opf,
            )
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.is_train = is_train
        self.penalty_factor = penalty_factor
        self.req_file = req_file
        self.use_gen_down = use_gen_down
        self.use_vp_rl = use_vp_rl
        self.use_opf = use_opf
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        obs_shape = tuple([self.num_envs] + list(self.single_observation_space.shape))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=obs_shape)
        act_shape = tuple([self.num_envs] + list(self.single_action_space.shape))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=act_shape)
        self.task_nums = []
        self.req_mod = 1
        self.gen_down = False

    def reset(self, task_nums, req_mod=1, gen_down=False):
        assert len(task_nums) == self.num_envs
        states = []
        self.task_nums = deepcopy(task_nums)
        self.req_mod = req_mod
        self.gen_down = gen_down
        infos = []
        for i in range(self.num_envs):
            state, info = self.envs[i].reset(
                task_nums[i],
                req_mod=req_mod,
                gen_down=gen_down,
            )
            states.append(state)
            infos.append(info)

        return np.array(states, dtype=float), infos

    def step(self, actions):
        assert len(actions) == self.num_envs
        states = []
        rewards = []
        dones = []
        tus = []
        infos = []
        for i in range(self.num_envs):
            state, reward, done, tu, info = self.envs[i].step(actions[i])
            rewards.append(reward)
            dones.append(done)
            tus.append(tu)

            if done or tu:
                info["final_obs"] = state.copy()
                state, _ = self.envs[i].reset(
                    self.task_nums[i],
                    req_mod=self.req_mod,
                    gen_down=self.gen_down,
                )
            infos.append(info)
            states.append(state)

        states = np.array(states, dtype=float)
        rewards = np.array(rewards, dtype=float)
        dones = np.array(dones, dtype=bool)
        tus = np.array(tus, dtype=bool)

        return states, rewards, dones, tus, infos

    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        # 一般来说，子环境要使用不同的随机种子，避免重复。
        for i, env in enumerate(self.envs):
            env.seed(seed + i * 100)

    def close(self):
        for env in self.envs:
            env.close()


class IEEE123_SafeRL(IEEE123_RL):
    def __init__(
        self, N_topo=5, N_scenerio=4, cost_type="lineP", cost_coef=1.0, **args
    ):
        super().__init__(N_topo=N_topo, N_scenerio=N_scenerio, **args)
        self.cost_type = cost_type
        if cost_type == "lineP":
            self.cost_name = "res_p_outline"
        elif cost_type == "nodeV":
            self.cost_name = "res_V_outline"
        self.cost_coef = cost_coef

    def step(self, action):
        state, reward, done, tu, info = super().step(action)
        cost = self.cost_coef * info[self.cost_name]
        return state, reward, cost, done, tu, info


class IEEE123_SafeRL_Vec:
    def __init__(
        self, num_envs=1, N_topo=5, N_scenerio=4, cost_type="lineP", cost_coef=1, **args
    ):
        self.envs = [
            IEEE123_SafeRL(N_topo, N_scenerio, cost_type, cost_coef)
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        obs_shape = tuple([self.num_envs] + list(self.single_action_space.shape))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=obs_shape)
        act_shape = tuple([self.num_envs] + list(self.single_action_space.shape))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=act_shape)

    def reset(self, task_nums, is_train=True):
        assert len(task_nums) == self.num_envs
        states = []
        infos = {}
        self.task_nums = deepcopy(task_nums)
        infos["task_nums"] = self.task_nums
        infos["env_info"] = []
        for i in range(self.num_envs):
            state, info = self.envs[i].reset(task_nums[i], is_train=is_train)
            states.append(state)
            infos["env_info"].append(info)

        return np.array(states, dtype=float), infos

    def step(self, actions):
        assert len(actions) == self.num_envs
        states = []
        rewards = []
        costs = []
        dones = []
        tus = []
        infos = {}
        infos["task_nums"] = self.task_nums
        infos["env_info"] = []
        for i in range(self.num_envs):
            state, reward, cost, done, tu, info = self.envs[i].step(actions[i])
            states.append(state)
            rewards.append(reward)
            costs.append(cost)
            dones.append(done)
            tus.append(tu)
            infos["env_info"].append(info)

        states = np.array(states, dtype=float)
        rewards = np.array(rewards, dtype=float)
        costs = np.array(costs, dtype=float)
        dones = np.array(dones, dtype=bool)
        tus = np.array(tus, dtype=bool)

        return states, rewards, costs, dones, tus, infos


if __name__ == "__main__":
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task_num', type=int, default=1)
    # args = parser.parse_args()
    # env = IEEE123_SafeRL()
    # obs, info = env.reset(task_num=args.task_num, is_train=True)
    # done = False
    # Vcosts = []
    # rets = []
    # Pcosts = []
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, cost, ter, tru, info = env.step(action)
    #     done = ter or tru
    #     print(reward, info['res_V_outline'], info['res_p_outline'])
    #     Vcosts.append(info['res_V_outline'])
    #     rets.append(reward)
    #     Pcosts.append(info['res_p_outline'])
    # print(np.sum(rets), np.sum(Vcosts), np.sum(Pcosts))
    envs = IEEE123_RL_Vec(num_envs=2)
    states, infos = envs.reset(task_nums=[0, 1])
    print(states.shape)
    print(infos)
    done = False
    while not done:
        actions = envs.action_space.sample()
        print(actions.shape)
        states, rewards, dones, tus, infos = envs.step(actions)
        print(states.shape)
        print(rewards)
        print(dones)
        print(tus)
        done = all(dones) or all(tus)
    del envs
    envs = IEEE123_SafeRL_Vec(num_envs=2)
    states, infos = envs.reset(task_nums=[0, 1])
    print(states.shape)
    print(infos)
    done = False
    while not done:
        actions = envs.action_space.sample()
        print(actions.shape)
        states, rewards, costs, dones, tus, infos = envs.step(actions)
        print(states.shape)
        print(rewards)
        print(costs)
        print(dones)
        print(tus)
        done = all(dones) or all(tus)
