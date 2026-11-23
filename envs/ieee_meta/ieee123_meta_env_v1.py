# import gym
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
# from gym.spaces import Box, Discrete, Dict
import copy
import pandas as pd
import pandapower as pp
import numpy as np
from envs.ieee_meta.pandapower_build_net import BuildNet
import random
import os


class IEEE123_Meta(gym.Env):
    def __init__(self, N_topo = 5, N_scenerio = 4, is_train = True, penalty_factor = 1000, req_file = "easy", use_gen_down = False, use_vp_rl = True, use_opf = False, **args):
        self.N_topo = N_topo
        self.N_scenerio = N_scenerio
        self.is_train = is_train
        self.penalty_factor = penalty_factor
        self.req_file = req_file
        self.use_gen_down = use_gen_down
        self.use_vp_rl = use_vp_rl
        self.region_num = 7
        self.region_element = {'load': [], 'ren':[], 'gen': [], 'storage':[]}
        self.region_max = {'load': [], 'ren':[], 'gen': []}
        self.max_T = 96
        self.pred_len = 4
        self.net_set = [BuildNet(opf_mode='gurobi', use_opf=use_opf, topo=i + 1) for i in range(self.N_topo)]

########################################initial region node set and max#####################################################################
        for j in range(self.region_num):
            region_load = self.net_set[0].bus.loc[self.net_set[0].bus['Region']==j+1, 'BusNo'].values
            region_ren = self.net_set[0].sgen.loc[self.net_set[0].sgen['Region']==j+1, 'BusNo'].values
            region_gen = self.net_set[0].gen.loc[self.net_set[0].gen['Region']==j+1, 'BusNo'].values
            region_storage = self.net_set[0].storage.loc[self.net_set[0].storage['Region']==j+1, 'BusNo'].values

            ren_sum_max = np.sum(self.net_set[0].net.sgen.loc[self.net_set[0].net.sgen['bus'].isin(region_ren) , 'max_p_mw'].values)
            load_sum_max = np.sum(self.net_set[0].net.load.loc[self.net_set[0].net.load['bus'].isin(region_load) , 'max_p_mw'].values)
            gen_sum_max = np.sum(self.net_set[0].net.gen.loc[self.net_set[0].net.gen['bus'].isin(region_gen) , 'max_p_mw'].values)

            self.region_element['load'].append(region_load)
            self.region_element['ren'].append(region_ren)
            self.region_element['gen'].append(region_gen)
            self.region_element['storage'].append(region_storage)

            self.region_max['load'].append(load_sum_max)
            self.region_max['ren'].append(ren_sum_max)
            self.region_max['gen'].append(gen_sum_max)
################################################read all mode data#####################################################################
        task_set_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_scenerio_task = [[pd.read_csv(task_set_dir + '/task_set/task' + str(scenerio_mode+1) + '/load'+str(i+1)+'_new.csv', encoding='utf-8')
                                     for i in range(5)] for scenerio_mode in range(N_scenerio)]
        self.solar_scenerio_task = [[pd.read_csv(task_set_dir + '/task_set/task' + str(scenerio_mode+1) + '/solar'+str(i+1)+'_new.csv', encoding='utf-8')
                                     for i in range(3)] for scenerio_mode in range(N_scenerio)]
        self.wind_scenerio_task = [[pd.read_csv(task_set_dir + '/task_set/task' + str(scenerio_mode+1) + '/wind'+str(i+1)+'_new.csv', encoding='utf-8')
                                     for i in range(2)] for scenerio_mode in range(N_scenerio)]
        self.price_scenerio_task = [pd.read_csv(task_set_dir + '/task_set/task' + str(scenerio_mode+1) + '/price_new.csv', encoding='utf-8') for scenerio_mode in range(N_scenerio)]

        topo_all = pd.read_csv(task_set_dir + '/task_set/base/graph.csv', encoding='utf-8')
        self.topo_task = [topo_all.loc[:, 'node_from_topo'+str(i+1):'node_to_topo'+str(i+1)].values.T for i in range(self.N_topo)]
#######################################initial all task z2 state##################################################################################
        edge_all = pd.read_csv(task_set_dir + '/task_set/base/edge.csv', encoding='utf-8')
        N_all_task = self.N_scenerio * self.N_topo
        self.z2_task_state = {'node': [], 'edge_index': [], 'edge_feature':[], 'price':[]}

        for i in range(N_all_task):
            z2_topo_index, z2_sce_index = self.reset_task(i)
            node_feature_100, price_feature_100 = self.make_state_task(z2_topo_index, z2_sce_index)#[300,7,2],[300]
            topo_feature_100 = self.topo_task[z2_topo_index]#[N_topo]
            edge_N = topo_feature_100.T
            edge_feature = []
            for e_n in range(edge_N.shape[0]):
                edge_from = edge_N[e_n, 0]
                edge_to = edge_N[e_n, 1]
                edge_feature_value = edge_all.loc[(edge_all['node_from'] == edge_from) & (edge_all['node_to'] == edge_to), 'limit'].values[0]
                edge_feature_D = [-edge_feature_value, edge_feature_value]
                edge_feature.append(edge_feature_D)
            edge_feature_100 = np.array(edge_feature)

            self.z2_task_state['node'].append(node_feature_100)
            self.z2_task_state['edge_index'].append(topo_feature_100)
            self.z2_task_state['edge_feature'].append(edge_feature_100)
            self.z2_task_state['price'].append(price_feature_100)
            #[N_task, ]

#######################################initial necesseray element####################################################################################3
        self.load_mode = {}
        self.ren_mode = {}
        self.price_mode = {}
        self.step_dict = {}
        self.pre_feature = {'node':[], 'price':[]}
        self.state = {}

        self.current_state_rl = None
        self.cur_topo_index = None
        self.cur_sce_index = None

        if self.use_vp_rl:
            self.action_space = Box(low=-1.0, high=1.0, shape=(22,))  #3个储能、4个机组有功、10个新能源、5个虚拟电厂
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(17,))  #3个储能、4个机组有功、10个新能源
        #TODO 更新ob_state
        self.observation_space = Box(low=-1.0, high=1.0, shape=(29,))

        self.env_base = pd.read_csv(task_set_dir + '/task_set/base/baseline.csv', encoding='utf-8')
        if self.req_file == "easy":
            self.request_set_train = pd.read_csv(task_set_dir + '/request_set/request_easy_train.csv', encoding='utf-8')
            self.request_set_test = pd.read_csv(task_set_dir + '/request_set/request_easy_test.csv', encoding='utf-8')
        elif self.req_file == "hard":
            self.request_set_train = pd.read_csv(task_set_dir + '/request_set/request_hard_train.csv', encoding='utf-8')
            self.request_set_test = pd.read_csv(task_set_dir + '/request_set/request_hard_test.csv', encoding='utf-8')
        elif self.req_file == "hard2":
            self.request_set_train = pd.read_csv(task_set_dir + '/request_set/request_hard2_train.csv', encoding='utf-8')
            self.request_set_test = pd.read_csv(task_set_dir + '/request_set/request_hard2_test.csv', encoding='utf-8')
        elif self.req_file == "hard3":
            self.request_set_train = pd.read_csv(task_set_dir + '/request_set/request_hard3_train.csv', encoding='utf-8')
            self.request_set_test = pd.read_csv(task_set_dir + '/request_set/request_hard3_test.csv', encoding='utf-8')
        elif self.req_file == "hard4":
            self.request_set_train = pd.read_csv(task_set_dir + '/request_set/request_hard4_train.csv', encoding='utf-8')
            self.request_set_test = pd.read_csv(task_set_dir + '/request_set/request_hard4_test.csv', encoding='utf-8')
        else:
            raise ValueError(f"Invalid request file: {self.req_file}")
        train_len = len(self.request_set_train.columns)
        test_len = len(self.request_set_test.columns)
        print(f"本环境需求文件: {self.req_file}， 训练集长度: {train_len}， 测试集长度: {test_len}")

        # 4 个火力发电机的状态，0 表示正常，1 表示维护
        self.gen_down_mask = np.zeros(4)
        if self.use_gen_down:
            print("本环境启用发电厂随机维护模式，训练模式随机开关，测试模式请指定开关")
        else:
            print("本环境未启用发电厂随机维护模式")

        if self.use_vp_rl:
            print("本环境使用强化学习控制虚拟电厂动作")
        else:
            print("本环境不使用虚拟电厂")

    def reset(self, task_num, req_mod=1, gen_down=False):
        # print(task_num)
        self.task_num = task_num
        topo_index, sce_index = self.reset_task(task_num)
        self.cur_topo_index = topo_index
        self.cur_sce_index = sce_index

        self.current_topo = copy.deepcopy(self.net_set[topo_index])
        self.current_step = 0
        self.load_mode = {}
        self.ren_mode = {}
        self.price_mode = {}
        self.request_mode = {}
    ####################################load mode#######################################
        if self.is_train:
            load_random_list = [random.randint(1,9) for _ in range(7)]
            # load_random_list = [1]*7
            self.load_mode['region1'] = self.load_scenerio_task[sce_index][0]['mod'+str(load_random_list[0])].values
            self.load_mode['region2'] = self.load_scenerio_task[sce_index][1]['mod'+str(load_random_list[1])].values
            self.load_mode['region3'] = self.load_scenerio_task[sce_index][2]['mod'+str(load_random_list[2])].values
            self.load_mode['region4'] = self.load_scenerio_task[sce_index][3]['mod'+str(load_random_list[3])].values
            self.load_mode['region5'] = self.load_scenerio_task[sce_index][4]['mod'+str(load_random_list[4])].values
            self.load_mode['region6'] = self.load_scenerio_task[sce_index][0]['mod'+str(load_random_list[5])].values
            self.load_mode['region7'] = self.load_scenerio_task[sce_index][1]['mod'+str(load_random_list[6])].values

            self.load_mode = pd.DataFrame(self.load_mode)
        else:
            self.load_mode['region1'] = self.load_scenerio_task[sce_index][0]['ave'].values
            self.load_mode['region2'] = self.load_scenerio_task[sce_index][1]['ave'].values
            self.load_mode['region3'] = self.load_scenerio_task[sce_index][2]['ave'].values
            self.load_mode['region4'] = self.load_scenerio_task[sce_index][3]['ave'].values
            self.load_mode['region5'] = self.load_scenerio_task[sce_index][4]['ave'].values
            self.load_mode['region6'] = self.load_scenerio_task[sce_index][0]['ave'].values
            self.load_mode['region7'] = self.load_scenerio_task[sce_index][1]['ave'].values

            self.load_mode = pd.DataFrame(self.load_mode)
    #######################################ren mode#######################################
        if self.is_train:
            sgen_random_list = [random.randint(1,9) for _ in range(len(self.current_topo.No_Seq_ren))]
            # sgen_random_list = [1]*len(self.current_topo.No_Seq_ren)
            for i in range(len(self.current_topo.No_Seq_ren)):
                if self.current_topo.Type_Seq_ren[i] == 'PV':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][0]['mod'+str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][1]['mod'+str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 3:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][2]['mod'+str(sgen_random_list[i])].values
                if self.current_topo.Type_Seq_ren[i] == 'WT':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][0]['mod'+str(sgen_random_list[i])].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][1]['mod'+str(sgen_random_list[i])].values

            self.ren_mode = pd.DataFrame(self.ren_mode)
        else:
            for i in range(len(self.current_topo.No_Seq_ren)):
                if self.current_topo.Type_Seq_ren[i] == 'PV':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][0]['ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][1]['ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 3:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.solar_scenerio_task[sce_index][2]['ave'].values
                if self.current_topo.Type_Seq_ren[i] == 'WT':
                    if self.current_topo.Mode_Seq_ren[i] == 1:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][0]['ave'].values
                    elif self.current_topo.Mode_Seq_ren[i] == 2:
                        self.ren_mode[self.current_topo.No_Seq_ren[i]] = self.wind_scenerio_task[sce_index][1]['ave'].values
            self.ren_mode = pd.DataFrame(self.ren_mode)
    ###########################price mode##############################################################
        if self.is_train:
            price_random = random.randint(1,9)
            # price_random = 1
            self.price_mode['ext_buy_price'] = self.price_scenerio_task[sce_index]['mod'+str(price_random)].values
        else:
            self.price_mode['ext_buy_price'] = self.price_scenerio_task[sce_index]['ave'].values
    #######################################request mode########################################################
        if self.is_train:
            # 从 request_set 的所有列中随机选择一列
            available_columns = [col for col in self.request_set_train.columns if col.startswith('mod')]
            selected_column = random.choice(available_columns)
            self.request_mode["request"] = self.request_set_train[selected_column].values
        else:
            self.request_mode["request"] = self.request_set_test['mod'+str(req_mod)].values
#################################update prediction#####################################################################
        self.pre_feature = {'node':[], 'price':[]}
        for t in range(100):
            step_ren_rate = self.ren_mode.loc[t, self.current_topo.No_Seq_ren].values
            a = self.load_mode.loc[t].values
            step_load_rate = []
            for i in range(a.shape[0]):
                step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
            step_load_rate = np.array(step_load_rate)
            step_price_rate = self.price_mode['ext_buy_price'][t]

            self.step_dict['ren'] = step_ren_rate
            self.step_dict['load'] = step_load_rate
            self.step_dict['price'] = step_price_rate

            self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)
            state_pre = self.make_state_prediction(topo_index, t)
            self.pre_feature['node'].append(state_pre['node'])
            self.pre_feature['price'].append(state_pre['price'])

##############################update real time data from mode##########################################################################
        step_ren_rate = self.ren_mode.loc[self.current_step, self.current_topo.No_Seq_ren].values
        a = self.load_mode.loc[self.current_step].values
        step_load_rate = []
        for i in range(a.shape[0]):
            step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
        step_load_rate = np.array(step_load_rate)
        step_price_rate = self.price_mode['ext_buy_price'][self.current_step]
        step_ext_request = self.request_mode['request'][self.current_step]

        self.step_dict['ren'] = step_ren_rate
        self.step_dict['load'] = step_load_rate
        self.step_dict['price'] = step_price_rate
        self.step_dict['request'] = step_ext_request

        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)
#####################################make state for RL##########################################################3
        # 火力发电厂维护状态, 0 表示正常，1 表示维护
        self.gen_down_mask = np.zeros(4)
        if self.use_gen_down:
            # 如果使用发电厂随机维护模式，那么训练时随机关闭发电厂，测试时使用指定的发电厂维护状态
            # 24号火力发电力随机处于维护状态
            if self.is_train:
                if np.random.rand() < 0.5:
                    self.gen_down_mask[0] = 1
                else:
                    self.gen_down_mask[0] = 0
            else:
                self.gen_down_mask[0] = gen_down


        state_real_time = self.make_state_rl(self.cur_topo_index)
        state_prediction_node = [self.pre_feature['node'][(self.current_step + i + 1) % self.max_T] for i in range(self.pred_len)]
        state_prediction_price = [self.pre_feature['price'][(self.current_step + i + 1) % self.max_T] for i in range(self.pred_len)]

        self.state['rt'] = state_real_time
        self.state['pre'] = {'node':state_prediction_node, 'price':state_prediction_price}
        info = {}

        return self.state, info

    def step(self, action):
        # if mode=='check':
        #     action = np.zeros((5,))
        #     self.step_dict['storage_rl'] = action[:3]
        #     self.step_dict['gen_rl'] = action[-2:]
        # elif mode=='train':
        #     self.step_dict['storage_rl'] = action[:3]
        #     self.step_dict['gen_rl'] = action[-2:]
        done = False
        tu = False

        if self.use_opf:
            self.step_dict['storage_opf'] = action[:3]
            self.step_dict['gen_opf'] = action[3:7]
            self.step_dict['ren_opf'] = action[7:17]
            self.step_dict['vp_opf'] = action[17:22]
        else:
            action = np.clip(action, -1.0, 1.0)
            self.step_dict['storage_rl'] = action[:3]
            self.step_dict['gen_rl'] = action[3:7]
            self.step_dict['ren_rl'] = action[7:17]
            if self.use_vp_rl:
                self.step_dict['vp_rl'] = action[17:22]
            else:
                self.step_dict['vp_rl'] = -np.ones(5)
            # 如果 24 号火力发电力处于维护状态， 那么 24 号火电机组出力为 0
            if self.use_gen_down:
                self.step_dict['gen_rl'] *= 1.0 - self.gen_down_mask

        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)

        if self.use_opf:
            gen_p_rtd, p_cur, p_vp, real_action, stor_penalty, soc, gen_penalty = self.current_topo.update_controllable_factor_opf(step_dict=self.step_dict)
        else:
            gen_p_rtd, p_cur, p_vp, real_action, stor_penalty, soc, gen_penalty = self.current_topo.update_controllable_factor_rl(step_dict=self.step_dict)

        if not self.is_train:
            stor_penalty = 0
            gen_penalty = 0

        #TODO 潮流计算逻辑
        try:
            pp.runpp(self.current_topo.net, numba=True)
        except:
            #TODO 潮流不收敛奖励惩罚
            reward = -100 + stor_penalty + gen_penalty
            info = {}
            info['p_ext'] = self.current_topo.net.res_ext_grid['p_mw'].values[0]  # 浮点数无需深拷贝
            info['last_request'] = self.step_dict['request']  # 浮点数无需深拷贝
            info['gen_down_mask'] = np.array(self.gen_down_mask)  # 这里一定要深拷贝！不然环境自动 reset 后可能会变化
            info['real_action'] = real_action
            info['stor_penalty'] = stor_penalty
            info['soc'] = soc
            info['gen_penalty'] = gen_penalty
            done = True
            # tu = False

            return self.state, reward, done, tu, info

        else:
        # TODO 潮流计算条件下的reward_function
            reward, reward_dict = self.reward_func(gen_p_rtd, p_cur, p_vp)
            reward += stor_penalty + gen_penalty
            reward_dict['stor_penalty'] = stor_penalty
            reward_dict['gen_penalty'] = gen_penalty
            reward_dict['reward'] = reward

            # info 包含节点电压潮流计算结果、节点电压上下限、含功率约束线路的有功潮流计算结果、线路有功功率上下限
            info = {}
            info['res_V_node'] = np.array(self.current_topo.net.res_bus['vm_pu'].values).reshape(-1)
            info['V_node_max'] = 1.05
            info['V_node_min'] = 0.95
            info['res_p_line'] = {}
            info['max_p_line'] = {}
            info['res_V_outline'] = (np.sum(info['res_V_node'] < info['V_node_min']) + np.sum(info['res_V_node'] > info['V_node_max'])) / info['res_V_node'].shape[0]

            info['res_p_outline'] = 0
            info['line_in_service'] = 0
            outline_penalty = 0.0
            outline_penalty_scale = -300
            for i,d in self.current_topo.net.line[-8:].iterrows():
                if d['in_service'] == True:
                    info['res_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])] = self.current_topo.net.res_line.loc[i, 'p_from_mw']
                    info['max_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])] = self.current_topo.branch.loc[i, 'max_P']
                    info['line_in_service'] += 1
                    if info['max_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])] < abs(info['res_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])]):
                        info['res_p_outline'] += 1
                        diff = abs(info['max_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])] - abs(info['res_p_line'][str(d['from_bus'])+'-'+str(d['to_bus'])]))
                        outline_penalty += outline_penalty_scale * diff
            if not self.is_train:
                outline_penalty = 0.0
            reward_dict['outline_penalty'] = outline_penalty
            reward += outline_penalty
            reward_dict['reward'] = reward
            info['res_p_outline'] = info['res_p_outline'] / info['line_in_service']
            info['p_ext'] = self.current_topo.net.res_ext_grid['p_mw'].values[0]  # 浮点数无需深拷贝
            info['last_request'] = self.step_dict['request']  # 浮点数无需深拷贝
            info['gen_down_mask'] = np.array(self.gen_down_mask)  # 这里一定要深拷贝！不然环境自动 reset 后可能会变化
            info['real_action'] = real_action
            info['stor_penalty'] = stor_penalty
            info['soc'] = soc
            info['gen_penalty'] = gen_penalty
            info['reward_dict'] = reward_dict

        self.current_step += 1

        if self.current_step >= self.max_T:
            done = True
        else:
            done = False

        step_ren_rate = self.ren_mode.loc[self.current_step, self.current_topo.No_Seq_ren].values
        a = self.load_mode.loc[self.current_step].values
        step_load_rate = []
        for i in range(a.shape[0]):
            step_load_rate += [a[i]] * self.current_topo.num_load_region[i]
        step_load_rate = np.array(step_load_rate)
        step_price_rate = self.price_mode['ext_buy_price'][self.current_step]
        step_ext_request = self.request_mode['request'][self.current_step]

        self.step_dict['ren'] = step_ren_rate
        self.step_dict['load'] = step_load_rate
        self.step_dict['price'] = step_price_rate
        self.step_dict['request'] = step_ext_request

        self.current_topo.update_uncontrollable_factor(step_dict=self.step_dict)

        #TODO 定义state
        state_real_time = self.make_state_rl(self.cur_topo_index)
        # if self.current_step < self.max_T:
        #     state_prediction_node = [self.pre_feature['node'][self.current_step + i + 1] for i in range(4)]
        #     state_prediction_price = [self.pre_feature['price'][self.current_step + i + 1] for i in range(4)]
        # else:
        #     state_prediction_node = [self.pre_feature['node'][self.current_step + i + 1] for i in range(3)]
        #     state_prediction_price = [self.pre_feature['price'][self.current_step + i + 1] for i in range(3)]
        state_prediction_node = [self.pre_feature['node'][(self.current_step + i + 1) % self.max_T] for i in range(self.pred_len)]
        state_prediction_price = [self.pre_feature['price'][(self.current_step + i + 1) % self.max_T] for i in range(self.pred_len)]

        self.state['rt'] = state_real_time
        self.state['pre'] = {'node':state_prediction_node, 'price':state_prediction_price}

        # self.current_topo.statistic(step_dict=self.step_dict)
        # tu = False

        return self.state, reward, done, tu, info

    def render(self, mode='human'):
        pass


    def make_state_rl(self, topo_index):
        current_state_dict = {}
        current_graph_state = []
        current_ext_feature = np.array([self.price_mode['ext_buy_price'][self.current_step]/80])
        request = self.request_mode['request'][40:64]
        # 这里给出未来 12 步的需求
        # current_ext_request = np.array([self.request_mode['request'][self.current_step]])
        current_ext_request = np.array(self.request_mode['request'][self.current_step:self.current_step+12])
        current_gen_down_mask = np.array(self.gen_down_mask)
        current_edge_index = self.topo_task[topo_index]

        for j in range(self.region_num):
            # region_vg = self.current_topo.vp.loc[self.current_topo.vp['Region']==j+1, 'BusNo'].values
            ren_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(self.region_element['ren'][j]) , 'max_p_mw'].values)
            load_sum = np.sum(self.current_topo.net.load.loc[self.current_topo.net.load['bus'].isin(self.region_element['load'][j]) , 'p_mw'].values)
            gen_sum = np.sum(self.current_topo.net.gen.loc[self.current_topo.net.gen['bus'].isin(self.region_element['gen'][j]) , 'p_mw'].values)
            storage_soc = np.sum(self.current_topo.net.storage.loc[self.current_topo.net.storage['bus'].isin(self.region_element['storage'][j]) , 'soc_percent'].values)/100
            # cur_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(region_ren) , 'p_mw'].values)
            # vg_sum = np.sum(self.current_topo.net.sgen.loc[self.net.sgen['bus'].isin(region_vg) , 'p_mw'].values)

            node_feature = [ren_sum/max(self.region_max['ren']), load_sum/max(self.region_max['load']), gen_sum/max(self.region_max['gen']), storage_soc]
            current_graph_state.append(np.array(node_feature))
        current_graph_state = np.array(current_graph_state)

        current_state_dict['node_real_time'] = current_graph_state
        current_state_dict['price_real_time'] = current_ext_feature
        current_state_dict['request_real_time'] = current_ext_request
        current_state_dict['request'] = request
        current_state_dict['gen_down_mask_real_time'] = current_gen_down_mask
        current_state_dict['edge'] = current_edge_index
        current_state_dict['time'] = self.current_step

        return current_state_dict

    def make_state_prediction(self, topo_index, t):
        current_state_dict = {}
        current_graph_state = []
        current_ext_feature = np.array([self.price_mode['ext_buy_price'][t]/80])
        current_edge_index = self.topo_task[topo_index]

        for j in range(self.region_num):
            # region_vg = self.current_topo.vp.loc[self.current_topo.vp['Region']==j+1, 'BusNo'].values
            ren_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(self.region_element['ren'][j]) , 'max_p_mw'].values)
            load_sum = np.sum(self.current_topo.net.load.loc[self.current_topo.net.load['bus'].isin(self.region_element['load'][j]) , 'p_mw'].values)
            # cur_sum = np.sum(self.current_topo.net.sgen.loc[self.current_topo.net.sgen['bus'].isin(region_ren) , 'p_mw'].values)
            # vg_sum = np.sum(self.current_topo.net.sgen.loc[self.net.sgen['bus'].isin(region_vg) , 'p_mw'].values)

            node_feature = [ren_sum/max(self.region_max['ren']), load_sum/max(self.region_max['load'])]
            current_graph_state.append(np.array(node_feature))
        current_graph_state = np.array(current_graph_state)

        current_state_dict['node'] = current_graph_state
        current_state_dict['price'] = current_ext_feature

        return current_state_dict

    def make_state_task(self, topo_index, sce_index):
        net_task = self.net_set[topo_index]
        load_task = self.load_scenerio_task[sce_index]
        solar_task = self.solar_scenerio_task[sce_index]
        wind_task = self.wind_scenerio_task[sce_index]
        task_seq = []

        for j in range(self.region_num):
            load = self.region_max['load'][j]
            load_mode = load_task[j % 5]['ave'].values
            load_seq = load * load_mode

            if j +1 in net_task.sgen['Region']:
                sgen_seq = np.zeros((101,))
                for i, d in net_task.sgen.iterrows():
                    if d['Region'] == j+1:
                        if d['Type'] == 'PV':
                            sgen = d['max_P']
                            sgen_mode = solar_task[d['mode'] - 1]['ave'].values
                            sgen_seq += sgen * sgen_mode
                        if d['Type'] == 'WT':
                            sgen = d['max_P']
                            sgen_mode = wind_task[d['mode'] - 1]['ave'].values
                            sgen_seq += sgen * sgen_mode
            else:
                sgen_seq = np.zeros((len(solar_task[0][0]),))

            region_seq = np.concatenate((load_seq[np.newaxis, :]/max(self.region_max['load']), sgen_seq[np.newaxis, :]/max(self.region_max['ren'])), axis = 0).T
            task_seq.append(region_seq)
        task_seq = [item[:, np.newaxis, :] for item in task_seq]
        task_seq = np.concatenate(task_seq, axis=1)

        price_seq = self.price_scenerio_task[sce_index]['ave'].values / 80

        return task_seq, price_seq

    def reward_func(self, gen_rtd, p_cur, p_vp):

        gen_rl = self.current_topo.gen.loc[self.current_topo.gen['Cal'] == 'RL', :]
        cost_gen = 0.0
        for n_gen in range(gen_rtd.shape[0]):
            cost_gen += gen_rl['Co_2'].values[n_gen] * gen_rtd[n_gen]**2 + gen_rl['Co_1'].values[n_gen] * gen_rtd[n_gen] + gen_rl['Co_0'].values[n_gen]

        cost_cur = 0
        for i,d in self.current_topo.sgen.iterrows():
            cost_cur += p_cur[i] * d['Co_cur']

        cost_vp = 0
        for i,d in self.current_topo.vp.iterrows():
            cost_vp += p_vp[i] * d['Co_Cut']

        cost_pl = np.sum(abs(self.current_topo.net.res_line['pl_mw'].values)) * 20

        p_ext = self.current_topo.net.res_ext_grid['p_mw'].values[0]

        # if p_ext >= 0:
        #     cost_ext = self.step_dict['price'] * p_ext
        # else:
        #     #TODO 越界惩罚
        #     cost_ext = abs(p_ext) * 100

        cost_ext = 0
        resp_penalty = 0
        # 需求包含未来 4 步，这里去当前时刻的需求
        request = self.step_dict['request']
        if request > -1:
            # 上级电网有需求
            # p_ext 为负数，表示下级电网向上级供电，这部分是供电偏离的惩罚
            resp_penalty = abs(request + p_ext) * self.penalty_factor
        else:
            # 上级电网无需求
            if p_ext >= 0:
                # 下级电网买电。这是下级电网的购电成本
                cost_ext = self.step_dict['price'] * p_ext
            else:
                # 下级电网卖电，上级电网按照 80% 的价格购买。这是下级电网的盈利
                cost_ext = self.step_dict['price'] * p_ext * 0.8
        reward = -(cost_gen + cost_cur + cost_vp + cost_pl + cost_ext + resp_penalty)
        reward_dict = {
            'cost_gen': -cost_gen,
            'cost_cur': -cost_cur,
            'cost_vp': -cost_vp,
            'cost_pl': -cost_pl,
            'cost_ext': -cost_ext,
            'resp_penalty': -resp_penalty,
            'reward': reward,
        }
        # print(f'cost_gen: {cost_gen},cost_cur: {cost_cur}, cost_vp: {cost_vp}, cost_pl: {cost_pl}, cost_ext: {cost_ext}')

        return reward, reward_dict

    def reset_task(self, n_task):
        '''
        N_scenerio = 4
        N_topo = 5
        n_task: 0-3; 4-7; 8-11; 12-15; 16-19
        '''
        topo_index = n_task // self.N_scenerio
        sce_index = n_task % self.N_scenerio

        return topo_index, sce_index

if __name__ == '__main__':
    a = IEEE123_Meta()
    b = a.reset(task_num=1)
    for n in range(288):
        ac = np.random.rand(5)
        ac = ac - 0.5
        current_state_rl, reward, done, info = a.step(action=ac, mode='train')
        print(current_state_rl.shape)
        # print(current_state_rl)
