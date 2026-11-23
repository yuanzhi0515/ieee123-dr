import pandapower as pp
import numpy as np
import pandas as pd
import random
import os


class BuildNet:
    def __init__(self, opf_mode='gurobi', use_opf=False, topo = 0):
        task_set_dir = os.path.dirname(os.path.abspath(__file__))
        self.base = pd.read_csv(task_set_dir + '/task_set/base/base.csv', encoding='utf-8')
        self.bus = pd.read_csv(task_set_dir + '/task_set/base/bus_region.csv', encoding='utf-8')
        self.relax_bus = pd.read_csv(task_set_dir + '/task_set/base/relax_bus.csv', encoding='utf-8')
        self.gen = pd.read_csv(task_set_dir + '/task_set/base/gen_region.csv', encoding='utf-8')
        self.sgen = pd.read_csv(task_set_dir + '/task_set/base/sgen_region.csv', encoding='utf-8')
        self.storage = pd.read_csv(task_set_dir + '/task_set/base/storage_region.csv', encoding='utf-8')
        self.vp = pd.read_csv(task_set_dir + '/task_set/base/vp_region.csv', encoding='utf-8')

        branch_case_dir = task_set_dir + '/task_set/branch_case' + str(topo) + '.csv'
        self.branch = pd.read_csv(branch_case_dir, encoding='utf-8')
        self.branch_log_seq = [(116, 1), (13, 18), (13,118), (18,115), (117,122), (67, 97), (60, 119), (54, 94), (81, 86)]

        self.opf_mode = opf_mode
        self.use_opf = use_opf

        self.Base_S = self.base['Base_S'][0]
        self.Base_V = self.base['Base_V'][0] / self.base['Co_V'][0]
        # self.Base_V = self.base['Base_V'][0]
        self.Base_R = self.Base_V ** 2 / self.Base_S

        self.time_interval = 0.25
        self.n_RL_storage = len(self.storage.loc[self.storage['Cal'] == 'RL', :])
        self.n_RL_gen = len(self.gen.loc[self.gen['Cal'] == 'RL', :])
        self.n_OPF_gen = len(self.gen.loc[self.gen['Cal'] == 'OPF', :])

        self.gen_rl_index = self.gen.loc[self.gen['Cal'] == 'RL', :].index
        self.gen_rl_max = self.gen.loc[self.gen_rl_index, 'max_P'].values
        self.gen_rl_min = self.gen.loc[self.gen_rl_index, 'min_P'].values
        self.gen_rl_cr = self.gen.loc[self.gen_rl_index, 'CR'].values

        self.gen_opf_index = self.gen.loc[self.gen['Cal'] == 'OPF', :].index
        self.gen_opf_max = self.gen.loc[self.gen_opf_index, 'max_P'].values
        self.gen_opf_min = self.gen.loc[self.gen_opf_index, 'min_P'].values
        self.gen_opf_cr = self.gen.loc[self.gen_opf_index, 'CR'].values

        self.num_load_region = []
        self.No_Seq_ren = []
        self.Type_Seq_ren = []
        self.Mode_Seq_ren = []

        for i in range(max(self.bus['Region'])):
            self.num_load_region.append(len(self.bus.loc[self.bus['Region'] == i + 1]))
        for index, row in self.sgen.iterrows():
            self.No_Seq_ren.append(row['Type']+'_'+str(row['Region'])+'_'+str(row['BusNo']))
            self.Type_Seq_ren.append(row['Type'])
            self.Mode_Seq_ren.append(row['mode'])

        self.net = pp.create_empty_network(sn_mva=self.Base_S)
        self.build_net()

        # OPF建模相关
        self.edge_list = []
        self.Constr_edge_list = []
        self.v_list = []
        self.dg_list = []
        self.bs_list = []
        self.grid_list = []
        self.vg_list = []
        self.curtail_list = []
        self.ren_list = []
        self.relax_bus_list = []

        # params
        self.r = {}
        self.x = {}
        self.edge_Constr = {}
        self.p_DG_Max = {}
        self.p_DG_Min = {}
        self.q_DG_Max = {}
        self.q_DG_Min = {}
        self.p_DG_cr = {}
        self.p_DG_rtd = {}
        self.p_DG_rtd_rl = {}
        self.v_DG_setting = {}

        self.alpha = {}
        self.beta = {}
        self.c = {}

        self.load_beta = {}
        self.price_buy = {}

        self.load_p = {}
        self.load_q = {}

        self.vg_p_limit = {}
        self.curtail_p_limit = {}
        self.curtail_q_limit = {}

        self.ren_p = {}
        self.ren_q = {}

        self.bs_p = {}

        # self.base_S = 5
        # self.base_V = 4.16
        # 统计相关
        self.ren_seq_reg = {}
        self.load_seq_reg = {}
        self.gen_seq_reg = {}
        self.vg_seq_reg = {}
        self.cur_seq_reg = {}
        self.relax_seq_reg = {}
        self.line_seq_reg = {}
        for (m,k) in self.branch_log_seq:
            self.line_seq_reg[str(m)+'-'+str(k)] = []

    def build_net(self):
        # 构建bus
        for i in range(self.base['Sum_Bus'][0]):
            pp.create_bus(self.net, vn_kv=self.Base_V, index=int(i + 1), max_vm_pu=1.05, min_vm_pu=0.95)
        # 外来电节点设置
        pp.create_ext_grid(self.net, bus=self.base['ext_Bus'][0], min_p_mw=self.base['min_ext_P'][0],
                           max_p_mw=self.base['max_ext_P'][0], min_q_mvar=-10, max_q_mvar=10, controllable=True)
        # TODO 经济参数读取列表
        pp.create_poly_cost(self.net, 0, "ext_grid", cp1_eur_per_mw=60, cp0_eur=0)
        # 线路信息导入
        for index, row in self.branch.iterrows():
            pp.create_line_from_parameters(self.net, from_bus=int(row['from']), to_bus=int(row['to']), length_km=1,
                                           r_ohm_per_km=float(row['r_pu']) * self.Base_R,
                                           x_ohm_per_km=float(row['x_pu']) * self.Base_R, c_nf_per_km=0,
                                           max_i_ka=float(row['max_P']), in_service=True if row['state']==True else False)
        # 节点负荷最大值信息导入
        for index, row in self.bus.iterrows():
            pp.create_load(self.net, bus=int(row['BusNo']), p_mw=float(row['Load_P_max']),
                           q_mvar=float(row['Load_Q_max']), max_p_mw=float(row['Load_P_max']),
                           max_q_mvar=float(row['Load_Q_max']), min_p_mw=0, min_q_mvar=0)
        # 发电机信息导入
        for index, row in self.gen.iterrows():
            # 发电机的技术参数
            # TODO 发电机爬坡约束
            pp.create_gen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), max_p_mw=float(row['max_P']),
                          min_p_mw=float(row['min_P']), max_q_mvar=float(row['max_Q']), min_q_mvar=float(row['min_Q']),
                          vm_pu=float(row['V']), controllable=True if row['Cal'] == 'OPF' else False)
            pp.create_poly_cost(self.net, index, "gen", cp1_eur_per_mw=float(row['Co_1']), cp0_eur=float(row['Co_0']),
                                cp2_eur_per_mw2=float(row['Co_2']))
        # 新能源信息导入
        for index, row in self.sgen.iterrows():
            pp.create_sgen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), q_mvar=float(row['rtd_Q']),
                           max_p_mw=float(row['max_P']), min_p_mw=float(row['min_P']), max_q_mvar=float(row['max_Q']),
                           min_q_mvar=float(row['min_Q']), controllable=True)
            pp.create_poly_cost(self.net, index, "sgen", cp1_eur_per_mw=float(row['Co_cur']), cp0_eur=0)
        # 虚拟电厂信息导入
        for index, row in self.vp.iterrows():
            pp.create_sgen(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_cut']), q_mvar=0,
                           max_p_mw=float(row['max_Cut']), min_p_mw=0, max_q_mvar=0, min_q_mvar=0, controllable=True)
            pp.create_poly_cost(self.net, index + len(self.sgen), "sgen",
                                cp1_eur_per_mw=float(row['Co_Cut']), cp0_eur=0)
        # 储能信息导入
        for index, row in self.storage.iterrows():
            pp.create_storage(self.net, bus=int(row['BusNo']), p_mw=float(row['rtd_P']), q_mvar=0,
                              max_e_mwh=float(row['MaxE']), soc_percent=float(row['init_SoC']),
                              max_p_mw=float(row['max_P']), min_p_mw=float(row['min_P']),
                              max_q_mvar=float(row['max_Q']),
                              min_q_mvar=float(row['min_Q']),
                              controllable=True if row['Cal'] == 'OPF' else False)

        #TODO 松弛节点构建

    def update_uncontrollable_factor(self, step_dict):
        # 更新新能源实际最大出力
        step_ren = step_dict['ren'] * self.sgen['max_P'].values
        self.net.sgen['p_mw'][0:len(self.sgen)] = step_ren
        step_ren_q = step_dict['ren'] * self.sgen['max_Q'].values
        self.net.sgen['q_mvar'][0:len(self.sgen)] = step_ren_q
        # 更新新能源弃电功率范围
        self.net.sgen['max_p_mw'][0:len(self.sgen)] = step_ren
        self.net.sgen['max_q_mvar'][0:len(self.sgen)] = step_ren_q
        # 更新负荷运行数据
        step_load_p = step_dict['load'] * self.net.load['max_p_mw'].values
        self.net.load['p_mw'] = step_load_p
        step_load_q = step_dict['load'] * self.net.load['max_q_mvar'].values
        self.net.load['q_mvar'] = step_load_q
        # 更新外部购电电价
        step_price = step_dict['price']
        self.net.poly_cost.loc[self.net.poly_cost['et'] == 'ext_grid', 'cp1_eur_per_mw'] = step_price
        # 更新虚拟电厂可控范围
        for index, row in self.vp.iterrows():
            self.net.sgen.loc[self.net.sgen['bus'] == int(row['BusNo']), 'max_p_mw'] = \
                self.net.load.loc[self.net.load['bus'] == int(row['BusNo']), 'p_mw'].values[0] * row['CR']


    #TODO 改强化学习确定的变量
    def update_controllable_factor_rl(self, step_dict):
        # 根据RL动作值更新储能实际执行功率
        action = step_dict['storage_rl']
        storage_p_rtd = np.zeros((self.n_RL_storage,))
        soc_result = np.zeros((self.n_RL_storage,))

        stor_penalty = 0.
        stor_penalty_scale = -30
        for index, row in self.storage.loc[self.storage['Cal'] == 'RL'].iterrows():
            storage_p_rtd[index] = action[index] * row['max_P']
            soc_now = self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'].values[0]

            if (soc_now / 100) <= row['min_SoC'] and action[index] < 0:
                stor_penalty += stor_penalty_scale * abs(action[index])
            elif (soc_now / 100) >= row['max_SoC'] and action[index] > 0:
                stor_penalty += stor_penalty_scale * abs(action[index])

            if storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                    + soc_now / 100 >= row['max_SoC']:
                # pred_soc = storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                #     + soc_now / 100
                # stor_penalty += stor_penalty_scale * abs(pred_soc - row['max_SoC'])
                storage_p_rtd[index] = (row['max_SoC'] - soc_now / 100) * row['MaxE'] \
                                       / row['charge_rate'] / self.time_interval
                self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = row['max_SoC'] * 100
            elif storage_p_rtd[index] * self.time_interval / row['MaxE'] / row['discharge_rate'] \
                    + soc_now / 100 <= row['min_SoC']:
                # pred_soc = storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                #     + soc_now / 100
                # stor_penalty += stor_penalty_scale * abs(pred_soc - row['min_SoC'])
                storage_p_rtd[index] = (row['min_SoC'] - soc_now / 100) * row['MaxE'] \
                                       * row['discharge_rate'] / self.time_interval
                self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = row['min_SoC'] * 100
            else:
                if storage_p_rtd[index] >= 0:
                    soc_rtd = 100 * storage_p_rtd[index] * row['charge_rate'] * self.time_interval / row['MaxE'] \
                              + soc_now
                    self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = soc_rtd
                else:
                    soc_rtd = 100 * storage_p_rtd[index] * self.time_interval / row['MaxE'] / row['discharge_rate'] \
                              + soc_now
                    self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = soc_rtd
            soc_result[index] = self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'].values[0] / 100
        self.net.storage.loc[self.storage['Cal'] == 'RL', 'p_mw'] = storage_p_rtd

        # 根据RL动作更新机组实际执行功率
        gen_penalty = 0.
        gen_penalty_scale = -30
        action_gen = step_dict['gen_rl']
        gen_p_init = self.net.gen.loc[self.gen_rl_index, 'p_mw'].values
        for i, ac in enumerate(action_gen):
            if gen_p_init[i] <= self.gen_rl_min[i] and ac < 0:
                gen_penalty += gen_penalty_scale * abs(ac)
            elif gen_p_init[i] >= self.gen_rl_max[i] and ac > 0:
                gen_penalty += gen_penalty_scale * abs(ac)
        gen_p_rtd = gen_p_init + action_gen * self.gen_rl_max * self.gen_rl_cr
        # for i, _ in enumerate(action_gen):
        #     if gen_p_rtd[i] < self.gen_rl_min[i]:
        #         gen_penalty += gen_penalty_scale * abs(gen_p_rtd[i] - self.gen_rl_min[i])
        #     elif gen_p_rtd[i] > self.gen_rl_max[i]:
        #         gen_penalty += gen_penalty_scale * abs(gen_p_rtd[i] - self.gen_rl_max[i])
        np.clip(gen_p_rtd, self.gen_rl_min, self.gen_rl_max, out=gen_p_rtd)

        self.net.gen.loc[self.gen_rl_index, 'p_mw'] = gen_p_rtd

        # 根据RL动作更新新能源实际功率
        action_ren = 0.5 - step_dict['ren_rl'] / 2
        action_cur = (1-action_ren) * self.net.sgen['max_p_mw'][0:len(self.sgen)].values
        action_ren = action_ren * self.net.sgen['max_p_mw'][0:len(self.sgen)].values
        self.net.sgen['p_mw'][0:len(self.sgen)] = action_ren

        # 根据RL动作更新实际负荷
        action_vp = step_dict['vp_rl'] / 2 + 0.5
        action_vp = action_vp * self.net.sgen['max_p_mw'][len(self.sgen):].values
        self.net.sgen['p_mw'][len(self.sgen):] = action_vp

        real_action = np.concatenate([storage_p_rtd, gen_p_rtd, action_ren, action_vp])

        return gen_p_rtd, action_cur, action_vp, real_action, stor_penalty, soc_result, gen_penalty

    #TODO 改强化学习确定的变量
    def update_controllable_factor_opf(self, step_dict):
        # 根据RL动作值更新储能实际执行功率
        storage_p_rtd = np.zeros((self.n_RL_storage,))
        soc_result = np.zeros((self.n_RL_storage,))

        stor_penalty = 0.
        stor_penalty_scale = -30
        for index, row in self.storage.loc[self.storage['Cal'] == 'RL'].iterrows():
            storage_p_rtd[index] = step_dict["storage_opf"][index]
            soc_now = self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'].values[0]

            if storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                    + soc_now / 100 >= row['max_SoC']:
                # pred_soc = storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                #     + soc_now / 100
                # stor_penalty += stor_penalty_scale * abs(pred_soc - row['max_SoC'])
                storage_p_rtd[index] = (row['max_SoC'] - soc_now / 100) * row['MaxE'] \
                                       / row['charge_rate'] / self.time_interval
                self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = row['max_SoC'] * 100
            elif storage_p_rtd[index] * self.time_interval / row['MaxE'] / row['discharge_rate'] \
                    + soc_now / 100 <= row['min_SoC']:
                # pred_soc = storage_p_rtd[index] * self.time_interval * row['charge_rate'] / row['MaxE'] \
                #     + soc_now / 100
                # stor_penalty += stor_penalty_scale * abs(pred_soc - row['min_SoC'])
                storage_p_rtd[index] = (row['min_SoC'] - soc_now / 100) * row['MaxE'] \
                                       * row['discharge_rate'] / self.time_interval
                self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = row['min_SoC'] * 100
            else:
                if storage_p_rtd[index] >= 0:
                    soc_rtd = 100 * storage_p_rtd[index] * row['charge_rate'] * self.time_interval / row['MaxE'] \
                              + soc_now
                    self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = soc_rtd
                else:
                    soc_rtd = 100 * storage_p_rtd[index] * self.time_interval / row['MaxE'] / row['discharge_rate'] \
                              + soc_now
                    self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'] = soc_rtd
            soc_result[index] = self.net.storage.loc[self.net.storage['bus'] == row['BusNo'], 'soc_percent'].values[0] / 100
        self.net.storage.loc[self.storage['Cal'] == 'RL', 'p_mw'] = storage_p_rtd

        # 根据RL动作更新机组实际执行功率
        gen_penalty = 0.
        gen_penalty_scale = -30
        self.net.gen.loc[self.gen_rl_index, 'p_mw'] = step_dict["gen_opf"]
        gen_p_rtd = step_dict["gen_opf"]

        # 根据RL动作更新新能源实际功率
        self.net.sgen['p_mw'][0:len(self.sgen)] = step_dict["ren_opf"]
        action_cur = self.net.sgen['max_p_mw'][0:len(self.sgen)].values - step_dict["ren_opf"]
        action_ren = step_dict["ren_opf"]

        # 根据RL动作更新实际负荷
        self.net.sgen['p_mw'][len(self.sgen):] = step_dict["vp_opf"]
        action_vp = step_dict["vp_opf"]

        real_action = np.concatenate([storage_p_rtd, gen_p_rtd, action_ren, action_vp])

        return gen_p_rtd, action_cur, action_vp, real_action, stor_penalty, soc_result, gen_penalty