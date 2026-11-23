## RL
建议使用CleanRL，目前SAC可以使用，例如
```bash
python sac.py --train_task_nums 1 --test_task_nums 1
python sac.py --train_task_nums 0 1 2 --test_task_nums 1 2
```
注意，task的编号为0-19共20个，每4个属于同一个拓扑，即0-3; 4-7; 8-11; 12-15; 16-19
## SafeRL
目前适配了PPO、PPO_Lag、CPO、CPPO_PID、PCPO、RCPO、FOCOPS、CUP，例如
```bash
python safepo/single_agent/ppo_lag.py --use-eval True --train-tasks 0 --test-tasks 0 --device cuda --num-envs 1
python safepo/single_agent/ppo_lag.py --use-eval True --train-tasks 0 1 2 --test-tasks 1 3 --device cuda --num-envs 2
```
注意，num-envs只影响训练的环境并行数量，每次会从train-tasks中采样num-envs个task，评估环境的并行数量始终等于test-tasks的数量
## 运行环境
建议参考environment.yml根据cuda版本等自行配置
