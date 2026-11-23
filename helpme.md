# IEEE123 即时响应

plot2.py 用于绘制 p_ext 曲线的对比情况，同时保存误差绝对值之和的结果。对应的绘制脚本为 plot2.sh，我们只需要绘制惩罚系数为 1000 的情况就行，内容如下：

```c
bash plot2.sh 0
bash plot2.sh 1
bash plot2.sh 2
bash plot2.sh 3
```

plot3.py 用于绘制响应误差的对比情况，事先要把plot2的误差结果读出来写在 plot3.py 中，使用方式为：

```c
python plot3.py
```

加载模型进行测试记录动作

```c
python sac_test.py --train_task_nums 0 1 2 3 --test_task_nums 0 1 2 3 --total_timesteps 110000 --test_freq 10000 --penalty_factor 1000 --use_emb 1 --log_dir test06082358_pen_1000 --run_name IEEE123__sac__0__train_0_1_2_3__test_0_1_2_3__20250609_000103
```