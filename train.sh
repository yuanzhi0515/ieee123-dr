#!/bin/bash
ALGO=$1
SEEDS=$2
TOPO=$3

source `which virtualenvwrapper.sh`
workon ieee123
echo -e "\nwelcome to ieee123!\n"

PATH_PREFIX=/aiarena/gpfs/ieee123_results/hard4_topo${TOPO}_1216_base+pred_lr1e-4+pred1000
echo "PATH_PREFIX: ${PATH_PREFIX}"

if [ ${TOPO} -eq -1 ]; then
    task_nums='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19'
elif [ ${TOPO} -eq 0 ]; then
    task_nums='0 1 2 3'
elif [ ${TOPO} -eq 1 ]; then
    task_nums='4 5 6 7'
elif [ ${TOPO} -eq 2 ]; then
    task_nums='8 9 10 11'
elif [ ${TOPO} -eq 3 ]; then
    task_nums='12 13 14 15'
elif [ ${TOPO} -eq 4 ]; then
    task_nums='16 17 18 19'
else
    echo "Invalid topology: ${TOPO}"
    exit 1
fi

echo "topo: ${TOPO}"
echo "task_nums: ${task_nums}"

for (( i=0; i<${SEEDS}; i++ ))
do
    echo "Running seed${i}..."
    python main_off_policy.py \
        --train_task_nums ${task_nums} \
        --test_task_nums ${task_nums} \
        --total_timesteps 1000000 \
        --learning_starts 2000 \
        --batch_size 1024 \
        --test_freq 20000 \
        --log_freq 2000 \
        --save_freq 10000 \
        --gamma 0.997 \
        --actor_lr 3e-4\
        --critic_lr 1e-3 \
        --penalty_factor 300 \
        --use_actor_encoder 0 \
        --use_actor_rnn 0 \
        --use_critic_encoder 0 \
        --use_critic_rnn 0 \
        --use_norm 1 \
        --use_weight 0 \
        --sampling_weight 0 \
        --reward_norm 0 \
        --time_enc_k 0 \
        --use_power_pred 1 \
        --pred_loss_weight 1000 \
        --pred_lr 1e-4 \
        --use_multi_goal 0 \
        --multi_goal_weight 0 \
        --use_gen_down 1 \
        --use_action_mask 1 \
        --compensate_update_freq 5 \
        --req_file hard4 \
        --n_step 1 \
        --use_vp_rl 0 \
        --algo ${ALGO} \
        --log_dir ${PATH_PREFIX}/${ALGO} \
        --seed $i &
    sleep 1
done

wait
echo -e "\ndone"
echo "results: ${PATH_PREFIX}/${ALGO}"
