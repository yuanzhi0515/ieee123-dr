#!/bin/bash
ALGO=$1
SEEDS=$2
TASK=$3

source `which virtualenvwrapper.sh`
workon ieee123
echo -e "\nwelcome to ieee123!\n"

PATH_PREFIX=/aiarena/gpfs/ieee123_results/hard4_task${TASK}_1215_base
echo "PATH_PREFIX: ${PATH_PREFIX}"

echo "task: ${TASK}"

for (( i=0; i<${SEEDS}; i++ ))
do
    echo "Running seed${i}..."
    python main_off_policy.py \
        --train_task_nums ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} ${TASK} \
        --test_task_nums ${TASK} \
        --total_timesteps 1000000 \
        --learning_starts 2000 \
        --batch_size 1024 \
        --test_freq 10000 \
        --log_freq 1000 \
        --save_freq 10000 \
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
        --use_power_pred 0 \
        --pred_loss_weight 0 \
        --pred_lr 1e-3 \
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
