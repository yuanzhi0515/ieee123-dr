#!/bin/bash
ALGO=$1
SEEDS=$2
TOPO=$3
LOG_DIR=$4

source `which virtualenvwrapper.sh`
workon ieee123
echo -e "\nwelcome to ieee123!\n"

echo "LOG_DIR: ${LOG_DIR}"

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
    python test_off_policy.py \
        --test_task_nums ${task_nums} \
        --penalty_factor 300 \
        --use_actor_encoder 0 \
        --use_actor_rnn 0 \
        --use_critic_encoder 0 \
        --use_critic_rnn 0 \
        --use_norm 1 \
        --time_enc_k 0 \
        --use_gen_down 1 \
        --req_file hard4 \
        --algo ${ALGO} \
        --log_dir ${LOG_DIR} \
        --use_opf 1 \
        --opf_dir /aiarena/gpfs/ieee123_results/opf_results/hard4_opf_1216 \
        --use_vp_rl 0 \
        --seed $i &
    sleep 1
done

wait
echo -e "\ndone"
