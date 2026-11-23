#!/bin/bash

for task in {0..19}; do
python plot_action3.py \
    --path /aiarena/gpfs/ieee123_results/hard2_topo-1_1108_base/sac/ \
    --opf_path /aiarena/gpfs/ieee123_results/opf_results/hard2_opf_1124 \
    --task ${task} \
    --req_mod 1 6 7 8 9 10 \
    --output ./figs/hard2_1124_topo-1_base_task${task}_req1678910_action.jpg
done
