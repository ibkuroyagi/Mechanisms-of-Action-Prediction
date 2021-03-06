#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

stage=1
verbose=1
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

No="009-2"
# for No in "007-1" "007-2" "007-3" "007-4"; do
conf="conf/tuning/node.v${No}.yaml"
tag="node/v${No}"
sbatch -J "M.${tag}" ./run.sh \
    --stage "${stage}" \
    --tag "${tag}" \
    --verbose "${verbose}" \
    --stop_stage 3 \
    --conf "${conf}" \
    --dpgmmdir ""
# done
# for No in 010 009; do
# No=010
# stage=2
# conf="conf/tuning/node.v${No}.yaml"
# tag="node/v${No}"
# echo "tag:${tag}, conf:${conf}"
# for step in 1000 2000 3000; do
#     checkpoint="checkpoint-${step}steps"
#     sbatch -J "M.${tag}" ./run.sh \
#         --stage "${stage}" \
#         --tag "${tag}" \
#         --verbose "${verbose}" \
#         --checkpoint "${checkpoint}" \
#         --step "${step}" \
#         --conf "${conf}"
# done
# checkpoint="best_loss"
# sbatch -J "M.${tag}" ./run.sh \
#     --stage "${stage}" \
#     --tag "${tag}" \
#     --verbose "${verbose}" \
#     --checkpoint "${checkpoint}" \
#     --step "best" \
#     --conf "${conf}"
# done