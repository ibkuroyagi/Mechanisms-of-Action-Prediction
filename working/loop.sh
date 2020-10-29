#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

stage=2
verbose=1
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;


No=000
conf="conf/tuning/node.v${No}.yaml"
tag="node/v${No}"
echo "tag:${tag}, conf:${conf}"
for step in 25000 30000 35000 40000; do
    checkpoint="checkpoint-${step}steps"
    sbatch -J "M.${tag}" ./run.sh \
        --stage "${stage}" \
        --tag "${tag}" \
        --verbose "${verbose}" \
        --checkpoint "${checkpoint}" \
        --step "${step}" \
        --conf "${conf}"
done
checkpoint="best_loss"
sbatch -J "M.${tag}" ./run.sh \
    --stage "${stage}" \
    --tag "${tag}" \
    --verbose "${verbose}" \
    --checkpoint "${checkpoint}" \
    --step "best" \
    --conf "${conf}"
