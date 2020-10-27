#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

stage=1
verbose=1
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;


No=000
conf="conf/tuning/node.v${No}.yaml"
tag="base/v${No}/"
echo "tag:${tag}, conf:${conf}"

sbatch -J "M.${tag}" ./run.sh \
    --stage "${stage}" \
    --tag "${tag}" \
    --verbose "${verbose}" \
    --conf "${conf}"
