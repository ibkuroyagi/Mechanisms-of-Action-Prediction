#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic setting
stage=-1       # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus for training
conf=conf/node.yaml
verbose=1      # verbosity level, higher is more logging

# directory related
expdir=exp          # directory to save experiments
tag="base/base"    # tag for manangement of the naming of experiments

# evaluation related
checkpoint=""          # path of checkpoint to be used for evaluation

. utils/parse_options.sh || exit 1;

set -euo pipefail
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Network training"
    outdir=${expdir}/${tag}
    log "Training start. See the progress via ${outdir}/train.log"
    # shellcheck disable=SC2086
    ${train_cmd} --gpu "${n_gpus}" "${outdir}/train.log" \
        python tab_base.py \
            --outdir "${outdir}" \
            --config "${conf}" \
            --verbose "${verbose}"
    log "Successfully finished the training."
fi

