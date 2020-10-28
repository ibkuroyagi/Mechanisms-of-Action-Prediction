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
stage=2       # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus for training
conf=conf/node.yaml
verbose=1      # verbosity level, higher is more logging

# directory related
expdir=exp          # directory to save experiments
tag="node/base"    # tag for manangement of the naming of experiments

# evaluation related
checkpoint="best_loss"          # path of checkpoint to be used for evaluation
step="best"

. utils/parse_options.sh || exit 1;

set -euo pipefail
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Network training."
    outdir=${expdir}/${tag}
    log "Training start. See the progress via ${outdir}/train.log"
    # shellcheck disable=SC2086
    ${train_cmd} --gpu "${n_gpus}" "${outdir}/train.log" \
        python node_train.py \
            --outdir "${outdir}" \
            --config "${conf}" \
            --verbose "${verbose}"
    log "Successfully finished the training."
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Network inference."
    outdir=${expdir}/${tag}/${step}
    checkpoints=""
    for fold in {0..4}; do
        checkpoints+="${outdir}/${checkpoint}${fold}fold.pkl "
    done
    log "Inference start. See the progress via ${outdir}/inference.log"
    # shellcheck disable=SC2086
    ${train_cmd} --gpu "${n_gpus}" "${outdir}/inference.log" \
        python node_inference.py \
            --outdir "${outdir}" \
            --config "${conf}" \
            --checkpoints ${checkpoints} \
            --verbose "${verbose}"
    log "Successfully finished the inference."
fi
