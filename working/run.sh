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
stage=0        # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus for training
n_jobs=8       # number of parallel jobs in feature extraction

conf=conf/node.yaml
verbose=1      # verbosity level, higher is more logging

# directory related
expdir=exp          # directory to save experiments
tag="node/base"    # tag for manangement of the naming of experiments
dpgmmdir="../input/dpgmm"
# evaluation related
checkpoint="best_loss"          # path of checkpoint to be used for evaluation
step="best"

. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 1: Calculate dpgmm."
    outdir=${expdir}/${tag}
    log "Calculate dpgmm. See the progress via ${outdir}/calculate_dpgmm.log"
    # shellcheck disable=SC2086
    ${train_cmd} --num_threads "${n_jobs}" "${outdir}/calculate_dpgmm.log" \
        python calculate_dpgmm.py \
            --outdir "${dpgmmdir}" \
            --config "${conf}" \
            --verbose "${verbose}"
    log "Successfully calculate dpgmm."
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Network training."
    outdir=${expdir}/${tag}
    log "Training start. See the progress via ${outdir}/train.log"
    # shellcheck disable=SC2086
    ${cuda_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/train.log" \
        python node_train.py \
            --outdir "${outdir}" \
            --config "${conf}" \
            --dpgmmdir "${dpgmmdir}" \
            --verbose "${verbose}"
    log "Successfully finished the training."
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Network inference."
    outdir=${expdir}/${tag}/${step}
    checkpoints=""
    for fold in {0..9}; do
        checkpoints+="${outdir}/${checkpoint}${fold}fold.pkl "
    done
    log "Inference start. See the progress via ${outdir}/inference.log"
    # shellcheck disable=SC2086
    ${cuda_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/inference.log" \
        python node_inference.py \
            --outdir "${outdir}" \
            --config "${conf}" \
            --checkpoints ${checkpoints} \
            --dpgmmdir "${dpgmmdir}" \
            --verbose "${verbose}"
    log "Successfully finished the inference."
fi
