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
verbose=1      # verbosity level, higher is more logging

# directory related
expdir=exp_tablenet            # directory to save experiments
cache_dir=""          # bert cache dir (if empty, automatically download)

# utt score related
save_type=pkl  # select json or pkl

tag="baseline"    # tag for manangement of the naming of experiments

# evaluation related
checkpoint=""          # path of checkpoint to be used for evaluation

. utils/parse_options.sh || exit 1;

set -euo pipefail
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 1: Network training"
    outdir=${expdir}/${tag}
    log "Training start. See the progress via ${outdir}/train.log"
    # shellcheck disable=SC2086
    ${train_cmd} --gpu "${n_gpus}" "${outdir}/train.log" \
        python tab_base.py
    log "Successfully finished the training."
fi