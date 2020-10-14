#!/bin/bash
cd /work2/i_kuroyanagi/kaggle/moa/working
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python tabularnet_baseline.py 
EOF
) >exp_tablenet//train.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp_tablenet//train.log
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( python tabularnet_baseline.py  ) &>>exp_tablenet//train.log
ret=$?
sync || truetime2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp_tablenet//train.log
echo '#' Accounting: end_time=$time2 >>exp_tablenet//train.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp_tablenet//train.log
echo '#' Finished at `date` with status $ret >>exp_tablenet//train.log
[ $ret -eq 137 ] && exit 100;
touch exp_tablenet/q/done.12749
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --ntasks-per-node=1  -p gpu --gres=gpu:1 --time 4:0:0  --open-mode=append -e exp_tablenet/q/train.log -o exp_tablenet/q/train.log  /work2/i_kuroyanagi/kaggle/moa/working/exp_tablenet/q/train.sh >>exp_tablenet/q/train.log 2>&1
