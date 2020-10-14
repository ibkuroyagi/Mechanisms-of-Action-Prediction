# cuda related
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# path related
if [ -e "${PWD}/../venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PWD}/../venv/bin/activate"
fi

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
export LC_CTYPE=en_US.UTF-8
