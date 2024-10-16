#!/bin/bash -i
#SBATCH --cpus-per-task=16

# Load CUDA, change to the cuda version on your environment if different
module load gcc/14.2.0 cuda/12.5
module load python3/3.11.8
nvidia-smi


source ${SRC_DIR}/find_port.sh
# Write server url to file
hostname=${SLURMD_NODENAME}
vllm_port_number=$(find_available_port $hostname 8080 65535)

echo "Server address: http://${hostname}:${vllm_port_number}/v1"
echo "http://${hostname}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}

# Activate vllm venv
if [ "$VENV_BASE" = "apptainer" ]; then
    export SINGULARITY_IMAGE=$WORK/static/llm-train-serve_aarch64.sif
    export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/libnccl.so.2.22.3
    module load tacc-apptainer
    apptainer exec $SINGULARITY_IMAGE ray stop
    apptainer exec --nv --bind $WORK/model-weights:/model-weights $SINGULARITY_IMAGE \
    python -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN}
else
    source ${VENV_BASE}/bin/activate
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --served-model-name ${JOB_NAME} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS} \
    --trust-remote-code \
    --max-model-len ${VLLM_MAX_MODEL_LEN}
fi
