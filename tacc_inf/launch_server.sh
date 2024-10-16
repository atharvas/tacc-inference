#!/bin/bash

# ================================= Read Named Args ======================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-family) model_family="$2"; shift ;;
        --model-variant) model_variant="$2"; shift ;;
        --partition) partition="$2"; shift ;;
        --qos) qos="$2"; shift ;;
        --time) walltime="$2"; shift ;;
        --num-nodes) num_nodes="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --max-model-len) max_model_len="$2"; shift ;;
        --vocab-size) vocab_size="$2"; shift ;;
        --data-type) data_type="$2"; shift ;;
        --venv) virtual_env="$2"; shift ;;
        --log-dir) log_dir="$2"; shift ;;
        --pipeline-parallelism) pipeline_parallelism="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

required_vars=(model_family model_variant partition qos walltime num_nodes num_gpus max_model_len vocab_size)

for var in "$required_vars[@]"; do
    if [ -z "$!var" ]; then
        echo "Error: Missing required --$var//_/- argument."
        exit 1
    fi
done

export MODEL_FAMILY=$model_family
export MODEL_VARIANT=$model_variant
export JOB_PARTITION=$partition
export QOS=$qos
export WALLTIME=$walltime
export NUM_NODES=$num_nodes
export NUM_GPUS=$num_gpus
export VLLM_MAX_MODEL_LEN=$max_model_len
export VLLM_MAX_LOGPROBS=$vocab_size
# For custom models, the following are set to default if not specified
export VLLM_DATA_TYPE="auto"
export VENV_BASE="base"
export LOG_DIR="default"
# Pipeline parallelism is disabled and can only be enabled if specified in models.csv as this is an experimental feature
export PIPELINE_PARALLELISM="false"

if [ -n "$data_type" ]; then
    export VLLM_DATA_TYPE=$data_type
fi

if [ -n "$virtual_env" ]; then
    export VENV_BASE=$virtual_env
fi

if [ -n "$log_dir" ]; then
    export LOG_DIR=$log_dir
fi

if [ -n "$pipeline_parallelism" ]; then
    export PIPELINE_PARALLELISM=$pipeline_parallelism
fi

# ================================= Set default environment variables ======================================
# Slurm job configuration
export JOB_NAME="$MODEL_FAMILY-$MODEL_VARIANT"
if [ "$LOG_DIR" = "default" ]; then
    export LOG_DIR="$HOME/.tacc-inf-logs/$MODEL_FAMILY"
fi
mkdir -p $LOG_DIR

# Model and entrypoint configuration. API Server URL (host, port) are set automatically based on the
# SLURM job and are written to the file specified at VLLM_BASE_URL_FILENAME
export SRC_DIR="$(dirname "$0")"
export MODEL_DIR="${SRC_DIR}/models/${MODEL_FAMILY}"
export VLLM_BASE_URL_FILENAME="${MODEL_DIR}/.${JOB_NAME}_url"
mkdir -p $(dirname $VLLM_BASE_URL_FILENAME)

# Variables specific to your working environment, below are examples for the Vector cluster
export VLLM_MODEL_WEIGHTS="/model-weights/$JOB_NAME"
# This is a hack for `vista.tacc.utexas.edu`. I ran:
# $ module load cuda/11.8
# $ echo $LD_LIBRARY_PATH
# export LD_LIBRARY_PATH="/opt/apps/ucx/1.17.0/lib:/opt/apps/nvidia24/cuda12/openmpi/5.0.5/lib:/home1/apps/nvidia/Linux_aarch64/24.7/cuda/12.5/extras/CUPTI/lib64:/home1/apps/nvidia/Linux_aarch64/24.7/cuda/12.5/lib64:/home1/apps/nvidia/Linux_aarch64/24.7/compilers/extras/qd/lib:/home1/apps/nvidia/Linux_aarch64/24.7/compilers/lib"


# ================================ Validate Inputs & Launch Server =================================

# Set data type to fp16 instead of bf16 for non-Ampere GPUs
fp16_partitions="t4v1 t4v2"

# choose from 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
if [[ $fp16_partitions =~ $JOB_PARTITION ]]; then
    export VLLM_DATA_TYPE="float16"
    echo "Data type set to due to non-Ampere GPUs used: $VLLM_DATA_TYPE"
fi

# Create a file to store the API server URL if it doesn't exist
if [ -f $VLLM_BASE_URL_FILENAME ]; then
    touch $VLLM_BASE_URL_FILENAME
fi

echo Job Name: $JOB_NAME
echo Partition: $JOB_PARTITION
echo Num Nodes: $NUM_NODES
echo GPUs per Node: $NUM_GPUS
echo QOS: $QOS
echo Walltime: $WALLTIME
echo Data Type: $VLLM_DATA_TYPE

is_special=""
if [ "$NUM_NODES" -gt 1 ]; then
    is_special="multinode_"
fi

sbatch --job-name $JOB_NAME \
    --partition $JOB_PARTITION \
    --nodes $NUM_NODES \
    --time $WALLTIME \
    --output $LOG_DIR/$JOB_NAME.%j.out \
    --error $LOG_DIR/$JOB_NAME.%j.err \
    $SRC_DIR/${is_special}vllm.slurm


# echo sbatch --job-name $JOB_NAME \
#     --partition $JOB_PARTITION \
#     --nodes $NUM_NODES \
#     --gres gpu:$NUM_GPUS \
#     --qos $QOS \
#     --time $WALLTIME \
#     --output $LOG_DIR/$JOB_NAME.%j.out \
#     --error $LOG_DIR/$JOB_NAME.%j.err \
#     $SRC_DIR/${is_special}vllm.slurm
