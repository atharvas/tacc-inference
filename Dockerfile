FROM nvcr.io/nvidia/pytorch:24.09-py3

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/compat/lib.real/"

RUN apt update

# Install tacc-inf
# RUN python -m pip install vllm ray pandas vllm-nccl-cu12 cupy-cuda12x
RUN python -m pip install git+https://github.com/atharvas/tacc-inference.git@main#egg=tacc-inf[dev]

# Install Flash Attention 2 backend
RUN python -m pip install flash-attn --no-build-isolation

# # Move nccl to accessible location
RUN mkdir -p /tacc-inf/nccl
RUN mv /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1 /tacc-inf/nccl/libnccl.so.2.18.1;

# Set the default command to start an interactive shell
CMD ["bash"]