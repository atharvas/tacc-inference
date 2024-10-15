# TACC Inference: Easy inference on Slurm clusters
This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the ~~Vector Institute~~ TACC cluster environment**. 

All credits go to the authors of `https://github.com/VectorInstitute/vector-inference`.

## Installation

Clone this repo and install the pip package. I can register this with pypi if there is enough interest.
```bash
# I'm using miniconda; feel free to use your favourite package manager.
# I'm assuming you've already made the directories. Use `mkdir -p <name>` otherwise
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O $WORK/bin/miniconda3/miniconda.sh
$ bash $WORK/bin/miniconda3/miniconda.sh -u -p $WORK/bin/miniconda3
(base) $ git clone <this repo>
(base) $ cd <repo directory>
(base) $ pip install .
(base) $ tacc-inf --help
```


## Install vLLM Singularity container. 


We run vLLM through a self-contained singularity container. There is a great document explaining how TACC interfaces with singularity [here](https://containers-at-tacc.readthedocs.io/en/latest/singularity/01.singularity_basics.html). To make this container, follow these steps:


```bash
# Option 1: Get atharvas vLLM Docker container.
$ ls /home1/08277/asehgal/work/vista/tacc-inference/static/llm-train-serve_aarch64.sif
# <should echo the path; if it gives an error I haven't set the permissions correctly and you should open a github issue.>
$ cp /home1/08277/asehgal/work/vista/tacc-inference/static/llm-train-serve_aarch64.sif static/llm-train-serve_aarch64.sif

# Option 2: Make the docker container yourself.
$ cd static/
# comission a node for 20 minutes
$ idev -p gh-dev -N 1 -n 1 -t 00:20:00
$ module load tacc-apptainer/1.3.3
# build the apptainer config from the llm-train-serve github (build for GH100 with an aarch64 microarchitecture)
$ apptainer build llm-train-serve_aarch64.sif docker://ghcr.io/abacusai/gh200-llm/llm-train-serve@sha256:4ba3de6b19e8247ce5d351bf7dd41aa41bb3bffe8c790b7a2f4077af74c1b4ab
# Confirm that the SIF file file is in $WORK/tacc-inference/static with this exact name.
$ ls $WORK/tacc-inference/static
llm-train-serve_aarch64.sif
# Free up the dev compute node.
$ logout

# Option 3: Compile your own version of vLLM
# I have no experience using this. You might need to change the vllm.slurm files and I cannot provide much assistance here.
# https://docs.vllm.ai/en/stable/getting_started/installation.html#use-an-existing-pytorch-installation
# Don't use the conda environment.
$ conda deactivate
# comission a node for 40 minutes as it might take longer
$ idev -p gh-dev -N 1 -n 1 -t 00:40:00
$ module load gcc/14.2.0  cuda/12.5
$ module load python3
$ python3 -m venv $WORK/tacc-inference/vllm_env
$ source activate $WORK/tacc-inference/vllm_env
# get PyTorch compiled for aarch64
(vllm_env) $ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
# Download, Setup, and Build the vLLM repo.
(vllm_env) $ git clone https://github.com/vllm-project/vllm.git
(vllm_env) $ cd vllm
(vllm_env) $ python use_existing_torch.py
(vllm_env) $ pip install -r requirements-build.txt
(vllm_env) $ pip install -e . --no-build-isolation
(vllm_env) $ pip install tacc-inference
# Integrate the source activate vllm_env call in the vllm.slurm files. 
```





> [!NOTE]  
> The rest of the README is unmodified from the [VectorInstitute/vector-inference](https://github.com/VectorInstitute/vector-inference) README. The rest of the README will be updated once the code works for TACC.


## Launch an inference server
We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:
```bash
tacc-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="400" alt="launch_img" src="https://github.com/user-attachments/assets/557eb421-47db-4810-bccd-c49c526b1b43">

The model would be launched using the [default parameters](tacc_inf/models/models.csv), you can override these values by providing additional options, use `--help` to see the full list. You can also launch your own customized model as long as the model architecture is [supported by vLLM](https://docs.vllm.ai/en/stable/models/supported_models.html), you'll need to specify all model launching related options to run a successful run.

You can check the inference server status by providing the Slurm job ID to the `status` command:
```bash
tacc-inf status 13014393
```

You should see an output like the following:

<img width="400" alt="status_img" src="https://github.com/user-attachments/assets/7385b9ca-9159-4ca9-bae2-7e26d80d9747">

There are 5 possible states:

* **PENDING**: Job submitted to Slurm, but not executed yet. Job pending reason will be shown.
* **LAUNCHING**: Job is running but the server is not ready yet.
* **READY**: Inference server running and ready to take requests.
* **FAILED**: Inference server in an unhealthy state. Job failed reason will be shown.
* **SHUTDOWN**: Inference server is shutdown/cancelled.

Note that the base URL is only available when model is in `READY` state, and if you've changed the Slurm log directory path, you also need to specify it when using the `status` command.

Finally, when you're finished using a model, you can shut it down by providing the Slurm job ID:
```bash
tacc-inf shutdown 13014393

> Shutting down model with Slurm Job ID: 13014393
```

You call view the full list of available models by running the `list` command:
```bash
tacc-inf list
```
<img width="1200" alt="list_img" src="https://github.com/user-attachments/assets/a4f0d896-989d-43bf-82a2-6a6e5d0d288f">

You can also view the default setup for a specific supported model by providing the model name, for example `Meta-Llama-3.1-70B-Instruct`:
```bash
tacc-inf list Meta-Llama-3.1-70B-Instruct
```
<img width="400" alt="list_model_img" src="https://github.com/user-attachments/assets/5dec7a33-ba6b-490d-af47-4cf7341d0b42">

`launch`, `list`, and `status` command supports `--json-mode`, where the command output would be structured as a JSON string.

## Send inference requests
Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/completions.py`, and you should expect to see an output like the following:
> {"id":"cmpl-c08d8946224747af9cce9f4d9f36ceb3","object":"text_completion","created":1725394970,"model":"Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"text":" is a question that many people may wonder. The answer is, of course, Ottawa. But if","logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":{"prompt_tokens":8,"total_tokens":28,"completion_tokens":20}}

**NOTE**: For multimodal models, currently only `ChatCompletion` is available, and only one image can be provided for each prompt.

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
Where the last number in the URL is the GPU number (gpu029 in this case). The example provided above is for the vector cluster, change the variables accordingly for your environment
