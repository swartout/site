---
title: torchtitan
---

# `torchtitan` repo

Notes on understanding everything within the `torchtitan` [repo](https://github.com/pytorch/torchtitan). `torchtitan` is a MVP hackable repo for large-scale LLM training in PyTorch. I'll be learning more about it to better understand distributed training and hopefully re-implement on UW HYAK.

Repo tree (as of [Sept 19](https://github.com/pytorch/torchtitan/tree/d2a4904f58accc683c17c66a360026cb3c8109af)):

```
.
├── [x] CODE_OF_CONDUCT.md
├── [x] CONTRIBUTING.md
├── [x] LICENSE
├── [ ] README.md
├── assets
│   └── images
│       ├── [x] llama2_loss_curves.png
│       ├── [x] llama3_1_405B_loss_curves.png
│       ├── [x] llama3_loss_curves.png
│       ├── [x] readme.md
│       └── [x] titan_play_video.png
├── [ ] create_seed_checkpoint.sh
├── [x] dev-requirements.txt -> .ci/docker/dev-requirements.txt
├── docs
│   ├── [ ] checkpoint.md
│   ├── [ ] composability.md
│   ├── [ ] float8.md
│   ├── [ ] fsdp.md
│   ├── [ ] license_header.txt
│   └── [ ] performance.md
├── [ ] estimation.py
├── [ ] multinode_trainer.slurm
├── [ ] pyproject.toml
├── [ ] requirements.txt -> .ci/docker/requirements.txt
├── [x] run_llama_train.sh
├── [ ] run_memory_estimation.sh
├── test
│   ├── [ ] __init__.py
│   ├── assets
│   │   ├── [ ] c4_test
│   │   │   └── [ ] data.json
│   │   └── [ ] test_tiktoken.model
│   ├── datasets
│   │   ├── [ ] __init__.py
│   │   └── [ ] test_checkpoint.py
│   ├── [ ] test_fused_rms_norm.py
│   └── [ ] test_job_config.py
├── [ ] test_runner.py
├── torchtitan
│   ├── [ ] checkpoint.py
│   ├── [ ] config_manager.py
│   ├── datasets
│   │   ├── [ ] __init__.py
│   │   ├── [ ] download_tokenizer.py
│   │   ├── [ ] hf_datasets.py
│   │   └── tokenizer
│   │       ├── [ ] __init__.py
│   │       ├── [ ] sentencepiece.py
│   │       ├── [ ] tiktoken.py
│   │       └── [ ] tokenizer.py
│   ├── [ ] float8.py
│   ├── [ ] logging.py
│   ├── [ ] metrics.py
│   ├── models
│   │   ├── [ ] __init__.py
│   │   ├── llama
│   │   │   ├── [ ] __init__.py
│   │   │   └── [ ] model.py
│   │   └── [ ] norms.py
│   ├── [ ] optimizer.py
│   ├── parallelisms
│   │   ├── [ ] __init__.py
│   │   ├── [ ] parallel_dims.py
│   │   ├── [ ] parallelize_llama.py
│   │   ├── [ ] pipeline_llama.py
│   │   ├── [ ] pipelining_utils.py
│   │   └── [ ] utils.py
│   ├── [ ] profiling.py
│   └── [ ] utils.py
├── [ ] train.py
├── train_configs
│   ├── [ ] debug_model.toml
│   ├── [ ] llama2_13b.toml
│   ├── [ ] llama2_70b.toml
│   ├── [ ] llama2_7b.toml
│   ├── [ ] llama3_405b.toml
│   ├── [ ] llama3_70b.toml
│   └── [ ] llama3_8b.toml
└── [ ] version.txt

```

## Files:

### `run_llama_train.sh`

Runs training on a Llama model, primarily calling `torchrun` while getting some config options. `torchrun` launches a distributed training job, automatically assigning worker IDs and scaling up and down with available hardware. As far as I'm aware, this is only for single-node training.

```
torchrun --nproc_per_node=${NGPU} \
         --rdzv_backend c10d \
         --rdzv_endpoint="localhost:0" \
         --local-ranks-filter ${LOG_RANK} \
         --role rank \
         --tee 3 \
         train.py \
         --job.config_file ${CONFIG_FILE} $overrides
```

Arg info:

- `--nproc_per_node=${NGPU}`: number of workers per node, tied to the number of GPUs. Each GPU needs its own worker to handle it.
- `--rdzv_backend c10d`: rendezvous backend, used to coordinate all workers *before* training starts. `c10d` is recommended and supported out-of-the-box.
- `--rdzv_endpoint="localhost:0"`: endpoint where backend is running.
- `--local-ranks-filter ${LOG_RANK}`: only show logs in stdout/stderr from the given ranks.
- `--role rank`: specifies the "role" that the workers should do (e.g. trainer vs evaluator). Only one role can be used per `torchrun` call, but multiple calls can be made.
- `--tee 3`: duplicates stdout/stderr to both console and log files.
- `train.py`: training script.
- `--job.config_file ${CONFIG_FILE}`: config file.
- `$overrides`: additional overrides.

### `train.py` notes:

Main training file for `torchtitan`.

Notes:

- `get_train_context`: creates a context manager to use loss parallelism and/or compiling backprop in training loops.
- Sets up a distributed "mesh", where different dimensions correspond to parallelism techniques. Supported techniques:
  - Data parallelism: duplicate the model across different devices, so that different data can be processed in parallel.
  - Tensor parallelism: different parts of the model (tensors) are put on different devices. Forward/backward passes are sent between different devices.
  - Pipeline parallelism: splitting the model apart into sequential portions, processing multiple mini-batches in parallel to reduce idle time. TODO: learn more
  - Loss parallelism: parallelizing loss calculation across multiple devices. TODO: learn more
- TODOs, skipped:
  - logging
  - tokenization (although this seems straightforward)
  - data loading
  - model configuration options
  - Float8
  - checkpointing

### Parallelization techniques:

- DDP:
  - data parallel

### Model notes:

- Use `@dataclass` for model config
- Use `@classmethod` for `from_model_args`
- Modular norms: I assume this is because of possible data parallelism?
- TODO:
  - learn rotary embeddings

### Dataset

- Dataset and dataloaders are "stateful"
  - Stateful datasets are shared between threads and the datasets themselves generate batches

### Parallelism:

- Uses a `ParallelDims` dataclass to validate and configure parallelization
  - Initializes the "device mesh"

### Training config:

```
# torchtitan Config.toml

[job]
dump_folder = "./outputs" # dumps job outputs
description = "Llama 3 debug training"
use_for_integration_test = true # only to add to integ test suite

[profiling]
enable_profiling = true # to use profiling or not
save_traces_folder = "profile_trace"
profile_freq = 10
enable_memory_snapshot = false
save_memory_snapshot_folder = "memory_snapshot"

[metrics]
log_freq = 1 # how often to log to tensorboard
enable_color_printing = true
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "debugmodel" # what model config to use, e.g. big/small
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm / fused_rmsnorm
# test tokenizer.model, for debug purpose only
tokenizer_path = "./test/assets/test_tiktoken.model"

[optimizer]
name = "AdamW"
lr = 8e-4

[training]
batch_size = 8
seq_len = 2048
warmup_steps = 2  # lr scheduler warm up, normally 20% of the train steps
max_norm = 1.0  # grad norm clipping
steps = 10
# DP degree, disabled at 1, weights are replicated across this many degrees, if shard_deg >1 HSDP, else DDP
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1 # when >1, weights are sharded across this many degrees. if dp_rep >1 HSDP, else FSDP
tensor_parallel_degree = 1
compile = false
dataset = "c4_test"  # supported datasets: c4_test (2K), c4 (177M)

[experimental]
pipeline_parallel_degree = 1
enable_async_tensor_parallel = false

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval_type = "steps"
interval = 5
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled"  # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = 'selective'  # ['none', 'selective', 'full']
selective_ac_option = '2'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[float8]
enable_float8_linear = false```

## log dumps:

### llama2 7b no compile, batch size 4

- uses FSDP2

```
2024-10-03 15:01:18,520 - root - INFO - step: 117  loss:  3.9053  memory: 36.20GiB(81.50%)  wps: 1,710  mfu: 23.50%
2024-10-03 15:01:23,311 - root - INFO - step: 118  loss:  3.9014  memory: 36.20GiB(81.50%)  wps: 1,710  mfu: 23.50%
2024-10-03 15:01:28,097 - root - INFO - step: 119  loss:  3.7841  memory: 36.20GiB(81.50%)  wps: 1,712  mfu: 23.52%
2024-10-03 15:01:32,876 - root - INFO - step: 120  loss:  3.9687  memory: 36.20GiB(81.50%)  wps: 1,715  mfu: 23.56%
2024-10-03 15:01:37,675 - root - INFO - step: 121  loss:  3.9105  memory: 36.20GiB(81.50%)  wps: 1,708  mfu: 23.46%
2024-10-03 15:01:42,463 - root - INFO - step: 122  loss:  3.7040  memory: 36.20GiB(81.50%)  wps: 1,712  mfu: 23.52%
2024-10-03 15:01:47,265 - root - INFO - step: 123  loss:  3.7633  memory: 36.20GiB(81.50%)  wps: 1,707  mfu: 23.45%
2024-10-03 15:01:52,045 - root - INFO - step: 124  loss:  3.7565  memory: 36.20GiB(81.50%)  wps: 1,714  mfu: 23.55%
2024-10-03 15:01:56,839 - root - INFO - step: 125  loss:  3.7816  memory: 36.20GiB(81.50%)  wps: 1,710  mfu: 23.49%
2024-10-03 15:02:01,631 - root - INFO - step: 126  loss:  3.7840  memory: 36.20GiB(81.50%)  wps: 1,710  mfu: 23.50%
2024-10-03 15:02:06,441 - root - INFO - step: 127  loss:  3.7705  memory: 36.20GiB(81.50%)  wps: 1,703  mfu: 23.40%
2024-10-03 15:02:11,234 - root - INFO - step: 128  loss:  3.7049  memory: 36.20GiB(81.50%)  wps: 1,710  mfu: 23.49%
2024-10-03 15:02:16,070 - root - INFO - step: 129  loss:  3.6569  memory: 36.20GiB(81.50%)  wps: 1,694  mfu: 23.28%
2024-10-03 15:02:20,869 - root - INFO - step: 130  loss:  3.5210  memory: 36.20GiB(81.50%)  wps: 1,707  mfu: 23.46%```

### llama2 7b with compile, batch size 4

- same as above, but with compile
- broke lol
  - `.Please create a symlink of libcuda.so to any of the files.`
  - there was a lot of issues i had to do fix
    - uninstalling triton
    - `export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib/libcuda.so`
- roughly 17% higher wps

```
2024-10-03 16:44:28,481 - root - INFO - step:  1  loss: 10.8235  memory: 26.73GiB(60.17%)  wps: 1,176  mfu: 16.15%
2024-10-03 16:44:32,658 - root - INFO - step:  2  loss: 10.6590  memory: 33.11GiB(74.54%)  wps: 1,962  mfu: 26.95%
2024-10-03 16:44:36,851 - root - INFO - step:  3  loss: 10.3083  memory: 33.11GiB(74.54%)  wps: 1,954  mfu: 26.85%
2024-10-03 16:44:41,065 - root - INFO - step:  4  loss:  9.9934  memory: 33.11GiB(74.54%)  wps: 1,944  mfu: 26.71%
2024-10-03 16:44:45,256 - root - INFO - step:  5  loss:  9.5579  memory: 33.11GiB(74.54%)  wps: 1,956  mfu: 26.87%
2024-10-03 16:44:49,442 - root - INFO - step:  6  loss:  9.4216  memory: 33.11GiB(74.54%)  wps: 1,957  mfu: 26.89%
2024-10-03 16:44:53,645 - root - INFO - step:  7  loss:  9.0326  memory: 33.11GiB(74.54%)  wps: 1,950  mfu: 26.79%
2024-10-03 16:44:57,844 - root - INFO - step:  8  loss:  8.9617  memory: 33.11GiB(74.54%)  wps: 1,952  mfu: 26.82%
2024-10-03 16:45:02,050 - root - INFO - step:  9  loss:  8.9453  memory: 33.11GiB(74.54%)  wps: 1,948  mfu: 26.77%
2024-10-03 16:45:06,242 - root - INFO - step: 10  loss:  8.7203  memory: 33.11GiB(74.54%)  wps: 1,955  mfu: 26.86%```


