---
title: torchtitan
---

# `torchtitan` repo

Notes on understanding everything within the `torchtitan` [repo](https://github.com/pytorch/torchtitan). `torchtitan` is a MVP hackable repo for large-scale LLM training in PyTorch. I'll be learning more about it to better understand distributed training and hopefully re-implement on UW HYAK.

Repo tree (as of [Sept 19](https://github.com/pytorch/torchtitan/tree/d2a4904f58accc683c17c66a360026cb3c8109af)):

```
.
├── [ ] CODE_OF_CONDUCT.md
├── [ ] CONTRIBUTING.md
├── [ ] LICENSE
├── [ ] README.md
├── assets
│   └── images
│       ├── [ ] llama2_loss_curves.png
│       ├── [ ] llama3_1_405B_loss_curves.png
│       ├── [ ] llama3_loss_curves.png
│       ├── [ ] readme.md
│       └── [ ] titan_play_video.png
├── [ ] create_seed_checkpoint.sh
├── [ ] dev-requirements.txt -> .ci/docker/dev-requirements.txt
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

### `train.py`

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
