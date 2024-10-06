---
title: Distributed DL Training
---

# Distributed DL Traning

[repo](https://github.com/swartout/distributed)

## Log:

### Oct 5:

I typically use nvim + tmux for dev, as well as a virtual environment for dependencies. On Hyak I'll use these in an apptainer container with a fakeroot overlay, but this is extremely slow + laggy. I used a NGC pytorch docker container with apptainer when running torchtitan, it worked ok but had two issues:

1. Still a real pain to use any new dependencies
2. Didn't have nvim + tmux installed

I'm trying to modify the NGC container with apptainer, but it is still difficult. `apptainer pull --name pytorch.sif docker://nvcr.io/nvidia/pytorch:24.09-py3` is incredibly slow for some reason. Some other people on the nodes are running extremely intensive processes, which might be overloading the system. With that being said, increasing the cpu/mem for the slurm request does seem to speed it up.

See: https://github.com/apptainer/singularity/issues/6055 for possibly more info, let the others know to add time as well.

necessary commands:

- `apt install tmux`
- `apt install libfuse2 # maybe can bind/load fuse? IDK`
- `wget NEOVIM appimage`
- `mv the appimage`
- `apt install ripgrep`

Right now this is running with a sandbox, ideally I'd build this all within an apptainer container.

There's also a typo with the hyak docs, https://hyak.uw.edu/docs/hyak101/containers/build under "build the container".

### Oct 4:

Starting to write a distributed DL training repo from scratch. I'll be using torchtitan as a reference when necessary, but trying to do as much myself as possible. The goal of this is to learn and get some experience how to do "serious" pytorch training.

I'll be using Llama 3 as the implementation and run the code on UW's Hyak supercomputer.

My first goal is to implement Llama 3 and verify it loads properly.
