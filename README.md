# HPC-launcher Repository

The HPC launcher repository contains a set of helpful scripts and
Python bindings for launching PyTorch (torchrun), LBANN 2.0
(PyTorch-core), or generic scripts on multiple leadership-class HPC
systems.  There are optimized routines for FLUX, SLURM, and LSF
launchers.  Additionally, there are optimized environments for systems
at known compute centers.  Currently there are supported systems at:
 - LLNL Livermore Computing (LC)
 - LBL NERSC (Pending)
 - ORNL OLCF (Pending)
 - RIKEN (Pending)

There are two main entry points into HPC-Launcher from the cli:
`launch` and `torchrun-hpc`.  `torchrun-hpc` is intended as a
replacement for `torchrun`, while `launch` is a generic interface for
launching parallel jobs.

## Installation

To install the package, install released versions from PyPI run:
```bash
pip install hpc-launcher
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/LBANN/HPC-launcher.git
```

GPU vendor management libraries used for autodetecting accelerators
(`amdsmi` for AMD, `nvidia-ml-py` for NVIDIA) are optional extras rather
than a default part of the install, since the base install has no way to
know what hardware the *runtime* machine will have -- only the one it
happens to be built on. Opt in to the one(s) matching your systems:
```bash
pip install hpc-launcher[rocm]
pip install hpc-launcher[cuda]
```

`amdsmi` is not version-constrained by `[rocm]`, because the version that
works is the one matching the ROCm install on the machine you *run* on,
which the package cannot know. When installing from source *on* the machine
you will run on, use `[rocm-auto]` instead -- it reads the local ROCm
version (`$ROCM_PATH/.info/version`, default `/opt/rocm`) at install time
and constrains `amdsmi` to the closest release PyPI offers:
```bash
pip install 'hpc-launcher[rocm-auto] @ git+https://github.com/LBANN/HPC-launcher.git'
```
(`[rocm-auto]` only helps for source installs: a pre-built wheel's
dependency metadata is frozen on the machine that built it.) If
autodetection fails with an `undefined symbol` error from `amdsmi`, install
the build matching your local ROCm, e.g. `pip install 'amdsmi~=6.4.2'` for
ROCm 6.4.2.

## Example Usage

Using the launch command to execute a command in parallel
```
launch -N1 -n1 hostname
```

Using the torchrun-hpc command to execute a PyTorch Python file in parallel on two nodes and four processes per node (8 in total):
```
torchrun-hpc -N2 -n4 file.py [arguments to Python file]
```

Using HPC-Launcher within existing PyTorch code with explicity invoking it from the command line (CLI).  Within the top level Python file, import `hpc_launcher.torch` first to ensure that `torch` is configured per HPC-Launcher's specification.
```
import hpc_launcher.torch
```

## HPC-Launcher command-line tools

- [`launch`](./launch_cli.md) - General purpose HPC job launcher
- [`torchrun-hpc`](./torchrun-hpc_cli.md) - PyTorch-specific distributed training launcher
- [`hpc-launcher-cancel`](./cancel_cli.md) - Cancel a launched job or job step

# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

![Block Diagram of LBANN v2.x](LBANN_2.0_Block_Diagram.png "Block
Diagram of LBANN v2.x architecture")

LBANN v2.x is composed of a custom backend LBANN device that is used
to provide processor-centric optimizations such as copy-elision for
AMD MI300A APUs.  Additionally, it is composed of Python, C++, CUDA,
and ROCm custom kernels that extend PyTorch 2.4+.  Libraries such as
DGraph, DistConv, and CheckMate, implement key algorithms using the
PyTorch 2.x API.  Each of these libraries should be both composable as
well as fully separable.  The suite of LBANN 2.x optimizations are
found in the [LBANN](https://github.com/orgs/LBANN/repositories)
GitHub group.

## Publications

A list of publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/LBANN/HPC-launcher/issues).
