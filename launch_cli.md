# HPC-launcher `launch` Command Documentation

## Overview

The `launch` command is a versatile tool for launching distributed jobs on HPC clusters or cloud environments. It provides a unified interface across different schedulers (SLURM, LSF, Flux) and supports both interactive and batch job submission.

## Synopsis

```bash
launch [options] command [args...]
```

## Command Structure

```bash
launch [-h] [--verbose] [-N NODES] [-n PROCS_PER_NODE] [--gpus-per-proc GPUS_PER_PROC]
       [-q QUEUE] [-t TIME_LIMIT] [-g GPUS_AT_LEAST] [--gpumem-at-least GPUMEM_AT_LEAST]
       [--exclusive] [--local] [--comm-backend {MPI,NCCL,RCCL,*CCL}]
       [-x KEY=VALUE [KEY=VALUE ...]] [--bg] [--batch-script BATCH_SCRIPT]
       [--scheduler {local,flux,slurm,lsf}]
       [-l [LAUNCH_DIR]] [-o OUTPUT_SCRIPT] [--setup-only] [--dry-run]
       [--account ACCOUNT] [--dependency DEPENDENCY] [-J JOB_NAME]
       [--reservation RESERVATION] [--save-hostlist]
       [-p KEY=VALUE [KEY=VALUE ...]] [--out OUT_LOG_FILE] [--err ERR_LOG_FILE]
       [--color-stderr] command [args...]
```

## Positional Arguments

| Argument | Description |
|----------|-------------|
| `command` | Command to be executed |
| `args` | Arguments to the command that should be executed |

## Optional Arguments

### General Options

| Option | Short Form | Description |
|--------|------------|-------------|
| `--help` | `-h` | Show help message and exit |
| `--verbose` | `-v` | Run in verbose mode, logging additional INFO-level messages about job setup and submission |

## Job Size Options

These options determine the number of nodes, accelerators, and ranks for the job.

| Option | Short Form | Description | Notes |
|--------|------------|-------------|-------|
| `--nodes` | `-N` | Specifies the number of requested nodes | |
| `--procs-per-node` | `-n` | Specifies the number of requested processes per node | Mutually exclusive with `-g` |
| `--gpus-per-proc` | | Specifies the number of requested GPUs per process | Default: 1 |
| `--cpus-per-task` | `-c` | Specifies the number of CPUs per task/process | Scheduler default if unset; inside an allocation gives the nested job step an exact CPU footprint so concurrent steps can pack side by side |
| `--queue` | `-q` | Specifies the queue to use | |
| `--time-limit` | `-t` | Set a time limit for the job in minutes | |
| `--gpus-at-least` | `-g` | Specifies the total number of accelerators requested | Mutually exclusive with `-n` and `-N` |
| `--gpumem-at-least` | | Constraint for accelerator memory needed (in GB) | System must be registered with launcher |
| `--exclusive` | | Request exclusive access from the scheduler | |
| `--local` | | Run locally (one process without batch scheduler) | |
| `--comm-backend` | | Indicate primary communication protocol (case-insensitive) | Choices: `MPI`, `NCCL`, `RCCL`, `*CCL`; any other value is a usage error |
| `--xargs` | `-x` | Specify scheduler and launch arguments | Format: `KEY=VALUE` |

### Notes on `--xargs`:
- Overrides (or adds) scheduler/launch arguments. Repeatable, and may take
  several space-separated `key=value` tokens: `-x k1=v1 k2=v2` or
  `-x k1=v1 -x k2=v2`.
- **Keys are passed through verbatim**, dashes included: overriding (or
  removing) a flag the launcher would otherwise set requires its exact
  spelling (`--ntasks` for SLURM, single-dash `-nnodes` for LSF, ...); any
  other key is added to the launch arguments exactly as given.
- **Dashed keys must use an attached form**: `-x--ntasks=8`,
  `--xargs=--ntasks=8`, or `-x-nnodes=2`. The space-separated form
  (`-x --ntasks=8`) is rejected because a token starting with `-` is parsed
  as another option.
- **Removal**: a leading tilde removes a key the launcher would otherwise set,
  e.g. `-x ~--ntasks` removes `--ntasks`. (A tilde-prefixed token may be
  space-separated; the tilde keeps it from looking like an option.)
- Values may contain `=`; the split is on the **first** `=` only, so
  `-x foo=a=b` sets `foo` to `a=b`.
- A double dash `--` is needed before the command if `-x` is the last option
  (otherwise the command token would be consumed as another `key=value`).

## Schedule Options

Arguments that determine when a job will run.

| Option | Description | Notes |
|--------|-------------|-------|
| `--bg` | Run job in background | Launcher won't wait for job start; uses timestamped directory by default |
| `--batch-script` | Launch a user-provided batch script | |
| `--scheduler` | Override default batch scheduler | Options: None, local, LocalScheduler, flux, FluxScheduler, slurm, SlurmScheduler, lsf, LSFScheduler |

## Script Options

Batch scheduler script parameters.

| Option | Short Form | Description | Notes |
|--------|------------|-------------|-------|
| `--launch-dir` | `-l` | Control launch directory creation | See detailed behavior below |
| `--output-script` | `-o` | Output job setup script file | Uses temporary file if not specified |
| `--setup-only` | | Only write job setup script without scheduling | |
| `--dry-run` | | Output results without side-effects | |
| `--account` | | Specify account/bank for the job | |
| `--dependency` | | Specify scheduler dependency | |
| `--job-name` | `-J` | Specify job name | |
| `--reservation` | | Add reservation argument | Typically for DAT runs |
| `--save-hostlist` | | Write hostlist to `hpc_launcher_hostlist.txt` | |

### `--launch-dir` Behavior:
- **No argument**: Creates timestamped launch directory
- **With argument**: Creates directory named `[LAUNCH_DIR]`
- **Argument = "."**: Creates launch script in current directory
- **Not set + blocking job**: Runs without creating files
- **Not set + non-blocking job (`--bg`)**: Creates a timestamped launch
  directory, the same as `-l` with no argument -- **not** the current
  directory (see the `--bg` row above)
- **Note**: Double dash `--` needed if this is the last argument

> **Important — the job runs from the launch directory.** When a launch
> directory is used, the generated job changes into it before running your
> command (`cd` for `--local`, `--chdir`/`--setattr=system.cwd` for the batch
> schedulers). The command itself is made absolute automatically, but **its
> arguments are passed through verbatim**, so a *relative* path argument (a
> script, a config file, a dataset) is resolved relative to the launch
> directory, not the directory you launched from. For example,
> `launch -l outdir python script.py` runs `cd outdir; python script.py` and
> fails to find `script.py` if it lives in your current directory. The launcher
> logs a warning when it detects such a relative-path argument. To avoid the
> problem, pass **absolute paths** for command arguments, or use `-l .` to run
> in the current directory.

## System Options

Provide system parameters from CLI - overrides built-in system descriptions and autodetection.

| Option | Short Form | Description | Format |
|--------|------------|-------------|--------|
| `--system-params` | `-p` | Specify system parameters | `KEY=VALUE` pairs |

### System Parameter Examples:
```bash
-p cores_per_node=128 gpus_per_node=8 gpu_arch=ampere mem_per_gpu=80 numa_domains=4 scheduler=slurm
```

Available parameters:
- `cores_per_node`: Integer value for CPU cores per node
- `gpus_per_node`: Integer value for GPUs per node
- `gpu_arch`: String value for GPU architecture
- `mem_per_gpu`: Float value for memory per GPU
- `numa_domains`: Integer value for NUMA domains
- `scheduler`: String value for scheduler type

**Note**: Double dash `--` needed if this is the last argument

## Logging Options

Control output and error logging.

| Option | Description |
|--------|-------------|
| `--out` | Capture standard output to a log file (console only if not specified) |
| `--err` | Capture standard error to a log file (console only if not specified) |
| `--color-stderr` | Use terminal colors to color stderr in red (doesn't affect output files) |

## Usage Examples

### Basic Examples

```bash
# Simple single-node job
launch -N 1 -n 1 hostname

# Multi-node MPI job
launch -N 4 -n 2 ./mpi_application

# GPU job with 2 GPUs per process
launch -N 2 -n 2 --gpus-per-proc 2 ./gpu_application
```

### Resource Specification

```bash
# Request specific number of GPUs total
launch -g 16 ./gpu_application

# Request allocation with at least 80GB GPU memory
launch --gpumem-at-least 80 ./memory_intensive_app

# Exclusive node access
launch -N 2 --exclusive ./exclusive_app

# Local execution without scheduler
launch -N 1 --local ./test_script.py
```

### Running Inside an Existing Allocation

When `launch` is invoked from inside a scheduler allocation (`salloc`,
`sbatch`, `flux alloc`, ...), a blocking launch runs as a *nested job step*
on the resources the allocation already holds -- it does not request a new
allocation.

```bash
# Inherit the allocation's node count (no job-size flag needed)
launch ./application

# Use a subset of the allocation
launch -N 1 ./application

# Two concurrent steps that pack side by side without sharing CPUs
launch -N 1 -c 8 ./application_a &
launch -N 1 -c 8 ./application_b &
```

Notes:

- With no `-N`/`-g`/`--gpumem-at-least`, the node count is inherited from the
  enclosing allocation.
- Requesting more nodes than the allocation holds is rejected upfront (SLURM
  would otherwise silently queue it as a brand-new allocation).
- On SLURM, allocation-selection flags (`--queue`/`--account`/
  `--reservation`) are dropped with a warning for a nested step: those
  properties are fixed by the enclosing allocation.
- On SLURM, nested steps are made concurrency-friendly so launches do not
  block each other on "step creation temporarily disabled": when a resource
  footprint is stated (`-c`/`--cpus-per-task` or a per-task GPU count),
  `--exact` is added so disjoint steps pack side by side; otherwise
  `--overlap` is added so steps may share resources (removable with
  `-x ~--overlap`). `--exclusive` suppresses both.
- Non-blocking (`--bg`) submissions are unaffected: submitting a batch job
  from inside an allocation deliberately creates a new job.
- A blocking ephemeral launch (one without `--launch-dir`) reports its live
  scheduler identity on stderr before starting user code. Flux prints
  `HPC_LAUNCHER_JOB_ID=<jobid>`; SLURM prints
  `HPC_LAUNCHER_STEP_ID=<jobid>.<stepid>`. Only rank zero emits the line.

### Job Scheduling

```bash
# Submit to specific queue with time limit
launch -q gpu_queue -t 120 -N 2 ./training_script.py

# Background job with custom name
launch --bg -J my_experiment -N 4 ./long_running_job

# Job with dependencies
launch --dependency afterok:12345 -N 1 ./dependent_job

# Use specific account/bank
launch --account project123 -N 2 ./billable_job
```

### Communication Backend

```bash
# MPI backend
launch --comm-backend MPI -N 4 -n 4 ./mpi_app

# NCCL backend for GPU communication
launch --comm-backend NCCL -N 2 -n 2 --gpus-per-proc 2 ./gpu_training
```

### Script and Directory Management

```bash
# Create timestamped launch directory
launch -l -N 2 ./my_job

# Use specific launch directory
launch -l experiment_001 -N 2 ./my_job

# Run in current directory
launch -l . -N 2 ./my_job

# Generate script without running
launch --setup-only -l -o job_script.sh -N 4 ./my_application

# Dry run to see what would be executed
launch --dry-run -N 4 -n 4 ./my_application
```

### System Override Examples

```bash
# Override system detection
launch -p cores_per_node=64 gpus_per_node=4 -N 2 ./custom_job

# Multiple system parameters
launch -p gpu_arch=sm_90 mem_per_gpu=32 scheduler=slurm -N 2 ./gpu_job
```

### Scheduler-Specific Arguments

```bash
# Multiple xargs
launch --xargs key1=val1 --xargs key2=val2 -N 2 ./job

# Override a flag the launcher would otherwise set (exact spelling, attached form)
launch -x--ntasks=8 -N 2 ./job

# Same for LSF's single-dash -nnodes
launch -x-nnodes=2 -N 2 ./job

# Remove a key the launcher would otherwise set
launch -x ~--ntasks -N 2 -- ./job
```

### Logging Configuration

```bash
# Capture output and error to files
launch -l --out output.log --err error.log -N 2 ./my_job

# Colored error output in terminal
launch --color-stderr -N 1 ./verbose_job

# Verbose mode with saved hostlist
launch -l --verbose --save-hostlist -N 4 ./debug_job
```

### Complex Example

```bash
# Production job with all options
launch \
  --verbose \
  -N 8 \
  -n 4 \
  --gpus-per-proc 2 \
  -q production \
  -t 480 \
  --exclusive \
  --comm-backend NCCL \
  --bg \
  --scheduler slurm \
  -l production_run_001 \
  --account ml_project \
  -J "ResNet Training" \
  --save-hostlist \
  --out output.log \
  --err error.log \
  python train_resnet.py --epochs 100 --batch-size 256
```

## Environment Variables

The `launch` command may set or use various environment variables depending on the scheduler and communication backend:

- For MPI jobs: Standard MPI environment variables
- For GPU jobs: CUDA-related environment variables
- For NCCL/RCCL: Communication library environment variables

## Exit Status

- `0`: Successful execution
- Non-zero: Error occurred (specific codes depend on failure type)

## Tips and Best Practices

1. **Use `--dry-run`** first to verify your command before submitting large jobs
2. **Use `--verbose`** for debugging to see detailed information about job submission
3. **Save scripts with `--setup-only`** to review and reuse job configurations
4. **Use timestamped directories** (`-l` without argument) for experiment tracking
5. **Specify `--account`** for proper resource accounting on shared systems
6. **Use `--save-hostlist`** when you need to know which nodes were allocated
7. **Set appropriate time limits** with `-t` to avoid jobs being killed prematurely
8. **Use `--exclusive`** for performance-critical jobs to avoid interference

## Scheduler Detection

The launcher automatically detects the available scheduler. Override with `--scheduler` if needed:
- `local`: Run without a scheduler
- `slurm`: SLURM workload manager
- `lsf`: IBM Spectrum LSF
- `flux`: Flux resource manager

## See Also

- [`torchrun-hpc`](./torchrun-hpc_cli.md) - PyTorch-specific distributed training launcher
- HPC-launcher documentation: https://github.com/LBANN/HPC-launcher
- LBANN documentation: https://lbann.readthedocs.io

---

*Generated from `launch -h` output*
