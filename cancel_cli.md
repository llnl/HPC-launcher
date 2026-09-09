# `hpc-launcher-cancel`

`hpc-launcher-cancel` cancels a job or job step using the native command for
the selected scheduler.

```text
usage: hpc-launcher-cancel [-h] [--scheduler {flux,slurm,lsf,local}]
                           [--verbose]
                           [identifier]
```

The identifier may be a raw scheduler ID, a marker copied directly from an
ephemeral launch, or the name of an exported marker:

```bash
hpc-launcher-cancel HPC_LAUNCHER_STEP_ID=1234.7
hpc-launcher-cancel --scheduler flux f5wWq9
export HPC_LAUNCHER_JOB_ID=f5wWq9
hpc-launcher-cancel
```

`HPC_LAUNCHER_STEP_ID` identifies Slurm automatically, and
`HPC_LAUNCHER_JOB_ID` identifies Flux. For a raw ID, the scheduler is
autodetected unless `--scheduler` is supplied. Use `--scheduler` when an ID is
ambiguous or when running somewhere the scheduler cannot be detected.

The native commands are:

| Scheduler | Command |
| --- | --- |
| Flux | `flux cancel ID` |
| Slurm | `scancel ID` |
| LSF | `bkill ID` |
| Local | `kill -TERM PID` |

The command exits with the native cancellation command's status. A missing
native command returns status 127.
