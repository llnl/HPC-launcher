# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
from typing import Optional


def num_nodes_in_current_allocation() -> Optional[int]:
    """
    The node count of the scheduler allocation this process is running
    inside, or ``None`` when not inside an allocation.

    Unlike ``Scheduler.num_nodes_in_allocation`` this is scheduler-agnostic:
    it is consulted *before* a scheduler has been selected (CLI argument
    validation), so it asks every scheduler class in turn rather than
    assuming one. Each class recognizes only its own allocation
    (``Scheduler.in_allocation``), so the first that reports a count wins.
    Flux is asked before Slurm on purpose: a Flux instance started inside a
    Slurm job is the allocation the user is actually working in.

    :return: Number of nodes in the enclosing allocation, or None.
    """
    seen = set()
    for scheduler in get_schedulers().values():
        if scheduler in seen:
            continue
        seen.add(scheduler)
        nodes = scheduler.num_nodes_in_allocation()
        if nodes is not None:
            return nodes
    return None


def get_schedulers():
    from .local import LocalScheduler
    from .flux import FluxScheduler
    from .slurm import SlurmScheduler
    from .lsf import LSFScheduler

    # Order matters to num_nodes_in_current_allocation: Flux before Slurm.
    return {
        None: LocalScheduler,
        "local": LocalScheduler,
        "LocalScheduler": LocalScheduler,
        "flux": FluxScheduler,
        "FluxScheduler": FluxScheduler,
        "slurm": SlurmScheduler,
        "SlurmScheduler": SlurmScheduler,
        "lsf": LSFScheduler,
        "LSFScheduler": LSFScheduler,
    }

def parse_env_list(*e) -> str:
    if len(e) == 1:
        m = e[0]
        return f"{m}\n"
    elif len(e) == 2:
        k,v = e
        return f"export {k}={v}\n"
    elif len(e) == 3:
        k,v,m = e
        return f"export {k}={v}\t\t# {m}\n"
    else:
        return f'{e}'
