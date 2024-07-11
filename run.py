#!/usr/bin/env python3
import argparse
import re
from typing import List, Callable, Union

import numpy as np
import pandas as pd

from sch.assignment import Assignment, VM, Host, Assignment
from sch.scheduler import Scheduler, FirstFit, CloseRadiusFit, RandomFit
from sch.lb import CloseRadiusLB
from sch.ub import PrefixUB

def load_vms_data(path: str) -> List[VM]:
    period_time_steps = 24 # Two hours
    vms = []

    if path.endswith(".json") >=0:
        df = pd.read_json(path)
        for index, row in df.iterrows():
            vms.append(VM(index=index,
                          cpu_util=np.array(row['vm_util'][:period_time_steps])))

    else:
        raise TypeError("File format is not supported")
    return vms

def evaluate(placers:List[Union[Scheduler, CloseRadiusLB, PrefixUB]],
                vms:List[VM],
                hosts:List[Host],
                p: float=0.95) -> pd.DataFrame:

    results = []

    for placer in placers:
        placer = placer()
        if placer.__class__.__name__ == "CloseRadiusLB":
            asg = CloseRadiusLB().place(hosts, p, vms)
            vms_num = len(asg.vms)

        elif placer.__class__.__name__ == "PrefixUB":
            # Calculate lower bound on number of VMs
            asg = CloseRadiusLB().place(hosts, p, vms)
            l_vms = len(asg.vms)
            # Pass lower bound to PrefixUb solver
            vms_num = PrefixUB().set_env(hosts, vms, p).solve(l_vms)

        else:
            # online scheduler
            asg = placer.place(hosts, p, vms)
            vms_num = len(asg.vms)

        _result = {"placed": vms_num, "Placer":placer.__class__.__name__}

        results.append(_result)

    return pd.DataFrame(results)


def run_example(filePath: str, p: float=0.95) -> None:
    VM.SYM = True #If turn on symmetryzer of CPU utilization
    VM.TP = 8 # Timestamps used for prediction of CPU statistics
    CORES_PER_HOST = 44 # Number of cores per host

    hosts_number = int(re.findall(".*/(\d+)_\d+.json", filePath)[0])
    vms:List[VM] = load_vms_data(filePath)
    hosts:List[Host] = [Host(index=i, cpu_capacity=CORES_PER_HOST) for i in range(hosts_number)]

    schedulers = [CloseRadiusFit, FirstFit, RandomFit, CloseRadiusLB]
    res = evaluate(schedulers, vms, hosts, p=p)
    print(res.sort_values("placed", ascending=False).to_string(index=False))

    print("\nLong time UpperBound calculation")
    schedulers = [PrefixUB]
    res = evaluate(schedulers, vms, hosts, p=p)
    print(res)

def process_input_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=str, required=True,
                        help='path to the .json containing trace')
    parser.add_argument('--placers', type=str, nargs='+', required=True,
                        help=f'algorithms to run, possible algorithms: '
                             f'{[alg.__name__ for alg in registry.values()]}')
    parser.add_argument('--p', type=float, required=True,
                        help='Probability of no hotspots')
    parser.add_argument('--sym', type=bool, required=False, default=True,
                        help='If true, turn on symmetrization for CPU utilization (bool, default=True)')
    parser.add_argument('--tp', type=int, required=False, default=8,
                        help='Number of timestamps used for prediction of CPU statistics, (int, default=8)')

    return parser.parse_args()

def run_from_command_line() -> None:
    args = process_input_arguments()
    VM.SYM = args.sym
    VM.TP = args.tp
    CORES_PER_HOST = 44

    hosts_number = int(re.findall(".*/(\d+)_\d+.json", args.trace)[0])
    vms:List[VM] = load_vms_data(args.trace)
    hosts:List[Host] = [Host(index=i, cpu_capacity=CORES_PER_HOST) for i in range(hosts_number)]
    placers = [registry[placer] for placer in args.placers]

    res = evaluate(placers, vms, hosts, p=args.p)
    print(res.sort_values("placed", ascending=False).to_string(index=False))

if __name__ == "__main__":
    np.random.seed(42)
    registry = {
            'FirstFit': FirstFit,
            'CloseRadiusFit': CloseRadiusFit,
            'RandomFit': RandomFit,
            'CloseRadiusLB': CloseRadiusLB,
            'PrefixUB': PrefixUB,
    }
    run_from_command_line()
    #run_example("dataset/50_1.json", p=0.95)
