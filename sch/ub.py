import math
from typing import List, Callable

import numpy as np
from ortools.linear_solver import pywraplp

from sch.gamma import calculate_gammas
from sch.assignment import Assignment, Host, VM

class PrefixUB:
    """PrefixUB calculates upper bound on the number of VMs which can
    be placed into hosts such that probability of hotspots is less than p.
    """
    def set_env(self, hosts: List[Host], vms: List[VM], p:float = 0.95):
        self.vms = np.array([[vm.util_center, vm.util_radius] for vm in vms])
        self.hosts = np.array([host.cpu_capacity for host in hosts], dtype=np.float64)
        self.gammas = calculate_gammas(Assignment.MAX_VMS_PH, 1.0 - p)
        self.gammas_conv = get_concave_gammas(Assignment.MAX_VMS_PH, 1.0 - p)
        return self

    def solve(self, l_vms:int = 0):
        total_cap = np.sum(self.hosts, axis=0)

        u_vms = len(self.vms)
        u_cap = self.capacity_lb(self.vms)

        if u_cap > total_cap:
            while u_vms - l_vms > 1:
                m_vms = (u_vms + l_vms) // 2
                cap = self.capacity_lb(self.vms[:m_vms])
                if cap > total_cap:
                    u_vms = m_vms
                else:
                    l_vms = m_vms
        else:
            l_vms = u_vms

        return l_vms


    def capacity_lb(self, vms:np.ndarray):
        r_order = np.argsort(vms[:, 1])[::-1]
        vms = vms[r_order]

        steps = self.branched_bin_search(0, len(vms) + 1, vms, self.gamma_lb)
        mask = np.zeros(len(vms), dtype=bool)
        last_value = 0
        for step, value in steps:
            delta = value - last_value
            while delta > 0:
                if not mask[step - 1]:
                    mask[step - 1] = True
                    delta -= 1
                step -= 1
            last_value = value
        return np.array(np.sum(vms[:, 0]) + np.sum(vms[mask, 1]))

    def gamma_lb(self, vms: np.ndarray, n: int):
        vms = vms[:n]
        if len(vms) == 0:
            return 0

        n_vms_nh = np.zeros(len(self.hosts), dtype=int)
        hosts_number = len(self.hosts)
        min_r = np.min(vms[:, 1])

        idx = np.argsort(vms[:, 0])
        vms = vms[idx]

        rems = np.copy(self.hosts)
        hid = 0
        for vm in vms:
            cpu_contribution = vm[0]
            if self.gammas[n_vms_nh[hid]] < self.gammas[n_vms_nh[hid] + 1]:
                cpu_contribution += min_r

            rems[hid] -= cpu_contribution
            n_vms_nh[hid] += 1
            if rems[hid] < 0:
                if hid + 1 < hosts_number:
                    rems[hid + 1] += rems[hid]
                rems[hid] = 0

            if rems[hid] == 0:
                hid += 1
                if hid == hosts_number:
                    break

        return math.ceil(np.sum(self.gammas_conv[n_vms_nh]))

    @staticmethod
    def branched_bin_search(start: int, end: int, vms: np.ndarray, expensive_f: Callable):
        f = {start: expensive_f(vms, start), end: expensive_f(vms, end)}
        intervals = [(start, end)]
        steps = []
        c = 0
        while intervals:
            c += 1
            lb, ub = intervals.pop()
            if ub - lb <= 1:
                steps.append((ub, f[ub]))
            else:
                mb = (lb + ub) // 2
                f[mb] = expensive_f(vms, mb)
                if f[mb] < f[ub]:
                    intervals.append((mb, ub))
                if f[lb] < f[mb]:
                    intervals.append((lb, mb))
        return steps

def get_concave_gammas(vm_num: int, probability: float):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    f = calculate_gammas(vm_num, probability)
    g = [solver.NumVar(0, solver.infinity(), 'g' + str(i)) for i in range(vm_num + 1)]

    solver.Add(g[0] <= f[0])
    solver.Add(g[vm_num] <= f[vm_num])
    for k in range(1, vm_num):
        solver.Add(g[k] <= f[k])
        solver.Add(g[k + 1] - g[k] <= g[k] - g[k - 1])

    solver.Maximize(sum(g))
    solver.Solve()
    g_ = [x.solution_value() for x in g]

    return np.array(g_)
