from typing import List
import numpy as np
from sch.assignment import Host, VM, Assignment

class CloseRadiusLB:
    """CloseRadiusLB provides lower bound on the number of VMs
    which can be placed into hosts.

    Binary search is used to find longest VM prefix.
    VMs inside each check prefix are sorted by their radii
    decreasingly and placed one by one.
    """
    def place(self, hosts: List[Host], p: float, vms: List[VM]) -> Assignment:
        asg = Assignment(hosts, p)
        n_vms = self.binary_search(asg, vms)
        ok = self.try_place_vms(asg, vms[:n_vms])
        assert ok
        return asg

    def try_place_vms(self, asg: Assignment, vms: List[VM]) -> bool:
        vm_order = np.argsort([vm.util_radius for vm in vms])[::-1]
        host_order = np.array([*range(asg.n_hosts)])
        asg.clear()
        for i in vm_order:
            if not self.try_place_vm(asg, vms[i], host_order):
                return False
        return True

    def try_place_vm(self, asg: Assignment, vm: VM, host_order:np.ndarray) -> bool:
        feasible_mask = ~asg.atleast_infeasible_mask(vm)
        host_indices = host_order[feasible_mask[host_order]]
        for host_index in host_indices:
            asg.include(vm, host_index)
            if asg.is_feasible(host_index):
                return True
            asg.exclude(vm)
        return False

    def binary_search(self, asg: Assignment, vms: List[VM]) -> int:
        if self.try_place_vms(asg, vms):
            return len(vms)

        lb = 0
        ub = len(vms)
        while ub - lb > 1:
            middle = int((ub + lb) / 2)
            if self.try_place_vms(asg, vms[:middle]):
                lb = middle
            else:
                ub = middle

        return lb

