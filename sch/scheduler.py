from abc import ABC
from typing import List
import numpy as np
from sch.assignment import Assignment, VM, Host

class Scheduler(ABC):
    """Interface for online scheduler"""

    def place(self, hosts: List[Host], p: float, vms:List[VM]) -> Assignment:
        asg = Assignment(hosts, p)
        for vm in vms:
            if not self.place_vm(asg, vm):
                return asg

        raise ValueError("Bad Trace. All VMs were placed.")


    def place_vm(self, asg: Assignment, vm: VM) -> bool:
        raise NotImplemented


class FirstFit(Scheduler):
    def place_vm(self, asg: Assignment, vm: VM) -> bool:
        for host_index in np.flatnonzero(~asg.atleast_infeasible_mask(vm)):
            asg.include(vm, host_index)
            if asg.is_feasible(host_index):
                return True
            asg.exclude(vm)
        return False

class RandomFit(Scheduler):
    def place_vm(self, asg: Assignment, vm: VM) -> bool:
        idx = np.flatnonzero(~asg.atleast_infeasible_mask(vm))
        np.random.shuffle(idx)
        for host_index in idx:
            asg.include(vm, host_index)
            if asg.is_feasible(host_index):
                return True
            asg.exclude(vm)
        return False


class CloseRadiusFit(Scheduler):
    def __init__(self):
        self.n_vms = 0 # number of added VMs
        self.cs = None # np array of centers for each added VM
        self.rs = None # np array of radii for each added VM

    def place_vm(self, asg: Assignment, vm: VM) -> bool:
        self.add_vm(vm)
        mask = ~asg.atleast_infeasible_mask(vm)
        host_indices = self.sort_hosts_by_preference(asg, vm)

        for host_index in host_indices[mask[host_indices]]:
            asg.include(vm, host_index)
            if asg.is_feasible(host_index):
                return True
            asg.exclude(vm)
        self.n_vms -= 1
        return False

    def sort_hosts_by_preference(self, asg: Assignment, vm: VM):
        """Return list of host indices sorted by preference with respect
        to a given VM.

        The most prefered host is the one which VM radius range corresponds
        exactly to a given VM. Next all the hosts are added which have
        progressively larger prefered VM radii. Finally, the remaining
        hosts are added."""
        radius_ranges = self.compute_prefered_radius_ranges(asg)
        shift = np.searchsorted(radius_ranges, vm.util_radius)
        return np.roll(np.arange(asg.n_hosts), -shift)

    def compute_prefered_radius_ranges(self, asg):
        """Compute prefered VM utilization radius range for each host.

        These are obtained in the following way. We compute relaxed
        pseudo-assignment where VMs are placed to hosts in a such way that:
        1) each host receives approximately the same center utilization
        2) each next host receives VMs of progressively smaller radii

        For each host we return largest VM radius it has received.
        """
        rs = self.rs[:self.n_vms]
        cs = self.cs[:self.n_vms]
        ch = np.sum(cs) / asg.n_hosts
        idx = np.argsort(rs)[::-1]
        r_space = []
        sums = np.cumsum(cs[idx])
        for i in range(asg.n_hosts):
            j = np.flatnonzero(sums >= ch * i)[0]
            r_space.append(rs[idx[j]])

        assert len(r_space) == asg.n_hosts
        return np.array(r_space)[::-1]

    def add_vm(self, vm):
        self.n_vms += 1
        if self.rs is None:
            self.rs = np.array([vm.util_radius])
            self.cs = np.array([vm.util_center])
        else:
            if len(self.rs) < self.n_vms:
                self.rs.resize((len(self.rs) * 2,))
                self.cs.resize((len(self.cs) * 2,))
            self.rs[self.n_vms - 1] = vm.util_radius
            self.cs[self.n_vms - 1] = vm.util_center

