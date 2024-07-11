from typing import List, ClassVar
from sortedcontainers import SortedSet
from dataclasses import dataclass
import numpy as np

from sch.gamma import calculate_gammas

@dataclass(eq=False)
class Host:
    index: int # index of the host in the problem
    cpu_capacity: float

@dataclass(eq=False)
class VM:
    SYM:ClassVar[bool] = False # If True, turn on distribution symmetryzer
    TP:ClassVar[int] = 8 # Number of timestamps used for prediction
    index: int # index of the VM in the problem
    cpu_util: np.ndarray = None
    min_util: float = None
    max_util: float = None
    util_center: float = None
    util_radius: float = None

    @staticmethod
    def symmetrizer(utils: np.ndarray):
        utils_reflect = np.max(utils) + np.min(utils) - utils
        s = np.max(np.sort(utils) - np.sort(utils_reflect))
        return np.min(utils) + s,  np.max(utils)


    def __post_init__(self):
        if VM.SYM:
            self.min_util, self.max_util = VM.symmetrizer(self.cpu_util[:VM.TP])
        else:
            self.min_util = np.min(self.cpu_utils[:VM.TP])
            self.max_util = np.max(self.cpu_utils[:VM.TP])

        self.util_center = (self.max_util + self.min_util) / 2.0
        self.util_radius = (self.max_util - self.min_util) / 2.0

class RobustContainer:
    def __init__(self, capacity: float, gammas: np.ndarray):
        self._gammas = gammas
        self._capacity = capacity

        self._utilization = 0 # sum of center values of all items
        self._robust_overhead = 0 # sum of radii of items in MaxSet

        self._vms = {} # VM.index -> VM
        self._max_set = SortedSet()
        self._min_set = SortedSet()

    def clear(self):
        self._utilization = 0
        self._robust_overhead = 0

        self._vms = {}
        self._max_set = SortedSet()
        self._min_set = SortedSet()

    def __contains__(self, vm):
        return vm.index in self._vms

    def __len__(self):
        return len(self._vms)

    def get_utilization(self):
        return self._utilization + self._robust_overhead

    def is_hotspot(self):
        return self.get_utilization() > self._capacity

    def include(self, vm: VM):
        assert vm not in self

        self._vms[vm.index] = vm
        self._utilization += vm.util_center

        item = (vm.util_radius, vm.index)
        if len(self._max_set) != 0 and vm.util_radius >= self._max_set[0][0]:
            self._max_set.add(item)
            self._robust_overhead += vm.util_radius
        else:
            self._min_set.add(item)

        self._fix_min_max_sets()

    def exclude(self, vm: VM):
        assert vm in self

        del self._vms[vm.index]
        self._utilization -= vm.util_center

        item = (vm.util_radius, vm.index)
        if item in self._max_set:
            self._max_set.remove(item)
            self._robust_overhead -= vm.util_radius
        elif item in self._min_set:
            self._min_set.remove(item)

        self._fix_min_max_sets()

    def _fix_min_max_sets(self):
        """After including or excluding VMs from container
        items in MinSet and MaxSet need to be adjusted
        """
        gamma = self._gammas[len(self)]

        while len(self._max_set) > gamma:
            item = self._max_set.pop(0)
            self._robust_overhead -= item[0]
            self._min_set.add(item)

        while len(self._max_set) < gamma:
            item = self._min_set.pop()
            self._max_set.add(item)
            self._robust_overhead += item[0]

        assert len(self._max_set) == gamma

class Assignment:
    """ Assignment for Online Scheduling with Bertsimas Gamma robust CPU-Constraints """
    MAX_VMS_PH = 1000

    def __init__(self, hosts: List[Host], p: float):
        self.n_hosts = len(hosts)
        self.hosts = hosts
        self.vms = {}
        self.mapping = {}

        self.load = np.zeros((len(hosts), 3), dtype=float) # center+radius, center, radius
        self.cpu_capacity = np.array([host.cpu_capacity for host in hosts])
        self.gammas = calculate_gammas(Assignment.MAX_VMS_PH, 1.0 - p)

        self.containers = [RobustContainer(host.cpu_capacity, self.gammas) for host in hosts]

    def clear(self):
        self.vms = {}
        self.mapping = {}

        self.load = np.zeros_like(self.load)

        for container in self.containers:
            container.clear()


    def include(self, vm: VM, host_index: int):
        "Include VM into container with given host ID"
        assert vm.index not in self.vms

        self.vms[vm.index] = vm
        self.mapping[vm.index] = host_index

        cpu = self.containers[host_index]
        cpu.include(vm)
        self.load[host_index, 0] = cpu.get_utilization()
        self.load[host_index, 1] += vm.util_center
        self.load[host_index, 2] += vm.util_radius

    def exclude(self, vm: VM):
        "Exclude VM from containers"
        assert vm.index in self.vms

        host_index = self.mapping[vm.index]

        cpu = self.containers[host_index]
        cpu.exclude(vm)
        self.load[host_index, 0] = cpu.get_utilization()
        self.load[host_index, 1] -= vm.util_center
        self.load[host_index, 2] -= vm.util_radius

        del self.vms[vm.index]
        del self.mapping[vm.index]

    def is_feasible(self, host_index: int):
        return not self.containers[host_index].is_hotspot()

    def atleast_infeasible_mask(self, vm: VM):
        """
        Fast way to get a mask of locations that are certainly infeasible
        No guarantees for feasability of unmasked locations
        """
        return self.load[:, 0] + vm.util_center > self.cpu_capacity

