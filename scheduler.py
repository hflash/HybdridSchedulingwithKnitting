import os
import numpy as np
from quantumcircuit.circuit import QuantumCircuit

class QPU
#一个subcircuit为一个task
class Task:
    def __init__(self, subcircuit):
        self.circuit = subcircuit
        self.fidelity = None
        self.poriority = 0.0
   
    def allocation(self, qpu:QPU, region, start_time, duration):
        self.start_time = start_time
        self.end_time = start_time + self.duration
        self.qpu = qpu
        self.region = region
        self.start_time = 0
        self.end_time = start_time + duration
        self.duration = duration
    
    def set_priority(self, priority):
        self.priority = priority
    def get_fidelity(self):
        if self.fidelity is None:
            #todo 计算保真度
            pass
            
        return self.fidelity

class Circuit:
    def __init__(self, qasmfile):
        self.name = os.path.basename(qasmfile)
        self.qc = QuantumCircuit(qasmfile)
        self.classical_cost = 0
        #多个分割方案 [tasks1,tasks2,...] tasks = list of Task
        self.partitions = []
        self.budgets = [] #每种分割方案的花费的预算
        self.selected_partition = None
        self.fidelity = None
    def cut(self, chips:list):
        #切割线路
        results = [] #todo: 调用切割算法
        for res in results:
            partition = []
            for subcircuit in res['subcircuits']:
                task = Task(subcircuit)
                partition.append(task)
            self.partitions.append(partition)
            self.budgets.append(res['budget'])
        #todo
        self.classical_cost = 0
    
    def finish_time(self):
        #返回线路完成时间
        return max(task.end_time for tasks in self.partitions for task in tasks )

    def get_fidelity(self):
        if self.fidelity is None:
            #todo 计算保真度
            pass
        return self.fidelity


class QPU:
    def __init__(self, backend):
        self.backend = backend
        self.num_qubits = backend.num_qubits
        self.qubit_busy_time = []
    def add_task(self, task:Task):
        pass

    def latest_free_time(self, region, duration):
        #返回机器在region上能开始执行任务的最早时间
        pass

    def end_time(self):
        #返回机器完成所有任务的时间
        pass


class Scheduler:
    def __init__(self, qpus:list[QPU], circuits:list):
        self.qpus = qpus
        self.circuits = circuits
    
    def cut_circuits(self):
        for circuit in self.circuits:
            circuit.cut(self.qpus)


    def schedule(self):
        self.cut_circuits()
        #todo 计算各个线路的优先级
        pass
        #todo调度任务
        
        max_end_time = max(qpu.end_time() for qpu in self.qpus)
        return max_end_time
        
    
    