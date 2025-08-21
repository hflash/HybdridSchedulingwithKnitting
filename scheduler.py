import os
import numpy as np
from CircuitCutting.performanceAccessing import compute_fidelity_requirement
import random
# 依赖：用于子线路在芯片上估计"最佳保真度"
from CircuitCutting.performanceAccessing import estimate_best_fidelity_and_logical_stats
from CircuitCutting.performanceAccessing import estimate_partition_budgets, _backend_from_name
import math
from qiskit import QuantumCircuit
from CircuitCutting.iterationCircuitCut import compute_subcircuits_with_budget

# 宏定义：shots分配的基数
SHOTS_SCALE_BASE = 4.0

# 宏定义：预算的最大值；经典资源上限 = 4^BUDGET_MAX
BUDGET_MAX = 10

# 宏定义：单位执行时间（基于3.4GHz时钟频率）
UNIT_EXECUTION_TIME = 1.0 / 3.4e9  # 秒

class QPU():
    pass
#一个subcircuit为一个task
class Task:
    def __init__(self, subcircuit, parent=None):
        self.circuit = subcircuit
        self.estimated_fidelity = None #估计的保真度
        self.estimated_seconds = None  #估计的执行时间(s)
        self.executed_fidelity = None  #执行后的保真度（若有）
        self.parent = parent #父节点
        self.poriority = 0.0
        self.shots = 1.0
   
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
    
    def get_estimated_fidelity(self):
        # 返回估计的保真度
        return self.estimated_fidelity
     
    def get_executed_fidelity(self):
        # 返回实际执行的保真度
        return self.executed_fidelity


class Circuit:
    def __init__(self, qasmfile):
        self.name = os.path.basename(qasmfile)
        self.qc = QuantumCircuit.from_qasm_file(qasmfile)
        self.qasmfile = qasmfile
        self.classical_cost_time = 0
        self.classical_cost_space = 0
        #多个分割方案 [tasks1,tasks2,...] tasks = list of Task
        self.partitions = []
        self.budgets = [] #每种分割方案的花费的预算
        self.kept_partition = None # 保留的分割方案
        self.selected_partition = None # 选择的分割方案
        self.required_fidelity = compute_fidelity_requirement(self.qc)
        self.fidelity = None
    
    def cut(self, chips:list):
        # 使用self.budgets中提供的多个预算，分别执行切割并生成多套子线路方案
        # 若未预先计算预算，则先计算
        if not getattr(self, 'budgets', None):
            try:
                self.compute_budget()
            except Exception:
                return
        budgets_list = list(self.budgets) if isinstance(self.budgets, (list, tuple)) else [self.budgets]
        # 清空旧方案与本次各方案的实际花费
        self.partitions = []
        self.partition_budgets = []
        # 按方案记录经典代价（时间/空间）
        self.partition_classical_cost_time = []
        self.partition_classical_cost_space = []

        # 读取原始电路（Qiskit对象）
        try:
            qiskit_circuit = QuantumCircuit.from_qasm_file(self.qasmfile)
        except Exception:
            return

        # 设备单次可用比特数上限：取可用QPU中的最大可用比特
        try:
            device_qubit_max = max(getattr(chip, 'num_qubits', 0) for chip in chips) if chips else qiskit_circuit.num_qubits
            if not device_qubit_max or device_qubit_max <= 0:
                device_qubit_max = qiskit_circuit.num_qubits
        except Exception:
            device_qubit_max = qiskit_circuit.num_qubits

        for budget in budgets_list:
            try:
                res = compute_subcircuits_with_budget(
                    circuit=qiskit_circuit,
                    budget=int(budget),
                    device_qubit_max=device_qubit_max,
                    chip_name=None,              # 如需交换优化与保真度评估，可传入实际芯片名
                    two_q_gate_name='cx',
                    max_regions=200,
                    excluded_qubits=None
                )
            except Exception:
                continue

            if not isinstance(res, dict) or res.get('status') != 'ok':
                continue

            subcircuits = res.get('subcircuits', []) or []
            membership = res.get('membership', []) or []
            per_counts = res.get('per_subcircuit_cut_counts', {}) or {}
            total_cuts = int(res.get('cuts', 0) or 0)

            # shots分配：按"子线路与其他子线路之间割数目"的归一化结果 * 4^(分割数目/2)
            groups = sorted(set(m for m in membership if m != -1))
            sum_counts = float(sum(per_counts.get(g, 0) for g in groups))
            scale = SHOTS_SCALE_BASE ** (total_cuts / 2)

            partition = []
            k = len(subcircuits) if subcircuits else 1
            for idx, sc in enumerate(subcircuits):
                g = groups[idx] if idx < len(groups) else idx
                cnt = float(per_counts.get(g, 0))
                if sum_counts > 0:
                    norm = cnt / sum_counts
                else:
                    norm = 1.0 / float(k)
                shots = norm * scale
                task = Task(sc, self)
                task.shots = shots
                partition.append(task)

            self.partitions.append(partition)
            # 记录该方案的实际花费预算（used_budget），若不可用则回退为输入budget
            used_b = res.get('used_budget', None)
            try:
                used_b = int(used_b) if used_b is not None else int(budget)
            except Exception:
                used_b = int(budget) if isinstance(budget, (int, float)) else 0
            self.partition_budgets.append(used_b)
            # 记录该方案的经典代价：shots 之和作为空间，乘以单位时间作为时间
            try:
                part_shots = float(sum(getattr(t, 'shots', 0.0) or 0.0 for t in partition))
            except Exception:
                part_shots = 0.0
            self.partition_classical_cost_space.append(part_shots)
            self.partition_classical_cost_time.append(part_shots * UNIT_EXECUTION_TIME)
        # 方案生成完成
        # classical_cost（总值）将在选择方案后设置为所选方案的代价
        self.classical_cost_time = 0.0
        self.classical_cost_space = 0.0
    
    def finish_time(self):
        #返回线路完成时间
        return max(task.end_time for tasks in self.partitions for task in tasks )

    def get_fidelity(self):
        if self.fidelity is None:
            #todo 计算保真度
            pass
        return self.fidelity

    def compute_budget(self):
        # 调用 estimate_partition_budgets 估计三档预算，并设置默认预算为中档(M)
        try:
            qc_qiskit = QuantumCircuit.from_qasm_file(self.qasmfile)
        except Exception:
            # 无法读取则设为0
            self.budget = 0
            return
        # 估计设备规模（缺少QPU上下文时用线路比特数回退）
        try:
            device_qubit_max = getattr(qc_qiskit, 'num_qubits', 0) or 0
            if device_qubit_max <= 0:
                device_qubit_max = self.qc.num_qubits
        except Exception:
            device_qubit_max = self.qc.num_qubits

        res = estimate_partition_budgets(
            qc_qiskit,
            device_qubit_max=device_qubit_max,
            B0=1.0, alpha=1.0, beta=1.0, eta=1.0,
            gammas=(0.5, 1.0, 2.0),
            required_fidelity=None,
            ratio_req_over_est=1.1,
        )
        try:
            budgets_int = res.get('budgets_int', {})
            # 采用中档预算作为默认
            # self.budget = int(budgets_int.get('M', 0) or 0)
            # 保存多档预算到 circuits：确保为去重、去零的有序列表 [L, M, H]
            cand_list = [int(budgets_int.get('L', 0) or 0), int(budgets_int.get('M', 0) or 0), int(budgets_int.get('H', 0) or 0)]
            cand_list = [b for b in cand_list if b > 0]
            # 去重并按升序排序
            cand_list = sorted(list(dict.fromkeys(cand_list)))
            cand_list = [b for b in cand_list if b <= BUDGET_MAX]
            if len(cand_list) < 3:
                cand_list.append(BUDGET_MAX)
            self.budgets = cand_list
            print(self.budgets)
            # 保留原始候选映射与详细信息
            # self.budget_candidates = budgets_int
            # 可选：保存细节供上层查看
            # self.budget_detail = res
        except Exception:
            self.budgets = None
        

class QPU:
    def __init__(self, backend):
        self.backend = _backend_from_name(backend)
        self.num_qubits = self.backend.num_qubits
        self.qubit_busy_time = []
        # 设备时间线管理 - 改为记录具体的量子比特分配
        self.allocations = []  # [(start_time, end_time, circuit, qubits_used), ...]
        self.backend_name = backend
        self.current_time = 0.0
        # 量子比特连通性信息
        self.connectivity = self._get_device_connectivity()
    
    def _get_device_connectivity(self):
        """获取设备的量子比特连通性"""
        try:
            # 尝试从backend获取连通性信息
            if hasattr(self.backend, 'configuration'):
                config = self.backend.configuration()
                if hasattr(config, 'coupling_map'):
                    return config.coupling_map
        except:
            pass
        
        # 默认连通性：相邻比特连通
        connectivity = []
        for i in range(self.num_qubits - 1):
            connectivity.append([i, i + 1])
            connectivity.append([i + 1, i])  # 双向连通
        return connectivity
    
    def add_task(self, task: Task):
        """添加任务到设备（已废弃，使用schedule_circuit替代）"""
        pass

    def schedule_circuit(self, circuit, start_time: float = None) -> tuple[float, float]:
        """
        为量子线路计算最早开始时间和执行时间，并实际分配资源
        输入：circuit (量子线路), current_time (当前时间) 传进来的是best_start_time
        输出：(earliest_start_time, execution_duration)
        """
        # if current_time is None:
        #     current_time = self.current_time
        
        required_qubits = circuit.num_qubits
        
        # 1. 计算最早开始时间
        # earliest_start = self._find_earliest_start_time(required_qubits, current_time)
        # earliest_start = start_time # 直接使用best_start_time

        # 2. 使用performanceAccessing函数估算执行时间（考虑当前时刻占用比特作为excluded_qubits）
        try:
            from CircuitCutting.performanceAccessing import estimate_best_fidelity_and_logical_stats
            occupied = set(self._get_occupied_qubits_at_time(start_time))
            fidelity_stats = estimate_best_fidelity_and_logical_stats(
                circuit,
                self.backend_name,
                two_q_gate_name='cx',
                max_regions=200,
                excluded_qubits=occupied if occupied else None,
            )
            if fidelity_stats and 'best_seconds' in fidelity_stats and 'best_fidelity' in fidelity_stats:
                estimated_seconds = fidelity_stats['best_seconds']
                executed_fidelity = fidelity_stats['best_fidelity']
            else:
                print("无法获取执行时间")
                estimated_seconds = 1000.0
                executed_fidelity = 0.0
        except ImportError:
            estimated_seconds = 1.0
        
        # 3. 计算实际执行时间（考虑设备状态）
        execution_duration = estimated_seconds
        
        # 4. 分配具体的量子比特
        allocated_qubits = self._allocate_qubits(required_qubits, start_time)
        
        # 5. 更新设备时间线
        end_time = start_time + execution_duration
        allocation = (start_time, end_time, circuit, allocated_qubits)
        self.allocations.append(allocation)
        
        # 6. 更新当前时间
        # self.current_time = max(self.current_time, end_time)
        
        return start_time, execution_duration, executed_fidelity
    
    def calculate_execution_times(self, circuit: QuantumCircuit, current_time: float) -> tuple[float, float]:
        """
        计算量子线路的最早开始时间和执行时间（不实际分配资源）
        输入：circuit (量子线路), current_time (当前时间)
        输出：(earliest_start_time, execution_duration)
        """
        required_qubits = circuit.num_qubits
        
        # 1. 计算最早开始时间
        earliest_start = self._find_earliest_start_time(required_qubits, current_time)
        
        # 2. 使用performanceAccessing函数估算执行时间（考虑当前时刻占用比特作为excluded_qubits）
        try:
            from CircuitCutting.performanceAccessing import estimate_best_fidelity_and_logical_stats
            # 使用 earliest_start 时刻的占用集合进行估计
            occupied = set(self._get_occupied_qubits_at_time(earliest_start))
            fidelity_stats = estimate_best_fidelity_and_logical_stats(
                circuit,
                self.backend_name,
                two_q_gate_name='cx',
                max_regions=200,
                excluded_qubits=occupied if occupied else None,
            )
            if fidelity_stats and 'best_seconds' in fidelity_stats:
                estimated_seconds = fidelity_stats['best_seconds']
            else:
                # 如果无法获取，使用默认值
                estimated_seconds = float('inf')
        except ImportError:
            # 如果无法导入模块，使用默认值
            estimated_seconds = 1.0
        
        # 3. 计算实际执行时间（考虑设备状态）
        execution_duration = estimated_seconds
        
        return earliest_start, execution_duration
    
    def _find_earliest_start_time(self, required_qubits: int, current_time: float) -> float:
        """找到满足容量约束的最早开始时间"""
        # 按时间顺序检查所有时间点
        check_times = [current_time]
        
        # 添加所有分配结束时间作为候选时间点
        for start, end, _, _ in self.allocations:
            if end > current_time:
                check_times.append(end)
        
        # 排序并去重
        check_times = sorted(set(check_times))
        
        # 找到第一个满足容量约束的时间点
        for time_point in check_times:
            if self._check_capacity_at_time(required_qubits, time_point):
                return time_point
        
        # 如果没有找到，返回当前时间（假设设备总是可用）
        return current_time
    
    def _check_capacity_at_time(self, required_qubits: int, time_point: float) -> bool:
        """检查在指定时间点是否有足够的连通量子比特"""
        # 获取当前被占用的量子比特
        occupied_qubits = self._get_occupied_qubits_at_time(time_point)
        
        # 获取可用的量子比特
        available_qubits = [i for i in range(self.num_qubits) if i not in occupied_qubits]
        
        # 检查是否有足够的连通量子比特
        return self._check_connectivity_constraint(available_qubits, required_qubits)
    
    def _get_occupied_qubits_at_time(self, time_point: float) -> list:
        """获取指定时间点被占用的量子比特列表"""
        occupied = []
        for start, end, _, qubits_used in self.allocations:
            if start <= time_point < end:
                occupied.extend(qubits_used)
        return occupied
    
    def _check_connectivity_constraint(self, available_qubits: list, required_qubits: int) -> bool:
        """检查是否有足够的连通量子比特"""
        if len(available_qubits) < required_qubits:
            return False
        
        # 检查连通性：找到最大的连通子集
        max_connected = self._find_max_connected_subset(available_qubits)
        return len(max_connected) >= required_qubits
    
    def _find_max_connected_subset(self, qubits: list) -> list:
        """找到最大的连通量子比特子集"""
        if not qubits:
            return []
        
        # 使用BFS找到最大连通分量
        visited = set()
        max_connected = []
        
        for start_qubit in qubits:
            if start_qubit in visited:
                continue
            
            # BFS搜索连通分量
            connected = []
            queue = [start_qubit]
            visited.add(start_qubit)
            
            while queue:
                current = queue.pop(0)
                connected.append(current)
                
                # 查找相邻的量子比特
                for neighbor in self._get_neighbors(current):
                    if neighbor in qubits and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(connected) > len(max_connected):
                max_connected = connected
        
        return max_connected
    
    def _get_neighbors(self, qubit: int) -> list:
        """获取指定量子比特的邻居"""
        neighbors = []
        for connection in self.connectivity:
            if connection[0] == qubit:
                neighbors.append(connection[1])
        return neighbors
    
    def _allocate_qubits(self, required_qubits: int, start_time: float) -> list:
        """分配具体的量子比特"""
        # 获取可用的量子比特
        occupied_qubits = self._get_occupied_qubits_at_time(start_time)
        available_qubits = [i for i in range(self.num_qubits) if i not in occupied_qubits]
        
        # 找到最大的连通子集
        connected_qubits = self._find_max_connected_subset(available_qubits)
        
        # 选择前required_qubits个量子比特
        if len(connected_qubits) >= required_qubits:
            return connected_qubits[:required_qubits]
        else:
            # 如果连通比特不够，选择任意可用的比特（简化处理）
            return available_qubits[:required_qubits]
    
    # def _estimate_execution_duration(self, circuit, base_duration: float, start_time: float) -> float:
    #     """根据设备状态估算执行时间"""
    #     required_qubits = circuit.num_qubits
        
    #     # 计算设备利用率
    #     utilization = self._get_utilization_at_time(start_time)
        
    #     # 检查连续量子比特可用性
    #     contiguous_available = self._get_contiguous_available_qubits(start_time, required_qubits)
        
    #     # 根据设备状态调整执行时间
    #     if contiguous_available >= required_qubits:
    #         # 有足够的连续比特，执行时间接近基础时间
    #         duration_factor = 1.0 + 0.1 * utilization
    #     else:
    #         # 需要重新映射或分片，执行时间显著增加
    #         duration_factor = 1.5 + 0.3 * utilization
        
    #     return base_duration * duration_factor
    
    def _get_utilization_at_time(self, time_point: float) -> float:
        """获取指定时间点的设备利用率"""
        occupied = len(self._get_occupied_qubits_at_time(time_point))
        return occupied / self.num_qubits
    
    def _get_contiguous_available_qubits(self, time_point: float, required_qubits: int) -> int:
        """获取指定时间点的连续可用量子比特数（简化实现）"""
        # 简化实现：返回总可用比特数
        occupied = len(self._get_occupied_qubits_at_time(time_point))
        return self.num_qubits - occupied
    
    def get_available_qubits_at_time(self, time_point: float) -> int:
        """获取指定时间点的可用量子比特数"""
        occupied = len(self._get_occupied_qubits_at_time(time_point))
        return self.num_qubits - occupied
    
    def get_utilization_at_time(self, time_point: float) -> float:
        """获取指定时间点的设备利用率"""
        return self._get_utilization_at_time(time_point)

    def latest_free_time(self, region, duration):
        #返回机器在region上能开始执行任务的最早时间
        pass

    def end_time(self):
        #返回机器完成所有任务的时间
        return self.current_time


class Scheduler:
    """量子电路调度器"""
    
    def __init__(self, qpus: list[QPU], circuits: list[Circuit]):
        self.qpus = qpus
        self.circuits: list[Circuit] = circuits
        self.executed_tasks = []
        # 经典资源：容量限制与时间线（limit = 4^BUDGET_MAX）
        self.classical_resource_limit = float(SHOTS_SCALE_BASE ** BUDGET_MAX)
        self.classical_allocations = []  # [{'circuit': c, 'start': s, 'end': e, 'space': x}]
        self.classical_jobs_scheduled = set()  # 已调度经典作业的电路集合
        # 移除重复的设备状态跟踪，使用QPU类内置的时间线管理
    
    def add_circuit(self, circuit: Circuit):
        """添加电路到调度器"""
        self.circuits.append(circuit)
    
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

    def _reconstruct_fidelity_geom(self, fidelities: list[float]) -> float | None:
        """
        线路执行结果重构保真度（几何均值）：
          Fidelity ≈ Π_k Fidelity_k^{1/K} = exp( mean(log Fidelity_k) )
        要求：alpha_k=1/K，K为子线路数。若fidelities为空或存在None返回None；若存在<=0返回0。
        """
        vals = [float(f) for f in fidelities if f is not None]
        if not vals or len(vals) == 0:
            return None
        # 若存在非正值，返回0
        if any(v <= 0.0 for v in vals):
            return 0.0
        avg_log = sum(math.log(v) for v in vals) / len(vals)
        return math.exp(avg_log)

    def offline_schedule(self, score_fn=None, prob_switch: float = 0.2, weights: tuple | None = None):
        """
        离线调度框架（加入更严格的第一轮筛选逻辑):
        1) 保真度过滤：仅保留满足 F_min^i 的方案。方案保真度估计为"该方案所有子线路在芯片上执行的最佳保真度"的重构保真度（几何均值，alpha_k=1/K）；若某线路无任何可行方案，则直接返回不可执行。
        2) 对剩余方案按 S = w1*max_qubits + w2*sum_seconds + w3*classical_time 评分（越小越好），为每条线路挑选S最小的方案进入初始tasks队列；
        3) 为tasks计算初始优先级；
        4) 迭代调度（与先前相同）。
        """
        # 预处理：评分函数
        score = score_fn if score_fn is not None else self._score_plan
        w1, w2, w3 = (weights if (isinstance(weights, (list, tuple)) and len(weights) == 3) else (3.0, 2.0, 1.0))

        # Step 1: 方案按 F_min 阈值过滤（方案保真度=子线路最佳保真度的重构值）
        circuit_to_filtered_plans = {}
        any_infeasible = False
        infeasible_circuit = None
        for c in self.circuits:
            plans = getattr(c, 'partitions', []) or []  # 每个元素是一个方案(list[Task])
            # F_min^i：优先使用 required_fidelity['fidelity_requirement']
            thr = 0.0
            rf = getattr(c, 'required_fidelity', None)
            if isinstance(rf, dict):
                try:
                    thr = float(rf.get('fidelity_requirement', 0.0))
                except Exception:
                    thr = 0.0
            else:
                try:
                    thr = float(rf) if rf is not None else 0.0
                except Exception:
                    thr = 0.0

            kept = []
            for plan in plans:
                # 逐子线路估计最佳保真度与时间
                best_fids = []
                for task in plan:
                    if getattr(task, 'estimated_fidelity', None) is None or getattr(task, 'estimated_seconds', None) is None:
                        best_fid = 0
                        best_seconds = float('inf')
                        for qpu in self.qpus:
                            try:
                                est = estimate_best_fidelity_and_logical_stats(task.circuit, qpu.backend_name)
                                best_fid = max(best_fid, est['best_fidelity']) if est and 'best_fidelity' in est else None
                                best_seconds = min(best_seconds, est['best_seconds']) if est and 'best_seconds' in est else None
                            except Exception:
                                best_fid = None
                                best_seconds = None
                        task.estimated_fidelity = best_fid
                        task.estimated_seconds = best_seconds
                    if task.estimated_fidelity is not None:
                        best_fids.append(float(task.estimated_fidelity))
                # 方案保真度：几何均值重构
                plan_fid = self._reconstruct_fidelity_geom(best_fids)
                if plan_fid is None:
                    continue
                if plan_fid >= thr:
                    kept.append(plan)
            circuit_to_filtered_plans[c] = kept
            c.kept_partition = kept
            if not kept:
                any_infeasible = True
                infeasible_circuit = c
                break

        if any_infeasible:
            return {
                'status': 'infeasible',
                'message': f"无法执行：任务{getattr(infeasible_circuit, 'name', '?')}在当前阈值与资源条件下无可行方案",
                'circuit': getattr(infeasible_circuit, 'name', None)
            }

        # Step 2: 评分与初选方案（S最小）
        chosen_plan_per_circuit = {}
        for c, kept in circuit_to_filtered_plans.items():
            if not kept:
                continue
            def plan_cost_S(plan):
                # 最大子线路qubits
                max_qubits = 0
                sum_seconds = 0.0
                # 优先使用预计算的方案级经典时间成本
                classical_time = None
                for t in plan:
                    try:
                        qn = int(getattr(t.circuit, 'num_qubits', 0))
                    except Exception:
                        qn = 0
                    if qn > max_qubits:
                        max_qubits = qn
                    # seconds
                    sec = getattr(t, 'estimated_seconds', None)
                    if sec is None:
                        try:
                            # 估计器已按多QPU取优，这里兜底
                            est = estimate_best_fidelity_and_logical_stats(t.circuit, self.qpus[0].backend_name if self.qpus else None)
                            sec = est['best_seconds'] if est and 'best_seconds' in est else 0.0
                            t.estimated_seconds = sec
                        except Exception:
                            sec = 0.0
                    try:
                        sum_seconds += float(sec or 0.0)
                    except Exception:
                        pass
                # classical time: 方案级
                try:
                    if hasattr(c, 'partitions') and plan in getattr(c, 'partitions', []):
                        idx = c.partitions.index(plan)
                        pct_list = getattr(c, 'partition_classical_cost_time', None)
                        if isinstance(pct_list, list) and idx < len(pct_list):
                            classical_time = float(pct_list[idx])
                except Exception:
                    classical_time = None
                if classical_time is None:
                    # 兜底回退为 shots*UNIT_EXECUTION_TIME 的和
                    tmp = 0.0
                    for t in plan:
                        try:
                            tmp += float(getattr(t, 'shots', 0.0) or 0.0) * UNIT_EXECUTION_TIME
                        except Exception:
                            pass
                    classical_time = tmp
                # S作为该线路需要的全部执行时间
                total_execution_time = (w1 * float(max_qubits)) + (w2 * float(sum_seconds)) + (w3 * float(classical_time))
                return total_execution_time

            scored = [(plan_cost_S(plan), plan) for plan in kept]
            scored.sort(key=lambda x: x[0])  # 越小越好
            chosen_plan_per_circuit[c] = {
                'all_plans': [p for _, p in scored],
                'best_plan': scored[0][1],
                'best_score': scored[0][0],
            }

        # 将被选方案的经典代价写回 circuit（用于后续经典资源利用率与排程）
        for c, meta in chosen_plan_per_circuit.items():
            try:
                best_plan = meta.get('best_plan')
                if hasattr(c, 'partitions') and best_plan in getattr(c, 'partitions', []):
                    idx = c.partitions.index(best_plan)
                    pct = getattr(c, 'partition_classical_cost_time', None)
                    pcs = getattr(c, 'partition_classical_cost_space', None)
                    if isinstance(pct, list) and idx < len(pct):
                        c.classical_cost_time = float(pct[idx])
                    if isinstance(pcs, list) and idx < len(pcs):
                        c.classical_cost_space = float(pcs[idx])
                    # 记录所选方案
                    c.selected_partition = best_plan
                    c.selected_partition_idx = idx
            except Exception:
                pass

        # 初始任务队列
        tasks = []
        for meta in chosen_plan_per_circuit.values():
            best = meta['best_plan']
            tasks.extend(best)

        # Step 3: 计算每个task的优先级
        time_now = 0.0
        self._compute_task_priorities(tasks, time_now)

        # Step 4: 调度循环（基于 P 优先，平手时按 EFT/比特数/轮转 规则）
        time_now = 0.0
        schedule_log = []
        remaining = set(tasks)
        # 轮转防饥饿：记录每个电路最近被选择的事件序号，越小代表越久未被选择
        event_index = 0
        last_pick_index_by_circuit = {}
        while remaining:
            ready_list = list(remaining)
            if not ready_list:
                break

            # 先找出最大优先级 P 的候选集合
            max_priority = max(getattr(t, 'priority', 0.0) for t in ready_list)
            if max_priority == 0.0:
                # 没有可执行任务
                # 推进时间到下一个节点
                time_now = self._next_resource_release_time(time_now)
                if time_now == float('inf'):
                    print("没有可执行任务")
                    break
                self._update_all_priorities(remaining, time_now)
                continue
            cand_list = [t for t in ready_list if getattr(t, 'priority', 0.0) == max_priority]
            # 平手规则：EFT 更小优先；若仍并列，qubits 更大优先；若仍并列，按轮转（最近最少被选）
            eft_map = self._compute_earliest_finish_times(cand_list, time_now)
            def tie_key(t):
                eft = eft_map.get(t, float('inf'))
                qubits = getattr(getattr(t, 'circuit', None), 'num_qubits', 0)
                circuit = getattr(t, 'parent', None)
                last_idx = last_pick_index_by_circuit.get(circuit, -1)
                return (eft, -qubits, last_idx)

            selected = min(cand_list, key=tie_key)

            # 放置于 Chip*(task, t)：按最小 EFT 的设备进行放置，位置为“最早可行开始时刻”
            start, end = self._execute_task(selected, time_now)
            schedule_log.append({'event': 'run', 'start': start, 'end': end, 'qpu': selected.assigned_qpu.backend_name})

            # 时间推进到放置事件时刻（事件驱动：放置/完成后更新）
            # time_now = max(time_now, start)
            owner_circuit = getattr(selected, 'parent', None)
            last_pick_index_by_circuit[owner_circuit] = event_index
            event_index += 1

            # 从就绪集中移除已完成任务，更新优先级（事件驱动）
            remaining.remove(selected)
            self._update_all_priorities(remaining, time_now)

            # 放置量子任务后，尝试调度经典作业（事件驱动：量子放置/完成后）
            self._schedule_ready_classical_jobs(time_now, schedule_log)
        
        while time_now < float('inf'):
            time_now = self._next_resource_release_time(time_now)
            self._schedule_ready_classical_jobs(time_now, schedule_log)
        return schedule_log

    # ============ 以下为可覆盖/可扩展的辅助方法（框架/占位） ============
    def _score_plan(self, circuit: Circuit, plan: list[Task]) -> float:
        """方案评分函数（占位）：可结合保真度、时长、预算等多因素。
        默认：按任务数量的负数（越少越好）+ 任务保真度最小值加权。
        """
        fids = [t.get_estimated_fidelity() for t in plan if t.get_estimated_fidelity() is not None]
        min_fid = min(fids) if fids else 0.0
        return (min_fid) - 0.01 * len(plan)

    def _compute_task_priorities(self, tasks: list[Task], time_now: float = 0.0):
        """计算task优先级（基于量子容量时间线的列表调度）
        实现动态优先级构造，考虑：
        1) 解锁经典资源的紧迫程度
        2) 经典资源利用率调制
        3) 尽早完工能力
        4) 装载友好性
        """
        # 计算经典资源利用率
        classical_utilization = self._compute_classical_utilization(time_now)
        modulation_factor = 1.0 - classical_utilization
        
        # 计算每个task所属circuit的剩余子线路数
        circuit_remaining_tasks = self._compute_circuit_remaining_tasks(tasks, time_now)
        
        # 计算就绪任务的最早完工时间
        eft_dict = self._compute_earliest_finish_times(tasks, time_now)
        
        # 计算装载友好性指标
        # Todo 应该是当前所有设备中，空闲量子比特数目最大的区域，并考虑正在执行的任务的占据情况
        max_device_qubits = max(qpu.num_qubits for qpu in self.qpus) if self.qpus else 1
        load_friendliness = self._compute_load_friendliness(tasks, max_device_qubits)
        
        # 优先级参数设置
        # 量子侧 α：可根据经典侧拥塞动态调节
        # 拥塞度定义：经典 ready 作业的总空间需求/limit 或 经典时间线当前利用率的组合
        alpha_base = 10.0
        alpha = self._dynamic_alpha(time_now, alpha_base)
        beta = 1.0    # 尽早完工权重
        gamma = 0.5   # 装载友好性权重
        lambda_param = 0.5  # 解锁紧迫度调制参数
        
                # 线性归一化：1 - norm(EFT)，越小越好
        eft_values = [v for v in eft_dict.values() if v != float('inf')]
        if eft_values:
            min_eft, max_eft = min(eft_values), max(eft_values)

        for t in tasks:
            # 1. 解锁经典资源的紧迫程度
            circuit = getattr(t, 'parent', None)
            if circuit is None:
                unlock_urgency = 0.0
            else:
                remaining_count = circuit_remaining_tasks.get(circuit, 0)
                if remaining_count == 1:
                    unlock_urgency = 1.0  # 只差最后一条子线路
                elif remaining_count > 1:
                    unlock_urgency = lambda_param / remaining_count
                else:
                    unlock_urgency = 0.0
            
            # 2. 尽早完工能力（归一化）
            eft = eft_dict.get(t, float('inf'))
            if eft == float('inf'):
                t.priority = 0
                continue
            if max_eft > min_eft:
                early_finish = 1.0 - (eft - min_eft) / (max_eft - min_eft)
            else:
                early_finish = 1.0
            # if eft == float('inf'):
            #     early_finish = 0.0
            # else:
            #     # 线性归一化：1 - norm(EFT)，越小越好
            #     eft_values = [v for v in eft_dict.values() if v != float('inf')]
            #     if eft_values:
            #         min_eft, max_eft = min(eft_values), max(eft_values)
            #         # if min_eft == float('inf'):
            #         #     t.priority = 0
            #         if max_eft > min_eft:
            #             early_finish = 1.0 - (eft - min_eft) / (max_eft - min_eft)
            #         else:
            #             early_finish = 1.0
            #     else:
            #         early_finish = 0.0
            
            # 3. 装载友好性（归一化）
            load_score = load_friendliness.get(t, 0.0)
            
            # 4. 综合优先级计算
            priority = (alpha * modulation_factor * unlock_urgency + 
                       beta * early_finish + 
                       gamma * load_score)
            
            t.priority = priority

    def _compute_classical_utilization(self, time_now: float) -> float:
        """计算经典资源利用率 U_C(t)"""
        total_classical_cost = 0.0
        classical_resource_limit = 1000.0  # 可配置的经典资源限制
        
        # 统计当前正在执行的经典后处理任务
        for circuit in self.circuits:
            # 这里需要根据实际调度状态计算
            # 简化实现：假设所有circuit的经典成本都在执行
            if hasattr(circuit, 'classical_cost_space'):
                total_classical_cost += circuit.classical_cost_space
        
        utilization = min(1.0, total_classical_cost / classical_resource_limit)
        return utilization

    def _compute_circuit_remaining_tasks(self, tasks: list[Task], time_now: float) -> dict:
        """计算每个circuit的剩余未完成任务数 R_i(t)"""
        circuit_remaining = {}
        
        for circuit in self.circuits:
            # 统计该circuit的所有任务
            circuit_tasks = [t for t in tasks if getattr(t, 'parent') == circuit]
            circuit_remaining[circuit] = len(circuit_tasks)
        
        return circuit_remaining

    def _compute_earliest_finish_times(self, tasks: list[Task], time_now: float) -> dict:
        """计算每个任务的最早完工时间 EFT(subcircuit, t)"""
        eft_dict = {}
        
        for task in tasks:
            min_eft = float('inf')
            flag_current_executable = False
            # 遍历所有设备，找到最早完工时间
            for qpu in self.qpus:
                # 使用QPU的calculate_execution_times方法（不实际分配资源）
                start_time, duration = qpu.calculate_execution_times(
                    task.circuit, 
                    time_now
                )
                if start_time > time_now:
                    continue
                flag_current_executable = True
                eft = start_time + duration
                min_eft = min(min_eft, eft)
            # if min_eft == float('inf'):
            #     eft_dict[task] = float('inf')
            #     break
            eft_dict[task] = min_eft
        
        return eft_dict

    def _compute_earliest_start_time(self, task: Task, qpu: QPU, time_now: float) -> float:
        """计算任务在指定设备上的最早开始时间 EST(subcircuit, Chip_k, t)"""
        # 使用QPU的calculate_execution_times方法
        start_time, _ = qpu.calculate_execution_times(
            task.circuit, 
            time_now
        )
        return start_time

    def _estimate_task_duration_on_device(self, task: Task, qpu: QPU, start_time: float) -> float:
        """根据设备状态动态估算任务执行时间"""
        # 使用QPU的方法计算执行时间
        return qpu._estimate_execution_duration(task.circuit, task.estimated_seconds or 1.0, start_time)

    def _get_device_timeline(self, qpu: QPU) -> dict:
        """获取设备的容量时间线"""
        # 返回QPU的时间线信息
        return {
            'num_qubits': qpu.num_qubits,
            'allocations': qpu.allocations,
            'current_time': qpu.current_time
        }

    def _check_capacity_constraint(self, qpu: QPU, start_time: float, required_qubits: int) -> bool:
        """检查在指定时间点是否满足容量约束"""
        return qpu._check_capacity_at_time(required_qubits, start_time)

    def _get_available_qubits_at_time(self, qpu: QPU, time: float) -> int:
        """获取指定时间点的可用量子比特数"""
        return qpu.get_available_qubits_at_time(time)

    def _get_occupied_qubits_at_time(self, qpu: QPU, time: float) -> int:
        """获取指定时间点被占用的量子比特数"""
        return len(qpu._get_occupied_qubits_at_time(time))

    def _get_device_utilization(self, qpu: QPU, time: float) -> float:
        """获取设备在指定时间点的利用率"""
        return qpu.get_utilization_at_time(time)

    def _get_contiguous_available_qubits(self, qpu: QPU, start_time: float, required_qubits: int) -> int:
        """获取从指定时间开始的连续可用量子比特数"""
        return qpu._get_contiguous_available_qubits(start_time, required_qubits)

    def _compute_load_friendliness(self, tasks: list[Task], max_device_qubits: int) -> dict:
        """计算装载友好性指标 K(subcircuit)"""
        load_scores = {}
        
        for task in tasks:
            task_qubits = task.circuit.num_qubits
            # 归一化：task_qubits / max_device_qubits
            load_score = task_qubits / max_device_qubits
            load_scores[task] = load_score
        
        return load_scores

    def _can_execute_task(self, task: Task, time_now: float) -> bool:
        """判断task在当前时刻是否可执行"""
        # 检查是否有设备满足容量约束
        for qpu in self.qpus:
            if self._check_capacity_constraint(qpu, time_now, task.circuit.num_qubits):
                return True
        return False

    def _execute_task(self, task: Task, time_now: float) -> tuple[float, float]:
        """执行task：分配QPU，更新设备状态，返回(start, end)"""
        # 找到最适合的设备
        best_qpu = None
        best_start_time = float('inf')
        best_eft = float('inf')
        
        for qpu in self.qpus:
            # 逐设备评估最早开始与完工时间
            start_time, duration = qpu.calculate_execution_times(
                task.circuit,
                time_now
            )
            eft = start_time + duration
            if eft < best_eft or (math.isfinite(eft) and not math.isfinite(best_eft)):
                best_eft = eft
                best_start_time = start_time
                best_qpu = qpu
        
        # 使用QPU的schedule_circuit方法
        start_time, duration, execute_fidelity = best_qpu.schedule_circuit(
            task.circuit, 
            best_start_time
        )
        if execute_fidelity == 0.0:
            print("执行失败")
            print(task.parent.name)
        end_time = start_time + duration
        
        # 更新任务状态
        task.start_time = start_time
        task.end_time = end_time
        task.assigned_qpu = best_qpu
        task.fidelity = execute_fidelity
        self.executed_tasks.append(task)
        return start_time, end_time

    def _update_all_priorities(self, tasks: set[Task], time_now: float):
        """根据时间推进/资源变化更新优先级"""
        # 重新计算优先级，考虑当前时间状态
        self._compute_task_priorities(list(tasks), time_now)


    def _next_resource_release_time(self, current_time: float) -> float:
        """获取下一个资源释放时间"""
        # 找到下一个任务结束时间
        next_release = float('inf')
        # 量子侧：已执行任务的结束时间
        for task in self.executed_tasks:
            if getattr(task, 'end_time', float('inf')) > current_time:
                next_release = min(next_release, task.end_time)
        # 经典侧：已调度经典作业的结束时间
        for job in getattr(self, 'classical_allocations', []) or []:
            end = job.get('end', float('inf'))
            if end > current_time:
                next_release = min(next_release, end)
        return next_release

    # ----------------- 经典作业调度 -----------------
    def _classical_timeline_load_at(self, time_point: float) -> float:
        """返回经典资源在 time_point 的占用（空间单位：shots 数）。"""
        load = 0.0
        for job in self.classical_allocations:
            if job['start'] <= time_point < job['end']:
                load += float(job.get('space', 0.0) or 0.0)
        return load

    def _classical_has_capacity(self, time_point: float, demand: float) -> bool:
        """检查 time_point 是否有足够经典容量可用。"""
        used = self._classical_timeline_load_at(time_point)
        return (used + demand) <= float(self.classical_resource_limit)

    def _find_earliest_classical_start(self, earliest_ready_time: float, demand_space: float, duration: float) -> float:
        """在经典时间线上，寻找不违反容量约束的最早起始时刻。"""
        check_times = [earliest_ready_time]
        # 使用现有经典分配的结束节点作为候选
        for job in self.classical_allocations:
            if job['end'] >= earliest_ready_time:
                check_times.append(job['end'])
        check_times = sorted(set(check_times))
        for t in check_times:
            if self._classical_has_capacity(t, demand_space):
                return t
        # 如未找到，回退为 earliest_ready_time（末尾轻微放宽）
        return earliest_ready_time

    def _enum_ready_classical_jobs(self, time_now: float):
        """枚举在 time_now 已就绪且未被调度的经典作业，返回 [(circuit, space, time)]。"""
        ready = []
        for c in self.circuits:
            if c in self.classical_jobs_scheduled:
                continue
            plan = getattr(c, 'selected_partition', None)
            if not plan:
                continue
            # 就绪条件：该线路所有选定子线路任务均已完成
            all_done = True
            latest_end = 0.0
            for t in plan:
                end = getattr(t, 'end_time', None)
                if end is None or end > time_now:
                    all_done = False
                    break
                latest_end = max(latest_end, end)
            if all_done:
                space = float(getattr(c, 'classical_cost_space', 0.0) or 0.0)
                dur = float(getattr(c, 'classical_cost_time', 0.0) or 0.0)
                ready.append((c, space, dur, latest_end))
        return ready

    def _schedule_ready_classical_jobs(self, time_now: float, schedule_log: list):
        """在经典资源时间线中放置就绪的经典作业。
        策略：最长处理时间优先（LPT），并在容量允许的最早时刻放置。
        """
        ready = self._enum_ready_classical_jobs(time_now)
        if not ready:
            return False
        # LPT：按持续时间从长到短
        ready.sort(key=lambda x: x[2], reverse=True)
        for circuit, space, dur, ready_time in ready:
            start_c = self._find_earliest_classical_start(max(time_now, ready_time), space, dur)
            # 检查容量
            if not self._classical_has_capacity(start_c, space):
                continue
            end_c = start_c + dur
            self.classical_allocations.append({'circuit': circuit, 'start': start_c, 'end': end_c, 'space': space})
            self.classical_jobs_scheduled.add(circuit)
            schedule_log.append({'event': 'run_classical', 'circuit': getattr(circuit, 'name', None), 'start': start_c, 'end': end_c, 'space': space})
        return True

    def _dynamic_alpha(self, time_now: float, alpha_base: float) -> float:
        """根据经典侧拥塞动态调整量子侧权重 α。
        拥塞度 proxy：
          - ready 的经典作业总空间占比 = sum(space_ready)/limit
          - 当前时刻经典时间线利用率 = used/limit
        取两者的最大值作为拥塞度 c ∈ [0, 1+]，并按 α = α_base * (1 - k*c) 调整。
        k 默认取 0.7，保证经典侧明显拥塞时显著下调 α。
        """
        try:
            ready = self._enum_ready_classical_jobs(time_now)
            sum_ready_space = sum(float(s or 0.0) for _, s, _, _ in ready) if ready else 0.0
            limit = float(self.classical_resource_limit)
            ratio_ready = (sum_ready_space / limit) if limit > 0 else 0.0
            used_now = self._classical_timeline_load_at(time_now)
            ratio_used = (used_now / limit) if limit > 0 else 0.0
            congestion = max(0.0, ratio_ready, ratio_used)
            k = 0.7
            return max(0.0, alpha_base * (1.0 - k * congestion))
        except Exception:
            return alpha_base