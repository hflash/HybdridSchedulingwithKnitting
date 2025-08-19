import os
import numpy as np
from quantumcircuit.circuit import QuantumCircuit
from performanceAccessing import compute_fidelity_requirement
import random
# 依赖：用于子线路在芯片上估计“最佳保真度”
from performanceAccessing import estimate_best_fidelity_and_logical_stats
import math

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
        self.required_fidelity = compute_fidelity_requirement(self.qc)
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
    

    def compute_budget(self):
        #todo 计算线路的预算
        self.budget = 0
        

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

    def offline_schedule(self, chip_name: str, score_fn=None, prob_switch: float = 0.2):
        """
        离线调度框架（加入更严格的第一轮筛选逻辑）:
        1) 保真度过滤：仅保留满足 F_min^i 的方案。方案保真度估计为“该方案所有子线路在芯片上执行的最佳保真度”的重构保真度（几何均值，alpha_k=1/K）；若某线路无任何可行方案，则直接返回不可执行。
        2) 对剩余方案评分（保留全部），但为每条线路挑选评分最高的方案进入初始tasks队列；
        3) 为tasks计算初始优先级；
        4) 迭代调度（与先前相同）。
        """
        # 预处理：评分函数
        score = score_fn if score_fn is not None else self._score_plan

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
                # 逐子线路估计最佳保真度
                best_fids = []
                for task in plan:
                    if getattr(task, 'fidelity', None) is None:
                        try:
                            est = estimate_best_fidelity_and_logical_stats(task.circuit, chip_name)
                            task.fidelity = est['best_fidelity'] if est and 'best_fidelity' in est else None
                        except Exception:
                            task.fidelity = None
                    if task.fidelity is not None:
                        best_fids.append(float(task.fidelity))
                # 方案保真度：几何均值重构
                plan_fid = self._reconstruct_fidelity_geom(best_fids)
                if plan_fid is None:
                    continue
                if plan_fid >= thr:
                    kept.append(plan)
            circuit_to_filtered_plans[c] = kept
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

        # Step 2: 评分与初选方案
        chosen_plan_per_circuit = {}
        for c, kept in circuit_to_filtered_plans.items():
            if not kept:
                continue
            scored = [(score(c, plan), plan) for plan in kept]
            scored.sort(key=lambda x: x[0], reverse=True)
            chosen_plan_per_circuit[c] = {
                'all_plans': [p for _, p in scored],
                'best_plan': scored[0][1],
                'best_score': scored[0][0],
            }

        # 初始任务队列
        tasks = []
        for meta in chosen_plan_per_circuit.values():
            best = meta['best_plan']
            tasks.extend(best)

        # Step 3: 计算每个task的优先级
        self._compute_task_priorities(tasks)

        # Step 4: 调度循环（与之前相同）
        time_now = 0.0
        schedule_log = []
        remaining = set(tasks)
        while remaining:
            ready_list = sorted(list(remaining), key=lambda t: getattr(t, 'priority', 0.0), reverse=True)
            if not ready_list:
                break
            selected = None
            for cand in ready_list:
                if self._can_execute_task(cand, time_now):
                    selected = cand
                    break
            if selected is not None:
                start, end = self._execute_task(selected, time_now)
                schedule_log.append({'event': 'run', 'task': selected, 'start': start, 'end': end})
                time_now = max(time_now, end)
                remaining.remove(selected)
                self._update_all_priorities(remaining, time_now)
                continue

            switched = False
            if random.random() < prob_switch:
                top_task = ready_list[0]
                owner_circuit = self._owner_circuit_of_task(top_task)
                meta = chosen_plan_per_circuit.get(owner_circuit)
                if meta and len(meta['all_plans']) > 1:
                    alt = meta['all_plans'][1]
                    old_tasks = set(meta['best_plan'])
                    remaining -= old_tasks
                    for t in alt:
                        remaining.add(t)
                    meta['all_plans'] = meta['all_plans'][1:] + meta['all_plans'][:1]
                    meta['best_plan'] = alt
                    self._compute_task_priorities(list(remaining))
                    switched = True
                    schedule_log.append({'event': 'switch_plan', 'circuit': owner_circuit})

            if not switched:
                dt = self._next_resource_release_time(time_now)
                time_now += dt
                schedule_log.append({'event': 'wait', 'until': time_now})
                self._update_all_priorities(remaining, time_now)

        return schedule_log

    # ============ 以下为可覆盖/可扩展的辅助方法（框架/占位） ============
    def _score_plan(self, circuit: Circuit, plan: list[Task]) -> float:
        """方案评分函数（占位）：可结合保真度、时长、预算等多因素。
        默认：按任务数量的负数（越少越好）+ 任务保真度最小值加权。
        """
        fids = [t.get_fidelity() for t in plan if t.get_fidelity() is not None]
        min_fid = min(fids) if fids else 0.0
        return (min_fid) - 0.01 * len(plan)

    def _compute_task_priorities(self, tasks: list[Task]):
        """计算task优先级（占位）。可用关键路径、截止期、保真度提升潜力等。
        默认：按保真度降序。
        """
        for t in tasks:
            fid = t.get_fidelity()
            pr = fid if fid is not None else 0.0
            # 也可加入更多因素
            t.priority = pr

    def _can_execute_task(self, task: Task, time_now: float) -> bool:
        """判断task在当前时刻是否可执行（占位）。
        可检查QPU可用性、region冲突、依赖完成等。
        """
        return True  # 占位：默认可执行

    def _execute_task(self, task: Task, time_now: float) -> tuple[float, float]:
        """执行task（占位）：分配QPU/region，返回(start, end)。"""
        duration = getattr(task, 'duration', 1.0) or 1.0
        start = time_now
        end = start + duration
        # TODO: 标记QPU资源占用
        return start, end

    def _update_all_priorities(self, tasks: set[Task], time_now: float):
        """根据时间推进/资源变化更新优先级（占位）。"""
        for t in tasks:
            # 示例：随时间略微衰减，推动早调度
            base = getattr(t, 'priority', 0.0)
            t.priority = max(0.0, base * 0.999)

    def _owner_circuit_of_task(self, task: Task) -> Circuit | None:
        """返回该task所属的Circuit（占位）。若未维护映射，返回None。"""
        return None

    def _next_resource_release_time(self, time_now: float) -> float:
        """计算距离下一次资源释放的时间间隔（占位）。"""
        return 1.0
        
    
    