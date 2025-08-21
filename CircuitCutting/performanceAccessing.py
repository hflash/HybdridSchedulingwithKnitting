from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import QuantumCircuit
import os
# from qiskit.providers.fake_provider import fake_provider
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeGuadalupeV2, FakeHanoiV2, FakeMelbourneV2, FakeManhattanV2
import math
from iterationCircuitCut import read_circuit, compute_subcircuits_with_budget

# 固定随机种子，稳定SABRE与其他启发式行为
SEED_TRANSPILE = 12345
import os
import numpy as np
# ------------------------
# 宏：执行正确率（可根据需要调整）
# ------------------------
TWO_Q_GATE_CORR = 0.9838782711905    # 两比特门执行正确率
ONE_Q_GATE_CORR = 0.9995892369557831   # 单比特门执行正确率
MEAS_CORR       = 0.9755874999999999    # 测量执行正确率（逐比特）
DECO_CORR       = 0.999   # 退相干时间正确率（逐深度层）
# 退相干模型宏：T为特征时间常数(秒)，AVG_GATE_TIME为平均门时长(秒)
DECO_T          = 8.115298082762845e-05  # 示例：100微秒
AVG_GATE_TIME   = 1.8133333333333334e-07   # 示例：50纳秒


def compute_macros_from_backend(backend, main_twoq_candidates=("cx", "ecr")):
    """
    基于单个IBM Fake后端，提取用于宏的统计样本（不做聚合，仅收集）：
    - 仅统计"主双比特门"（优先 'cx'，若无则 'ecr'），并将qargs按无向边去重
    返回 dict:
      {
        'twoq_corrs': [float, ...],        # 主双比特门的(1-error)，无向去重
        'oneq_corrs': [float, ...],        # 单比特门的(1-error)
        'meas_corrs': [float, ...],        # 每个比特的(1-readout_error)
        'gate_durations': [float, ...],    # 每个门的持续时间(秒)（含单双比特门），双比特无向去重
        't1s': [float, ...],               # 每个比特的T1(秒)
        'meta': { 'main_twoq': str|None }
      }
    """
    samples = {
        'twoq_corrs': [],
        'oneq_corrs': [],
        'meas_corrs': [],
        'gate_durations': [],
        't1s': [],
        'meta': {'main_twoq': None},
    }
    try:
        target = getattr(backend, 'target', None)
        dt = None
        try:
            dt = getattr(target, 'dt', None)
        except Exception:
            dt = getattr(backend, 'dt', None)

        # 选择主双比特门
        main_twoq = None
        try:
            opset = set(getattr(target, 'operation_names', []) or []) if target is not None else set()
            for cand in main_twoq_candidates:
                if cand in opset:
                    main_twoq = cand
                    break
        except Exception:
            main_twoq = None
        samples['meta']['main_twoq'] = main_twoq

        if target is not None:
            # 预先构建无向去重集合
            seen_twoq_pairs = set()    # {(min(a,b), max(a,b))}
            seen_dur_twoq = set()

            for opname in getattr(target, 'operation_names', []):
                props_map = target.get(opname)
                if not props_map:
                    continue
                for qargs, props in props_map.items():
                    if props is None:
                        continue

                    # 处理duration为秒
                    try:
                        dur = getattr(props, 'duration', None)
                        if dur is not None:
                            dval = float(dur)
                            if dt and dval > 0 and float(dval).is_integer():
                                dsec = dval * float(dt)
                            else:
                                dsec = dval
                        else:
                            dsec = None
                    except Exception:
                        dsec = None

                    # 单/双比特分类
                    try:
                        if qargs is None:
                            continue
                        arity = len(qargs)
                    except Exception:
                        arity = 0

                    # 正确率 = 1 - error
                    corr = None
                    try:
                        if hasattr(props, 'error') and props.error is not None:
                            corr = max(0.0, 1.0 - float(props.error))
                    except Exception:
                        corr = None

                    # 测量门
                    if str(opname).lower() == 'measure' and arity == 1:
                        if corr is not None:
                            samples['meas_corrs'].append(corr)
                        # duration也纳入总体门时长样本
                        if dsec is not None and dsec > 0:
                            samples['gate_durations'].append(dsec)
                        continue

                    # 单比特门
                    if arity == 1:
                        if corr is not None:
                            samples['oneq_corrs'].append(corr)
                        if dsec is not None and dsec > 0:
                            samples['gate_durations'].append(dsec)
                        continue

                    # 双比特门（仅统计主门，且无向去重）
                    if arity == 2 and main_twoq is not None and str(opname).lower() == str(main_twoq).lower():
                        try:
                            a, b = int(qargs[0]), int(qargs[1])
                        except Exception:
                            # 若qargs不可转int，就跳过
                            a, b = None, None
                        if a is None or b is None:
                            continue
                        u, v = (a, b) if a < b else (b, a)
                        pair = (u, v)
                        if pair not in seen_twoq_pairs:
                            if corr is not None:
                                samples['twoq_corrs'].append(corr)
                            if dsec is not None and dsec > 0 and pair not in seen_dur_twoq:
                                samples['gate_durations'].append(dsec)
                                seen_dur_twoq.add(pair)
                            seen_twoq_pairs.add(pair)
                        continue

                    # 其他门（非主双比特门）仅用于时长样本（不计入twoq_corrs）
                    if dsec is not None and dsec > 0:
                        samples['gate_durations'].append(dsec)

        # T1 from qubit_properties
        if hasattr(backend, 'qubit_properties'):
            try:
                phys = list(getattr(target, 'physical_qubits', [])) if target is not None else None
                if phys is None or len(phys) == 0:
                    n = getattr(backend, 'num_qubits', 0) or 0
                    phys = list(range(int(n)))
                for i in phys:
                    try:
                        qp = backend.qubit_properties(int(i))
                    except Exception:
                        qp = None
                    if qp is None:
                        continue
                    t1 = getattr(qp, 't1', None)
                    if t1 is not None:
                        try:
                            t1v = float(t1)
                            if t1v > 0:
                                samples['t1s'].append(t1v)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass
    return samples


def _robust_summary(arr, trim_ratio=0.1):
    """
    对数列给出稳健统计：均值、截尾均值(默认10%)、中位数、P10、P90。
    返回 dict: {'mean': float|None, 'tmean': float|None, 'median': float|None, 'p10': float|None, 'p90': float|None}
    """
    if not arr:
        return {'mean': None, 'tmean': None, 'median': None, 'p10': None, 'p90': None}
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {'mean': None, 'tmean': None, 'median': None, 'p10': None, 'p90': None}
    a_sorted = np.sort(a)
    mean = float(np.mean(a_sorted))
    median = float(np.median(a_sorted))
    p10 = float(np.percentile(a_sorted, 10))
    p90 = float(np.percentile(a_sorted, 90))
    # 截尾均值
    n = a_sorted.size
    k = int(n * trim_ratio)
    if k * 2 >= n:
        tmean = mean
    else:
        tmean = float(np.mean(a_sorted[k:n - k]))
    return {'mean': mean, 'tmean': tmean, 'median': median, 'p10': p10, 'p90': p90}


def compute_ibm_fake_macro_averages():
    """
    计算宏的建议值，基于四个IBM Fake设备：
      - FakeHanoiV2 (27)
      - FakeGuadalupeV2 (16)
      - FakeManhattanV2 (65)
      - FakeMelbourneV2 (15)
    规则：
      - 仅统计主双比特门（优先'cx'，若无则'ecr'），对qargs按无向去重
      - 使用稳健统计（中位数/截尾均值），先在每台设备内聚合，再在设备间求平均
      - 返回顶层键作为推荐值（采用"设备内中位数"在设备间的平均），并附带详细统计以便审阅

    返回：
      {
        'TWO_Q_GATE_CORR': float|None,   # 推荐值（跨设备平均的"设备内中位数"）
        'ONE_Q_GATE_CORR': float|None,
        'MEAS_CORR': float|None,
        'DECO_T': float|None,            # T1为代表时间尺度
        'AVG_GATE_TIME': float|None,
        'details': {
            'by_backend': [
                { 'name': str, 'main_twoq': str|None,
                  'twoq': {'mean','tmean','median','p10','p90'},
                  'oneq': {...}, 'meas': {...}, 't1': {...}, 'dur': {...}
                }, ...
            ],
            'across_devices': {
                'twoq_medians': {'mean','tmean','median','p10','p90'},
                'oneq_medians': {...},
                'meas_medians': {...},
                't1_medians': {...},
                'dur_medians': {...}
            }
        }
      }
    """
    backends = [FakeHanoiV2(), FakeGuadalupeV2(), FakeManhattanV2(), FakeMelbourneV2()]

    per_backend = []
    # 设备内聚合：计算各项稳健统计
    for be in backends:
        name = type(be).__name__
        s = compute_macros_from_backend(be)
        twoq_stats = _robust_summary(s.get('twoq_corrs', []))
        oneq_stats = _robust_summary(s.get('oneq_corrs', []))
        meas_stats = _robust_summary(s.get('meas_corrs', []))
        t1_stats   = _robust_summary(s.get('t1s', []))
        dur_stats  = _robust_summary(s.get('gate_durations', []))
        per_backend.append({
            'name': name,
            'main_twoq': s.get('meta', {}).get('main_twoq'),
            'twoq': twoq_stats,
            'oneq': oneq_stats,
            'meas': meas_stats,
            't1': t1_stats,
            'dur': dur_stats,
        })

    # 设备间再聚合：取"设备内中位数"的集合，再做稳健统计
    def _collect_medians(key):
        vals = []
        for item in per_backend:
            v = item.get(key, {}).get('median')
            if v is not None:
                vals.append(v)
        return vals

    twoq_meds = _collect_medians('twoq')
    oneq_meds = _collect_medians('oneq')
    meas_meds = _collect_medians('meas')
    t1_meds   = _collect_medians('t1')
    dur_meds  = _collect_medians('dur')

    across = {
        'twoq_medians': _robust_summary(twoq_meds),
        'oneq_medians': _robust_summary(oneq_meds),
        'meas_medians': _robust_summary(meas_meds),
        't1_medians': _robust_summary(t1_meds),
        'dur_medians': _robust_summary(dur_meds),
    }

    # 推荐值：采用"设备内中位数"的简单平均（若为空则None）
    def _avg(vals):
        return float(sum(vals) / len(vals)) if vals else None

    macros = {
        'TWO_Q_GATE_CORR': _avg(twoq_meds),
        'ONE_Q_GATE_CORR': _avg(oneq_meds),
        'MEAS_CORR': _avg(meas_meds),
        'DECO_T': _avg(t1_meds),
        'AVG_GATE_TIME': _avg(dur_meds),
        'details': {
            'by_backend': per_backend,
            'across_devices': across,
        }
    }
    return macros


def _extract_circuit_stats(qc):
    """
    提取线路统计信息：两比特门数、单比特门数、测量比特数、线路深度。
    兼容Qiskit样式(具有data、depth等)或自定义电路对象（尽量回退）。
    返回: (num_twoq, num_oneq, num_meas_qubits, depth)
    """
    num_twoq = 0
    num_oneq = 0
    meas_qubits = set()
    depth = 0

    # 1) 深度（优先使用属性/方法）
    try:
        if hasattr(qc, 'depth'):
            d = qc.depth() if callable(qc.depth) else qc.depth
            depth = int(d) if d is not None else 0
    except Exception:
        depth = 0

    # 2) 遍历指令（Qiskit风格）
    try:
        data = getattr(qc, 'data', None)
        if data is not None:
            for inst, qargs, _ in data:
                name = getattr(inst, 'name', '').lower()
                qlen = len(qargs) if qargs is not None else 0
                if name == 'measure':
                    # 记录被测量的量子比特索引
                    for q in (qargs or []):
                        idx = getattr(q, '_index', getattr(q, 'index', None))
                        if idx is not None:
                            meas_qubits.add(int(idx))
                else:
                    if qlen == 2:
                        num_twoq += 1
                    elif qlen == 1:
                        num_oneq += 1
            return num_twoq, num_oneq, len(meas_qubits), depth
    except Exception:
        pass

    # 3) 回退：若有操作计数接口（如count_ops或ops字典）
    try:
        if hasattr(qc, 'count_ops') and callable(getattr(qc, 'count_ops')):
            ops = qc.count_ops()
            # 常见双比特门
            for k, v in ops.items():
                key = str(k).lower()
                if key in {'cx', 'cz', 'swap', 'ecr'}:
                    num_twoq += int(v)
            # 常见单比特门（粗略）
            for k, v in ops.items():
                key = str(k).lower()
                if key in {'x','y','z','h','s','sdg','t','tdg','rx','ry','rz','u','u3','id','sx'}:
                    num_oneq += int(v)
            # 测量数量未知情况下：取0（可按需扩展接口）
            return num_twoq, num_oneq, len(meas_qubits), depth
    except Exception:
        pass

    # 4) 全部失败则返回零统计
    return num_twoq, num_oneq, len(meas_qubits), depth


def compute_fidelity_requirement(qc):
    """
    基于宏常量与线路属性，计算"保真度要求"。
    模型：
      fidelity_req = (TWO_Q_GATE_CORR ^ 两比特门数)
                     * (ONE_Q_GATE_CORR ^ 单比特门数)
                     * (MEAS_CORR ^ 测量比特数)
                     * (DECO_CORR ^ 线路深度)
    入参：
      qc: Qiskit 量子线路或兼容对象（需尽可能具有 data/depth 接口，或 count_ops 回退）
    返回：
      dict {
        'two_qubit_gates': int,
        'one_qubit_gates': int,
        'measured_qubits': int,
        'depth': int,
        'fidelity_requirement': float,
        'macros': { 'TWO_Q_GATE_CORR': float, 'ONE_Q_GATE_CORR': float, 'MEAS_CORR': float, 'DECO_CORR': float }
      }
    """
    num_twoq, num_oneq, _ignored_meas, depth = _extract_circuit_stats(qc)
    # 测量比特数 = 线路比特数（不依赖最终是否显式测量）
    try:
        num_meas_qubits = int(getattr(qc, 'num_qubits', 0))
    except Exception:
        num_meas_qubits = 0
    # 退相干因子改为: (1 - exp(-t / T)) ^ 执行时间，其中 执行时间 = depth * AVG_GATE_TIME
    # 注意：此处"^执行时间"为按秒的实数次幂（即连续时间近似）。
    # 若需使用离散步长，可自行改为按层累乘。
    import math
    t_seconds = depth * AVG_GATE_TIME
    deco_factor = (1.0 - math.exp(-t_seconds / DECO_T)) ** (t_seconds if t_seconds > 0 else 0.0)

    fidelity_req = (TWO_Q_GATE_CORR ** int(num_twoq)) \
                 * (ONE_Q_GATE_CORR ** int(num_oneq)) \
                 * (MEAS_CORR       ** int(num_meas_qubits)) \
                 * float(deco_factor)
    return {
        'two_qubit_gates': int(num_twoq),
        'one_qubit_gates': int(num_oneq),
        'measured_qubits': int(num_meas_qubits),
        'depth': int(depth),
        'exec_time_seconds': float(t_seconds),
        'deco_factor': float(deco_factor),
        'fidelity_requirement': float(fidelity_req),
        'macros': {
            'TWO_Q_GATE_CORR': TWO_Q_GATE_CORR,
            'ONE_Q_GATE_CORR': ONE_Q_GATE_CORR,
            'MEAS_CORR': MEAS_CORR,
            'DECO_CORR': DECO_CORR,
            'DECO_T': DECO_T,
            'AVG_GATE_TIME': AVG_GATE_TIME,
        }
    }


def _schedule_circuit_for_timing(circ, backend):
    """
    使用后端的Target时序信息对电路进行调度，便于获取总时长（dt为时间步长）。
    返回已调度电路与dt（秒）。
    """
    scheduled = transpile(circ, backend=backend, scheduling_method='asap', optimization_level=0, seed_transpiler=SEED_TRANSPILE)
    dt = getattr(backend, 'dt', None)
    if dt is None and hasattr(backend, 'target'):
        dt = getattr(backend.target, 'dt', None)
    return scheduled, dt


def estimate_exec_time_seconds(circ, backend):
    """
    估计电路在指定后端上的执行时间（秒）。需要backend提供dt并完成调度。
    """
    scheduled, dt = _schedule_circuit_for_timing(circ, backend)
    total_dt = getattr(scheduled, 'duration', None)
    if total_dt is None or dt is None:
        return None
    return float(total_dt) * float(dt)


def _get_instruction_error_from_target(backend, inst_name, qargs_tuple):
    """
    从BackendV2的Target中读取某个指令在特定qargs上的error（若无则返回0）。
    """
    try:
        target = backend.target
        if inst_name in target:
            props = target[inst_name].get(qargs_tuple)
            if props is not None and hasattr(props, 'error') and props.error is not None:
                return float(props.error)
    except Exception:
        pass
    return 0.0


def estimate_gate_product_fidelity(circ, backend):
    """
    将每个门的(1 - error)相乘，得到门误差近似的整体保真度（粗略上界）。
    若某些门无error数据，视为0误差。
    """
    fidelity = 1.0
    for inst, qargs, _ in circ.data:
        name = inst.name
        # 安全地获取每个量子比特的索引（优先 _index，其次 index）
        indices = []
        for q in qargs:
            if hasattr(q, '_index'):
                indices.append(q._index)
            else:
                indices.append(getattr(q, 'index', None))
        # 若有索引缺失，则跳过误差查询（按0误差处理）
        if any(idx is None for idx in indices):
            p_err = 0.0
        else:
            qidx = tuple(indices)
            p_err = _get_instruction_error_from_target(backend, name, qidx)
        fidelity *= max(0.0, 1.0 - p_err)
    return fidelity


def estimate_decoherence_factor(backend, total_seconds, num_qubits):
    """
    基于全局时长的粗略退相干因子估计：按每个物理比特叠乘exp(-T/T1)与exp(-T/T2)。
    注意：这是保守的近似（真实需精确调度逐比特空闲/占用时间）。
    """
    factor = 1.0
    if hasattr(backend, 'qubit_properties'):
        for q in range(num_qubits):
            try:
                qp = backend.qubit_properties(q)
            except Exception:
                qp = None
            if qp is None:
                continue
            t1 = getattr(qp, 't1', None)
            t2 = getattr(qp, 't2', None)
            if t1:
                factor *= math.exp(-total_seconds / float(t1))
            if t2:
                factor *= math.exp(-total_seconds / float(t2))
    return factor


def _get_measure_error_for_qubit(backend, qidx):
    """
    获取单个量子比特的测量误差。
    优先从Target的measure指令读取；若无，则尝试legacy BackendProperties。
    返回值范围[0,1]，若未知则返回0.0。
    """
    # 优先：Target中的measure指令
    try:
        target = getattr(backend, 'target', None)
        if target is not None and 'measure' in target:
            props = target['measure'].get((qidx,))
            if props is not None and hasattr(props, 'error') and props.error is not None:
                return float(props.error)
    except Exception:
        pass

    # 备选：BackendProperties（老接口）
    try:
        props_fn = getattr(backend, 'properties', None)
        if callable(props_fn):
            bp = props_fn()
            # 直接方法
            if hasattr(bp, 'readout_error'):
                try:
                    val = bp.readout_error(qidx)
                    if val is not None:
                        return float(val)
                except Exception:
                    pass
            # 遍历参数项
            if hasattr(bp, 'qubits') and bp.qubits and qidx < len(bp.qubits):
                for param in bp.qubits[qidx]:
                    name = getattr(param, 'name', '')
                    if name == 'readout_error':
                        return float(getattr(param, 'value', 0.0))
    except Exception:
        pass

    return 0.0


def estimate_readout_fidelity(circ, backend):
    """
    统计电路中的测量操作，按测量误差(1 - p_err)累乘，返回读取保真度。
    若电路中无显式测量，则返回1.0。
    """
    measured_qubits = set()
    for inst, qargs, _ in circ.data:
        if inst.name == 'measure':
            # 安全获取索引
            for q in qargs:
                qidx = q._index if hasattr(q, '_index') else getattr(q, 'index', None)
                if qidx is not None:
                    measured_qubits.add(qidx)
    if not measured_qubits:
        return 1.0
    fid = 1.0
    for qidx in measured_qubits:
        p_err = _get_measure_error_for_qubit(backend, qidx)
        fid *= max(0.0, 1.0 - p_err)
    return fid


def estimate_time_and_fidelity(circ, backend):
    """
    估计电路总执行时间（秒）与综合保真度（门误差×测量误差×退相干的粗略估计）。
    返回: (seconds, fidelity)
    """
    scheduled, dt = _schedule_circuit_for_timing(circ, backend)
    total_dt = getattr(scheduled, 'duration', None)
    seconds = None
    if total_dt is not None and dt is not None:
        seconds = float(total_dt) * float(dt)
    gate_fid = estimate_gate_product_fidelity(scheduled, backend)
    meas_fid = estimate_readout_fidelity(scheduled, backend)
    deco_fid = estimate_decoherence_factor(backend, seconds if seconds is not None else 0.0, scheduled.num_qubits)
    overall = gate_fid * meas_fid * deco_fid
    return seconds, overall


# --------------------------
# 区域映射辅助工具
# --------------------------

def _region_edges_for_gate(backend, region_qubits, two_q_gate_name='cx'):
    """
    从后端Target中提取指定双比特门在region中的耦合边（无向去重）。
    返回[[u, v], ...] 形式的列表。
    """
    edges = set()
    try:
        qargs_set = backend.target.qargs_for_operation_name(two_q_gate_name)
        for qargs in qargs_set:
            if len(qargs) == 2:
                a, b = qargs
                if a in region_qubits and b in region_qubits:
                    u, v = sorted((a, b))
                    edges.add((u, v))
    except Exception:
        # 若Target不可用，回退为空
        pass
    return [[u, v] for (u, v) in sorted(edges)]


essential_basis = ["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg", "measure"]

def transpile_to_region(circ, backend, region_qubits, seed=SEED_TRANSPILE, opt_level=1):
    """
    将电路映射到后端芯片的指定物理比特区域内。
    要求: circ.num_qubits <= len(region_qubits)。
    """
    if circ.num_qubits > len(region_qubits):
        raise ValueError("区域物理比特数量不足以容纳电路虚拟比特")
    # 取后端支持的基础门，与必需门并集
    try:
        basis = sorted(set(backend.target.operation_names) | set(essential_basis))
    except Exception:
        basis = essential_basis
    # 区域耦合图（仅region内部）
    region_edges = _region_edges_for_gate(backend, set(region_qubits), two_q_gate_name='cx')
    # 初始布局：将前n个虚拟比特放到region中的前n个物理比特
    initial_layout = list(region_qubits)[: circ.num_qubits]
    # 在受限耦合图下运行SABRE路由，避免跑出区域
    mapped = transpile(
        circ,
        basis_gates=basis,
        coupling_map=region_edges,
        initial_layout=initial_layout,
        layout_method='trivial',
        routing_method='sabre',
        optimization_level=opt_level,
        seed_transpiler=seed,
    )
    return mapped

 

def _get_coupling_edges(backend, two_q_gate_name='cx'):
    """
    从BackendV2的Target或后备字段提取双比特门耦合边（无向去重）。
    返回[[u, v], ...]。
    """
    edges = set()
    # 优先：Target提供的qargs
    try:
        target = getattr(backend, 'target', None)
        if target is not None:
            try:
                qargs_set = target.qargs_for_operation_name(two_q_gate_name)
                for qargs in qargs_set:
                    if len(qargs) == 2:
                        a, b = qargs
                        u, v = (a, b) if a < b else (b, a)
                        edges.add((u, v))
            except Exception:
                pass
    except Exception:
        pass
    # 备选：直接读取backend的coupling_map（若存在）
    if not edges:
        cm = getattr(backend, 'coupling_map', None)
        if cm:
            try:
                for a, b in cm:
                    u, v = (a, b) if a < b else (b, a)
                    edges.add((u, v))
            except Exception:
                pass
    return [[u, v] for (u, v) in sorted(edges)]


def _bfs_region(adj, start, k, excluded=None):
    """
    从start出发的BFS，收集前k个可达节点，返回列表（若不足k则返回实际数量）。
    excluded: 需要跳过的比特集合。
    """
    excluded = set(excluded) if excluded else set()
    visited = set()
    order = []
    if start in excluded:
        return order
    queue = [start]
    while queue and len(order) < k:
        node = queue.pop(0)
        if node in visited or node in excluded:
            continue
        visited.add(node)
        order.append(node)
        # 邻居排序以稳定输出
        for nb in sorted(adj.get(node, [])):
            if nb not in visited and nb not in excluded:
                queue.append(nb)
    return order


def find_feasible_regions_for_circuit(circ, backend, two_q_gate_name='cx', max_regions=10, excluded_qubits=None):
    """
    基于backend耦合图，找到大小为电路比特数的连通可行区域。
    - 仅保证连通（非完全图）；返回多个候选（去重），最多max_regions个。
    - excluded_qubits: 需要排除的物理比特ID集合/列表，这些比特不会出现在返回的区域中。
    返回: List[List[int]]（每个是物理比特列表）。
    """
    k = circ.qc.num_qubits
    edges = _get_coupling_edges(backend, two_q_gate_name=two_q_gate_name)
    if not edges:
        raise ValueError("未能从后端获取耦合边，无法搜索可行区域")
    # 构建邻接表
    adj = {}
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    nodes = sorted(nodes)
    excluded = set(excluded_qubits) if excluded_qubits else set()
    # 简单BFS从每个起点收集k个节点作为一个连通候选（排除excluded）
    seen = set()
    regions = []
    for s in nodes:
        if s in excluded:
            continue
        region = _bfs_region(adj, s, k, excluded=excluded)
        if len(region) == k:
            key = tuple(sorted(region))
            if key not in seen:
                seen.add(key)
                regions.append(list(key))
                if len(regions) >= max_regions:
                    break
    return regions



def _region_internal_edges(edges, region_set):
    cnt = 0
    for u, v in edges:
        if u in region_set and v in region_set:
            cnt += 1
    return cnt


def _region_connectivity_score(edges, region):
    n = len(region)
    if n <= 1:
        return 1.0
    region_set = set(region)
    e = _region_internal_edges(edges, region_set)
    max_e = n * (n - 1) / 2
    return e / max_e if max_e > 0 else 0.0


def _avg_gate_fidelity_in_region(backend, region, two_q_gate_name='cx'):
    """
    计算区域内的平均门保真度：综合单比特门与双比特(two_q_gate_name)门。
    返回: (gate_avg, sq_avg or None, tq_avg or None)
    """
    region_set = set(region)
    target = getattr(backend, 'target', None)

    # 单比特门候选名称集合（与后端交集）
    single_gate_candidates = {
        'x', 'sx', 'rz', 'ry', 'rx', 'h', 's', 'sdg', 't', 'tdg', 'id', 'u', 'u3'
    }
    try:
        if target is not None:
            single_ops = sorted(single_gate_candidates & set(target.operation_names))
        else:
            single_ops = list(single_gate_candidates)
    except Exception:
        single_ops = list(single_gate_candidates)

    # 单比特平均保真度
    sq_vals = []
    for q in region:
        qubit_vals = []
        for op in single_ops:
            try:
                props = target[op].get((q,)) if target is not None else None
                if props is not None and hasattr(props, 'error') and props.error is not None:
                    qubit_vals.append(max(0.0, 1.0 - float(props.error)))
            except Exception:
                continue
        if qubit_vals:
            sq_vals.append(sum(qubit_vals) / len(qubit_vals))
    sq_avg = (sum(sq_vals) / len(sq_vals)) if sq_vals else None

    # 双比特平均保真度（仅区域内部允许的边）
    tq_vals = []
    try:
        if target is not None and two_q_gate_name in target:
            # 尝试双向查询
            for a in region:
                for b in region:
                    if a == b:
                        continue
                    props = target[two_q_gate_name].get((a, b))
                    if props is not None and hasattr(props, 'error') and props.error is not None:
                        tq_vals.append(max(0.0, 1.0 - float(props.error)))
    except Exception:
        pass
    # 若仅有有向边，以上会重复计数；取平均无妨
    tq_avg = (sum(tq_vals) / len(tq_vals)) if tq_vals else None

    # 组合总体平均
    parts = []
    if sq_avg is not None:
        parts.append(sq_avg)
    if tq_avg is not None:
        parts.append(tq_avg)
    gate_avg = sum(parts) / len(parts) if parts else 0.0
    return gate_avg, sq_avg, tq_avg


def find_best_region_by_avg_performance(circ, backend, two_q_gate_name='cx', excluded_qubits=None, weight=0.5, max_regions=200):
    """
    在芯片上找到一个"平均性能最好"的连通区域，区域大小等于电路比特数。
    性能度量 = weight * 平均门保真度 + (1-weight) * 区域连通性(边密度)。
    - 平均门保真度: 区域内单/双比特门的平均(1-error)
    - 连通性: 区域子图的边密度 = |E_region| / (n*(n-1)/2)

    返回: dict { 'region': List[int], 'score': float, 'gate_avg': float, 'sq_avg': float|None, 'tq_avg': float|None, 'connectivity': float }
    若找不到则返回None。
    """
    k = circ.num_qubits
    edges = _get_coupling_edges(backend, two_q_gate_name=two_q_gate_name)
    if not edges:
        return None
    # 邻接表
    adj = {}
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    nodes = sorted(nodes)

    excluded = set(excluded_qubits) if excluded_qubits else set()

    seen = set()
    best = None
    count = 0
    for s in nodes:
        if s in excluded:
            continue
        region = _bfs_region(adj, s, k, excluded=excluded)
        if len(region) != k:
            continue
        key = tuple(sorted(region))
        if key in seen:
            continue
        seen.add(key)
        count += 1
        # 计算性能
        gate_avg, sq_avg, tq_avg = _avg_gate_fidelity_in_region(backend, region, two_q_gate_name=two_q_gate_name)
        conn = _region_connectivity_score(edges, region)
        score = weight * gate_avg + (1.0 - weight) * conn
        item = {
            'region': list(key),
            'score': score,
            'gate_avg': gate_avg,
            'sq_avg': sq_avg,
            'tq_avg': tq_avg,
            'connectivity': conn,
        }
        if best is None or item['score'] > best['score']:
            best = item
        if count >= max_regions:
            break
    return best



def allocate_regions_for_circuits(circuits, backend, two_q_gate_name='cx', excluded_qubits=None, weight=0.5, max_regions_each=200):
    """
    为多条线路按优先级（列表下标越小优先级越高）分配芯片上的连通区域。
    约束：区域大小等于各线路比特数；各区域互不重叠；可排除指定物理比特。
    目标：优先级越高的线路，其区域"平均性能"（门保真度与连通性加权）越好。

    返回：列表，与circuits等长。每项为dict：
      {
        'index': idx,
        'region': List[int] | None,
        'score': float | None,
        'gate_avg': float | None,
        'sq_avg': float | None,
        'tq_avg': float | None,
        'connectivity': float | None,
        'mapped_circuit': QuantumCircuit | None,
        'est_seconds': float | None,
        'est_fidelity': float | None,
        'status': 'ok' | 'no_region' | 'error',
        'message': str | None,
      }
    """
    results = []
    excluded = set(excluded_qubits) if excluded_qubits else set()

    for idx, circ in enumerate(circuits):
        try:
            # 先尝试寻找平均性能最优的区域（在当前excluded之外）
            best = find_best_region_by_avg_performance(
                circ,
                backend,
                two_q_gate_name=two_q_gate_name,
                excluded_qubits=excluded,
                weight=weight,
                max_regions=max_regions_each,
            )
            # 如找不到，退化为可行区域搜索后再评分
            if best is None or not best.get('region'):
                candidates = find_feasible_regions_for_circuit(
                    circ,
                    backend,
                    two_q_gate_name=two_q_gate_name,
                    max_regions=max_regions_each,
                    excluded_qubits=excluded,
                )
                if not candidates:
                    results.append({
                        'index': idx,
                        'region': None,
                        'score': None,
                        'gate_avg': None,
                        'sq_avg': None,
                        'tq_avg': None,
                        'connectivity': None,
                        'mapped_circuit': None,
                        'est_seconds': None,
                        'est_fidelity': None,
                        'status': 'no_region',
                        'message': '无可行连通区域',
                    })
                    continue
                # 选取评分最高的候选
                edges = _get_coupling_edges(backend, two_q_gate_name=two_q_gate_name)
                sel = None
                for region in candidates:
                    gate_avg, sq_avg, tq_avg = _avg_gate_fidelity_in_region(backend, region, two_q_gate_name=two_q_gate_name)
                    conn = _region_connectivity_score(edges, region)
                    score = weight * gate_avg + (1.0 - weight) * conn
                    item = {
                        'region': region,
                        'score': score,
                        'gate_avg': gate_avg,
                        'sq_avg': sq_avg,
                        'tq_avg': tq_avg,
                        'connectivity': conn,
                    }
                    if sel is None or item['score'] > sel['score']:
                        sel = item
                best = sel

            region = best['region']
            # 标记占用，保证后续不重叠
            excluded.update(region)

            # 在该区域上转译与评估
            mapped = transpile_to_region(circ, backend, region)
            secs, fid = estimate_time_and_fidelity(mapped, backend)

            results.append({
                'index': idx,
                'region': region,
                'score': best.get('score'),
                'gate_avg': best.get('gate_avg'),
                'sq_avg': best.get('sq_avg'),
                'tq_avg': best.get('tq_avg'),
                'connectivity': best.get('connectivity'),
                'mapped_circuit': mapped,
                'est_seconds': secs,
                'est_fidelity': fid,
                'status': 'ok',
                'message': None,
            })
        except Exception as e:
            results.append({
                'index': idx,
                'region': None,
                'score': None,
                'gate_avg': None,
                'sq_avg': None,
                'tq_avg': None,
                'connectivity': None,
                'mapped_circuit': None,
                'est_seconds': None,
                'est_fidelity': None,
                'status': 'error',
                'message': str(e),
            })
    return results


def compute_swap_cnot_counts_per_logical(original_circ, mapped_circ, initial_layout):
    """
    按逻辑比特统计：
    - swaps_on_logical[l]: 执行路由后电路时，该逻辑比特经历的SWAP次数（通过跟踪映射动态变化；每个SWAP计两端参与的逻辑比特）
    - orig_cx_on_logical[l]: 原始电路中，该逻辑比特参与的CNOT次数（每个CNOT计两端参与的逻辑比特）

    要求：提供 initial_layout（长度为 original_circ.num_qubits），表示初始时逻辑比特l映射到的物理比特ID。

    返回：dict { logical_index: { 'swaps': int, 'orig_cx': int, 'ratio': float|None } }
    """
    n = original_circ.num_qubits
    if initial_layout is None or len(initial_layout) < n:
        raise ValueError("需要提供长度不少于电路比特数的initial_layout")

    # 1) 原始电路：按逻辑比特统计CNOT次数
    orig_cx_on_logical = [0] * n
    for inst, qargs, _ in original_circ.data:
        if inst.name == 'cx' and len(qargs) == 2:
            a = qargs[0]._index if hasattr(qargs[0], '_index') else getattr(qargs[0], 'index', None)
            b = qargs[1]._index if hasattr(qargs[1], '_index') else getattr(qargs[1], 'index', None)
            if a is not None and 0 <= a < n:
                orig_cx_on_logical[a] += 1
            if b is not None and 0 <= b < n:
                orig_cx_on_logical[b] += 1

    # 2) 路由后电路：动态跟踪逻辑<->物理映射，统计每个逻辑比特经历的SWAP次数
    swaps_on_logical = [0] * n
    log_to_phys = {l: int(initial_layout[l]) for l in range(n)}
    phys_to_log = {int(initial_layout[l]): l for l in range(n)}

    # 构建仅包含双比特CX/显式SWAP的序列，忽略所有单比特门
    ops = []  # 元素: (name, p0, p1)
    for inst, qargs, _ in mapped_circ.data:
        if len(qargs) == 2 and inst.name in ('cx', 'swap'):
            p0 = qargs[0]._index if hasattr(qargs[0], '_index') else getattr(qargs[0], 'index', None)
            p1 = qargs[1]._index if hasattr(qargs[1], '_index') else getattr(qargs[1], 'index', None)
            if p0 is not None and p1 is not None:
                ops.append((inst.name, int(p0), int(p1)))
    i = 0
    while i < len(ops):
        name, a0, b0 = ops[i]
        # 显式swap门
        if name == 'swap':
            l0 = phys_to_log.get(a0)
            l1 = phys_to_log.get(b0)
            if l0 is not None and 0 <= l0 < n:
                swaps_on_logical[l0] += 1
            if l1 is not None and 0 <= l1 < n:
                swaps_on_logical[l1] += 1
            # 更新映射（真实swap）
            phys_to_log[a0], phys_to_log[b0] = l1, l0
            if l0 is not None:
                log_to_phys[l0] = b0
            if l1 is not None:
                log_to_phys[l1] = a0
            i += 1
            continue
        # 连续三个相同的cx（忽略单比特门），判为swap；映射不更新
        if name == 'cx' and i + 2 < len(ops):
            n1 = ops[i+1]
            n2 = ops[i+2]
            if n1[0] == 'cx' and n2[0] == 'cx':
                a1, b1 = n1[1], n1[2]
                a2, b2 = n2[1], n2[2]
                if a0 == a1 == a2 and b0 == b1 == b2:
                    l0 = phys_to_log.get(a0)
                    l1 = phys_to_log.get(b0)
                    if l0 is not None and 0 <= l0 < n:
                        swaps_on_logical[l0] += 1
                    if l1 is not None and 0 <= l1 < n:
                        swaps_on_logical[l1] += 1
                    i += 3
                    continue
        i += 1

    # 3) 组装输出（含比值）
    out = {}
    for l in range(n):
        orig_cx = orig_cx_on_logical[l]
        swaps = swaps_on_logical[l]
        denom = orig_cx if orig_cx > 0 else 1
        ratio = swaps / denom
        if orig_cx == 0:
            ratio = 0.0
        out[l] = {'swaps': swaps, 'orig_cx': orig_cx, 'ratio': ratio}
    return out

def print_circuit_gates(circ, title=None):
    """
    打印电路中所有门的名称与其作用的量子比特索引（按当前电路中的物理索引）。
    """
    if title:
        print(title)
    for i, (inst, qargs, _) in enumerate(circ.data):
        qidx = []
        for q in qargs:
            qidx.append(q._index if hasattr(q, '_index') else getattr(q, 'index', None))
        print(f"{i}: {inst.name} -> qubits {qidx}")



def _backend_from_name(chip_name: str):
    """
    根据芯片名称字符串返回对应的Fake后端实例。
    支持: 'FakeManilaV2', 'FakeGuadalupeV2', 'FakeHanoiV2', 'FakeMelbourneV2', 'FakeManhattanV2'
    """
    name = chip_name.strip()
    mapping = {
        'FakeManilaV2': FakeManilaV2,
        'FakeGuadalupeV2': FakeGuadalupeV2,
        'FakeHanoiV2': FakeHanoiV2,
        'FakeMelbourneV2': FakeMelbourneV2,
        'FakeManhattanV2': FakeManhattanV2,
    }
    if chip_name not in mapping:
        raise ValueError(f"不支持的芯片名称: {chip_name}")
    return mapping[name]()


def assign_regions_and_estimates(circuits, chip_name: str, weight: float = 0.5, excluded_qubits=None, two_q_gate_name: str = 'cx', max_regions_each: int = 200):
    """
    1) 给定多条量子线路（按列表顺序高优先级在前）和芯片名称，
       为每条线路分配互不重叠的连通区域（优先级高的区域"平均性能"更好），
       并返回每条线路的区域、执行时间和保真度。

    返回: List[{
      'index': int,
      'region': List[int] | None,
      'est_seconds': float | None,
      'est_fidelity': float | None,
      'score': float | None,            # 平均性能评分（用于参考）
      'gate_avg': float | None,
      'connectivity': float | None,
      'status': str,                    # ok / no_region / error
      'message': str | None,
    }]
    """
    backend = _backend_from_name(chip_name)
    alloc = allocate_regions_for_circuits(
        circuits,
        backend,
        two_q_gate_name=two_q_gate_name,
        excluded_qubits=excluded_qubits,
        weight=weight,
        max_regions_each=max_regions_each,
    )
    # 简化返回字段
    out = []
    for item in alloc:
        out.append({
            'index': item.get('index'),
            'region': item.get('region'),
            'est_seconds': item.get('est_seconds'),
            'est_fidelity': item.get('est_fidelity'),
            'score': item.get('score'),
            'gate_avg': item.get('gate_avg'),
            'connectivity': item.get('connectivity'),
            'status': item.get('status'),
            'message': item.get('message'),
        })
    return out


def estimate_best_fidelity_and_logical_stats(circuit, chip_name: str, two_q_gate_name: str = 'cx', max_regions: int = 200, excluded_qubits=None):
    """
    2) 给定单条量子线路与芯片名称，快速估计其在该芯片上的最高保真度与对应执行时间，
       并返回所有逻辑比特的原始cx数目、swap数目及其比值（按逻辑比特统计）。

    返回: {
      'best_region': List[int],
      'best_fidelity': float,
      'best_seconds': float,
      'logical_stats': { logical_idx: { 'swaps': int, 'orig_cx': int, 'ratio': float|None } }
    } 或 None（若无可行区域）
    """
    backend = _backend_from_name(chip_name)

    # 候选区域（连通、规模匹配），限制数量以提升速度
    # candidates = find_feasible_regions_for_circuit(
    #     circuit,
    #     backend,
    #     two_q_gate_name=two_q_gate_name,
    #     max_regions=max_regions,
    #     excluded_qubits=excluded_qubits,
    # )
    candidates = find_best_region_by_avg_performance(circuit, backend, two_q_gate_name=two_q_gate_name, max_regions=max_regions, excluded_qubits=excluded_qubits)

    if not candidates:
        return None

    best = None
    mapped = transpile_to_region(circuit, backend, candidates['region'])
    secs, fid = estimate_time_and_fidelity(mapped, backend)
    item = {
        'region': candidates['region'],
        'mapped': mapped,
        'seconds': secs if secs is not None else float('inf'),
        'fidelity': fid if fid is not None else 0.0,
    }
    if best is None or item['fidelity'] > best['fidelity']:
        best = item

    if best is None:
        return None

    # 统计逻辑比特的原始cx / swap / 比值（需初始布局）
    init_layout = list(best['region'])[: circuit.num_qubits]
    logical_stats = compute_swap_cnot_counts_per_logical(circuit, best['mapped'], init_layout)

    return {
        'best_region': best['region'],
        'best_fidelity': best['fidelity'],
        'best_seconds': best['seconds'],
        'logical_stats': logical_stats,
    }


def estimate_partition_budgets(
    qc,
    device_qubit_max: int,
    B0: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 1.0,
    gammas: tuple = (0.5, 1.0, 2.0),
    required_fidelity: float | None = None,
    ratio_req_over_est: float | None = 1.1,
):
    """
    基于给定材料估计单条线路的三档分割预算（仅计采样聚合开销）。

    定义：
      - 宽度占比 r_Q(i) = |Q_i| / Q_max^{dev}
      - 双比特门密度 d_{2q}(i) = |G_{2q,i}| / |Q_i|
      - 组合大小因子 S_i = r_Q(i)^alpha * d_{2q}(i)^beta
      - 集群噪声近似 \bar{F}_{cluster}(i) ≈ (1-ε_dec(t))^{|Q_i|} (1-ε_1q)^{|G_{1q,i}|} (1-ε_2q)^{|G_{2q,i}|} (1-ε_ro)^{|M_i|}
        其中：
          ε_1q = 1 - ONE_Q_GATE_CORR, ε_2q = 1 - TWO_Q_GATE_CORR, ε_ro = 1 - MEAS_CORR；
          退相干误差 ε_dec(t) = 1 - exp(-t / DECO_T)；t = depth * AVG_GATE_TIME。
      - 需求权重 H_i = (ratio)^eta，其中 ratio = (需求/估计) 的比值；若未显式给定 ratio_req_over_est，则回退为 (F_min^i / \bar{F}_{cluster}(i))。
      - 基线预算 B_i^{base} = B0 * S_i * H_i
      - 三档预算 {B^{(L)}, B^{(M)}, B^{(H)}} = {Γ_L, Γ_M, Γ_H} * B_i^{base}

    入参：
      - qc: 线路（Qiskit QuantumCircuit 或兼容对象）
      - device_qubit_max: 设备可用最大比特数 Q_max^{dev}
      - B0, alpha, beta, eta: 标定常数
      - gammas: (Γ_L, Γ_M, Γ_H)
      - required_fidelity: 若提供，直接作为 F_min^i；否则调用 compute_fidelity_requirement 估计（仅用于回退比值）
      - ratio_req_over_est: 外部给定的"需求/估计"比值，默认1.1；若为None，则回退为 (F_min^i / \bar{F}_{cluster}(i))。

    返回：dict
      {
        'width': |Q_i|,
        'twoq_gates': |G_{2q,i}|,
        'oneq_gates': |G_{1q,i}|,
        'meas_qubits': |M_i|,
        'r_Q': float,
        'd_2q': float,
        'S': float,
        'F_cluster': float,            # 含时间相关退相干项
        'F_cluster_factors': {         # 各因子便于检查
            'F_deco': float,           # (1-ε_dec(t))^{|Q|}
            'F_1q': float,
            'F_2q': float,
            'F_ro': float,
            't_seconds': float,
        },
        'F_min': float,
        'ratio_used': float,           # 用于计算H的(需求/估计)比值
        'H': float,
        'B_base': float,
        'budgets_float': {'L': float, 'M': float, 'H': float},
        'budgets_int':   {'L': int,   'M': int,   'H': int  },  # ceil 为整数预算
      }
    """
    # 线路规模
    try:
        width = int(getattr(qc, 'num_qubits', 0))
    except Exception:
        width = 0
    if width <= 0 or device_qubit_max <= 0:
        return {
            'width': width,
            'twoq_gates': 0,
            'oneq_gates': 0,
            'meas_qubits': 0,
            'r_Q': 0.0,
            'd_2q': 0.0,
            'S': 0.0,
            'F_cluster': 0.0,
            'F_cluster_factors': {'F_deco': 0.0, 'F_1q': 0.0, 'F_2q': 0.0, 'F_ro': 0.0, 't_seconds': 0.0},
            'F_min': 0.0,
            'ratio_used': 0.0,
            'H': 0.0,
            'B_base': 0.0,
            'budgets_float': {'L': 0.0, 'M': 0.0, 'H': 0.0},
            'budgets_int':   {'L': 0,   'M': 0,   'H': 0  },
        }

    # 计数信息
    num_twoq, num_oneq, _ignored_meas, _depth = _extract_circuit_stats(qc)
    # 测量比特数 = 线路比特数
    meas_qubits = width

    # 尺寸与密度
    r_Q = float(width) / float(device_qubit_max)
    d_2q = (float(num_twoq) / float(width)) if width > 0 else 0.0
    # 组合大小因子
    S = (r_Q ** float(alpha)) * (d_2q ** float(beta)) if (r_Q > 0 and d_2q > 0) else 0.0

    # 时间：t = depth * AVG_GATE_TIME
    try:
        depth = int(getattr(qc, 'depth')() if callable(getattr(qc, 'depth', None)) else getattr(qc, 'depth', 0))
    except Exception:
        depth = 0
    t_seconds = float(depth) * float(AVG_GATE_TIME)

    # 集群噪声近似（用宏平均误差 + 时间相关退相干）
    eps_1q = max(0.0, 1.0 - float(ONE_Q_GATE_CORR))
    eps_2q = max(0.0, 1.0 - float(TWO_Q_GATE_CORR))
    eps_ro = max(0.0, 1.0 - float(MEAS_CORR))
    # 退相干平均误差（基于宏DECO_T）：ε_dec(t) = 1 - exp(-t/DECO_T)
    if float(DECO_T) > 0.0 and t_seconds > 0.0:
        eps_dec = 1.0 - math.exp(-t_seconds / float(DECO_T))
        F_deco = (1.0 - eps_dec) ** int(width)  # 等价于 exp(-width * t / T)
    else:
        eps_dec = 0.0
        F_deco = 1.0

    F_1q = (1.0 - eps_1q) ** int(num_oneq)
    F_2q = (1.0 - eps_2q) ** int(num_twoq)
    F_ro = (1.0 - eps_ro) ** int(meas_qubits)
    F_cluster = F_deco * F_1q * F_2q * F_ro

    # 需求保真度 F_min（仅用于回退比值）
    if required_fidelity is None:
        try:
            req_stats = compute_fidelity_requirement(qc)
            F_min = float(req_stats.get('fidelity_requirement', 0.0))
        except Exception:
            F_min = 0.0
    else:
        F_min = float(required_fidelity)

    # 需求/估计保真度比（外部给定优先，None时回退为F_min/F_cluster）
    if ratio_req_over_est is not None:
        ratio_used = max(0.0, float(ratio_req_over_est))
    else:
        if F_cluster <= 0.0:
            ratio_used = float('inf') if F_min > 0 else 0.0
        else:
            ratio_used = max(0.0, F_min / F_cluster)
    H = ratio_used ** float(eta)

    # 基线预算与三档预算
    B_base = float(B0) * float(S) * float(H)
    gamma_L, gamma_M, gamma_H = gammas if isinstance(gammas, (list, tuple)) and len(gammas) == 3 else (0.5, 1.0, 2.0)
    B_L = gamma_L * B_base
    B_M = gamma_M * B_base
    B_H = gamma_H * B_base

    # 输出（同时提供向上取整的整数预算）
    def _ceil_pos(x: float) -> int:
        if not math.isfinite(x) or x <= 0:
            return 0
        return int(math.ceil(x))

    return {
        'width': int(width),
        'twoq_gates': int(num_twoq),
        'oneq_gates': int(num_oneq),
        'meas_qubits': int(meas_qubits),
        'r_Q': float(r_Q),
        'd_2q': float(d_2q),
        'S': float(S),
        'F_cluster': float(F_cluster),
        'F_cluster_factors': {
            'F_deco': float(F_deco),
            'F_1q': float(F_1q),
            'F_2q': float(F_2q),
            'F_ro': float(F_ro),
            't_seconds': float(t_seconds),
        },
        'F_min': float(F_min),
        'ratio_used': float(ratio_used),
        'H': float(H),
        'B_base': float(B_base),
        'budgets_float': {'L': float(B_L), 'M': float(B_M), 'H': float(B_H)},
        'budgets_int':   {'L': _ceil_pos(B_L), 'M': _ceil_pos(B_M), 'H': _ceil_pos(B_H)},
    }

if __name__ == "__main__":

    base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    # folder = base_path + "pra_benchmark/small_scale"  # 可换 qaoa/qft/rca/vqe 等
    folder = "/home/normaluser/fzchen/HybdridSchedulingwithKnitting/testqasm/"
    device_qubit_max = 65
    budget = 10
    chip_name = "FakeManhattanV2"
    limit = 5  # 只测前5个文件

    qasm_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".qasm")][:limit]

    for fname in qasm_files:
        fpath = os.path.join(folder, fname)
        print(f"\n=== 文件: {fname} | 预算={budget} | device_qubit_max={device_qubit_max} | 芯片={chip_name} ===")
        
        try:
            circuit = read_circuit(fpath)
            
            budgets_info = estimate_partition_budgets(circuit, device_qubit_max)
            print(budgets_info['budgets_int'])
            # res = compute_subcircuits_with_budget(
            #     circuit=circuit,
            #     budget=budget,
            #     device_qubit_max=device_qubit_max,
            #     chip_name=chip_name,
            #     two_q_gate_name='cx',
            #     max_regions=200,
            #     excluded_qubits=None
            # )
            # print(f"status: {res['status']}, message: {res['message']}")
            # if res['status'] == 'ok':
            #     print(f"used_budget: {res['used_budget']}, cuts: {res['cuts']}, parts: {res['parts']}")
            #     print(f"post_swap_applied: {res['post_swap_applied']}")
            #     print(f"生成子线路数: {len(res['subcircuits'])}")
            #     for i, sc in enumerate(res['subcircuits']):
            #         print(f"  子线路 {i}: qubits={sc.num_qubits}, depth={sc.depth()}")
            #     # 新增：打印每个子线路与其他子线路之间的割数量(权重和)
            #     per_counts = res.get('per_subcircuit_cut_counts', {})
            #     if per_counts:
            #         print("各子线路与其他子线路之间的割数量(权重和):")
            #         for g in sorted(per_counts.keys()):
            #             print(f"  组 {g}: {per_counts[g]}")
            # else:
            #     print("无法执行，需增加预算")
        except Exception as e:
            print(f"[跳过] {fname}: {e}")



    # macros = compute_ibm_fake_macro_averages()
    # print(macros)
    # 可选：将全局宏覆盖为建议均值
    # if macros['TWO_Q_GATE_CORR'] is not None: TWO_Q_GATE_CORR = macros['TWO_Q_GATE_CORR']
    # if macros['ONE_Q_GATE_CORR'] is not None: ONE_Q_GATE_CORR = macros['ONE_Q_GATE_CORR']
    # if macros['MEAS_CORR'] is not None: MEAS_CORR = macros['MEAS_CORR']
    # if macros['DECO_T'] is not None: DECO_T = macros['DECO_T']
    # if macros['AVG_GATE_TIME'] is not None: AVG_GATE_TIME = macros['AVG_GATE_TIME']
    base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    test_circuit_path = os.path.join(base_path, "pra_benchmark", "small_scale", "cm82a_208.qasm")
    # circuit = QuantumCircuit.from_qasm_file(test_circuit_path)
    circuit = read_circuit(test_circuit_path)
    # coupling_map_all = []
    # num_qubits = 100
    # # for i in range(num_qubits):
    # #     for j in range(num_qubits):
    # #         if i != j:
    # #             coupling_map_all.append([i, j])

    # for i in range(num_qubits - 1) :
    #     coupling_map_all.append([i, i+1])

    # backend = GenericBackendV2(basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],
    #                             num_qubits=num_qubits, coupling_map=coupling_map_all)
    backend2 = FakeMelbourneV2()
    # circuit_transpile = transpile(circuit, basis_gates=["x", "y", "z", "h", "s", "t", "cx", "rx", "rz", "ry", "tdg"],layout_method='sabre', routing_method='sabre', backend=backend2, seed_transpiler=SEED_TRANSPILE)

    # # print(circuit_transpile.layout)
    # print(circuit_transpile.depth())
    # # print(circuit_transpile.)

    # # 估计时间与保真度
    # est_seconds, est_fidelity = estimate_time_and_fidelity(circuit_transpile, backend2)
    # print(f"Estimated exec time (s): {est_seconds}")
    # print(f"Estimated overall fidelity: {est_fidelity}")

    # # 选择芯片上的一片区域（物理比特编号）
    # # region = [3, 4, 7, 8, 11, 12, 13, 14]  # 举例

    # # # 将原始/转译电路映射到该区域
    # # mapped_circ = transpile_to_region(circuit, backend2, region)

    # # # 继续评估时间与保真度
    # # secs, fid = estimate_time_and_fidelity(mapped_circ, backend2)
    # # print(secs, fid)
    # # 示例：基于FakeMelbourneV2在芯片上寻找与电路规模匹配的连通区域
    # try:
    #     excluded = {0, 1, 2}  # 不使用的物理比特
    #     regions = find_feasible_regions_for_circuit(circuit, backend2, excluded_qubits=excluded, max_regions=5)
    #     print(regions)
    # except Exception as e:
    #     print("Region search failed:", e)


    # # # 将原始/转译电路映射到该区域
    # mapped_circ = transpile_to_region(circuit, backend2, regions[2])

    # # # 继续评估时间与保真度
    # secs, fid = estimate_time_and_fidelity(mapped_circ, backend2)
    # print(secs, fid)


    # 示例：搜索平均性能最优的区域 执行
    try:
        best_region = find_best_region_by_avg_performance(circuit, backend2, two_q_gate_name='cx', excluded_qubits=None, weight=0.5, max_regions=200)
        print("Best region by avg performance:", best_region)
    except Exception as e:
        print("Best region search failed:", e)


    # # # 将原始/转译电路映射到该区域
    # # print
    mapped_circ = transpile_to_region(circuit, backend2, best_region['region'])

    # # # 继续评估时间与保真度
    # secs, fid = estimate_time_and_fidelity(mapped_circ, backend2)
    # print(secs, fid)


    # # 测试多个线路一起执行
    # backend3 = FakeHanoiV2()
    # # 示例（如需测试，取消注释）：
    # circuits_list = [circuit]*2  # 多条线路按优先级排列
    # alloc = allocate_regions_for_circuits(
    #     circuits_list, backend3,
    #     two_q_gate_name='cx',
    #     excluded_qubits=None,  # 可传入初始不可用比特
    #     weight=0.6,            # 保真度 vs 连通性 权重
    #     max_regions_each=200   # 每条线路候选上限
    # )
    # for item in alloc:
    #     print(item['index'], item['status'], item['region'], item['score'], item['est_seconds'], item['est_fidelity'])



    # # 测试对cnot数目和swap数目的计算
    # mapped = transpile_to_region(circuit, backend2, best_region['region'])
    # initial_layout = list(best_region['region'])[: circuit.num_qubits]  # 逻辑→物理初始映射
    # stats_logical = compute_swap_cnot_counts_per_logical(circuit, mapped, initial_layout)
    # # 找影响最大的逻辑比特（比值最大）
    # # print_circuit_gates(mapped)
    # most_logical = max(
    #     (l for l, v in stats_logical.items() if v['ratio'] is not None),
    #     key=lambda l: stats_logical[l]['ratio'],
    #     default=None
    # )
    # print(stats_logical, most_logical)

    # print(assign_regions_and_estimates([circuit]*2, 'FakeHanoiV2', weight=0.5, excluded_qubits=None, two_q_gate_name='cx', max_regions_each=200))

    # print(estimate_best_fidelity_and_logical_stats(circuit, 'FakeHanoiV2', two_q_gate_name='cx', max_regions=200, excluded_qubits=None))


    # stats = compute_fidelity_requirement(mapped_circ)
    # print("two_qubit_gates:", stats['two_qubit_gates'])
    # print("one_qubit_gates:", stats['one_qubit_gates'])
    # print("measured_qubits:", stats['measured_qubits'])
    # print("depth:", stats['depth'])
    # print("exec_time_seconds:", stats['exec_time_seconds'])
    # print("deco_factor:", stats['deco_factor'])
    # print("fidelity_requirement:", stats['fidelity_requirement'])
    # print("macros:", stats['macros'])

    # # 路径与后端
    # base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    # qasm_path = base_path + "pra_benchmark/small_scale/cm82a_208.qasm"
    # backend = FakeMelbourneV2()
    # qc = read_circuit(qasm_path)

    # # 找一个最佳区域并映射，用其执行时间作为预算函数的 t
    # best = find_best_region_by_avg_performance(qc, backend, two_q_gate_name='cx', excluded_qubits=None, weight=0.5, max_regions=200)
    # region = best['region']
    # mapped = transpile_to_region(qc, backend, region)
    # t_seconds, _ = estimate_time_and_fidelity(mapped, backend)

    # # 设备规模（示例：Melbourne 15）
    # device_qubit_max = backend.num_qubits

    # # 估计三档预算
    # res = estimate_partition_budgets(
    #     qc,
    #     device_qubit_max=device_qubit_max,
    #     B0=1.0, alpha=1.0, beta=1.0, eta=1.0,
    #     gammas=(0.5, 1.0, 2.0)
    # )

    # print("budgets_float:", res['budgets_float'])
    # print("budgets_int:", res['budgets_int'])
    # print("F_cluster:", res['F_cluster'])
    # print("F_cluster_factors:", res['F_cluster_factors'])
    # print("S, r_Q, d_2q:", res['S'], res['r_Q'], res['d_2q'])
    # print("F_min, H, B_base:", res['F_min'], res['H'], res['B_base'])
    