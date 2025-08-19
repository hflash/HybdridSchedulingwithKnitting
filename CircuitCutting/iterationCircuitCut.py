import numpy as np
import pymetis
from qiskit import QuantumCircuit
import networkx as nx
import matplotlib.pyplot as plt
import os
import math
from qiskit.circuit import QuantumRegister
# 导入性能评估相关函数
try:
    from performanceAccessing import (
        estimate_time_and_fidelity, 
        _backend_from_name,
        transpile_to_region,
        compute_swap_cnot_counts_per_logical,
        find_feasible_regions_for_circuit,
        find_best_region_by_avg_performance
    )
except ImportError:
    print("Warning: 无法导入performanceAccessing模块的函数，部分功能可能不可用")

def create_interaction_graph(circuit):
    """
    为量子线路创建通信图（相互作用图）
    
    Args:
        circuit (QuantumCircuit): 输入的量子线路
        
    Returns:
        G (networkx.Graph): 表示量子比特之间相互作用的无向图
    """
    num_qubits = circuit.num_qubits
    G = nx.Graph()
    
    # 添加所有量子比特作为节点
    for i in range(num_qubits):
        G.add_node(i)
    
    # 遍历所有门操作，添加边
    for instruction in circuit.data:
        qubits = instruction[1]
        # 如果门作用于多个量子比特，在这些量子比特之间添加边
        if len(qubits) > 1:
            for i in range(len(qubits)):
                for j in range(i+1, len(qubits)):
                    qubit1 = qubits[i]._index
                    qubit2 = qubits[j]._index
                    # 添加边或增加边的权重
                    if G.has_edge(qubit1, qubit2):
                        G[qubit1][qubit2]['weight'] += 1
                    else:
                        G.add_edge(qubit1, qubit2, weight=1)
    
    return G

def visualize_interaction_graph(G, title="量子线路通信图"):
    """
    可视化量子比特之间的通信图并保存为文件
    
    Args:
        G (networkx.Graph): 量子比特通信图
        title (str): 图的标题
    """
    plt.clf()  # 清除当前图形
    plt.figure(figsize=(12, 8))
    
    # 使用kamada_kawai_layout，这是一种更稳定的布局算法
    pos = nx.kamada_kawai_layout(G)
    
    # 获取边的权重
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights)
    
    # 归一化边的宽度，使其在视觉上更合适
    normalized_weights = [2.0 * w / max_weight for w in weights]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.7)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos,
                          width=normalized_weights,
                          edge_color='gray',
                          alpha=0.6)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos,
                          font_size=12,
                          font_weight='bold')
    
    # 添加边标签（显示权重）
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
                               edge_labels=edge_labels,
                               font_size=10)
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')  # 关闭坐标轴
    
    # 保存图像到文件
    output_file = "quantum_interaction_graph.png"
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形
    print(f"图形已保存到文件：{output_file}")

def weight4metis(cx_weight):
    xadj = [0]
    now_index = 0
    adjncy = []
    eweight = []
    for i in range(len(cx_weight)):
        for j in range(len(cx_weight)):
            # print(cx_weight[i][j])
            if cx_weight[i][j]:
                adjncy.append(j)
                eweight.append(cx_weight[i][j])
                now_index = now_index + 1
        xadj.append(now_index)
    return xadj, adjncy, eweight

def find_highest_degree_node(G, num_nodes=1):
    """
    找到图中度数最高的num_nodes个节点
    
    Args:
        G: networkx图
        num_nodes: 要找到的节点数量
        
    Returns:
        list of tuples: [(节点1, 度数1), (节点2, 度数2), ...]
    """
    degrees = dict(G.degree(weight='weight'))  # 使用带权重的度数
    # 按度数降序排序
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    # 返回前num_nodes个节点
    return sorted_nodes[:num_nodes]

def remove_highest_degree_node(G, num_nodes=1):
    """
    删除图中度数最高的num_nodes个节点
    
    Args:
        G: networkx图
        num_nodes: 要删除的节点数量
        
    Returns:
        tuple: (处理后的图, 被删除的节点列表)
    """
    G_copy = G.copy()
    highest_nodes = find_highest_degree_node(G_copy, num_nodes)
    removed_nodes = []
    
    for node, degree in highest_nodes:
        # 删除节点
        G_copy.remove_node(node)
        removed_nodes.append(node)
    
    return G_copy, removed_nodes

def metis_zmz(G, k, removed_nodes=None):
    """
    执行metis分割，并在结果中标记被删除的节点
    
    Args:
        G: networkx图
        k: 分割数
        removed_nodes: 被删除的节点列表
    """
    num_parts = k
    xadj, adjncy, eweight = weight4metis(nx.adjacency_matrix(G).todense().tolist())
    cuts, membership = pymetis.part_graph(nparts=num_parts, adjncy=adjncy, xadj=xadj, eweights=eweight, recursive=True, contiguous=True)

    # 如果有被删除的节点，将其加入到membership中
    if removed_nodes:
        # 找到最大的节点编号
        max_node = max(max(G.nodes()) if G.nodes() else 0, max(removed_nodes))
        
        # 创建完整的membership列表
        full_membership = []
        current_node = 0
        
        for i in range(max_node + 1):
            if i in removed_nodes:
                full_membership.append(-1)  # 被删除的节点标记为-1
            else:
                if current_node < len(membership):
                    full_membership.append(membership[current_node])
                    current_node += 1
        
        membership = full_membership

    return cuts, membership

# 打印通信图
def print_interaction_graph(G):
    edges = G.edges(data=True)
    sorted_edges = sorted(edges, key=lambda x: (x[0], x[1]))
    print("\n量子比特之间的相互作用关系：")
    print("格式：(比特1, 比特2): 权重")
    print("-" * 30)
    for edge in sorted_edges:
        qubit1, qubit2, data = edge
        weight = data['weight']
        print(f"({qubit1}, {qubit2}): {weight}")

def print_circuit_info(circuit, circuit_name="量子线路"):
    """
    打印量子线路的详细信息
    
    Args:
        circuit: 量子线路
        circuit_name: 线路名称
    """
    print(f"\n{'-'*20} {circuit_name} 信息 {'-'*20}")
    print(f"量子比特数量: {circuit.num_qubits}")
    print(f"经典比特数量: {circuit.num_clbits}")
    print(f"线路深度: {circuit.depth()}")
    print(f"总门操作数量: {len(circuit.data)}")
    
    # 统计不同类型的门操作
    gate_counts = {}
    two_qubit_gates = 0
    for instruction in circuit.data:
        gate_name = instruction[0].name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        if len(instruction[1]) == 2:  # 双量子比特门
            two_qubit_gates += 1
    
    print("\n门操作统计:")
    print(f"双量子比特门数量: {two_qubit_gates}")
    print("各类型门数量:")
    for gate_name, count in sorted(gate_counts.items()):
        print(f"  - {gate_name}: {count}")
    
    # 打印每个量子比特参与的门操作数量
    qubit_gate_counts = [0] * circuit.num_qubits
    for instruction in circuit.data:
        for qubit in instruction[1]:
            qubit_gate_counts[qubit._index] += 1
    
    print("\n各量子比特参与的门操作数量:")
    for i, count in enumerate(qubit_gate_counts):
        print(f"  - 量子比特 {i}: {count} 个门操作")

    # 打印每个量子比特参与的CX门数量
    qubit_cx_counts = [0] * circuit.num_qubits
    for instruction in circuit.data:
        if instruction[0].name == 'cx':
            for qubit in instruction[1]:
                qubit_cx_counts[qubit._index] += 1
    print("\n各量子比特参与的CX门数量:")
    for i, count in enumerate(qubit_cx_counts):
        print(f"  - 量子比特 {i}: {count} 个CX")
    
    print("-" * (42 + len(circuit_name)))

def read_circuit(circuit_path):
    circuit = QuantumCircuit.from_qasm_file(circuit_path)
    return circuit


def circuit_iteration_cut(circuit, budget, qubit_number_max):
    interaction_graph = create_interaction_graph(circuit)
    qubit_number = len(interaction_graph.nodes())
    intial_cut_result = []
    budget_remain = budget
    if qubit_number - budget_remain <= qubit_number_max:
        print("need not cutting")
        if budget_remain > qubit_number:
            return 0, [-1] * qubit_number, budget_remain - qubit_number
        else:
            reduced_graph, removed_nodes = remove_highest_degree_node(interaction_graph, budget_remain)
            membership = [0] * qubit_number
            for i in removed_nodes:
                membership[i] = -1
            return 0, membership, 0
    i = 1
    while True:
        reduced_graph, removed_nodes = remove_highest_degree_node(interaction_graph, i)
        qubit_number = len(reduced_graph.nodes())
        parts_number = math.ceil(qubit_number / qubit_number_max)
        cuts, membership = metis_zmz(reduced_graph, parts_number, removed_nodes)
        # print(parts_number, cuts, membership, budget_remain)
        budget_remain = budget - cuts - i
        intial_cut_result = [cuts, membership]
        i += 1
        if budget_remain > 0:
            return intial_cut_result, budget_remain
        if 0 > budget_remain and i < budget:
            continue;
        else:
            return False
            


def extract_sub_circuit(circuit, membership):
    """
    根据membership从原始线路中提取子线路
    
    Args:
        circuit: 原始量子线路
        membership: 每个比特所属的分组（-1表示被删除的比特）
    
    Returns:
        list: 子线路列表
    """
    # 获取所有分组编号（不包括-1）
    groups = sorted(list(set(x for x in membership if x != -1)))
    
    # 为每个分组创建子线路
    subcircuits = []
    for group in groups:
        # 创建新的量子线路
        subcircuit = QuantumCircuit()
        
        # 添加该分组的量子比特
        group_qubits = []
        original_to_new_index = {}  # 原始索引到新索引的映射
        new_index = 0
        
        for i, m in enumerate(membership):
            if m == group:
                group_qubits.append(circuit.qubits[i])
                original_to_new_index[i] = new_index
                new_index += 1
        
        # 如果这个分组有量子比特，创建寄存器并添加到子线路
        if group_qubits:
            subcircuit.add_register(QuantumRegister(len(group_qubits), f'q_{group}'))
            
            # 遍历原始线路中的所有门操作
            for instruction in circuit.data:
                gate = instruction[0]
                qubits = instruction[1]
                
                # 检查这个门操作的所有量子比特是否都属于当前分组
                qubit_indices = [q._index for q in qubits]
                if all(membership[idx] == group for idx in qubit_indices):
                    # 将原始比特索引映射到新的子线路中的索引
                    new_qubits = [subcircuit.qubits[original_to_new_index[q._index]] for q in qubits]
                    # 添加门操作到子线路
                    subcircuit.append(gate, new_qubits)
        
        subcircuits.append(subcircuit)
    
    return subcircuits


def compute_cut_weight_for_membership(G, membership):
    """
    计算在给定membership（分组方案）下的割权重（跨分组的边权之和）。
    约定：membership[i]为-1的比特视为被删除，不参与割统计。
    """
    cut_weight = 0
    for u, v, data in G.edges(data=True):
        mu = membership[u] if u < len(membership) else None
        mv = membership[v] if v < len(membership) else None
        if mu is None or mv is None:
            continue
        if mu == -1 or mv == -1:
            continue
        if mu != mv:
            cut_weight += data.get('weight', 1)
    return cut_weight


def evaluate_swap_cut_change(G, membership, q1, q2):
    """
    评估交换两个比特q1与q2的分组标签后，割权重的变化。
    返回：{
        'q1': q1,
        'q2': q2,
        'orig_labels': (membership[q1], membership[q2]),
        'base_cut': base_cut,
        'swapped_cut': swapped_cut,
        'delta': swapped_cut - base_cut
    }
    """
    if q1 >= len(membership) or q2 >= len(membership):
        raise ValueError("q1或q2超出membership长度")
    base_cut = compute_cut_weight_for_membership(G, membership)
    new_membership = list(membership)
    new_membership[q1], new_membership[q2] = new_membership[q2], new_membership[q1]
    swapped_cut = compute_cut_weight_for_membership(G, new_membership)
    return {
        'q1': q1,
        'q2': q2,
        'orig_labels': (membership[q1], membership[q2]),
        'base_cut': base_cut,
        'swapped_cut': swapped_cut,
        'delta': swapped_cut - base_cut
    }


def find_high_impact_qubits(circuit, chip_name, membership, num_top=5):
    """
    从子线路中找到原始CX数目、SWAP数目最高的比特。
    
    Args:
        circuit: 原始量子线路
        chip_name: 芯片名称
        membership: 分组方案
        num_top: 返回前几个高影响比特
    
    Returns:
        dict: {
            'orig_cx_top': [(qubit, cx_count), ...],  # 按原始CX数目排序
            'swap_top': [(qubit, swap_count), ...],   # 按SWAP数目排序
            'combined_top': [(qubit, combined_score), ...]  # 按综合评分排序
        }
    """
    try:
        backend = _backend_from_name(chip_name)
        
        # 为每个分组找一个连通区域并统计
        groups = sorted(list(set(x for x in membership if x != -1)))
        all_stats = {}
        
        for group in groups:
            group_qubits = [i for i, m in enumerate(membership) if m == group]
            if len(group_qubits) == 0:
                continue
                
            # 简单映射到连续物理比特（实际应用中可能需要更复杂的映射）
            region = list(range(len(group_qubits)))
            
            try:
                mapped = transpile_to_region(circuit, backend, region)
                initial_layout = list(range(len(group_qubits)))
                stats = compute_swap_cnot_counts_per_logical(circuit, mapped, initial_layout)
                
                # 将逻辑比特映射回原始比特编号
                for local_idx, global_idx in enumerate(group_qubits):
                    if local_idx in stats:
                        all_stats[global_idx] = stats[local_idx]
            except Exception as e:
                print(f"Warning: 无法为分组{group}计算统计信息: {e}")
                continue
        
        # 按不同指标排序
        valid_qubits = [(q, s) for q, s in all_stats.items() if s['orig_cx'] > 0]
        
        orig_cx_top = sorted(valid_qubits, key=lambda x: x[1]['orig_cx'], reverse=True)[:num_top]
        swap_top = sorted(valid_qubits, key=lambda x: x[1]['swaps'], reverse=True)[:num_top]
        
        # 综合评分：原始CX + SWAP数
        combined_top = sorted(valid_qubits, 
                            key=lambda x: x[1]['orig_cx'] + x[1]['swaps'], 
                            reverse=True)[:num_top]
        
        return {
            'orig_cx_top': orig_cx_top,
            'swap_top': swap_top,
            'combined_top': combined_top,
            'all_stats': all_stats
        }
        
    except Exception as e:
        print(f"Error in find_high_impact_qubits: {e}")
        return None


def evaluate_qubit_swaps_impact(circuit, chip_name, G, membership, high_impact_qubits):
    """
    评估高影响比特两两交换后的割数目变化和子线路保真度变化。
    
    Args:
        circuit: 原始量子线路
        chip_name: 芯片名称
        G: 通信图
        membership: 分组方案
        high_impact_qubits: 高影响比特列表 [(qubit, stats), ...]
    
    Returns:
        list: 交换结果列表，每项包含割变化和保真度变化
    """
    results = []
    qubits = [q for q, _ in high_impact_qubits]
    
    try:
        backend = _backend_from_name(chip_name)
        
        # 计算原始子线路的保真度
        orig_subcircuits = extract_sub_circuit(circuit, membership)
        orig_fidelities = []
        
        for i, subcircuit in enumerate(orig_subcircuits):
            try:
                # 简单映射
                region = list(range(subcircuit.num_qubits))
                mapped = transpile_to_region(subcircuit, backend, region)
                _, fidelity = estimate_time_and_fidelity(mapped, backend)
                orig_fidelities.append(fidelity)
            except:
                orig_fidelities.append(None)
        
        # 评估所有两两交换
        for i in range(len(qubits)):
            for j in range(i+1, len(qubits)):
                q1, q2 = qubits[i], qubits[j]
                
                # 只考虑不同分组间的交换
                if membership[q1] == membership[q2]:
                    continue
                
                # 计算割变化
                cut_change = evaluate_swap_cut_change(G, membership, q1, q2)
                
                # 计算交换后的保真度变化
                new_membership = list(membership)
                new_membership[q1], new_membership[q2] = new_membership[q2], new_membership[q1]
                
                try:
                    new_subcircuits = extract_sub_circuit(circuit, new_membership)
                    new_fidelities = []
                    
                    for k, subcircuit in enumerate(new_subcircuits):
                        try:
                            region = list(range(subcircuit.num_qubits))
                            mapped = transpile_to_region(subcircuit, backend, region)
                            _, fidelity = estimate_time_and_fidelity(mapped, backend)
                            new_fidelities.append(fidelity)
                        except:
                            new_fidelities.append(None)
                    
                    # 计算保真度变化
                    fidelity_changes = []
                    for k in range(min(len(orig_fidelities), len(new_fidelities))):
                        if orig_fidelities[k] is not None and new_fidelities[k] is not None:
                            fidelity_changes.append(new_fidelities[k] - orig_fidelities[k])
                        else:
                            fidelity_changes.append(None)
                    
                    results.append({
                        'q1': q1,
                        'q2': q2,
                        'q1_group': membership[q1],
                        'q2_group': membership[q2],
                        'cut_change': cut_change,
                        'orig_fidelities': orig_fidelities,
                        'new_fidelities': new_fidelities,
                        'fidelity_changes': fidelity_changes,
                        'total_fidelity_change': sum(fc for fc in fidelity_changes if fc is not None)
                    })
                    
                except Exception as e:
                    print(f"Warning: 无法计算交换({q1},{q2})的保真度变化: {e}")
                    results.append({
                        'q1': q1,
                        'q2': q2,
                        'q1_group': membership[q1],
                        'q2_group': membership[q2],
                        'cut_change': cut_change,
                        'fidelity_error': str(e)
                    })
        
        return results
        
    except Exception as e:
        print(f"Error in evaluate_qubit_swaps_impact: {e}")
        return []


def max_ratio_qubit_per_subcircuit(circuit, membership, chip_name, two_q_gate_name='cx', max_regions=100, excluded_qubits=None):
    """
    给定分割后的membership与目标设备，返回每个子线路中 (swap数 / 原始cx数) 比值最高的逻辑比特。
    返回: List[{
        'group': int,
        'best_logical': int,       # 逻辑比特在原始电路中的索引
        'orig_cx': int,
        'swaps': int,
        'ratio': float | None,
        'region': List[int],       # 用于该子线路评估的设备区域
    }]
    """
    backend = _backend_from_name(chip_name)
    results = []

    groups = sorted(list(set(x for x in membership if x != -1)))
    for group in groups:
        # 取该子线路的原始逻辑比特集合
        group_qubits = [i for i, m in enumerate(membership) if m == group]
        if not group_qubits:
            continue

        # 构建该子线路对应的子电路（按membership已有逻辑）
        subcircuits = extract_sub_circuit(circuit, membership)
        # 这里subcircuits的顺序与groups顺序一致
        subcircuit = subcircuits[groups.index(group)]

        # 在设备上为该子线路找一个"平均性能"最优的区域（规模=子线路比特数）
        best = find_best_region_by_avg_performance(
            subcircuit, backend,
            two_q_gate_name=two_q_gate_name,
            excluded_qubits=excluded_qubits,
            max_regions=max_regions
        )
        if not best or 'region' not in best or not best['region']:
            results.append({
                'group': group,
                'best_logical': None,
                'orig_cx': 0,
                'swaps': 0,
                'ratio': None,
                'region': [],
            })
            continue
        region = best['region']

        # 将该子线路映射到所选区域
        mapped = transpile_to_region(subcircuit, backend, region)
        # 初始布局：子线路的逻辑索引 -> 物理区域顺序
        init_layout = list(region)[: subcircuit.num_qubits]
        # 统计子线路内部的逻辑比特负载
        stats = compute_swap_cnot_counts_per_logical(subcircuit, mapped, init_layout)

        # 选择比值最高的逻辑比特（注意orig_cx为0的返回None）
        best_logical = None
        best_ratio = -1
        best_swaps = 0
        best_cx = 0
        for local_idx, s in stats.items():
            cx = s.get('orig_cx', 0)
            sp = s.get('swaps', 0)
            ratio = (sp / cx) if cx > 0 else None
            if ratio is None:
                continue
            if ratio > best_ratio:
                best_ratio = ratio
                best_logical = local_idx
                best_swaps = sp
                best_cx = cx

        # 将子线路逻辑索引映射回原始电路逻辑索引
        if best_logical is not None:
            # group_qubits按原始索引排序，local_idx与extract_sub_circuit映射一致
            global_logical = group_qubits[best_logical]
            results.append({
                'group': group,
                'best_logical': global_logical,
                'orig_cx': best_cx,
                'swaps': best_swaps,
                'ratio': best_ratio,
                'region': region,
            })
        else:
            results.append({
                'group': group,
                'best_logical': None,
                'orig_cx': 0,
                'swaps': 0,
                'ratio': None,
                'region': region,
            })

    return results


def swap_top_ratio_qubits_and_evaluate(circuit, membership, chip_name, two_q_gate_name='cx', max_regions=200, excluded_qubits=None):
    """
    在两个子线路中分别找到 swap数/原始cx数 比值最高的逻辑比特，交换它们，
    返回交换后的割数目变化以及两个子线路的保真度变化、子线路深度与执行时间变化。

    返回:
    {
      'q1': int,
      'q2': int,
      'q1_group': int,
      'q2_group': int,
      'cut_change': {...},
      'orig_fidelities': {group_id: fidelity or None},
      'new_fidelities': {group_id: fidelity or None},
      'fidelity_changes': {group_id: delta or None},
      'orig_depths': {group_id: int},
      'new_depths': {group_id: int},
      'depth_changes': {group_id: int},
      'orig_seconds': {group_id: float|None},
      'new_seconds': {group_id: float|None},
      'seconds_changes': {group_id: float|None},
    }
    """
    backend = _backend_from_name(chip_name)

    groups = sorted(list(set(x for x in membership if x != -1)))
    if len(groups) < 2:
        raise ValueError("需要至少两个子线路进行交换评估")
    # 仅取前两个子线路
    g1, g2 = groups[0], groups[1]

    subcircuits = extract_sub_circuit(circuit, membership)
    # subcircuits顺序与groups一致
    sc1 = subcircuits[groups.index(g1)]
    sc2 = subcircuits[groups.index(g2)]

    # 为每个子线路选择平均性能最优区域，并计算逻辑比特swap/cx比值
    def _best_ratio_logical(subcircuit, group_qubits):
        best = find_best_region_by_avg_performance(
            subcircuit, backend,
            two_q_gate_name=two_q_gate_name,
            excluded_qubits=excluded_qubits,
            max_regions=max_regions
        )
        if not best or 'region' not in best or not best['region']:
            return None, None, None, None, []
        region = best['region']
        mapped = transpile_to_region(subcircuit, backend, region)
        init_layout = list(region)[: subcircuit.num_qubits]
        stats = compute_swap_cnot_counts_per_logical(subcircuit, mapped, init_layout)
        best_local, best_ratio, best_swaps, best_cx = None, -1, 0, 0
        for lidx, s in stats.items():
            cx = s.get('orig_cx', 0)
            sp = s.get('swaps', 0)
            denom = cx if cx > 0 else 1
            ratio = sp / denom
            if ratio > best_ratio:
                best_ratio, best_local, best_swaps, best_cx = ratio, lidx, sp, cx
        if best_local is None:
            return None, None, None, None, region
        return group_qubits[best_local], best_ratio, best_swaps, best_cx, region

    g1_qubits = [i for i, m in enumerate(membership) if m == g1]
    g2_qubits = [i for i, m in enumerate(membership) if m == g2]

    q1, _, _, _, region1 = _best_ratio_logical(sc1, g1_qubits)
    q2, _, _, _, region2 = _best_ratio_logical(sc2, g2_qubits)

    if q1 is None or q2 is None:
        raise RuntimeError("无法找到比值最高的逻辑比特（可能orig_cx均为0）")

    # 交换q1与q2所在分组，计算割变化
    G = create_interaction_graph(circuit)
    cut_change = evaluate_swap_cut_change(G, membership, q1, q2)

    # 原始两个子线路的保真度与时间、深度
    def _subcircuit_metrics(subcircuit):
        best = find_best_region_by_avg_performance(
            subcircuit, backend,
            two_q_gate_name=two_q_gate_name,
            excluded_qubits=excluded_qubits,
            max_regions=max_regions
        )
        if not best or 'region' not in best or not best['region']:
            return None, None
        region = best['region']
        mapped = transpile_to_region(subcircuit, backend, region)
        seconds, fid = estimate_time_and_fidelity(mapped, backend)
        return seconds, fid

    orig_depths = {g1: sc1.depth(), g2: sc2.depth()}
    orig_seconds = {}
    orig_fids = {}
    s, f = _subcircuit_metrics(sc1)
    orig_seconds[g1], orig_fids[g1] = s, f
    s, f = _subcircuit_metrics(sc2)
    orig_seconds[g2], orig_fids[g2] = s, f

    # 交换后的membership与两个子线路保真度/时间/深度
    new_membership = list(membership)
    new_membership[q1], new_membership[q2] = new_membership[q2], new_membership[q1]
    new_subcircuits = extract_sub_circuit(circuit, new_membership)
    new_groups = sorted(list(set(x for x in new_membership if x != -1)))
    sc1_new = new_subcircuits[new_groups.index(g1)]
    sc2_new = new_subcircuits[new_groups.index(g2)]

    new_depths = {g1: sc1_new.depth(), g2: sc2_new.depth()}
    new_seconds = {}
    new_fids = {}
    s, f = _subcircuit_metrics(sc1_new)
    new_seconds[g1], new_fids[g1] = s, f
    s, f = _subcircuit_metrics(sc2_new)
    new_seconds[g2], new_fids[g2] = s, f

    depth_changes = {g1: new_depths[g1] - orig_depths[g1], g2: new_depths[g2] - orig_depths[g2]}
    seconds_changes = {
        g1: (new_seconds[g1] - orig_seconds[g1]) if (new_seconds[g1] is not None and orig_seconds[g1] is not None) else None,
        g2: (new_seconds[g2] - orig_seconds[g2]) if (new_seconds[g2] is not None and orig_seconds[g2] is not None) else None,
    }

    fid_changes = {}
    for g in (g1, g2):
        if orig_fids[g] is not None and new_fids[g] is not None:
            fid_changes[g] = new_fids[g] - orig_fids[g]
        else:
            fid_changes[g] = None

    return {
        'q1': q1,
        'q2': q2,
        'q1_group': g1,
        'q2_group': g2,
        'cut_change': cut_change,
        'orig_fidelities': orig_fids,
        'new_fidelities': new_fids,
        'fidelity_changes': fid_changes,
        'orig_depths': orig_depths,
        'new_depths': new_depths,
        'depth_changes': depth_changes,
        'orig_seconds': orig_seconds,
        'new_seconds': new_seconds,
        'seconds_changes': seconds_changes,
    }


def swap_top_ratio_qubits_all_pairs(circuit, membership, chip_name, two_q_gate_name='cx', max_regions=200, excluded_qubits=None):
    """
    对于两个及以上的子线路：为每个子线路找到其 swap/cx 比值最高的逻辑比特；
    忽略比值==0的子线路；对剩余子线路做两两组合，交换对应两个逻辑比特，
    返回每对交换的割数目变化与两个子线路的保真度变化。

    返回：List[{
      'group_pair': (g1, g2),
      'q1': int, 'q2': int,
      'q1_ratio': float, 'q2_ratio': float,
      'cut_change': {...},
      'orig_fidelities': {g1: fid1 or None, g2: fid2 or None},
      'new_fidelities':  {g1: fid1' or None, g2: fid2' or None},
      'fidelity_changes': {g1: Δfid1 or None, g2: Δfid2 or None}
    }]
    """
    backend = _backend_from_name(chip_name)
    groups = sorted(list(set(x for x in membership if x != -1)))
    if len(groups) < 2:
        return []

    # 提前构建原始子线路、其最佳区域保真度、以及每组的top逻辑比特
    subcircuits = extract_sub_circuit(circuit, membership)
    group_to_sub = {g: subcircuits[groups.index(g)] for g in groups}

    def _best_region_metrics(subcircuit):
        best = find_best_region_by_avg_performance(
            subcircuit, backend,
            two_q_gate_name=two_q_gate_name,
            excluded_qubits=excluded_qubits,
            max_regions=max_regions
        )
        if not best or 'region' not in best or not best['region']:
            return None, None, []
        region = best['region']
        mapped = transpile_to_region(subcircuit, backend, region)
        seconds, fid = estimate_time_and_fidelity(mapped, backend)
        return seconds, fid, region

    # 计算每组top逻辑比特（global index）及其ratio
    group_top = {}
    for g in groups:
        sc = group_to_sub[g]
        # 选最优区域
        sec_tmp, fid_tmp, region = _best_region_metrics(sc)
        if region == []:
            continue
        mapped = transpile_to_region(sc, backend, region)
        init_layout = list(region)[: sc.num_qubits]
        stats = compute_swap_cnot_counts_per_logical(sc, mapped, init_layout)
        # 找max ratio（ratio定义已在统计函数中处理了cx=0分母）
        best_local, best_ratio = None, -1.0
        best_swaps, best_cx = 0, 0
        for lidx, s in stats.items():
            r = s.get('ratio', 0.0)
            if r > best_ratio:
                best_ratio = r
                best_local = lidx
                best_swaps = s.get('swaps', 0)
                best_cx = s.get('orig_cx', 0)
        if best_local is None or best_ratio == 0.0:
            # 忽略ratio为0的子线路
            continue
        # 映射回原始逻辑比特索引
        group_qubits = [i for i, m in enumerate(membership) if m == g]
        global_idx = group_qubits[best_local]
        group_top[g] = {
            'logical': global_idx,
            'ratio': best_ratio,
            'swaps': best_swaps,
            'orig_cx': best_cx,
        }

    valid_groups = sorted(group_top.keys())
    if len(valid_groups) < 2:
        return []

    # 预先计算原始每组保真度/时间（最优区域）与深度
    base_fids = {}
    base_secs = {}
    base_depths = {}
    for g in valid_groups:
        sc = group_to_sub[g]
        base_depths[g] = sc.depth()
        sec, fid, _ = _best_region_metrics(sc)
        base_secs[g] = sec
        base_fids[g] = fid

    G = create_interaction_graph(circuit)
    results = []

    for i in range(len(valid_groups)):
        for j in range(i+1, len(valid_groups)):
            g1, g2 = valid_groups[i], valid_groups[j]
            q1 = group_top[g1]['logical']
            q2 = group_top[g2]['logical']
            r1 = group_top[g1]['ratio']
            r2 = group_top[g2]['ratio']

            # 割变化
            cut_change = evaluate_swap_cut_change(G, membership, q1, q2)

            # 交换后新membership
            new_membership = list(membership)
            new_membership[q1], new_membership[q2] = new_membership[q2], new_membership[q1]
            new_subcircs = extract_sub_circuit(circuit, new_membership)
            new_groups = sorted(list(set(x for x in new_membership if x != -1)))
            sc1_new = new_subcircs[new_groups.index(g1)]
            sc2_new = new_subcircs[new_groups.index(g2)]

            # 计算新保真度/时间（最优区域）与深度
            new_depths = {g1: sc1_new.depth(), g2: sc2_new.depth()}
            sec1_new, fid1_new, _ = _best_region_metrics(sc1_new)
            sec2_new, fid2_new, _ = _best_region_metrics(sc2_new)

            fid1_old = base_fids.get(g1)
            fid2_old = base_fids.get(g2)
            sec1_old = base_secs.get(g1)
            sec2_old = base_secs.get(g2)

            delta1 = (fid1_new - fid1_old) if (fid1_new is not None and fid1_old is not None) else None
            delta2 = (fid2_new - fid2_old) if (fid2_new is not None and fid2_old is not None) else None

            dsec1 = (sec1_new - sec1_old) if (sec1_new is not None and sec1_old is not None) else None
            dsec2 = (sec2_new - sec2_old) if (sec2_new is not None and sec2_old is not None) else None

            results.append({
                'group_pair': (g1, g2),
                'q1': q1, 'q2': q2,
                'q1_ratio': r1, 'q2_ratio': r2,
                'cut_change': cut_change,
                'orig_fidelities': {g1: fid1_old, g2: fid2_old},
                'new_fidelities': {g1: fid1_new, g2: fid2_new},
                'fidelity_changes': {g1: delta1, g2: delta2},
                'orig_depths': {g1: base_depths.get(g1), g2: base_depths.get(g2)},
                'new_depths': new_depths,
                'depth_changes': {g1: (new_depths[g1] - base_depths.get(g1)) if base_depths.get(g1) is not None else None,
                                  g2: (new_depths[g2] - base_depths.get(g2)) if base_depths.get(g2) is not None else None},
                'orig_seconds': {g1: sec1_old, g2: sec2_old},
                'new_seconds': {g1: sec1_new, g2: sec2_new},
                'seconds_changes': {g1: dsec1, g2: dsec2},
            })

    return results


def _fmt_float(x, nan='-'):
    try:
        if x is None:
            return nan
        return f"{x:.6g}"
    except Exception:
        return nan


def print_swap_pairs_report(pairs_eval, top_k=None):
    """
    按"割更小更好、保真度更高更好"的原则，对两两交换结果进行排序并清晰打印。
    - 排序键：先按割变化delta升序（负数更优），再按两子线路保真度总变化降序，其次按两子线路执行时间总变化升序。
    - 每条目打印：分组对、参与交换的逻辑比特、各自ratio、割变化、两子线路保真度/深度/时间前后与变化、结论。
    """
    if not pairs_eval:
        print("无可评估的交换结果")
        return

    def total_df(item):
        fc = item.get('fidelity_changes', {})
        vals = [v for v in fc.values() if v is not None]
        return sum(vals) if vals else 0.0

    def total_dt(item):
        sc = item.get('seconds_changes', {})
        vals = [v for v in sc.values() if v is not None]
        return sum(vals) if vals else 0.0

    def sort_key(item):
        cut_delta = item.get('cut_change', {}).get('delta', 0)
        return (cut_delta, -total_df(item), total_dt(item))

    items = sorted(pairs_eval, key=sort_key)
    if top_k is not None:
        items = items[:top_k]

    print("\n==== 子线路交换评估报告 ====")
    for idx, it in enumerate(items, 1):
        g1, g2 = it.get('group_pair', (None, None))
        q1, q2 = it.get('q1'), it.get('q2')
        r1, r2 = it.get('q1_ratio'), it.get('q2_ratio')
        cut = it.get('cut_change', {})
        dcut = cut.get('delta')
        base_cut = cut.get('base_cut')
        swap_cut = cut.get('swapped_cut')
        of = it.get('orig_fidelities', {})
        nf = it.get('new_fidelities', {})
        fc = it.get('fidelity_changes', {})
        od = it.get('orig_depths', {})
        nd = it.get('new_depths', {})
        dd = it.get('depth_changes', {})
        os = it.get('orig_seconds', {})
        ns = it.get('new_seconds', {})
        ds = it.get('seconds_changes', {})
        df1 = fc.get(g1)
        df2 = fc.get(g2)
        sum_df = total_df(it)
        dt_sum = total_dt(it)
        # 结论规则：优先割下降与保真度上升；时间增加过大可视作减分（此处仅展示）
        verdict = '劣'
        if (dcut is not None and dcut < 0) and (sum_df is not None and sum_df > 0):
            verdict = '优'
        elif (dcut is not None and dcut <= 0) and (sum_df is not None and sum_df >= 0):
            verdict = '可'

        print(f"[{idx}] 组对({g1},{g2}) | 逻辑比特 {q1}↔{q2} | ratio { _fmt_float(r1) } / { _fmt_float(r2) }")
        print(f"     割: { _fmt_float(base_cut) } -> { _fmt_float(swap_cut) }  Δ={ _fmt_float(dcut) }")
        print(f"     保真度: 组{g1}: { _fmt_float(of.get(g1)) } -> { _fmt_float(nf.get(g1)) }  Δ={ _fmt_float(df1) }; 组{g2}: { _fmt_float(of.get(g2)) } -> { _fmt_float(nf.get(g2)) }  Δ={ _fmt_float(df2) } | 总Δ={ _fmt_float(sum_df) }")
        print(f"     深度: 组{g1}: { od.get(g1) } -> { nd.get(g1) }  Δ={ dd.get(g1) }; 组{g2}: { od.get(g2) } -> { nd.get(g2) }  Δ={ dd.get(g2) }")
        print(f"     时间(s): 组{g1}: { _fmt_float(os.get(g1)) } -> { _fmt_float(ns.get(g1)) }  Δ={ _fmt_float(ds.get(g1)) }; 组{g2}: { _fmt_float(os.get(g2)) } -> { _fmt_float(ns.get(g2)) }  Δ={ _fmt_float(ds.get(g2)) } | 总Δ={ _fmt_float(dt_sum) }")
        print(f"     结论: {verdict}")
    print("==== 结束 ====")


def test_pra_benchmark_swaps(chip_name,
                            root_base="/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/",
                            bench_subdir="pra_benchmark/small_scale",
                            parts_k=3,
                            limit=None,
                            top_k=5):
    """
    遍历pra benchmark文件夹中的所有.qasm线路：
    - 对每个线路进行metis分割（parts_k个子线路）
    - 计算两两子线路交换"swap/cx最高逻辑比特"的割变化与保真度变化
    - 打印清晰报告并返回汇总结果

    Args:
        chip_name: 目标设备名（如'FakeHanoiV2'）
        root_base: 基础Benchmark路径
        bench_subdir: 子路径（默认pra_benchmark/small_scale）
        parts_k: 分割子线路数
        limit: 可选，最多处理的文件数量
        top_k: 报告中每个线路展示的前K条交换建议

    Returns:
        List[{
            'file': str,
            'num_qubits': int,
            'pairs_eval': list,   # 由swap_top_ratio_qubits_all_pairs返回
        }]
    """
    folder = os.path.join(root_base, bench_subdir)
    if not os.path.isdir(folder):
        print(f"目标目录不存在: {folder}")
        return []

    qasm_files = [f for f in sorted(os.listdir(folder)) if f.endswith('.qasm')]
    if limit is not None:
        qasm_files = qasm_files[:limit]

    results = []
    for fname in qasm_files:
        fpath = os.path.join(folder, fname)
        try:
            circuit = read_circuit(fpath)
            interaction_graph = create_interaction_graph(circuit)
            cut_result = metis_zmz(interaction_graph, parts_k)
            membership = cut_result[1]

            print(f"\n=== 文件: {fname} | qubits={circuit.num_qubits} | parts={parts_k} ===")
            pairs_eval = swap_top_ratio_qubits_all_pairs(
                circuit, membership=membership,
                chip_name=chip_name, two_q_gate_name='cx',
                max_regions=200, excluded_qubits=None
            )
            print_swap_pairs_report(pairs_eval, top_k=top_k)

            results.append({
                'file': fname,
                'num_qubits': circuit.num_qubits,
                'pairs_eval': pairs_eval,
            })
        except Exception as e:
            print(f"[跳过] {fname}: {e}")
            continue
    return results


def test_prabench_manhattan_all(root_base="/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/",
                               bench_subdirs=("pra_benchmark/small_scale", "pra_benchmark/middle_scale", "pra_benchmark/large_scale"),
                               chip_name='FakeManhattanV2',
                               manhattan_qubits=65,
                               limit=None,
                               top_k=5):
    """
    使用FakeManhattanV2（或同规模设备）对pra_benchmark下多个子目录的线路进行批量测试：
    - 对每个线路，设置 parts_k = ceil(线路比特数 / 65)
    - 计算两两子线路交换（基于各子线路swap/cx最高逻辑比特）的割变化与保真度变化
    - 打印清晰报告并返回汇总

    Returns: List[{ 'folder': str, 'file': str, 'num_qubits': int, 'parts_k': int, 'pairs_eval': list }]
    """
    results = []
    for subdir in bench_subdirs:
        folder = os.path.join(root_base, subdir)
        if not os.path.isdir(folder):
            print(f"目标目录不存在: {folder}")
            continue
        qasm_files = [f for f in sorted(os.listdir(folder)) if f.endswith('.qasm')]
        if limit is not None:
            qasm_files = qasm_files[:limit]
        for fname in qasm_files:
            fpath = os.path.join(folder, fname)
            try:
                circuit = read_circuit(fpath)
                n = circuit.num_qubits
                parts_k = max(1, math.ceil(n / manhattan_qubits))
                interaction_graph = create_interaction_graph(circuit)
                cut_result = metis_zmz(interaction_graph, parts_k)
                membership = cut_result[1]
                print(f"\n=== [{subdir}] 文件: {fname} | qubits={n} | parts_k={parts_k} (ceil({n}/{manhattan_qubits})) ===")
                pairs_eval = swap_top_ratio_qubits_all_pairs(
                    circuit, membership=membership,
                    chip_name=chip_name, two_q_gate_name='cx',
                    max_regions=200, excluded_qubits=None
                )
                print_swap_pairs_report(pairs_eval, top_k=top_k)
                results.append({
                    'folder': subdir,
                    'file': fname,
                    'num_qubits': n,
                    'parts_k': parts_k,
                    'pairs_eval': pairs_eval,
                })
            except Exception as e:
                print(f"[跳过] {subdir}/{fname}: {e}")
                continue
    return results


def compute_subcircuits_with_budget(circuit, budget, device_qubit_max,
                                   chip_name=None,
                                   two_q_gate_name='cx',
                                   max_regions=200,
                                   excluded_qubits=None):
    """
    给定预算与设备单次可用比特数上限（device_qubit_max），对原始电路进行分割，
    返回：子线路列表、总预算、已用预算、割数目。

    原则：尽量用满预算来获得更小的子线路（潜在更高保真度）。
    - 预算消耗模型：已用预算 = 跨分组边数量(cuts) + 删除的比特数量(removed_nodes_count)
    - 即使电路本身不超过device_qubit_max，也会尝试用预算做更多分割（增加parts）。
    - 若提供chip_name，并且初次分割后仍有剩余预算，则进一步尝试：
      1) 借助两两子线路交换（优先交换各子线路swap/cx比值最高的逻辑比特），若交换新增cut不超过剩余预算且提升总保真度，则应用交换
      2) 否则尝试"二次冻结"：增加删除的最高度数比特数目，在预算内寻找更充分利用预算的方案
    - 若所有尝试均超预算，则返回"无法执行，需增加预算"。

    返回:
      dict {
        'status': 'ok' | 'insufficient_budget',
        'message': Optional[str],
        'subcircuits': List[QuantumCircuit] | None,
        'total_budget': int,
        'used_budget': Optional[int],
        'cuts': Optional[int],
        'membership': Optional[List[int]],
        'removed_nodes': Optional[List[int]],
        'parts': Optional[int],
        'post_swap_applied': bool,                 # 是否应用了交换优化
        'post_swap_info': Optional[dict],          # 交换细节（若有）
      }
    """
    G = create_interaction_graph(circuit)

    def try_partition_with_removed(k_remove):
        Greduced, removed_nodes = remove_highest_degree_node(G, k_remove)
        remaining = len(Greduced.nodes())
        base_parts = max(1, math.ceil(remaining / device_qubit_max))
        best_local = None  # (used, cuts, membership, removed_nodes, parts)
        for parts in range(base_parts, remaining + 1):
            cuts, membership = metis_zmz(Greduced, parts, removed_nodes=removed_nodes)
            used = cuts + len(removed_nodes)
            if used <= budget:
                if (best_local is None) or (used > best_local[0]):
                    best_local = (used, cuts, membership, removed_nodes, parts)
        return best_local

    best = None  # (used, cuts, membership, removed_nodes, parts)
    # 在删除0..budget个节点范围内搜索，并对每个删除数目尝试增加parts
    for k in range(0, max(0, int(budget)) + 1):
        local = try_partition_with_removed(k)
        if local is None:
            continue
        used = local[0]
        if used <= budget:
            if (best is None) or (used > best[0]):
                best = local

    if best is None:
        return {
            'status': 'insufficient_budget',
            'message': '无法执行，需增加预算',
            'subcircuits': None,
            'total_budget': int(budget),
            'used_budget': None,
            'cuts': None,
            'membership': None,
            'removed_nodes': None,
            'parts': None,
            'post_swap_applied': False,
            'post_swap_info': None,
        }

    used, cuts, membership, removed_nodes, parts = best
    leftover = int(budget) - int(used)
    post_swap_applied = False
    post_swap_info = None

    # 若有剩余预算且有芯片信息，尝试进行交换优化
    if leftover > 0 and chip_name is not None:
        try:
            pairs_eval = swap_top_ratio_qubits_all_pairs(
                circuit, membership=membership,
                chip_name=chip_name, two_q_gate_name=two_q_gate_name,
                max_regions=max_regions, excluded_qubits=excluded_qubits
            )
            # 挑选满足 cut_delta <= leftover 且 总Δ保真度 > 0 的最佳交换
            best_pair = None
            best_gain = 0.0
            for it in pairs_eval:
                dcut = it.get('cut_change', {}).get('delta', 0)
                fc = it.get('fidelity_changes', {})
                gain = sum(v for v in fc.values() if v is not None) if fc else 0.0
                # 允许dcut<0（减少cut），或 0<=dcut<=leftover
                if (dcut <= leftover) or (dcut < 0):
                    if gain > 0 and (best_pair is None or gain > best_gain or (gain == best_gain and dcut < best_pair['cut_change'].get('delta', 0))):
                        best_pair = it
                        best_gain = gain
            if best_pair is not None:
                # 应用交换：更新membership与预算
                q1 = best_pair['q1']; q2 = best_pair['q2']
                dcut = best_pair['cut_change']['delta']
                new_membership = list(membership)
                new_membership[q1], new_membership[q2] = new_membership[q2], new_membership[q1]
                membership = new_membership
                cuts = cuts + dcut
                used = cuts + len(removed_nodes)
                leftover = int(budget) - int(used)
                post_swap_applied = True
                post_swap_info = best_pair
        except Exception:
            pass

    # 若仍有剩余预算，尝试"二次冻结"最高度数比特
    if leftover > 0:
        try:
            # 在更大删除范围内重新搜索（不小于当前删除数）
            current_removed = len(removed_nodes)
            improved = None
            for k in range(current_removed, current_removed + leftover + 1):
                Greduced2, rem2 = remove_highest_degree_node(G, k)
                remaining2 = len(Greduced2.nodes())
                base_parts2 = max(1, math.ceil(remaining2 / device_qubit_max))
                for parts2 in range(base_parts2, remaining2 + 1):
                    cuts2, mem2 = metis_zmz(Greduced2, parts2, removed_nodes=rem2)
                    used2 = cuts2 + len(rem2)
                    if used2 <= budget:
                        if (improved is None) or (used2 > improved[0]):
                            improved = (used2, cuts2, mem2, rem2, parts2)
            if improved is not None and improved[0] > used:
                used, cuts, membership, removed_nodes, parts = improved
                leftover = int(budget) - int(used)
        except Exception:
            pass

    subcircuits = extract_sub_circuit(circuit, membership)

    # 统计每个子线路与其他子线路之间的割数量
    per_group_cut_counts = compute_per_group_cut_counts(G, membership)

    return {
        'status': 'ok',
        'message': None,
        'subcircuits': subcircuits,
        'total_budget': int(budget),
        'used_budget': int(used),
        'cuts': int(cuts),
        'membership': membership,
        'removed_nodes': removed_nodes,
        'parts': parts,
        'post_swap_applied': post_swap_applied,
        'post_swap_info': post_swap_info,
        'per_subcircuit_cut_counts': per_group_cut_counts,
    }


def compute_per_group_cut_counts(G, membership):
    """
    按分组统计每个子线路与其他子线路之间的割数量（带权重）。
    规则：
    - membership[i] == -1 的比特忽略；
    - 对于跨组边(u,v)，其权重同时计入两个端点所属分组的计数。
    返回：dict[group_id -> int]
    """
    group_ids = sorted(set(m for m in membership if m != -1))
    counts = {g: 0 for g in group_ids}
    for u, v, data in G.edges(data=True):
        mu = membership[u] if u < len(membership) else None
        mv = membership[v] if v < len(membership) else None
        if mu is None or mv is None:
            continue
        if mu == -1 or mv == -1:
            continue
        if mu != mv:
            w = data.get('weight', 1)
            if mu in counts:
                counts[mu] += w
            if mv in counts:
                counts[mv] += w
    return counts


if __name__ == "__main__":
    # 读取电路
    # base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    # test_circuit_path = os.path.join(base_path, "pra_benchmark", "small_scale", "z4_268.qasm")
    # circuit = read_circuit(test_circuit_path)

    # # 打印原始线路信息
    # print_circuit_info(circuit, f"原始线路 ({os.path.basename(test_circuit_path)})")

    # # 执行线路切割
    # # cut_result = circuit_iteration_cut(circuit, 5, 3)
    # # print("\n切割结果：", cut_result)
    # interaction_graph = create_interaction_graph(circuit)
    # cut_result = metis_zmz(interaction_graph, 3)
    # if cut_result:
    #     subcircuits = extract_sub_circuit(circuit, cut_result[1])
    #     print(f"\n{'='*20} 子线路信息 {'='*20}")
    #     print(f"生成的子线路数量：{len(subcircuits)}")
        
    #     # 打印每个子线路的信息
    #     for i, subcircuit in enumerate(subcircuits):
    #         print_circuit_info(subcircuit, f"子线路 {i}")
            
    #     # 打印一些总体统计
    #     total_gates = sum(len(sc.data) for sc in subcircuits)
    #     max_depth = max(sc.depth() for sc in subcircuits)
    #     print(f"\n{'='*20} 切割后总体统计 {'='*20}")
    #     print(f"原始线路门操作数量: {len(circuit.data)}")
    #     print(f"子线路门操作总数: {total_gates}")
    #     print(f"门操作开销: {total_gates - len(circuit.data)}")
    #     print(f"最大子线路深度: {max_depth}")
    #     print(f"原始线路深度: {circuit.depth()}")
    #     print("=" * 50)


# # 创建通信图
# interaction_graph = create_interaction_graph(circuit)

# # 打印边和权重信息
# # print("原始图：")
# # print_interaction_graph(interaction_graph)
# print("\n原始图的metis分割结果：")
# print(metis_zmz(interaction_graph, 2))

# # 删除最高度数节点并进行metis分割
# num_nodes_to_remove = 2  # 可以修改这个数字来删除不同数量的节点
# reduced_graph, removed_nodes = remove_highest_degree_node(interaction_graph, num_nodes_to_remove)
# print(f"\n删除{num_nodes_to_remove}个最高度数节点后的metis分割结果（-1表示被删除的节点）：")
# print(metis_zmz(reduced_graph, 2, removed_nodes))

    # interaction_graph = create_interaction_graph(circuit)
    # # 假设已有 membership（例如来自 metis_zmz 的结果）
    # # res = evaluate_swap_cut_change(G, cut_result[1], q1=0, q2=5)
    # # print(estimate_best_fidelity_and_logical_stats(circuit, 'FakeHanoiV2', two_q_gate_name='cx', max_regions=200, excluded_qubits=None))
    # cut_result = metis_zmz(interaction_graph, 3)
    # print(res['base_cut'], res['swapped_cut'], res['delta'])
    # print(max_ratio_qubit_per_subcircuit(circuit, cut_result[1], 'FakeHanoiV2'))

    # print(swap_top_ratio_qubits_and_evaluate(circuit, cut_result[1], 'FakeHanoiV2'))

    # pairs_eval = swap_top_ratio_qubits_all_pairs(
    #     circuit, membership=cut_result[1],
    #     chip_name='FakeHanoiV2', two_q_gate_name='cx',
    #     max_regions=200, excluded_qubits=None
    # )
    # print(pairs_eval)

    # print_swap_pairs_report(pairs_eval)

    # # 针对FakeHanoiV2，分割为3个子线路，只打印每个文件前5条交换建议
    # test_pra_benchmark_swaps(
    #     chip_name='FakeHanoiV2',
    #     root_base="/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/",
    #     bench_subdir="pra_benchmark/small_scale",
    #     parts_k=3,
    #     limit=None,
    #     top_k=5
    # )

    # 针对FakeManhattanV2，批量测试多个子目录
    # test_prabench_manhattan_all(
    #     root_base="/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/",
    #     bench_subdirs=("pra_benchmark/small_scale", "pra_benchmark/qaoa", "pra_benchmark/qft", "pra_benchmark/rca", "pra_benchmark/vqe"),
    #     chip_name='FakeManhattanV2',
    #     manhattan_qubits=65,
    #     limit=None,
    #     top_k=5
    # )
    # base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    # folder = base_path + "pra_benchmark/small_scale"  # 换其他: qaoa/qft/rca/vqe
    # device_qubit_max = 65
    # budget = 1
    # limit = 5  # 只测前5个

    # qasm_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".qasm")]
    # qasm_files = qasm_files[:limit]

    # for fname in qasm_files:
    #     fpath = os.path.join(folder, fname)
    #     print(f"\n=== 文件: {fname} | 预算={budget} | device_qubit_max={device_qubit_max} ===")
    #     try:
    #         circuit = read_circuit(fpath)
    #         res = compute_subcircuits_with_budget(circuit, budget=budget, device_qubit_max=device_qubit_max)
    #         print(f"status: {res['status']}, message: {res['message']}")
    #         if res['status'] == 'ok':
    #             print(f"used_budget: {res['used_budget']}, cuts: {res['cuts']}, parts: {res['parts']}")
    #             print(f"生成子线路数: {len(res['subcircuits'])}")
    #             for i, sc in enumerate(res['subcircuits']):
    #                 print(f"  子线路 {i}: qubits={sc.num_qubits}, depth={sc.depth()}")
    #         else:
    #             print("无法执行，需增加预算")
    #     except Exception as e:
    #         print(f"[跳过] {fname}: {e}")

    # 一、单文件测试：分割+余量预算优化（交换/二次冻结）
    # base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    # qasm_path = base_path + "pra_benchmark/small_scale/z4_268.qasm"  # 替换为你想测试的文件
    # device_qubit_max = 65
    # budget = 10
    # chip_name = "FakeManhattanV2"  # 用于保真度评估与交换判断

    # circuit = read_circuit(qasm_path)

    # res = compute_subcircuits_with_budget(
    #     circuit=circuit,
    #     budget=budget,
    #     device_qubit_max=device_qubit_max,
    #     chip_name=chip_name,           # 开启交换优化需要提供芯片名
    #     two_q_gate_name='cx',
    #     max_regions=200,
    #     excluded_qubits=None
    # )

    # print(f"\n=== 单文件测试: {qasm_path} ===")
    # print(f"status: {res['status']}, message: {res['message']}")
    # print(f"total_budget: {res['total_budget']}")
    # if res['status'] == 'ok':
    #     print(f"used_budget: {res['used_budget']}, cuts: {res['cuts']}, parts: {res['parts']}")
    #     print(f"post_swap_applied: {res['post_swap_applied']}")
    #     if res['post_swap_applied']:
    #         info = res['post_swap_info']
    #         print(f"  swap info: groups={info.get('group_pair')}, q1={info.get('q1')}, q2={info.get('q2')}, "
    #               f"dcut={info.get('cut_change',{}).get('delta')}, "
    #               f"Δfid={sum(v for v in info.get('fidelity_changes',{}).values() if v is not None)}")
    #     print(f"生成子线路数: {len(res['subcircuits'])}")
    #     for i, sc in enumerate(res['subcircuits']):
    #         print(f"  子线路 {i}: qubits={sc.num_qubits}, depth={sc.depth()}")
    # else:
    #     print("无法执行，需增加预算")

    # 二、批量测试：遍历一个子目录的多个文件

    base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/"
    folder = base_path + "pra_benchmark/small_scale"  # 可换 qaoa/qft/rca/vqe 等
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
            res = compute_subcircuits_with_budget(
                circuit=circuit,
                budget=budget,
                device_qubit_max=device_qubit_max,
                chip_name=chip_name,
                two_q_gate_name='cx',
                max_regions=200,
                excluded_qubits=None
            )
            print(f"status: {res['status']}, message: {res['message']}")
            if res['status'] == 'ok':
                print(f"used_budget: {res['used_budget']}, cuts: {res['cuts']}, parts: {res['parts']}")
                print(f"post_swap_applied: {res['post_swap_applied']}")
                print(f"生成子线路数: {len(res['subcircuits'])}")
                for i, sc in enumerate(res['subcircuits']):
                    print(f"  子线路 {i}: qubits={sc.num_qubits}, depth={sc.depth()}")
                # 新增：打印每个子线路与其他子线路之间的割数量(权重和)
                per_counts = res.get('per_subcircuit_cut_counts', {})
                if per_counts:
                    print("各子线路与其他子线路之间的割数量(权重和):")
                    for g in sorted(per_counts.keys()):
                        print(f"  组 {g}: {per_counts[g]}")
            else:
                print("无法执行，需增加预算")
        except Exception as e:
            print(f"[跳过] {fname}: {e}")



