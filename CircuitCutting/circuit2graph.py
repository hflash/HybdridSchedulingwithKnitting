# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2024/2/5 20:35
# @Author   : HFLASH @ LINKE
# @File     : circuit2graph.py
# @Software : PyCharm
import os
import random

import math
import numpy as np
# from qiskit import QuantumCircuit as IBMquantumcircuit
import networkx as nx
# import matplotlib.pyplot as plt
from metisCut import metis_zmz
from pyoee import OEE
from quantumcircuit.circuit import QuantumCircuit
from qiskit import QuantumCircuit as IBMquantumcircuit


# 这个文件的内容：
#   readCircuitIBM 从路径读取量子线路到IBM QuantumCricuit格式，但是后面用不上了
#   circuit2Graph 读取量子线路中的两比特量子门，并将之转化为无向带权图
#   本来打算用图的betweeness做划分的，后来发现不是特别合适
#   circuitPartition 将图通过metis方法（metis_zmz方法，带权图划分）划分成k个子图，k = 线路量子比特数目/分布式节点量子数目，假设分布式量子设备的计算比特数目是相同的
#       返回remote_operations 远程操作的门id
#       circuit_dagtable 量子线路的dagtable
#       gate_list 量子线路的门列表，其中的每一个元素都是LINKEQ的quantumgate
#       subcircuit_communication 子线路通信的邻接数组
#       qubit_loc_dic 量子比特与子线路的对应关系字典
#       subcircuit_qubit_partitions subcircuit i 中有哪些qubit
# def readCircuitIBM(path):
#     circuit = IBMquantumcircuit.from_qasm_file(path)
#     # print(circuit)
#     return circuit


# def circuit2Graph(circuit: IBMquantumcircuit):
#     circuitGraph = nx.Graph()
#     circuitGraph.add_nodes_from([i for i in range(circuit.num_qubits)])
#     gates = circuit.data
#     for gate in gates:
#         # print(gate.operation.name, end=': ')
#         # print(gate.operation.num_qubits, end=' ')
#         if gate.operation.num_qubits == 2:
#             if (gate.qubits[0].index, gate.qubits[1].index) not in circuitGraph.edges:
#                 # print(gate.qubits[0].index, gate.qubits[1].index)
#                 circuitGraph.add_edge(gate.qubits[0].index, gate.qubits[1].index, weight=1)
#             else:
#                 nowweight = circuitGraph[gate.qubits[0].index][gate.qubits[1].index]['weight']
#                 circuitGraph.add_edge(gate.qubits[0].index, gate.qubits[1].index, weight=nowweight + 1)
#     # betweenness = nx.betweenness_centrality(circuitGraph, normalized=True, weight='weight')
#     # print(betweenness)
#     # nx.draw(circuitGraph, with_labels=True, font_weight='bold')
#     # plt.show()
#     #
#     # 检测 线路 cx 3 0和cx 0 3是否同为一条边
#     # weights = [circuitGraph[u][v]['weight'] for u, v in circuitGraph.edges()]
#     # for u, v in circuitGraph.edges():
#     #     weight = circuitGraph[u][v]['weight']
#     #     print(u, v, weight)
#     return circuitGraph
#
#     # 画出这张图 https://stackoverflow.com/questions/14943439/how-to-draw-multigraph-in-networkx-using-matplotlib-or-graphviz
#     # pos = nx.random_layout(circuitGraph)
#     # names = {name: name for name in circuitGraph.nodes}
#     # nx.draw_networkx_nodes(circuitGraph, pos, node_color='b', node_size=250, alpha=1)
#     # nx.draw_networkx_labels(circuitGraph, pos, names, font_size=12, font_color='w')
#     # ax = plt.gca()
#     # for e in circuitGraph.edges:
#     #     ax.annotate("",
#     #                 xy=pos[e[1]], xycoords='data',
#     #                 xytext=pos[e[0]], textcoords='data',
#     #                 arrowprops=dict(arrowstyle="->", color="0",
#     #                                 shrinkA=10, shrinkB=10,
#     #                                 patchA=None, patchB=None,
#     #                                 connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
#     #                                                                        ),
#     #                                 ),
#     #                 )
#     # plt.axis('off')
#     # plt.show()

def circuit2Graph(circuit: QuantumCircuit):
    circuitGraph = nx.Graph()
    circuitGraph.add_nodes_from([i for i in range(circuit.qubit_number)])
    circuit_dagtable = circuit.to_dagtable()
    gate_list = circuit.gate_list
    for gate in gate_list:
        qubits = gate.get_qubits()
        if len(qubits) == 2:
            if (qubits[0], qubits[1]) not in circuitGraph.edges:
                # print(gate.qubits[0].index, gate.qubits[1].index)
                circuitGraph.add_edge(qubits[0], qubits[1], weight=1)
            else:
                nowweight = circuitGraph[qubits[0]][qubits[1]]['weight']
                circuitGraph.add_edge(qubits[0], qubits[1], weight=nowweight + 1)
    return circuitGraph


def graphBetweenessCentrality(graph):
    pass

def trivialPartition(circuitGraph, k):
    qubit_num = circuitGraph.number_of_nodes()
    each_node_num = math.ceil(qubit_num / k)
    membership = []
    for i in range(qubit_num):
        membership.append(i // each_node_num)
    cut_weight_sum = 0
    for i in range(qubit_num):
        for j in range(i + 1, qubit_num):
            if membership[i] != membership[j] and (i, j) in circuitGraph.edges:
                cut_weight_sum += circuitGraph[i][j]['weight']

    return cut_weight_sum, membership, cut_weight_sum


def circuitPartition(path, device_qubit_number, randomseed, partition_method='metis'):
    # circuitIBM = readCircuitIBM(path)
    random.seed(randomseed)
    np.random.seed(randomseed)
    circuitLINKEQ = QuantumCircuit.from_QASM(path)
    circuit_dagtable = circuitLINKEQ.to_dagtable()
    gate_list = circuitLINKEQ.gate_list
    circuitGraph = circuit2Graph(circuitLINKEQ)
    circuitGraph = circuit2graphIBM(path)
    k = math.ceil(circuitLINKEQ.qubit_number / device_qubit_number)
    # k = 9
    # gate_list 与global_operations也就是dagtable对应，但是gatelist中有一个init门，在后两者中自动隐去
    # cuts 与 cut_weight_sum一致
    # membership: 节点属于哪个子图
    # cut_weight_sum: 被分割掉的边的数目
    if partition_method == 'metis':
        cuts, membership, cut_weight_sum = metis_zmz(circuitGraph, k, randomseed)
    elif partition_method == 'oee':
        cuts, membership, cut_weight_sum = OEE(circuitLINKEQ, k)
    elif partition_method == 'trivial':
        cut_weight_sum, membership, cut_weight_sum = trivialPartition(circuitGraph, k)
    # print(cuts, membership, cut_weight_sum)
    # subcircuits = []
    remote_operations = []

    # qubit 与 子线路之间的关系， subcircuit_qubit_partitions[0]中存储子线路0上的所有qubit
    subcircuit_qubit_partitions = [[] for _ in range(k)]

    # qubit 到 partition的字典
    ## qubit_loc_dic: 比特属于哪个子线路 {0: 1, 1: 1, 2: 0, 3: 0}
    ## subcircuit_qubit_partitions: 子线路包含哪些比特 [[2, 3], [0, 1]]
    qubit_loc_dic = {}
    for index, value in enumerate(membership):
        qubit_loc_dic[index] = value
    for key in qubit_loc_dic.keys():
        subcircuit_qubit_partitions[qubit_loc_dic[key]].append(key)
    subcircuit_communication = np.zeros([k, k], dtype=int)
    # for cut in range(cuts):
    #     subcircuit_qubit = [index for index, value in enumerate(membership) if value == cut]
    #     subcircuit_qubit_partitions.append(subcircuit_qubit)
    notSameLocationNum = 0
    for num, gate in enumerate(gate_list):
        if num == 0:
            # 忽略第一个门
            assert gate.name == 'Init'
            continue
        qubits = gate.get_qubits()
        if len(qubits) > 1:
            # assert gate.name == "CX"
            same_location_flag = (qubit_loc_dic[qubits[0]] == qubit_loc_dic[qubits[1]])
            # print(same_location_flag)
            if not same_location_flag:
                subcircuit_communication[qubit_loc_dic[qubits[0]]][qubit_loc_dic[qubits[1]]] += 1
                subcircuit_communication[qubit_loc_dic[qubits[1]]][qubit_loc_dic[qubits[0]]] += 1
                notSameLocationNum += 1
                remote_operations.append(num)
    # 切割掉的边的权重之和应当与上面过程中属于不同子线路的CX门数目一致
    assert cut_weight_sum == notSameLocationNum and notSameLocationNum == len(remote_operations)
    # print(remote_operations)
    return remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions

def circuitPartitionDebug(path, device_qubit_number, randomseed, partition_method='metis'):
    # circuitIBM = readCircuitIBM(path)
    random.seed(randomseed)
    np.random.seed(randomseed)
    circuitLINKEQ = QuantumCircuit.from_QASM(path)
    circuit_dagtable = circuitLINKEQ.to_dagtable()
    gate_list = circuitLINKEQ.gate_list
    circuitGraph = circuit2Graph(circuitLINKEQ)
    k = math.ceil(circuitLINKEQ.qubit_number / device_qubit_number)
    # gate_list 与global_operations也就是dagtable对应，但是gatelist中有一个init门，在后两者中自动隐去
    # cuts 与 cut_weight_sum一致
    # membership: 节点属于哪个子图
    # cut_weight_sum: 被分割掉的边的数目
    circuit_adjacency = [[0 for _ in range(circuitLINKEQ.qubit_number)] for _ in range(circuitLINKEQ.qubit_number)]
    for num, gate in enumerate(gate_list):
        if num == 0:
            # 忽略第一个门
            assert gate.name == 'Init'
            continue
        qubits = gate.get_qubits()
        if len(qubits) == 2:
            circuit_adjacency[qubits[0]][qubits[1]] += 1
            circuit_adjacency[qubits[1]][qubits[0]] += 1
    print(circuit_adjacency)
    return circuit_adjacency
    # if partition_method == 'metis':
    #     cuts, membership, cut_weight_sum = metis_zmz(circuitGraph, k, randomseed)
    # elif partition_method == 'oee':
    #     cuts, membership, cut_weight_sum = OEE(circuitLINKEQ, k)
    # elif partition_method == 'trivial':
    #     cut_weight_sum, membership, cut_weight_sum = trivialPartition(circuitGraph, k)
    # # print(cuts, membership, cut_weight_sum)
    # # subcircuits = []
    # remote_operations = []

    # # qubit 与 子线路之间的关系， subcircuit_qubit_partitions[0]中存储子线路0上的所有qubit
    # subcircuit_qubit_partitions = [[] for _ in range(k)]

    # # qubit 到 partition的字典
    # ## qubit_loc_dic: 比特属于哪个子线路 {0: 1, 1: 1, 2: 0, 3: 0}
    # ## subcircuit_qubit_partitions: 子线路包含哪些比特 [[2, 3], [0, 1]]
    # qubit_loc_dic = {}
    # for index, value in enumerate(membership):
    #     qubit_loc_dic[index] = value
    # for key in qubit_loc_dic.keys():
    #     subcircuit_qubit_partitions[qubit_loc_dic[key]].append(key)
    # subcircuit_communication = np.zeros([k, k], dtype=int)
    # # for cut in range(cuts):
    # #     subcircuit_qubit = [index for index, value in enumerate(membership) if value == cut]
    # #     subcircuit_qubit_partitions.append(subcircuit_qubit)
    # notSameLocationNum = 0
    # for num, gate in enumerate(gate_list):
    #     if num == 0:
    #         # 忽略第一个门
    #         assert gate.name == 'Init'
    #         continue
    #     qubits = gate.get_qubits()
    #     if len(qubits) > 1:
    #         # assert gate.name == "CX"
    #         same_location_flag = (qubit_loc_dic[qubits[0]] == qubit_loc_dic[qubits[1]])
    #         # print(same_location_flag)
    #         if not same_location_flag:
    #             subcircuit_communication[qubit_loc_dic[qubits[0]]][qubit_loc_dic[qubits[1]]] += 1
    #             subcircuit_communication[qubit_loc_dic[qubits[1]]][qubit_loc_dic[qubits[0]]] += 1
    #             notSameLocationNum += 1
    #             remote_operations.append(num)
    # # 切割掉的边的权重之和应当与上面过程中属于不同子线路的CX门数目一致
    # assert cut_weight_sum == notSameLocationNum and notSameLocationNum == len(remote_operations)
    # # print(remote_operations)
    # return remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions

def analyze_circuit_adjacency_list(circuit_adjacency_list):
    for index, adjacency_matrix in enumerate(circuit_adjacency_list):
        # 将邻接矩阵转换为 NumPy 数组
        adjacency_matrix = np.array(adjacency_matrix)
        
        # 矩阵的形状
        shape = adjacency_matrix.shape
        
        # 非零元素的数量
        non_zero_count = np.count_nonzero(adjacency_matrix)
        
        # 平均连接强度（非零元素的平均值）
        if non_zero_count > 0:
            average_weight = np.sum(adjacency_matrix) / non_zero_count
        else:
            average_weight = 0
        
        # 打印分析结果
        print(f"量子电路 {index + 1}:")
        print(f"  矩阵形状: {shape}")
        print(f"  非零元素数量: {non_zero_count}")
        print(f"  平均连接强度: {average_weight:.2f}")
        print()

def circuitPartitionUneven(path, target_sizes, tolerance_k, randomseed):
    """
    不均匀量子线路分割
    
    参数:
    - path: QASM文件路径
    - target_sizes: 目标节点数列表，如[100, 50, ...]
    - tolerance_k: 松弛参数，允许的节点数范围是[ni-k, ni]
    - randomseed: 随机种子
    
    返回:
    - remote_operations: 远程操作的门id
    - circuit_dagtable: 量子线路的dagtable
    - gate_list: 量子线路的门列表
    - subcircuit_communication: 子线路通信的邻接数组
    - qubit_loc_dic: 量子比特与子线路的对应关系字典
    - subcircuit_qubit_partitions: subcircuit i 中有哪些qubit
    - partition_info: 分割结果信息，包含实际的分区大小
    """
    from metisCut import metis_uneven_partition_iterative
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    circuitLINKEQ = QuantumCircuit.from_QASM(path)
    circuit_dagtable = circuitLINKEQ.to_dagtable()
    gate_list = circuitLINKEQ.gate_list
    circuitGraph = circuit2Graph(circuitLINKEQ)
    
    # 检查目标大小是否合理
    total_qubits = circuitLINKEQ.qubit_number
    total_target = sum(target_sizes)
    if total_target > total_qubits + len(target_sizes) * tolerance_k:
        raise ValueError(f"目标大小总和({total_target})过大，超出了总量子比特数({total_qubits})加上容忍度的范围")
    
    # 使用新的不均匀分割方法
    cuts, membership, cut_weight_sum, actual_partition_sizes = metis_uneven_partition_iterative(
        circuitGraph, target_sizes, tolerance_k, randomseed
    )
    
    k = len(target_sizes)
    remote_operations = []

    # qubit 与 子线路之间的关系
    subcircuit_qubit_partitions = [[] for _ in range(k)]

    # qubit 到 partition的字典
    qubit_loc_dic = {}
    for index, value in enumerate(membership):
        qubit_loc_dic[index] = value
    for key in qubit_loc_dic.keys():
        subcircuit_qubit_partitions[qubit_loc_dic[key]].append(key)
    
    subcircuit_communication = np.zeros([k, k], dtype=int)
    
    notSameLocationNum = 0
    for num, gate in enumerate(gate_list):
        if num == 0:
            # 忽略第一个门
            assert gate.name == 'Init'
            continue
        qubits = gate.get_qubits()
        if len(qubits) > 1:
            same_location_flag = (qubit_loc_dic[qubits[0]] == qubit_loc_dic[qubits[1]])
            if not same_location_flag:
                subcircuit_communication[qubit_loc_dic[qubits[0]]][qubit_loc_dic[qubits[1]]] += 1
                subcircuit_communication[qubit_loc_dic[qubits[1]]][qubit_loc_dic[qubits[0]]] += 1
                notSameLocationNum += 1
                remote_operations.append(num)
    
    # 切割掉的边的权重之和应当与上面过程中属于不同子线路的CX门数目一致
    assert cut_weight_sum == notSameLocationNum and notSameLocationNum == len(remote_operations)
    
    # 创建分割结果信息
    partition_info = {
        'target_sizes': target_sizes,
        'actual_sizes': actual_partition_sizes,
        'tolerance_k': tolerance_k,
        'cut_weight_sum': cut_weight_sum,
        'total_qubits': total_qubits
    }
    
    return remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions, partition_info

def circuitPartitionWithIBM(path, device_qubit_number, randomseed, partition_method='metis'):
    """
    使用IBM量子线路格式进行电路分割
    
    参数:
    - path: QASM文件路径
    - device_qubit_number: 每个设备上的量子比特数量
    - randomseed: 随机种子
    - partition_method: 分割方法，可选'metis'、'oee'或'trivial'
    
    返回:
    - remote_operations: 远程操作的门id列表
    - gate_list: IBM量子线路中的指令列表
    - subcircuit_communication: 子线路间的通信矩阵
    - qubit_loc_dic: 量子比特位置字典
    - subcircuit_qubit_partitions: 每个子线路包含的量子比特列表
    """
    random.seed(randomseed)
    np.random.seed(randomseed)
    
    # 读取IBM格式的量子线路
    circuit = IBMquantumcircuit.from_qasm_file(path)
    gate_list = circuit.data
    
    # 构建电路图
    circuitGraph = circuit2graphIBM(path)
    
    # 计算需要的子电路数量
    k = math.ceil(circuit.num_qubits / device_qubit_number)
    
    # 根据选择的方法进行分割
    if partition_method == 'metis':
        cuts, membership, cut_weight_sum = metis_zmz(circuitGraph, k, randomseed)
    elif partition_method == 'oee':
        # 为了使用OEE方法，需要先将IBM电路转换为LINKEQ格式
        circuitLINKEQ = QuantumCircuit.from_QASM(path)
        cuts, membership, cut_weight_sum = OEE(circuitLINKEQ, k)
    elif partition_method == 'trivial':
        cut_weight_sum, membership, cut_weight_sum = trivialPartition(circuitGraph, k)
    
    remote_operations = []
    
    # 初始化子线路的量子比特分配
    subcircuit_qubit_partitions = [[] for _ in range(k)]
    
    # 建立量子比特到子线路的映射
    qubit_loc_dic = {}
    for index, value in enumerate(membership):
        qubit_loc_dic[index] = value
    for key in qubit_loc_dic.keys():
        subcircuit_qubit_partitions[qubit_loc_dic[key]].append(key)
    
    # 初始化子线路间的通信矩阵
    subcircuit_communication = np.zeros([k, k], dtype=int)
    
    # 统计跨子线路的操作
    notSameLocationNum = 0
    for num, instruction in enumerate(gate_list):
        qubits = instruction.qubits
        if len(qubits) == 2:  # 对于两比特门
            qubit1_idx = qubits[0].index
            qubit2_idx = qubits[1].index
            same_location_flag = (qubit_loc_dic[qubit1_idx] == qubit_loc_dic[qubit2_idx])
            
            if not same_location_flag:
                subcircuit_communication[qubit_loc_dic[qubit1_idx]][qubit_loc_dic[qubit2_idx]] += 1
                subcircuit_communication[qubit_loc_dic[qubit2_idx]][qubit_loc_dic[qubit1_idx]] += 1
                notSameLocationNum += 1
                remote_operations.append(num)
    
    # 验证切割边的权重和与跨子线路操作数量一致
    assert cut_weight_sum == notSameLocationNum and notSameLocationNum == len(remote_operations)
    
    return remote_operations, None, None, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions

    # remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions

def circuit2graphIBM(qasm_path: str) -> nx.Graph:
    """
    从 QASM 文件读取量子线路并转换为无向带权图（使用 Qiskit）
    
    Args:
        qasm_path: QASM 文件的路径
        
    Returns:
        nx.Graph: 转换后的无向带权图，其中：
            - 节点表示量子比特
            - 边表示两个量子比特之间的两比特门
            - 边的权重表示两比特门的数量
    """
    # 读取 QASM 文件
    circuit = IBMquantumcircuit.from_qasm_file(qasm_path)
    
    # 创建空图
    circuitGraph = nx.Graph()
    
    # 添加节点（量子比特）
    circuitGraph.add_nodes_from(range(circuit.num_qubits))
    
    # 遍历所有量子门
    for instruction in circuit.data:
        # 获取量子门作用的量子比特
        qubits = instruction.qubits
        
        # 只处理两比特门
        if len(qubits) == 2:
            qubit1, qubit2 = qubits[0].index, qubits[1].index
            
            # 如果边不存在，添加边并设置权重为1
            if not circuitGraph.has_edge(qubit1, qubit2):
                circuitGraph.add_edge(qubit1, qubit2, weight=1)
            else:
                # 如果边已存在，增加权重
                current_weight = circuitGraph[qubit1][qubit2]['weight']
                circuitGraph[qubit1][qubit2]['weight'] = current_weight + 1

        if len(qubits) > 2:
            print(instruction.operation.name)
    return circuitGraph


if __name__ == "__main__":
    # randomseed = np.random.seed()
    # circuit_adjacency_list = []
    # for root, dirs, files in os.walk('/home/normaluser/fzchen/qnet_exp/qnet_iwqos/qnet_iwqos/exp_src_data/test_benchmark'):
    #     for file in files:
    #         if file.endswith('.qasm'):
    #             path = os.path.join(root, file) 
    #             circuit_adjacency = circuitPartitionDebug(path, device_qubit_number = 9, randomseed = randomseed, partition_method='metis')
    #             circuit_adjacency_list.append(circuit_adjacency)
    
    # analyze_circuit_adjacency_list(circuit_adjacency_list)
    randomseed = np.random.seed()
    path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/pra_benchmark/qft/qft_100.qasm"
    # path = "/home/normaluser/fzchen/qnet_exp/qnet_iwqos/qnet_iwqos/exp_src_data/test_benchmark/qugan_n111.qasm"
    # path = "/home/normaluser/fzchen/qnet_exp/qnet_iwqos/qnet_iwqos/exp_src_data/test_benchmark/multiplier_n75.qasm"
    print(len(circuitPartition(path, device_qubit_number = math.ceil(100/2), randomseed = randomseed, partition_method='metis')[0]))
    print(len(circuitPartition(path, device_qubit_number = math.ceil(100/2), randomseed = randomseed, partition_method='oee')[0]))
    print(len(circuitPartition(path, device_qubit_number = math.ceil(100/2), randomseed = randomseed, partition_method='trivial')[0]))
    # print(circuitPartition(path, device_qubit_number = 12, randomseed = randomseed, partition_method='metis')[0])
    # print(circuitPartition(path, device_qubit_number = 12, randomseed = randomseed, partition_method='oee')[0])
    # print(circuitPartition(path, device_qubit_number = 12, randomseed = randomseed, partition_method='trivial')[0])
    # dir = ''
    # path = './adder_n4.qasm'
    # circuit = readCircuit(path)
    # circuit2Graph(circuit)
    # path = '../exp_circuit_benchmark/large_scale'
    # # print(circuitPartition(path))
    # small_data = {}
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith('.qasm'):
    #             circuit_name = file.split('.')[0]
    #             qasm_path = os.path.join(root, file)
    #             print(qasm_path)
    #             with open(qasm_path, 'r') as f:
    #                 circuit = QuantumCircuit.from_QASM(qasm_path)
    #                 qubit_num = circuit.qubit_number
    #                 depth = circuit.get_circuit_depth()
    #                 two_qubit_gate_num = 0
    #                 gate_list = circuit.gate_list
    #                 for gate in gate_list:
    #                     qubits = gate.get_qubits()
    #                     if len(qubits) == 2:
    #                         two_qubit_gate_num += 1
    #                 path = os.path.join(root, file)
    #                 remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions = circuitPartition(
    #                     path, 40, randomseed=np.random.seed())
    #                 small_data[circuit_name] = [qubit_num, depth, two_qubit_gate_num, len(remote_operations)]
    # print(small_data)

    # graph = nx.Graph()
    # graph.add_weighted_edges_from([(1, 2, 2), (1, 3, 1), (2, 3, 1), (3, 4, 1), (2, 4, 1)])
    # betweeness = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    # print(betweeness)
