import networkx as nx
import random

import numpy as np
import pymetis


def partGraphInit():
    # 创建一个带权的随机图
    G = nx.Graph()
    # 添加10个节点
    G.add_nodes_from(range(10))
    # 随机添加边和权重
    for i in range(10):
        for j in range(i + 1, 10):
            if random.random() < 0.5:  # 以一定概率添加边
                G.add_edge(i, j, weight=random.randint(1, 10))

    # 转换图为pymetis的输入格式
    adjacency_list = [list(G.neighbors(i)) for i in range(10)]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 使用pymetis的part_graph_kway方法划分图
    num_parts = 3  # 划分成3个子图
    cuts, membership = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    # 打印节点的分组
    print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in range(num_parts)]

    # 打印每个子图的节点
    for i, sg in enumerate(subgraphs, start=1):
        print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum

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

def metis_zmz(G, k, randomseed):
    random.seed(randomseed)
    np.random.seed(randomseed)
    m = nx.adjacency_matrix(G)
    # print(m)
    # print(nx.adjacency_matrix(G).todense())
    num_parts = k
    xadj, adjncy, eweight = weight4metis(nx.adjacency_matrix(G).todense().tolist())
    # 使用contiguous=True, 20250225
    cuts, membership = pymetis.part_graph(nparts=num_parts, adjncy=adjncy, xadj=xadj, eweights=eweight, recursive = True, contiguous = True)
    # cuts, membership = pymetis.part_graph(nparts=num_parts, adjncy=adjncy, xadj=xadj, eweights=eweight)
    # print(cuts)
    # print(membership)

    # 打印节点的分组
    # print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in
                 range(num_parts)]

    # 打印每个子图的节点
    # for i, sg in enumerate(subgraphs, start=1):
    #     print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    # print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum

def partGraph(G, k):
    # G: 量子线路生成的带权图 k: 划分成的子图数目

    # 转换图为pymetis的输入格式
    adjacency_list = [list(G.neighbors(i)) for i in range(len(G.nodes))]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 使用pymetis的part_graph方法划分图
    num_parts = k  # 划分成3个子图
    cuts, membership = pymetis.part_graph(num_parts, eweights=weights, adjacency=adjacency_list)
    # cuts, membership = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    # 打印节点的分组
    print("Node memberships:", membership)

    # 创建子图
    subgraphs = [G.subgraph([node for node in range(len(membership)) if membership[node] == part]) for part in range(num_parts)]

    # 打印每个子图的节点
    for i, sg in enumerate(subgraphs, start=1):
        print(f"Subgraph {i}: Nodes {list(sg.nodes())}")

    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
            cut_weight_sum += G[u][v]['weight']

    print("Total Weight of Cut Edges:", cut_weight_sum)
    return cuts, membership, cut_weight_sum
#
# def partGraphKway():
#     import networkx as nx
#     import pymetis
#     import random
#
#     # 创建一个带权重的随机图
#     G = nx.gnm_random_graph(10, 20)
#     for (u, v) in G.edges():
#         G.edges[u, v]['weight'] = random.randint(1, 10)
#
#     # 将NetworkX图转换为Pymetis的输入格式
#     adjacency_list = [list(G.neighbors(i)) for i in range(len(G))]
#     # 为Pymetis准备边权重
#     edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
#
#     # 调用pymetis分区函数，将图划分为3个子图
#     cuts, membership = pymetis.part_graph_kway(3, adjacency=adjacency_list, eweights=edge_weights)
#
#     # 输出每个节点的子图分配
#     print("Node Subgraph Membership:", membership)
#
#     # 计算被分割的边的权重总和
#     cut_weight_sum = 0
#     for u, v in G.edges():
#         if membership[u] != membership[v]:  # 如果这条边的两个节点属于不同的子图
#             cut_weight_sum += G[u][v]['weight']
#
#     print("Total Weight of Cut Edges:", cut_weight_sum)


def metis_uneven_partition(G, target_sizes, tolerance_k, randomseed):
    """
    使用METIS进行不均匀图分割
    
    参数:
    - G: NetworkX图对象
    - target_sizes: 目标节点数列表，如[100, 50, ...]
    - tolerance_k: 松弛参数k，允许的节点数范围是[ni-k, ni]
    - randomseed: 随机种子
    
    返回:
    - cuts: 被分割的边数
    - membership: 节点分组结果
    - cut_weight_sum: 被分割边的权重总和
    """
    random.seed(randomseed)
    np.random.seed(randomseed)
    
    total_nodes = G.number_of_nodes()
    num_parts = len(target_sizes)
    
    # 计算目标权重（每个分区占总节点数的比例）
    tpwgts = []
    for size in target_sizes:
        weight = size / total_nodes
        tpwgts.append(weight)
    
    # 计算负载不平衡容忍度
    # ubvec定义了允许的最大不平衡度，公式为: ubvec[i] * tpwgts[i] * total_nodes
    # 我们希望允许的范围是[ni-k, ni]，所以最大值是ni，最小期望是ni-k
    # 不平衡度 = ni / (tpwgts[i] * total_nodes) = ni / target_size
    ubvec = []
    for i, size in enumerate(target_sizes):
        # 允许的最大节点数是size（即ni）
        # 目标节点数也是size，所以不平衡度是 1.0
        # 但我们需要确保最小值不会小于size-k
        # 计算允许的不平衡度：max_allowed_nodes / target_nodes
        max_imbalance = size / (size - tolerance_k) if size > tolerance_k else 2.0
        ubvec.append(max_imbalance)
    
    # 转换图为pymetis的输入格式
    xadj, adjncy, eweight = weight4metis(nx.adjacency_matrix(G).todense().tolist())
    
    # 使用pymetis进行分割，传入目标权重和不平衡容忍度
    cuts, membership = pymetis.part_graph(
        nparts=num_parts, 
        adjncy=adjncy, 
        xadj=xadj, 
        eweights=eweight,
        tpwgts=tpwgts,
        ubvec=ubvec,
        recursive=True, 
        contiguous=True
    )
    
    # 计算切割权重总和
    cut_weight_sum = 0
    for u, v in G.edges():
        if membership[u] != membership[v]:
            cut_weight_sum += G[u][v]['weight']
    
    return cuts, membership, cut_weight_sum


def metis_uneven_partition_iterative(G, target_sizes, tolerance_k, randomseed, max_iterations=10):
    """
    迭代式不均匀图分割，尝试多次以找到最佳结果
    
    参数:
    - G: NetworkX图对象
    - target_sizes: 目标节点数列表，如[100, 50, ...]
    - tolerance_k: 松弛参数k，允许的节点数范围是[ni-k, ni]
    - randomseed: 随机种子
    - max_iterations: 最大尝试次数
    
    返回:
    - best_cuts: 最佳的被分割边数
    - best_membership: 最佳的节点分组结果
    - best_cut_weight_sum: 最佳的被分割边权重总和
    - partition_sizes: 实际的分区大小
    """
    best_cuts = float('inf')
    best_membership = None
    best_cut_weight_sum = float('inf')
    best_partition_sizes = None
    
    for iteration in range(max_iterations):
        try:
            cuts, membership, cut_weight_sum = metis_uneven_partition(
                G, target_sizes, tolerance_k, randomseed + iteration
            )
            
            # 检查分区大小是否满足约束
            partition_sizes = [0] * len(target_sizes)
            for node_partition in membership:
                partition_sizes[node_partition] += 1
            
            # 检查是否所有分区都在允许的范围内
            valid = True
            for i, (actual_size, target_size) in enumerate(zip(partition_sizes, target_sizes)):
                if actual_size < max(1, target_size - tolerance_k) or actual_size > target_size:
                    valid = False
                    break
            
            # 如果有效且更优，则更新最佳结果
            if valid and cut_weight_sum < best_cut_weight_sum:
                best_cuts = cuts
                best_membership = membership
                best_cut_weight_sum = cut_weight_sum
                best_partition_sizes = partition_sizes
                
        except Exception as e:
            print(f"第{iteration}次尝试失败: {e}")
            continue
    
    if best_membership is None:
        # 如果所有尝试都失败，回退到标准的均匀分割
        print("警告：无法满足约束条件，回退到均匀分割")
        return metis_zmz(G, len(target_sizes), randomseed)
    
    return best_cuts, best_membership, best_cut_weight_sum, best_partition_sizes


if __name__ == "__main__":
    metis_zmz()