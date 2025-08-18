from circuit2graph import circuit2Graph
from circuit2graph import circuitPartitionWithIBM as traditional_cut
from circuit2graph import circuitPartitionUneven as uneven_cut
from circuit2graph import circuit2graphIBM
import numpy as np
import os

def demo_traditional_cut():
    """演示传统的均匀分割方法"""
    print("=== 传统的均匀分割方法 ===")
    path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/pra_benchmark/qft/qft_100.qasm"
    result = traditional_cut(path, 34, np.random.seed(), 'metis')
    print(f"传统方法结果: {len(result[0])} 个远程操作")
    print(f"子线路量子比特分配: {result[5]}")
    print(f"每个子线路的量子比特数: {[len(partition) for partition in result[5]]}")
    print()

def demo_uneven_cut():
    """演示新的不均匀分割方法"""
    print("=== 新的不均匀分割方法 ===")
    path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/pra_benchmark/qft/qft_100.qasm"
    
    # 示例：300个节点的图，目标分割成100, 50的子图，松弛参数k=10
    target_sizes = [40, 30, 30]  # 目标节点数
    tolerance_k = 10  # 松弛参数
    
    try:
        result = uneven_cut(path, target_sizes, tolerance_k, 1)
        remote_operations, circuit_dagtable, gate_list, subcircuit_communication, qubit_loc_dic, subcircuit_qubit_partitions, partition_info = result
        
        print(f"目标分区大小: {partition_info['target_sizes']}")
        print(f"实际分区大小: {partition_info['actual_sizes']}")
        print(f"松弛参数k: {partition_info['tolerance_k']}")
        print(f"总量子比特数: {partition_info['total_qubits']}")
        print(f"切割权重总和: {partition_info['cut_weight_sum']}")
        print(f"远程操作数: {len(remote_operations)}")
        print(f"子线路量子比特分配: {subcircuit_qubit_partitions}")
        
        # 验证约束是否满足
        print("\n约束验证:")
        for i, (target, actual) in enumerate(zip(target_sizes, partition_info['actual_sizes'])):
            min_allowed = max(1, target - tolerance_k)
            max_allowed = target
            is_valid = min_allowed <= actual <= max_allowed
            print(f"  分区{i}: 目标={target}, 实际={actual}, 范围=[{min_allowed}, {max_allowed}], 有效={is_valid}")
            
    except Exception as e:
        print(f"不均匀分割失败: {e}")
    print()

def compare_methods():
    """比较两种方法的性能"""
    print("=== 方法性能比较 ===")
    path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/pra_benchmark/qft/qft_100.qasm"
    
    # 传统方法
    try:
        traditional_result = traditional_cut(path, 34, 1, 'metis')
        traditional_cuts = len(traditional_result[0])
        traditional_partitions = [len(partition) for partition in traditional_result[5]]
        print(f"传统均匀分割:")
        print(f"  远程操作数: {traditional_cuts}")
        print(f"  分区大小: {traditional_partitions}")
    except Exception as e:
        print(f"传统方法失败: {e}")
    
    # 不均匀方法
    target_sizes = [35, 35, 30]
    tolerance_k = 15
    try:
        uneven_result = uneven_cut(path, target_sizes, tolerance_k, 1)
        partition_info = uneven_result[6]
        print(f"新的不均匀分割:")
        print(f"  远程操作数: {partition_info['cut_weight_sum']}")
        print(f"  目标分区大小: {partition_info['target_sizes']}")
        print(f"  实际分区大小: {partition_info['actual_sizes']}")
    except Exception as e:
        print(f"不均匀方法失败: {e}")

def test_medium_large_circuits():
    """测试 QASMBench 中 medium 和 large 规模量子线路的二分割"""
    print("=== QASMBench Medium & Large 线路二分割测试 ===")
    
    # 基础路径
    base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/QASMBench"
    
    # 遍历 medium 和 large 文件夹
    for size in ['medium', 'large']:
        size_path = os.path.join(base_path, size)
        if not os.path.exists(size_path):
            print(f"错误：找不到路径 {size_path}")
            continue
            
        print(f"\n{size.upper()} 规模线路测试:")
        print("-" * 40)
        
        # 获取该规模下的所有线路
        circuits = [d for d in os.listdir(size_path) if os.path.isdir(os.path.join(size_path, d))]
        
        for circuit in sorted(circuits):
            qasm_path = os.path.join(size_path, circuit, f"{circuit}.qasm")
            if not os.path.exists(qasm_path):
                print(f"警告：找不到文件 {qasm_path}")
                continue
                
            print(f"\n测试线路: {circuit}")
            
            try:
                # 获取线路的总节点数以确定合适的分割大小
                graph = circuit2Graph(qasm_path)
                total_nodes = len(graph.nodes())
                target_size = total_nodes // 2  # 二分割
                
                # 使用传统的均匀分割方法
                result = traditional_cut(qasm_path, target_size, 1, 'metis')
                cuts = len(result[0])
                partitions = [len(partition) for partition in result[5]]
                
                print(f"总节点数: {total_nodes}")
                print(f"分割结果:")
                print(f"  - 远程操作数: {cuts}")
                print(f"  - 子线路量子比特数: {partitions}")
                
            except Exception as e:
                print(f"处理失败: {e}")
    
    print("\n测试完成")

def test_qasmbench_bipartition():
    """测试 QASMBench 中 medium 和 large 规模量子线路的二分割"""
    print("=== QASMBench Medium & Large 线路二分割测试 ===")
    
    # 基础路径
    base_path = "/home/normaluser/hflash/HybdridSchedulingwithKnitting/Benchmark/QASMBench"
    #'medium' , 'large'
    # 遍历 medium 和 large 文件夹
    for size in ['large']:
        size_path = os.path.join(base_path, size)
        if not os.path.exists(size_path):
            print(f"错误：找不到路径 {size_path}")
            continue
            
        print(f"\n{size.upper()} 规模线路测试:")
        print("-" * 40)
        
        # 获取该规模下的所有线路文件夹
        circuits = [d for d in os.listdir(size_path) if os.path.isdir(os.path.join(size_path, d))]
        
        for circuit in sorted(circuits):
            qasm_path = os.path.join(size_path, circuit, f"{circuit}.qasm")
            if not os.path.exists(qasm_path):
                print(f"警告：找不到文件 {qasm_path}")
                continue
                
            print(f"\n测试线路: {circuit}")
            
            # try:
                # 使用新的 IBM 方法生成图
            graph = circuit2graphIBM(qasm_path)
            total_nodes = len(graph.nodes())
            target_size = np.floor(total_nodes / 2)  # 二分割
            
            # 计算总的两比特门数量
            total_two_qubit_gates = sum(graph[u][v]['weight'] for u, v in graph.edges())
            
            # 使用传统的均匀分割方法
            result = traditional_cut(qasm_path, target_size, 1, 'metis')
            cuts = len(result[0])
            partitions = [len(partition) for partition in result[5]]
            
            print(f"量子比特数: {total_nodes}")
            print(f"两比特门总数: {total_two_qubit_gates}")
            print(f"分割结果:")
            print(f"  - 远程操作数: {cuts}")
            print(f"  - 远程操作比例: {cuts/total_two_qubit_gates:.2%}")
            print(f"  - 子线路量子比特数: {partitions}")
                
            # except Exception as e:
                # print(f"处理失败: {e}")
    
    print("\n测试完成")

if __name__ == "__main__":
    print("量子线路分割方法测试")
    print("="*50)
    
    # demo_traditional_cut()
    # demo_uneven_cut()  
    # compare_methods()
    # test_medium_large_circuits()
    test_qasmbench_bipartition()
