import numpy as np
from quantumcircuit.circuit import QuantumCircuit

class Node:
    def __init__(self,id) -> None:
        self.id = id
        self.col = 0
        self.neibors = []

class OEER:
    def __init__(self, adj_m, k) -> None:
        self.qubit_number = len(adj_m)
        self.k = k
        self.node_number = self.qubit_number
        if self.node_number%k != 0:
            self.node_number += k-self.node_number%k
        self.node_list = [Node(i) for i in range(self.node_number)]
        self.adj_matrix = np.zeros(shape=(self.node_number, self.node_number), dtype=int)
        for i in range(self.qubit_number):
            for j in range(i+1, self.qubit_number):
                self.adj_matrix[i,j] = adj_m[i,j]
                self.adj_matrix[j,i] = adj_m[i,j]

        self.D_matrix = np.zeros(shape=(self.node_number, self.k), dtype=int)
        #init color
        n = self.node_number//k
        for i in range(k):
            for j in range(n):
                self.node_list[i*n+j].col=i
    
    def w_node_i_col_j(self, i:Node, j:int):
        w = 0
        for node in self.node_list:
            if node.col == j:
                w += self.adj_matrix[i.id, node.id]
        return w
    def w_node_i_col_node_j(self, i:Node, j:Node):
        return self.w_node_i_col_j(i, j.col)

    def D_node_i_col_j(self, i:Node, j:int):
        return self.w_node_i_col_j(i, j) - self.w_node_i_col_j(i, i.col)
    
    def D_node_i_col_node_j(self, i:Node, j:Node):
        return self.D_node_i_col_j(i, j.col)

    def reduced_cost(self, i:Node, j:Node):
        return self.D_matrix[i.id, j.col] + self.D_matrix[j.id, i.col] - 2*self.adj_matrix[i.id, j.id]

    def init_D(self):
        for i in range(len(self.node_list)):
            for j in range(self.k):
                self.D_matrix[i,j] = self.D_node_i_col_j(self.node_list[i], j)

    def exchange(self, i:Node, j:Node):
        i.col, j.col = j.col, i.col
        self.init_D()

    def reverse(self, exchange_list, m):
        for i in range(m, len(exchange_list)):
            a,b = exchange_list[i]
            a.col, b.col = b.col, a.col
        self.init_D()

    def run(self):           
        while True:
            self.init_D()
            #step 1
            C = [1]*self.node_number
            left_node = self.node_number
            exchange_list = []
            gmax = -99999999
            g_total = 0
            m = 0
            index = 0
            while True:
                #step 2
                node_a = None
                node_b = Node
                max_g_i_j = -99999999
                for i in range(self.node_number):
                    if C[i] == 0:
                        continue
                    for j in range(i+1, self.node_number):
                        if C[j] == 0:
                            continue
                        tmp_cost = self.reduced_cost(self.node_list[i], self.node_list[j])
                        if tmp_cost > max_g_i_j:
                            node_a = self.node_list[i]
                            node_b = self.node_list[j]
                            max_g_i_j = tmp_cost
                #step 3
                self.exchange(node_a, node_b)
                C[node_a.id] = 0
                C[node_b.id] = 0
                left_node -= 2
                index += 1
                exchange_list.append([node_a,node_b])
                #step 5
                g_total += max_g_i_j
                if (g_total > gmax):
                    gmax = g_total
                    m = index

                #step 4
                if left_node <= 1:
                    break

            #step 6
            self.reverse(exchange_list, m)
            if gmax <= 0:
                break
                
    def result(self):
        cost = 0
        for i in range(self.qubit_number):
            a = self.node_list[i]
            for j in range(i+1, self.qubit_number):
                b = self.node_list[j]
                if a.col != b.col:
                    cost += self.adj_matrix[a.id,b.id]
        partiton = [node.col for node in self.node_list]
        return cost, partiton, cost


def circuit2adjmatrix(circuit: QuantumCircuit):
    qubit_number = circuit.qubit_number
    adj_m = np.zeros(shape=(qubit_number,qubit_number),dtype=np.int64)
    for gate in circuit.gate_list:
        quits = gate.get_qubits()
        if len(quits) == 2:
            adj_m[quits[0]][quits[1]] += 1
            adj_m[quits[1]][quits[0]] += 1
    return adj_m

def OEE(circuit:QuantumCircuit, k):
    adj_m = circuit2adjmatrix(circuit)
    er = OEER(adj_m, k)
    er.run()
    return er.result()


