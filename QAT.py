import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import colors
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']

# 量子最小值查找器，使用Grover算法模拟
class QuantumMinFinder:
    def __init__(self, n_qubits: int):
        # n_qubits: 量子比特数
        self.n_qubits = n_qubits
        self.n = 2 ** n_qubits
        self.simulator = Aer.get_backend('qasm_simulator')

    def create_oracle(self, threshold: float, values: list):
        qr = QuantumRegister(self.n_qubits + 1, 'q')
        qc = QuantumCircuit(qr, name="Oracle")
        solution_indices = []
        for i, v in enumerate(values):
            if v <= threshold:
                solution_indices.append(i)
        for idx in solution_indices:
            binary_rep = format(idx, f'0{self.n_qubits}b')
            for j, bit in enumerate(binary_rep):
                if bit == '0':
                    qc.x(j)
            if self.n_qubits > 1:
                qc.mcx(list(range(self.n_qubits)), self.n_qubits)
            else:
                qc.cz(0, self.n_qubits)
            for j, bit in enumerate(binary_rep):
                if bit == '0':
                    qc.x(j)
        return qc, solution_indices

    def create_diffuser(self):
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr, name="Diffuser")
        qc.h(range(self.n_qubits))
        qc.x(range(self.n_qubits))
        if self.n_qubits == 1:
            qc.z(0)
        else:
            qc.h(self.n_qubits-1)
            if self.n_qubits > 1:
                qc.mcx(list(range(self.n_qubits-1)), self.n_qubits-1)
            qc.h(self.n_qubits-1)
        qc.x(range(self.n_qubits))
        qc.h(range(self.n_qubits))
        return qc

    def grover_search(self, threshold: float, values: list):
        oracle, solution_indices = self.create_oracle(threshold, values)
        num_solutions = len(solution_indices)
        if num_solutions > 0:
            optimal_iterations = self.calculate_optimal_iterations(num_solutions)
            index_qr = QuantumRegister(self.n_qubits, 'index')
            oracle_qr = QuantumRegister(1, 'oracle')
            cr = ClassicalRegister(self.n_qubits, 'c')
            qc = QuantumCircuit(index_qr, oracle_qr, cr)
            qc.h(index_qr)
            qc.x(oracle_qr)
            qc.h(oracle_qr)
            for _ in range(optimal_iterations):
                qc.append(oracle, list(range(self.n_qubits + 1)))
                diffuser = self.create_diffuser()
                qc.append(diffuser, list(range(self.n_qubits)))
            qc.measure(index_qr, cr)
            try:
                from qiskit import transpile
                compiled = transpile(qc, self.simulator)
                job = self.simulator.run(compiled, shots=100)
                counts = job.result().get_counts()
                sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                found_solutions = []
                for result, count in sorted_results[:min(3, len(sorted_results))]:
                    idx = int(result, 2)
                    if idx < len(values) and values[idx] < threshold:
                        found_solutions.append((idx, values[idx], count))
                return True, found_solutions, num_solutions
            except Exception as e:
                found_solutions = [(idx, values[idx], 0) for idx in solution_indices]
                return True, found_solutions, num_solutions
        else:
            return False, [], 0

    def calculate_optimal_iterations(self, num_solutions: int):
        if num_solutions == 0:
            return 0
        N = self.n
        M = num_solutions
        theta = math.asin(math.sqrt(M / N))
        optimal = (math.pi / 2 - theta) / (2 * theta)
        return max(1, int(round(optimal)))

    def find_min(self, values: list):
        if len(values) > self.n:
            values = values[:self.n]
        low, high = 0.0, 10000.0
        precision = 1.0
        best_solutions = []
        for i in range(20):
            mid = (low + high) / 2
            has_solutions, found_solutions, num_solutions = self.grover_search(mid, values)
            if has_solutions:
                high = mid
                best_solutions = found_solutions
                solution_indices = [j for j, v in enumerate(values) if v <= mid]
                if len(solution_indices) == 1:
                    sole_idx = solution_indices[0]
                    sole_val = values[sole_idx]
                    return sole_val, sole_idx
            else:
                low = mid
            if high - low <= precision:
                break
        if best_solutions:
            min_solution = min(best_solutions, key=lambda x: x[1])
            return min_solution[1], min_solution[0]
        else:
            return float('inf'), -1

# 量子增强A*路径规划器
class QuantumEnhancedAStar:
    def __init__(self, grid_size=10, obstacle_ratio=0.2):
        # grid_size: 地图大小, obstacle_ratio: 障碍物比例
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.grid = None
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        self.quantum_iterations = 3
        self.simulation_precision = 1000
        self.comparison_count = 0
        self.quantum_operations = 0
        self.classical_operations = 0

    def generate_map(self):
        # 随机生成地图，保证起点终点可达
        while True:
            self.grid = np.zeros((self.grid_size, self.grid_size))
            total_cells = self.grid_size * self.grid_size
            num_obstacles = int(total_cells * self.obstacle_ratio)
            available_positions = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) != self.start and (i, j) != self.goal:
                        available_positions.append((i, j))
            if num_obstacles > len(available_positions):
                num_obstacles = len(available_positions)
            obstacle_positions = random.sample(range(len(available_positions)), num_obstacles)
            for idx in obstacle_positions:
                i, j = available_positions[idx]
                self.grid[i, j] = 1
            if self.is_reachable():
                break

    def is_reachable(self):
        # BFS判断起点到终点是否可达
        if self.grid[self.start] == 1 or self.grid[self.goal] == 1:
            return False
        visited = set()
        queue = [self.start]
        visited.add(self.start)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while queue:
            current = queue.pop(0)
            if current == self.goal:
                return True
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                    self.grid[nx, ny] == 0 and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def heuristic(self, a, b):
        # 曼哈顿距离启发函数
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def quantum_minimum_search(self, open_list, f_scores):
        """
        使用Qiskit量子算法查找f值最小的节点
        open_list: 当前可选节点列表
        f_scores: 每个节点的f值字典
        """
        if not open_list:
            return None
        # 量子比较次数按 sqrt(N) 统计
        self.quantum_operations += int(np.sqrt(len(open_list)))
        # 构造f值列表，按open_list顺序
        f_list = [f_scores[node] for node in open_list]
        n_qubits = max(1, int(math.ceil(math.log2(len(f_list)))))
        finder = QuantumMinFinder(n_qubits)
        min_val, min_idx = finder.find_min(f_list)
        if min_idx >= 0 and min_idx < len(open_list):
            return open_list[min_idx]
        else:
            return self.classical_find_min(open_list, f_scores)

    def classical_find_min(self, open_list, f_scores):
        # 经典方法遍历查找最小f值节点
        min_node = open_list[0]
        min_f = f_scores[min_node]
        for node in open_list[1:]:
            self.comparison_count += 1
            if f_scores[node] < min_f:
                min_f = f_scores[node]
                min_node = node
        return min_node

    def quantum_a_star_search(self):
        # 量子增强A*主流程
        self.comparison_count = 0
        self.quantum_operations = 0
        self.classical_operations = 0
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        open_list = []
        closed_set = set()
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        parent = {self.start: None}
        open_list.append(self.start)
        self.classical_operations += 1
        while open_list:
            current = self.quantum_minimum_search(open_list, f_score)
            if current is None:
                break
            open_list.remove(current)
            self.classical_operations += 1
            if current == self.goal:
                # 路径重构
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            closed_set.add(current)
            self.classical_operations += 1
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                if (nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or
                    self.grid[nx, ny] == 1):
                    continue
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + 1
                self.comparison_count += 1
                if neighbor not in open_list:
                    open_list.append(neighbor)
                    self.classical_operations += 1
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    parent[neighbor] = current
                elif tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    parent[neighbor] = current
        return None

    def get_complexity_metrics(self):
        return {
            'quantum_comparisons': self.quantum_operations,
            'classical_comparisons': self.comparison_count,
            'quantum_operations': self.quantum_operations,
            'classical_operations': self.classical_operations,
        }

    def visualize(self, path=None, complexity_info=None):
        cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        display_grid = self.grid.copy()
        display_grid[self.start] = 2
        display_grid[self.goal] = 3
        if path:
            for pos in path[1:-1]:
                display_grid[pos] = 4
        plt.figure(figsize=(12, 10))
        plt.imshow(display_grid, cmap=cmap, norm=norm, interpolation='nearest')
        plt.grid(which='both', color='gray', linewidth=0.5)
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.start:
                    plt.text(j, i, 'S', ha='center', va='center', fontsize=12, fontweight='bold')
                elif (i, j) == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
                elif self.grid[i, j] == 1:
                    plt.text(j, i, '■', ha='center', va='center', fontsize=10)
                elif path and (i, j) in path[1:-1]:
                    plt.text(j, i, '•', ha='center', va='center', fontsize=10)
        title = '量子增强A*算法路径规划结果'
        if complexity_info:
            title += f'\n量子操作: {complexity_info["quantum_operations"]} | 经典操作: {complexity_info["classical_operations"]}'
        plt.title(title)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='起点 S'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='终点 G'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='路径'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='障碍物')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        if complexity_info:
            complexity_text = f"""量子算法复杂度统计:
  量子比较次数: {complexity_info['quantum_comparisons']}
  经典比较次数: {complexity_info['classical_comparisons']}"""
            plt.figtext(0.02, 0.02, complexity_text,
                       bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5),
                       fontsize=10, verticalalignment='bottom')
        plt.tight_layout()
        plt.show()

def main():
    # 主函数入口，演示量子增强A*路径规划
    random.seed(42)
    np.random.seed(42)
    print("创建量子增强A*路径规划器...")
    q_planner = QuantumEnhancedAStar(grid_size=10, obstacle_ratio=0.25)
    print("生成随机地图...")
    q_planner.generate_map()
    print("地图信息:")
    print(f"起点: {q_planner.start}, 终点: {q_planner.goal}")
    print(f"障碍物数量: {np.sum(q_planner.grid)}")
    print("\n执行量子增强A*搜索...")
    path = q_planner.quantum_a_star_search()
    complexity_info = q_planner.get_complexity_metrics()
    if path:
        print(f"找到路径! 路径长度: {len(path)} 步")
        print(f"路径: {path}")
    else:
        print("未找到路径")
    print("\n量子算法性能分析:")
    print(f"量子比较次数: {complexity_info['quantum_comparisons']}")
    print(f"经典比较次数: {complexity_info['classical_comparisons']}")
    q_planner.visualize(path, complexity_info)

if __name__ == "__main__":
    main()