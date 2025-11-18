import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import colors

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定默认字体为支持中文的字体

class QuantumEnhancedAStar:
    def __init__(self, grid_size=10, obstacle_ratio=0.2):
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.grid = None
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        
        # 量子搜索参数
        self.quantum_iterations = 3  # Grover迭代次数
        self.simulation_precision = 1000  # 量子模拟精度
        
        # 复杂度计数
        self.comparison_count = 0
        self.quantum_operations = 0
        self.classical_operations = 0
    
    def generate_map(self):
        """生成随机地图，确保起点和终点可达"""
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
        """使用BFS验证起点到终点是否可达"""
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
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def quantum_minimum_search(self, open_list, f_scores):
        """
        使用Grover算法思想搜索f值最小的节点
        这里使用概率模拟来实现量子搜索的效果
        """
        if not open_list:
            return None
        
        self.quantum_operations += len(open_list)
        
        # 模拟量子叠加状态：为每个节点分配振幅
        amplitudes = np.ones(len(open_list)) / np.sqrt(len(open_list))
        
        # 模拟Grover迭代：放大最小值对应的振幅
        for iteration in range(self.quantum_iterations):
            # Oracle操作：标记f值较小的节点（标记相位）
            min_f = min(f_scores[node] for node in open_list)
            max_f = max(f_scores[node] for node in open_list)
            f_range = max_f - min_f if max_f > min_f else 1
            
            for i, node in enumerate(open_list):
                # f值越小的节点，相位翻转越大（概率放大越多）
                f_normalized = (f_scores[node] - min_f) / f_range
                phase_shift = 1.0 - 2.0 * (1.0 - f_normalized)  # 小f值得到更大的相位翻转
                amplitudes[i] *= phase_shift
            
            # 扩散操作：关于平均振幅反转
            avg_amplitude = np.mean(amplitudes)
            amplitudes = 2 * avg_amplitude - amplitudes
            
            # 确保概率归一化（模拟量子测量约束）
            amplitude_norm = np.sqrt(np.sum(amplitudes**2))
            if amplitude_norm > 0:
                amplitudes /= amplitude_norm
        
        # 基于放大后的概率选择节点
        probabilities = amplitudes**2
        probabilities = np.maximum(probabilities, 0)  # 确保非负
        prob_sum = np.sum(probabilities)
        
        if prob_sum > 0:
            probabilities /= prob_sum
            selected_index = np.random.choice(len(open_list), p=probabilities)
            return open_list[selected_index]
        else:
            # 量子搜索失败，回退到经典方法
            return self.classical_find_min(open_list, f_scores)
    
    def classical_find_min(self, open_list, f_scores):
        """经典方法查找最小f值节点（备用）"""
        min_node = open_list[0]
        min_f = f_scores[min_node]
        
        for node in open_list[1:]:
            self.comparison_count += 1
            if f_scores[node] < min_f:
                min_f = f_scores[node]
                min_node = node
                
        return min_node
    
    def quantum_a_star_search(self):
        """量子增强的A*算法"""
        # 重置计数
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
            # 使用量子搜索找到f值最小的节点
            current = self.quantum_minimum_search(open_list, f_score)
            
            if current is None:
                break
                
            # 从开放列表中移除当前节点
            open_list.remove(current)
            self.classical_operations += 1
            
            if current == self.goal:
                # 重构路径
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            closed_set.add(current)
            self.classical_operations += 1
            
            # 探索邻居节点
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
        """获取复杂度度量指标"""
        return {
            'quantum_operations': self.quantum_operations,
            'classical_operations': self.classical_operations,
            'comparisons': self.comparison_count,
            'total_operations': self.quantum_operations + self.classical_operations + self.comparison_count,
            'quantum_speedup_ratio': (self.classical_operations + self.comparison_count) / 
                                   max(1, self.quantum_operations)
        }
    
    def visualize(self, path=None, complexity_info=None):
        """可视化地图和路径"""
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
量子操作数: {complexity_info['quantum_operations']}
经典操作数: {complexity_info['classical_operations']}
比较次数: {complexity_info['comparisons']}
总操作数: {complexity_info['total_operations']}
量子加速比: {complexity_info['quantum_speedup_ratio']:.2f}"""
            
            plt.figtext(0.02, 0.02, complexity_text, 
                       bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5), 
                       fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数演示"""
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
    print(f"量子操作次数: {complexity_info['quantum_operations']}")
    print(f"经典操作次数: {complexity_info['classical_operations']}")
    print(f"比较次数: {complexity_info['comparisons']}")
    print(f"总操作数: {complexity_info['total_operations']}")
    print(f"量子加速比: {complexity_info['quantum_speedup_ratio']:.2f}")
    
    q_planner.visualize(path, complexity_info)

if __name__ == "__main__":
    main()