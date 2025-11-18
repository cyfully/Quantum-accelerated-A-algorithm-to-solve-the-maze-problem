import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定默认字体为支持中文的字体

class AStarPlannerSimple:
    def __init__(self, grid_size=10, obstacle_ratio=0.2):
        """
        初始化A*路径规划器（简单列表版本）
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            obstacle_ratio: 障碍物所占比例 (0.0-1.0)
        """
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.grid = None
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        
        # 复杂度计数
        self.comparison_count = 0  # 比较次数
        self.node_processing_count = 0  # 节点处理次数
        self.list_operations_count = 0  # 列表操作次数
        
    def generate_map(self):
        """生成随机地图，确保起点和终点可达"""
        while True:
            # 初始化网格
            self.grid = np.zeros((self.grid_size, self.grid_size))
            
            # 随机生成障碍物
            total_cells = self.grid_size * self.grid_size
            num_obstacles = int(total_cells * self.obstacle_ratio)
            
            # 排除起点和终点位置
            available_positions = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) != self.start and (i, j) != self.goal:
                        available_positions.append((i, j))
            
            # 随机选择障碍物位置
            if num_obstacles > len(available_positions):
                num_obstacles = len(available_positions)
                
            obstacle_positions = np.random.choice(
                len(available_positions), num_obstacles, replace=False
            )
            
            for idx in obstacle_positions:
                i, j = available_positions[idx]
                self.grid[i, j] = 1  # 1表示障碍物
            
            # 验证起点到终点是否可达
            if self.is_reachable():
                break
    
    def is_reachable(self):
        """使用BFS验证起点到终点是否可达"""
        if self.grid[self.start] == 1 or self.grid[self.goal] == 1:
            return False
            
        visited = set()
        queue = [self.start]
        visited.add(self.start)
        
        # 四方向移动: 上, 右, 下, 左
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
        """曼哈顿距离启发函数（适用于四方向移动）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_min_f_node(self, open_list, f_scores):
        """
        从开放列表中找出f值最小的节点（手动查找，非堆优化）
        这里会有O(n)的比较复杂度
        """
        if not open_list:
            return None
            
        min_node = open_list[0]
        min_f = f_scores[min_node]
        
        # 遍历查找最小f值节点
        for node in open_list[1:]:
            self.comparison_count += 1  # 每次比较计数
            if f_scores[node] < min_f:
                min_f = f_scores[node]
                min_node = node
                
        return min_node
    
    def a_star_search_simple(self):
        """A*算法实现（简单列表版本）"""
        # 重置计数
        self.comparison_count = 0
        self.node_processing_count = 0
        self.list_operations_count = 0
        
        # 定义四方向移动
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # 使用简单列表作为开放列表和关闭列表
        open_list = []  # 简单列表，非优先队列
        closed_set = set()
        
        # 存储每个节点的代价和父节点
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        parent = {self.start: None}
        
        # 初始化起点
        open_list.append(self.start)
        self.list_operations_count += 1
        
        start_time = time.time()
        
        while open_list:
            # 从开放列表中找出f值最小的节点（O(n)操作）
            current = self.find_min_f_node(open_list, f_score)
            
            # 从开放列表中移除当前节点
            open_list.remove(current)
            self.list_operations_count += 1
            
            self.node_processing_count += 1
            
            # 如果到达终点，重构路径
            if current == self.goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                processing_time = time.time() - start_time
                return path[::-1], processing_time
            
            # 将当前节点加入关闭列表
            closed_set.add(current)
            self.list_operations_count += 1
            
            # 探索邻居节点
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # 检查边界和障碍物
                if (nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or 
                    self.grid[nx, ny] == 1):
                    continue
                
                # 如果邻居在关闭列表中，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算临时g值
                tentative_g = g_score[current] + 1
                self.comparison_count += 1  # 比较计数
                
                # 如果邻居不在开放列表中，或者找到更优路径
                if neighbor not in open_list:
                    open_list.append(neighbor)
                    self.list_operations_count += 1
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    parent[neighbor] = current
                elif tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    parent[neighbor] = current
        
        processing_time = time.time() - start_time
        return None, processing_time  # 没有找到路径
    
    def get_complexity_metrics(self):
        """获取复杂度度量指标"""
        return {
            'comparisons': self.comparison_count,
            'nodes_processed': self.node_processing_count,
            'list_operations': self.list_operations_count,
            'total_operations': self.comparison_count + self.node_processing_count + self.list_operations_count
        }
    
    def visualize(self, path=None, complexity_info=None):
        """可视化地图和路径"""
        # 创建颜色映射
        cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # 创建显示用的网格
        display_grid = self.grid.copy()
        
        # 标记起点和终点
        display_grid[self.start] = 2  # 绿色起点
        display_grid[self.goal] = 3   # 红色终点
        
        # 标记路径
        if path:
            for pos in path[1:-1]:  # 排除起点和终点
                display_grid[pos] = 4  # 蓝色路径
        
        plt.figure(figsize=(12, 10))
        
        # 主地图
        plt.subplot(1, 1, 1)
        plt.imshow(display_grid, cmap=cmap, norm=norm, interpolation='nearest')
        
        # 添加网格线
        plt.grid(which='both', color='gray', linewidth=0.5)
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        
        # 添加坐标标签
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
        
        title = 'A*算法路径规划结果 (简单列表实现)'
        if complexity_info:
            title += f'\n比较次数: {complexity_info["comparisons"]} | 节点处理: {complexity_info["nodes_processed"]}'
        plt.title(title)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='起点 S'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='终点 G'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='路径'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='障碍物')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 添加复杂度信息文本框
        if complexity_info:
            complexity_text = f"""复杂度统计:
比较次数: {complexity_info['comparisons']}
节点处理数: {complexity_info['nodes_processed']}
列表操作数: {complexity_info['list_operations']}
总操作数: {complexity_info['total_operations']}"""
            
            plt.figtext(0.02, 0.02, complexity_text, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5), 
                       fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    
    # 创建A*规划器（简单列表版本）
    planner = AStarPlannerSimple(grid_size=10, obstacle_ratio=0.25)
    
    print("正在生成随机地图...")
    planner.generate_map()
    
    print("地图生成完成！")
    print(f"起点: {planner.start}")
    print(f"终点: {planner.goal}")
    print(f"障碍物数量: {np.sum(planner.grid)}")
    
    print("\n正在使用A*算法（简单列表版本）寻找最短路径...")
    path, processing_time = planner.a_star_search_simple()
    
    # 获取复杂度信息
    complexity_info = planner.get_complexity_metrics()
    
    if path:
        print(f"找到路径！路径长度: {len(path)} 步")
        print(f"处理时间: {processing_time:.4f} 秒")
        print(f"路径坐标: {path}")
        
        # 计算路径长度
        path_length = len(path) - 1  # 步数 = 路径点数量 - 1
        print(f"从起点到终点需要 {path_length} 步")
    else:
        print("未找到路径！")
    
    # 输出复杂度信息
    print("\n=== 复杂度分析 ===")
    print(f"比较次数: {complexity_info['comparisons']}")
    print(f"节点处理次数: {complexity_info['nodes_processed']}")
    print(f"列表操作次数: {complexity_info['list_operations']}")
    print(f"总操作数: {complexity_info['total_operations']}")
    print(f"处理时间: {processing_time:.4f} 秒")
    
    # 可视化结果
    planner.visualize(path, complexity_info)

if __name__ == "__main__":
    main()