import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib import colors

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定默认字体为支持中文的字体

class AStarPlanner:
    def __init__(self, grid_size=10, obstacle_ratio=0.2):
        """
        初始化A*路径规划器
        
        参数:
            grid_size: 网格大小 (grid_size x grid_size)
            obstacle_ratio: 障碍物所占比例 (0.0-1.0)
        """
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.grid = None
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        
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
    
    def a_star_search(self):
        """A*算法实现"""
        # 定义四方向移动
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # 初始化开放列表（优先队列）和关闭列表
        open_list = []
        closed_set = set()
        
        # 节点信息字典: {位置: (父节点, g值, f值)}
        node_info = {}
        
        # 初始化起点
        heapq.heappush(open_list, (0, self.start))
        node_info[self.start] = (None, 0, self.heuristic(self.start, self.goal))
        
        while open_list:
            # 获取f值最小的节点
            current_f, current_pos = heapq.heappop(open_list)
            
            # 如果到达终点，重构路径
            if current_pos == self.goal:
                path = []
                while current_pos:
                    path.append(current_pos)
                    current_pos = node_info[current_pos][0]
                return path[::-1]
            
            # 将当前节点加入关闭列表
            closed_set.add(current_pos)
            
            # 探索邻居节点
            for dx, dy in directions:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                neighbor = (nx, ny)
                
                # 检查边界和障碍物
                if (nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or 
                    self.grid[nx, ny] == 1):
                    continue
                
                # 如果邻居在关闭列表中，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算新的g值
                tentative_g = node_info[current_pos][1] + 1
                
                # 如果邻居不在开放列表中，或者找到更优路径
                if neighbor not in node_info or tentative_g < node_info[neighbor][1]:
                    g_value = tentative_g
                    h_value = self.heuristic(neighbor, self.goal)
                    f_value = g_value + h_value
                    
                    node_info[neighbor] = (current_pos, g_value, f_value)
                    heapq.heappush(open_list, (f_value, neighbor))
        
        return None  # 没有找到路径
    
    def visualize(self, path=None):
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
        
        plt.figure(figsize=(10, 10))
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
        
        plt.title('A*算法路径规划结果 (10×10网格)')
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
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    
    # 创建A*规划器
    planner = AStarPlanner(grid_size=10, obstacle_ratio=0.25)
    
    print("正在生成随机地图...")
    planner.generate_map()
    
    print("地图生成完成！")
    print(f"起点: {planner.start}")
    print(f"终点: {planner.goal}")
    print(f"障碍物数量: {np.sum(planner.grid)}")
    
    print("\n正在使用A*算法寻找最短路径...")
    path = planner.a_star_search()
    
    if path:
        print(f"找到路径！路径长度: {len(path)} 步")
        print(f"路径坐标: {path}")
        
        # 计算路径长度
        path_length = len(path) - 1  # 步数 = 路径点数量 - 1
        print(f"从起点到终点需要 {path_length} 步")
    else:
        print("未找到路径！")
    
    # 可视化结果
    planner.visualize(path)

if __name__ == "__main__":
    main()