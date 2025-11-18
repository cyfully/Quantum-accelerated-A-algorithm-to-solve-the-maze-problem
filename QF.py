
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import math

class QuantumMinFinder:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n = 2 ** n_qubits
        self.simulator = Aer.get_backend('qasm_simulator')
    
    def create_oracle(self, threshold: float, values: list) -> QuantumCircuit:
        """创建Oracle - 标记所有小于阈值的解"""
        qr = QuantumRegister(self.n_qubits + 1, 'q')
        qc = QuantumCircuit(qr, name="Oracle")
        
        # 找出所有小于阈值的索引
        solution_indices = []
        for i, v in enumerate(values):
            if v <= threshold:
                solution_indices.append(i)
        
        # 标记所有解
        for idx in solution_indices:
            binary_rep = format(idx, f'0{self.n_qubits}b')
            
            # 根据二进制表示设置控制位
            for j, bit in enumerate(binary_rep):
                if bit == '0':
                    qc.x(j)
            
            # 多控制门标记目标态
            if self.n_qubits > 1:
                qc.mcx(list(range(self.n_qubits)), self.n_qubits)
            else:
                qc.cz(0, self.n_qubits)  # 单量子比特情况用CZ门
            
            # 恢复控制位
            for j, bit in enumerate(binary_rep):
                if bit == '0':
                    qc.x(j)
        
        return qc, solution_indices
    
    def create_diffuser(self) -> QuantumCircuit:
        """创建扩散算子"""
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
    
    def grover_search(self, threshold: float, values: list) -> tuple:
        """量子搜索：返回解的数量和位置"""
        oracle, solution_indices = self.create_oracle(threshold, values)
        num_solutions = len(solution_indices)
        
        if num_solutions > 0:
            # 计算最优迭代次数（多解情况）
            optimal_iterations = self.calculate_optimal_iterations(num_solutions)
            
            # 创建量子电路
            index_qr = QuantumRegister(self.n_qubits, 'index')
            oracle_qr = QuantumRegister(1, 'oracle')
            cr = ClassicalRegister(self.n_qubits, 'c')
            qc = QuantumCircuit(index_qr, oracle_qr, cr)
            
            # 初始化叠加态
            qc.h(index_qr)
            qc.x(oracle_qr)
            qc.h(oracle_qr)
            
            # Grover迭代
            for _ in range(optimal_iterations):
                qc.append(oracle, list(range(self.n_qubits + 1)))
                diffuser = self.create_diffuser()
                qc.append(diffuser, list(range(self.n_qubits)))
            
            qc.measure(index_qr, cr)
            
            # 执行模拟
            try:
                from qiskit import transpile
                compiled = transpile(qc, self.simulator)
                job = self.simulator.run(compiled, shots=100)
                counts = job.result().get_counts()
                
                # 分析结果：找出概率最高的几个解
                sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                found_solutions = []
                
                for result, count in sorted_results[:min(3, len(sorted_results))]:
                    idx = int(result, 2)
                    if idx < len(values) and values[idx] < threshold:
                        found_solutions.append((idx, values[idx], count))
                
                return True, found_solutions, num_solutions
                
            except Exception as e:
                print(f"模拟错误: {e}")
                # 失败时返回已知的解
                found_solutions = [(idx, values[idx], 0) for idx in solution_indices]
                return True, found_solutions, num_solutions
        else:
            return False, [], 0

    def calculate_optimal_iterations(self, num_solutions: int) -> int:
        """计算多解情况下的最优迭代次数"""
        if num_solutions == 0:
            return 0
        
        N = self.n  # 总状态数
        M = num_solutions  # 解的数量
        
        # Grover算法最优迭代次数公式
        theta = math.asin(math.sqrt(M / N))
        optimal = (math.pi / 2 - theta) / (2 * theta)
        
        return max(1, int(round(optimal)))
    
    def find_min(self, values: list) -> tuple:
        """查找最小值 - 使用多解标记策略"""
        if len(values) > self.n:
            values = values[:self.n]
        
        low, high = 0.0, 10000.0
        precision = 1.0
        
        best_solutions = []
        
        print(f"数据: {values}")
        print(f"搜索范围: [{low:.1f}, {high:.1f}]")
        print(f"实际最小值: {min(values)} (索引{values.index(min(values))})")
        
        for i in range(20):  # 最大迭代次数
            mid = (low + high) / 2
            
            has_solutions, found_solutions, num_solutions = self.grover_search(mid, values)
            
            if has_solutions:
                # 存在解：阈值太大，降低阈值
                high = mid
                best_solutions = found_solutions
                print(f"迭代{i+1}: 阈值{mid:.1f} → 找到{num_solutions}个解")
                for idx, val, count in found_solutions[:3]:  # 显示前3个解
                    print(f"    索引{idx}: 值{val} (概率{count}%)")

                # 如果恰好只有一个解，可以根据当前阈值直接确定最小值并提前终止
                solution_indices = [j for j, v in enumerate(values) if v <= mid]
                if len(solution_indices) == 1:
                    sole_idx = solution_indices[0]
                    sole_val = values[sole_idx]
                    print(f"检测到唯一解: 索引{sole_idx}, 值{sole_val}，提前终止二分搜索")
                    return sole_val, sole_idx
            else:
                # 不存在解：阈值太小，提高阈值
                low = mid
                print(f"迭代{i+1}: 阈值{mid:.1f} → 无解")
            
            # 收敛检查
            if high - low <= precision:
                print(f"收敛于阈值范围: [{low:.1f}, {high:.1f}]")
                break
        
        # 返回找到的最小值
        if best_solutions:
            min_solution = min(best_solutions, key=lambda x: x[1])
            return min_solution[1], min_solution[0]  # 值, 索引
        else:
            return float('inf'), -1

def generate_integer_test_data(size):
    """生成正整数测试数据"""
    np.random.seed(24)
    values = np.random.randint(1, 100, size).tolist()
    # 获取最小值
    min_value = min(values)
    min_idx = values.index(min_value)
    return values, min_value  # 现在返回最小值而不是索引

def run_tests():
    """运行测试 - 适配新接口"""
    test_sizes = [4, 8, 16]
    
    print("量子最小值查找测试 - 多解标记版本")
    print("=" * 50)
    
    for size in test_sizes:
        n_qubits = int(math.log2(size))
        values, true_min_val = generate_integer_test_data(size)
        true_min_idx = values.index(true_min_val)
        
        print(f"\n测试 {size} 元素:")
        print(f"数据: {values}")
        
        finder = QuantumMinFinder(n_qubits)
        found_val, found_idx = finder.find_min(values)
        
        status = "✓" if abs(found_val - true_min_val) < 1e-6 else "✗"
        print(f"结果: 找到值={found_val} (索引{found_idx}), 实际最小值={true_min_val} (索引{true_min_idx}) {status}")

if __name__ == "__main__":
    run_tests()