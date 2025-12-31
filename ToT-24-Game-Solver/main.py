import math
from typing import List, Optional, Any, Callable, Dict, Set
from dataclasses import dataclass, field
from collections import deque
import itertools


# ==========================================
# 2.1 核心算法实现
# ==========================================

@dataclass
class ThoughtNode:
    """思维树节点"""
    state: List[Dict[str, Any]]  # 例如: [{'val': 3.0, 'exp': '3'}, {'val': 8.0, 'exp': '8'}]
    parent: Optional['ThoughtNode'] = None
    operation: str = "init"
    depth: int = 0
    score: float = 0.0

    def __hash__(self):
        # 为了去重，我们将状态中的数值排序后作为 hash 依据
        # 注意：这里只hash数值，忽略表达式字符串的差异
        values = tuple(sorted([round(x['val'], 6) for x in self.state]))
        return hash(values)

    def get_expression(self) -> str:
        """回溯获取最终表达式，如果只剩一个元素，直接返回其表达式"""
        if len(self.state) == 1:
            return self.state[0]['exp']
        return ""


class TreeOfThoughts:
    """Tree of Thoughts 实现"""

    def __init__(
            self,
            thought_generator: Callable[[ThoughtNode], List[ThoughtNode]],
            state_evaluator: Callable[[ThoughtNode], float],
            goal_checker: Callable[[ThoughtNode], bool],
            strategy: str = 'bfs'
    ):
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.goal_checker = goal_checker
        self.strategy = strategy
        self.visited: Set[int] = set()

    def search(self, initial_state: Any) -> Optional[ThoughtNode]:
        """执行搜索"""
        root = ThoughtNode(state=initial_state, depth=0)

        if self.strategy == 'bfs':
            return self._bfs(root)
        elif self.strategy == 'dfs':
            return self._dfs(root)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _bfs(self, root: ThoughtNode) -> Optional[ThoughtNode]:
        queue = deque([root])
        self.visited.add(hash(root))

        while queue:
            current_node = queue.popleft()

            # 检查是否达成目标
            if self.goal_checker(current_node):
                return current_node

            # 剪枝/评估：如果评估分数为0（代表不可能或无意义），则跳过
            # 在简单版24点中，除非除数为0等非法操作，通常都可以继续

            # 生成新的思维（下一层状态）
            next_thoughts = self.thought_generator(current_node)

            for thought in next_thoughts:
                h = hash(thought)
                if h not in self.visited:
                    self.visited.add(h)
                    queue.append(thought)

        return None

    def _dfs(self, root: ThoughtNode) -> Optional[ThoughtNode]:
        # 使用栈实现迭代式 DFS
        stack = [root]
        self.visited.add(hash(root))

        while stack:
            current_node = stack.pop()

            if self.goal_checker(current_node):
                return current_node

            # 生成思维
            next_thoughts = self.thought_generator(current_node)

            # 对于DFS，通常反向入栈以保持生成顺序（或者无所谓）
            for thought in next_thoughts:
                h = hash(thought)
                if h not in self.visited:
                    self.visited.add(h)
                    stack.append(thought)
        return None


class Point24Solver:
    """24点求解器"""

    def __init__(self):
        # 初始化 ToT 引擎
        self.tot = TreeOfThoughts(
            thought_generator=self.generate_thoughts,
            state_evaluator=self.evaluate_state,
            goal_checker=self.check_goal,
            strategy='bfs'  # 默认使用 BFS
        )

    def solve(self, numbers: List[int]) -> Optional[str]:
        """
        求解24点
        Args:
            numbers: 4个数字 [1-13]
        Returns:
            表达式字符串，无解返回None
        """
        # 1. 构造初始状态
        # 状态格式: list of dict {'val': float, 'exp': str}
        initial_state = [{'val': float(x), 'exp': str(x)} for x in numbers]

        # 2. 清空缓存
        self.tot.visited.clear()

        # 3. 运行搜索
        result_node = self.tot.search(initial_state)

        if result_node:
            # 找到解，返回表达式，并附加 = 24
            expr = result_node.state[0]['exp']
            return f"{expr} = 24"
        return None

    # --- ToT 必要的组件方法 ---

    def generate_thoughts(self, node: ThoughtNode) -> List[ThoughtNode]:
        """思维生成器：从当前状态生成所有可能的下一步状态"""
        current_list = node.state
        next_nodes = []

        # 如果只剩1个或0个数字，无法继续运算
        if len(current_list) < 2:
            return []

        # 从列表中任取两个数进行运算
        # itertools.combinations 返回不重复的组合索引会有问题，我们需要取索引
        indices = range(len(current_list))
        for i, j in itertools.combinations(indices, 2):
            item1 = current_list[i]
            item2 = current_list[j]

            # 剩下的列表（未参与运算的部分）
            remaining = [current_list[k] for k in indices if k != i and k != j]

            # 尝试四则运算
            # 加法
            next_nodes.append(self._create_node(
                item1['val'] + item2['val'],
                f"({item1['exp']} + {item2['exp']})",
                remaining, node
            ))

            # 乘法
            next_nodes.append(self._create_node(
                item1['val'] * item2['val'],
                f"({item1['exp']} * {item2['exp']})",
                remaining, node
            ))

            # 减法 (注意顺序: a-b 和 b-a)
            next_nodes.append(self._create_node(
                item1['val'] - item2['val'],
                f"({item1['exp']} - {item2['exp']})",
                remaining, node
            ))
            next_nodes.append(self._create_node(
                item2['val'] - item1['val'],
                f"({item2['exp']} - {item1['exp']})",
                remaining, node
            ))

            # 除法 (注意分母不为0)
            if abs(item2['val']) > 1e-6:
                next_nodes.append(self._create_node(
                    item1['val'] / item2['val'],
                    f"({item1['exp']} / {item2['exp']})",
                    remaining, node
                ))
            if abs(item1['val']) > 1e-6:
                next_nodes.append(self._create_node(
                    item2['val'] / item1['val'],
                    f"({item2['exp']} / {item1['exp']})",
                    remaining, node
                ))

        return next_nodes

    def _create_node(self, new_val: float, new_exp: str, remaining: List[Dict], parent: ThoughtNode) -> ThoughtNode:
        """辅助函数：创建新节点"""
        new_state = remaining + [{'val': new_val, 'exp': new_exp}]
        return ThoughtNode(state=new_state, parent=parent, depth=parent.depth + 1)

    def evaluate_state(self, node: ThoughtNode) -> float:
        """简单评估函数"""
        # 如果找到了24，满分
        if self.check_goal(node):
            return 1.0
        return 0.5

    def check_goal(self, node: ThoughtNode) -> bool:
        """目标检测：剩下一个数，且约为24"""
        if len(node.state) == 1:
            return math.isclose(node.state[0]['val'], 24.0, rel_tol=1e-5)
        return False


# 测试
if __name__ == "__main__":
    solver = Point24Solver()

    test_cases = [
        [3, 3, 8, 8],  # 经典: 8/(3-8/3)
        [1, 1, 1, 1],  # 无解
        [1, 2, 3, 4],  # 简单: 1*2*3*4
        [5, 5, 5, 1],  # 分数: 5 * (5 - 1/5)
    ]

    print(f"{'Input':<15} | {'Result'}")
    print("-" * 40)

    for nums in test_cases:
        result = solver.solve(nums)
        print(f"{str(nums):<15} | {result}")