import numpy as np

class Node:
    def __init__(self, x, y, cost, bound, level):
        self.x = x
        self.y = y
        self.cost = cost
        self.bound = bound
        self.level = level

def branch_and_bound(x, y):
    n = len(x)
    best_x = None
    best_cost = float('inf')
    q = [Node(np.zeros(n), y, 0, 0, -1)]
    while q:
        node = q.pop()
        if node.level == n - 1:
            if node.cost < best_cost:
                best_x = node.x
                best_cost = node.cost
        else:
            left_x = node.x.copy()
            left_x[node.level + 1] = 0
            left_cost = node.cost - node.y[node.level] * left_x[node.level]
            left_bound = left_cost
            for j in range(node.level + 1, n):
                if left_x[j] == 0:
                    left_bound -= node.y[j]
            if left_bound < best_cost:
                q.append(Node(left_x, node.y, left_cost, left_bound, node.level + 1))
            right_x = node.x.copy()
            right_x[node.level + 1] = 1
            right_cost = node.cost + node.y[node.level] * right_x[node.level]
            right_bound = right_cost
            for j in range(node.level + 1, n):
                if right_x[j] == 0:
                    right_bound -= node.y[j]
            if right_bound < best_cost:
                q.append(Node(right_x, node.y, right_cost, right_bound, node.level + 1))
    return best_x, best_cost
