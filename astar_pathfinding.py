"""
A* Pathfinding Algorithm
Description:
    A clean, modular implementation of the A* search algorithm on a 2D grid. Demonstrates heuristics, priority queues, and object-oriented design.
"""

from heapq import heappush, heappop


class Node:
    """A node in the search grid."""
    def __init__(self, x, y, cost=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    @property
    def total_cost(self):
        return self.cost + self.heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(node):
    """Reconstructs the final path from end node."""
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]


def astar_search(grid, start, goal):
    """
    A* algorithm on a grid.
    grid: 2D list with 0 = free and 1 = obstacle
    start: (x, y)
    goal: (x, y)
    """
    rows, cols = len(grid), len(grid[0])
    open_list = []
    closed = set()

    start_node = Node(*start, heuristic=heuristic(start, goal))
    heappush(open_list, start_node)

    while open_list:
        current = heappop(open_list)

        if (current.x, current.y) == goal:
            return reconstruct_path(current)

        closed.add((current.x, current.y))

        # explore neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current.x + dx, current.y + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 1:
                continue
            if (nx, ny) in closed:
                continue

            new_cost = current.cost + 1
            node = Node(
                nx, ny,
                cost=new_cost,
                heuristic=heuristic((nx, ny), goal),
                parent=current
            )
            heappush(open_list, node)

    return None


if __name__ == "__main__":
    grid_map = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0]
    ]

    path = astar_search(grid_map, start=(0, 0), goal=(2, 3))
    print("Shortest path:", path)
