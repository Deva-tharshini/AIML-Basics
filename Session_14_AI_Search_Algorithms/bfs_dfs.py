# =============================
# Session 14 – BFS and DFS
# =============================

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}


def bfs(graph, start, goal):

    visited = []

    queue = [[start]]

    while queue:

        path = queue.pop(0)

        node = path[-1]

        if node == goal:
            return path

        if node not in visited:

            visited.append(node)

            for neighbour in graph[node]:

                new_path = list(path)

                new_path.append(neighbour)

                queue.append(new_path)


def dfs(graph, start, goal):

    visited = []

    stack = [[start]]

    while stack:

        path = stack.pop()

        node = path[-1]

        if node == goal:
            return path

        if node not in visited:

            visited.append(node)

            for neighbour in graph[node]:

                new_path = list(path)

                new_path.append(neighbour)

                stack.append(new_path)


start = "A"

goal = "F"

print("BFS Path:", bfs(graph, start, goal))

print("DFS Path:", dfs(graph, start, goal))