# ===============================
#  Session 14 – A* Algorithm
# ==============================

import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):

    open_list = []

    heapq.heappush(open_list,(0, start))

    came_from = {}

    g_cost = {start: 0}

    directions = [(0, 1),(1, 0),(0, -1),(-1, 0)]

    while open_list:

        current = heapq.heappop(open_list)[1]

        if current == goal:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[current]

            path.append(start)

            return path[::-1]

        for d in directions:

            neighbour = (current[0] + d[0],current[1] + d[1])

            if (0 <= neighbour[0] < len(grid) and
                0 <= neighbour[1] < len(grid[0]) and
                grid[neighbour[0]][neighbour[1]] == 0
            ):

                new_cost = (g_cost[current] + 1)

                if (neighbour not in g_cost or new_cost < g_cost[neighbour]):

                    g_cost[neighbour] = new_cost

                    f_cost = (new_cost+heuristic(neighbour,goal))

                    heapq.heappush(open_list,(f_cost,neighbour))

                    came_from[neighbour] = current


grid = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]

start = (0,0)

goal = (3,3)

print("Optimal Path:",astar(grid,start,goal))