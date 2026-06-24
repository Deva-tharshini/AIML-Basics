# Session 14 – Search Algorithms (BFS, DFS, A*)

## Objective

To understand and implement graph and pathfinding algorithms including Breadth First Search (BFS), Depth First Search (DFS), and A* Search Algorithm for exploring paths and finding optimal solutions.

## Dataset Used

Custom Graph Data Structure

Custom Grid Map for A* Search

No external dataset required.

## Features

### BFS and DFS Graph

Nodes:

* A
* B
* C
* D
* E
* F

Connections:

* A → B, C
* B → D, E
* C → F
* E → F

### A* Search Grid

Grid Values:

* 0 → Valid Path
* 1 → Obstacle

Grid Size:

* 4 × 4

Start Node:

* (0,0)

Goal Node:

* (3,3)

## Operations Performed

### Breadth First Search (BFS)

* Created graph representation
* Traversed level by level
* Maintained queue structure
* Found shortest path

### Depth First Search (DFS)

* Traversed graph using stack
* Explored deeper nodes first
* Generated traversal path

### A* Search Algorithm

* Defined heuristic function
* Calculated cost values
* Explored neighboring nodes
* Generated optimal path
* Avoided obstacles

## Model Outputs

BFS Outputs:

* Traversal Path

DFS Outputs:

* Traversal Path

A* Outputs:

* Optimal Path

## Real-world Applications

* GPS navigation systems
* Robot path planning
* Route optimization
* Maze solving
* Game AI
* Network routing
* Search engines

## Output Files

* console_output.png

## Learning Outcome

Learned how BFS explores level by level, DFS explores depth first, and A* combines path cost and heuristics to efficiently discover optimal paths.

## Technologies Used

* Python
* Heap Queue (heapq)
* Graph Theory
* Search Algorithms
* Artificial Intelligence
