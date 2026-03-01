# Dynamic Pathfinding Agent

A Python + Pygame application that visually demonstrates AI pathfinding
algorithms on an interactive grid with dynamic obstacles.

---

## How to Install & Run

**Step 1 — Install Pygame:**
```
pip install pygame
```

**Step 2 — Run the program:**
```
python pathfinding.py
```

---

## What the Program Does

The agent finds the shortest path from a **Start node (green)** to a
**Goal node (red)** on a grid. You can place walls, generate random mazes,
and watch the agent navigate in real time while new obstacles appear.

---

## Algorithms

### A* Search
- Formula: `f(n) = g(n) + h(n)`
- `g(n)` = cost from start to current node (steps already taken)
- `h(n)` = estimated cost to goal (heuristic)
- Always finds the **shortest path**

### Greedy Best-First Search (GBFS)
- Formula: `f(n) = h(n)`
- Only uses the heuristic, ignores how far it has traveled
- **Faster** than A* but may not find the shortest path

---

## Heuristics

### Manhattan Distance
```
distance = |row1 - row2| + |col1 - col2|
```
Best for grid movement (up/down/left/right only).

### Euclidean Distance
```
distance = sqrt((row1-row2)^2 + (col1-col2)^2)
```
Straight-line distance between two points.

---

## How to Use

### Grid Controls
| Action | How |
|--------|-----|
| Draw a wall | Select "Wall" mode → click or drag on grid |
| Erase a wall | Click an existing wall to remove it |
| Move start | Select "Start" mode → click any empty cell |
| Move goal | Select "Goal" mode → click any empty cell |

### Buttons
| Button | What it does |
|--------|--------------|
| RUN SEARCH | Runs the algorithm and animates the search |
| Clear Path | Removes path display, keeps walls |
| Reset Grid | Clears everything |
| Apply Grid Size | Resizes grid using the Rows/Cols you typed |
| Generate Random Map | Creates a random maze using obstacle density |
| Dynamic Mode ON/OFF | Enables walls spawning while agent moves |
| START TRAVERSAL | Agent walks the path step by step |

### Input Fields
| Field | Description |
|-------|-------------|
| Rows / Cols | Grid size (e.g. 20 x 20) |
| Obstacle Density | 0.0 = empty, 1.0 = full (try 0.30) |
| Spawn Chance | Chance of new wall per step (e.g. 0.03 = 3%) |
| Agent Speed | ms between each step (lower number = faster) |

---

## Color Guide

| Color | Meaning |
|-------|---------|
| Green | Start node |
| Red (dark) | Goal node |
| Orange | Moving agent |
| Cyan | Final path |
| Blue/Purple | Visited (explored) nodes |
| Yellow | Frontier (nodes in queue) |
| Dark grey | Static wall |
| Bright red | Dynamic wall (spawned during run) |

---

## Dynamic Mode

1. Enable **Dynamic Mode**
2. Set a **Spawn Chance** (e.g. 0.03)
3. Click **START TRAVERSAL**

New walls appear randomly while the agent moves. If a wall blocks the
remaining path, the agent **replans immediately** from its current position.
If the wall is not on the path, the agent keeps going without replanning.

---

## Stats Panel

| Stat | Meaning |
|------|---------|
| Algorithm | A* or GBFS |
| Heuristic | Manhattan or Euclidean |
| Nodes Visited | Total cells explored |
| Path Length | Number of steps in final path |
| Time (ms) | How long the search took |
| Re-plans | How many times agent had to recalculate |

---

## Files

```
pathfinding.py   ← run this file
README.md        ← this guide
```
