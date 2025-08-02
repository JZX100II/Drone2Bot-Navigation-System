import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

class AStarPlanner:
    def __init__(self, grid_res=0.1, obstacle_thresh=5, inflation_radius=1):
        self.grid_res = grid_res
        self.obstacle_thresh = obstacle_thresh
        self.inflation_radius = inflation_radius
        self.grid = None
        self.min_x = None
        self.min_y = None
        self.points = None

    def load_points(self, file_path):
        map_points = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    x, y = map(float, line.strip().split(','))
                    map_points.append((x, y))
                except:
                    continue
        self.points = np.array(map_points)
        return self.points

    def build_grid(self, points):
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = int(np.ceil((max_x - min_x) / self.grid_res)) + 1
        height = int(np.ceil((max_y - min_y) / self.grid_res)) + 1

        grid = np.zeros((height, width), dtype=np.uint8)
        count = {}

        for x, y in points:
            col = int((x - min_x) / self.grid_res)
            row = int((y - min_y) / self.grid_res)
            count[(row, col)] = count.get((row, col), 0) + 1

        for (row, col), c in count.items():
            if c >= self.obstacle_thresh:
                if 0 <= row < height and 0 <= col < width:
                    grid[row, col] = 1

        temp_grid = grid.copy()
        for row in range(height):
            for col in range(width):
                if temp_grid[row, col] == 1:
                    for i in range(max(0, row - self.inflation_radius), min(height, row + self.inflation_radius + 1)):
                        for j in range(max(0, col - self.inflation_radius), min(width, col + self.inflation_radius + 1)):
                            grid[i, j] = 1

        self.grid = grid
        self.min_x = min_x
        self.min_y = min_y
        return grid, min_x, min_y

    def real_to_grid(self, pos):
        return (
            int((pos[1] - self.min_y) / self.grid_res),
            int((pos[0] - self.min_x) / self.grid_res),
        )

    def grid_to_real(self, row, col):
        return (col * self.grid_res + self.min_x, row * self.grid_res + self.min_y)

    def heuristic(self, a, b):
        base_cost = np.linalg.norm(np.array(a) - np.array(b))
        h, w = self.grid.shape
        min_dist = float('inf')
        for i in range(max(0, a[0] - 5), min(h, a[0] + 6)):
            for j in range(max(0, a[1] - 5), min(w, a[1] + 6)):
                if self.grid[i, j] == 1:
                    dist = np.linalg.norm(np.array(a) - np.array([i, j]))
                    min_dist = min(min_dist, dist)
        penalty = 0
        if min_dist <= 5:
            penalty = min(25, (5 - min_dist) * 2)
        return max(base_cost + penalty, 0.1)

    def a_star(self, start_real, goal_real):
        start = self.real_to_grid(start_real)
        goal = self.real_to_grid(goal_real)

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        h, w = self.grid.shape
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        open_set = PriorityQueue()
        open_set.put((f_score[start], start))

        visited = set()

        while not open_set.empty():
            current = open_set.get()[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            visited.add(current)

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < h and 0 <= ny < w and self.grid[nx, ny] == 0:
                    neighbor = (nx, ny)
                    if neighbor in visited:
                        continue
                    tentative_g = g_score[current] + self.heuristic(current, neighbor)
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        open_set.put((f_score[neighbor], neighbor))

        return None

    def draw_path(self, path, start_real, goal_real):
        if path is None or self.points is None:
            print("No path or map loaded.")
            return

        plt.figure(figsize=(10, 8))
        plt.plot(self.points[:, 0], self.points[:, 1], 'k.', label='LIDAR Points', alpha=0.3)

        path_coords = [self.grid_to_real(row, col) for row, col in path]
        px, py = zip(*path_coords)
        plt.plot(px, py, 'g-', linewidth=2, label='Path')

        plt.plot(*start_real, 'ro', markersize=8, label='Start')
        plt.plot(*goal_real, 'bo', markersize=8, label='Goal')

        plt.legend()
        plt.title("A* Path on LIDAR Map")
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def visualize_grid(self, grid, start, goal, path, min_x, min_y, res):
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='binary', origin='lower', extent=[min_x, min_x + grid.shape[1]*res, min_y, min_y + grid.shape[0]*res])
        
        if path:
            path_coords = [(col * res + min_x, row * res + min_y) for row, col in path]
            px, py = zip(*path_coords)
            plt.plot(px, py, 'g-', linewidth=2, label='Path')
        
        plt.plot(start[1] * res + min_x, start[0] * res + min_y, 'ro', markersize=8, label='Start')
        plt.plot(goal[1] * res + min_x, goal[0] * res + min_y, 'bo', markersize=8, label='Goal')

        plt.title("Occupancy Grid with Path")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

start_real = (-4.94, -8.22)
goal_real = (-2.03, -8.22)

# start_real = (-2.0, -8.23)
# goal_real = (-0.91, -6.7)

planner = AStarPlanner()
points = planner.load_points(r"C:\Robotics Final Project Git In Progress\controllers\mavic_lidar_map\map_lidar_2m_full_map_final_version.txt")
# points = planner.load_points(r"C:\Robotics Final Project Git In Progress\controllers\Map Points\map_lidar_mavic.txt")
grid, min_x, min_y = planner.build_grid(points)
path = planner.a_star(start_real, goal_real)

GRID_RES = 0.1
start = planner.real_to_grid(start_real)
goal = planner.real_to_grid(goal_real)
planner.visualize_grid(grid, start, goal, path, min_x, min_y, GRID_RES)
# planner.draw_path(path, start_real, goal_real)