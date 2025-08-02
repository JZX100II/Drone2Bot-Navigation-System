import numpy as np
import matplotlib.pyplot as plt

def load_points(file_path):
    map_points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        step = max(1, len(lines) // 1000)
        for i, line in enumerate(lines):
            if i % step == 0:
                x, y = map(float, line.strip().split(','))
                map_points.append((x, y))
    return np.array(map_points)

def find_doors(points, axis='horizontal', gap_min=0.65, gap_max=0.9, step=0.05, tolerance=0.05):
    doors = []
    axis_idx = 1 if axis == 'horizontal' else 0  # y for horizontal, x for vertical
    sweep_range = np.arange(np.min(points[:, axis_idx]), np.max(points[:, axis_idx]), step)

    for val in sweep_range:
        # Select points close to this line (within tolerance)
        mask = np.abs(points[:, axis_idx] - val) < tolerance
        line_points = points[mask]

        if len(line_points) < 2:
            continue

        # Sort along the other axis (x if horizontal, y if vertical)
        other_idx = 1 - axis_idx
        sorted_pts = line_points[np.argsort(line_points[:, other_idx])]
        coords = sorted_pts[:, other_idx]
        gaps = np.diff(coords)

        for i, gap in enumerate(gaps):
            if gap_min <= gap <= gap_max:
                pt1 = sorted_pts[i]
                pt2 = sorted_pts[i + 1]
                midpoint = (pt1 + pt2) / 2
                orientation = 0 if axis == 'horizontal' else np.pi / 2
                doors.append((midpoint[0], midpoint[1], orientation))

    return doors

if __name__ == '__main__':
    file_path = r"C:\Robotic Project\Another One\controllers\Burger\temps\map_lidar.txt"
    points = load_points(file_path)
    print(points[:1])

    # Detect doors along both axes
    doors_h = find_doors(points, axis='horizontal')
    doors_v = find_doors(points, axis='vertical')
    doors = doors_h + doors_v

    print(f"Detected doors at: {doors}")

    plt.figure(figsize=(10, 8))
    plt.plot(points[:, 0], points[:, 1], 'b.', label='Map Points')
    for i, (door_x, door_y, angle) in enumerate(doors):
        plt.plot(door_x, door_y, 'ro', label='Door' if i == 0 else "")
        dx = 0.4 * np.cos(angle)
        dy = 0.4 * np.sin(angle)
        plt.plot([door_x - dx, door_x + dx], [door_y - dy, door_y + dy], 'r-')
    plt.title("Door Detection in Lidar Map")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()