import cv2
import sys
import math
import numpy as np
from controller import Robot
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
from filterpy.kalman import ExtendedKalmanFilter
from Detect_Door_Image_Processing import detect_door_white

QR_POSITIONS = {
    '003': np.array([0.0, -7.5]), # door1
    "005": np.array([-6.97, -7.24]), # door3
    "001": np.array([-6.24, -8.43]), # door
    "007": np.array([-4.94, -8.43]), # door7
    "002": np.array([-2.15, -8.43]) # door2
}

map_points = []

def load_points(file_path):
    global map_points
    map_points = []
    try:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                x, y = map(float, line.strip().split(','))
                map_points.append([x, y])
        print(f"Loaded {len(map_points)} points from {file_path}")
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
    except ValueError:
        print(f"⚠️ Invalid data format in {file_path}")

def initialize_ekf():
    ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)  # State: [x, y, theta], Measurement: [dx, dy]
    ekf.x = np.array([0.0, 0.0, 0.0])  # Initial state: [x, y, theta]
    ekf.P = np.diag([0.5, 0.5, np.deg2rad(5.0)])  # Initial covariance
    ekf.R = np.diag([0.1, 0.1])  # Measurement noise for [dx, dy]
    ekf.Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])  # Process noise
    return ekf

def h(x, landmark):
    dx = landmark[0] - x[0]
    dy = landmark[1] - x[1]
    return np.array([dx, dy])  # Only position differences

def H_jacobian(x, landmark):
    return np.array([
        [-1, 0, 0],
        [0, -1, 0]
    ])  # 2x3 matrix, no yaw dependence

class TurtleBot3(Robot):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        print(f"Time step: {self.time_step} ms")

        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()

        self.front_lidar = self.getDevice("lidar(1)") # this one is only used to find out where the qr code is
        self.front_lidar.enable(self.time_step)
        self.front_lidar.enablePointCloud()

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_encoder = self.getDevice("left wheel sensor")
        self.right_encoder = self.getDevice("right wheel sensor")
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)

        self.pitch_motor = self.getDevice("camera_pitch_motor")

        self.kf = initialize_ekf()
        self.prev_left_pos = 0.0
        self.prev_right_pos = 0.0
        self.pose = [0.0, 0.0, 0.0]
        # self.pose = [0.0, 0.0, math.radians(-180.0)]
        self.qr_detected = False
        self.target_index = 0
        self.current_qr = None
        self.know_rough_location = False
        self.LIDAR_MAX_RANGE = 6
        self.LIDAR_MAX_RANGE_FRONT = 3.5
        self.target_goal = None
        self.called_waypoint = False
        self.qr_code_correction = False
        self.go_straight = False
        self.out_of_room = False
        self.known_door = None
        self.room_center = None
        self.room_centered = False
        self.found_door_img_processing = False
        self.need_to_rotate = False

        self.map = []
        self.front_lidar_map = []

    def update_odometry(self):
        # --- Encoder readings ---
        left_pos = self.left_encoder.getValue()
        right_pos = self.right_encoder.getValue()
        dl = left_pos - self.prev_left_pos
        dr = right_pos - self.prev_right_pos
        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos

        wheel_radius = 0.033
        base_width = 0.16
        ds = wheel_radius * (dl + dr) / 2.0

        # --- Get yaw from InertialUnit ---
        imu_yaw = self.imu.getRollPitchYaw()[2]  # Already in radians

        # Optional smoothing with encoder-based yaw
        # encoder_dtheta = wheel_radius * (dr - dl) / base_width
        # encoder_yaw = self.pose[2] + encoder_dtheta
        # alpha = 0.1  # Use more IMU than encoder
        # self.pose[2] = alpha * encoder_yaw + (1 - alpha) * imu_yaw

        # Use pure IMU yaw (recommended in Webots)
        self.pose[2] = imu_yaw

        # Normalize yaw to [-pi, pi]
        self.pose[2] = math.atan2(math.sin(self.pose[2]), math.cos(self.pose[2]))

        # --- Position update ---
        self.pose[0] += ds * math.cos(self.pose[2])
        self.pose[1] += ds * math.sin(self.pose[2])

        # --- Update Kalman filter (if used) ---
        self.kf.x = np.array(self.pose)

    def update_map(self):
        curr_pose = self.kf.x[0], self.kf.x[1]
        yaw = self.imu.getRollPitchYaw()[2]
        points = self.lidar.getPointCloud()
        for point in points:
            if not np.isfinite(point.x) or not np.isfinite(point.y):
                continue
            
            world_x = curr_pose[0] + point.x * math.cos(yaw) - point.y * math.sin(yaw)
            world_y = curr_pose[1] + point.x * math.sin(yaw) + point.y * math.cos(yaw)
            dist = np.sqrt(point.x**2 + point.y**2) 
            if not np.isfinite(dist) or dist > self.LIDAR_MAX_RANGE:
                continue  # Skip points beyond max range or invalid
            self.map.append((world_x, world_y))  # Append valid points
    
    # **************************************************************************************************************************************
    # dude i can't accept that, lidar and front lidar are picking very two different data, they're not even similar and the door detected in front lidar is pretty far from what it should be
    # this front_lidar_map data is so curvy i don't get it
    def update_front_lidar_map(self):
        curr_pose = self.kf.x[0], self.kf.x[1]
        yaw = self.imu.getRollPitchYaw()[2]
        points = self.front_lidar.getPointCloud()
        for point in points:
            if not np.isfinite(point.x) or not np.isfinite(point.y):
                continue
            
            world_x = curr_pose[0] + point.x * math.cos(yaw) - point.y * math.sin(yaw)
            world_y = curr_pose[1] + point.x * math.sin(yaw) + point.y * math.cos(yaw)

            # world_x = curr_pose[0] + point.x
            # world_y = curr_pose[1] + point.y

            dist = np.sqrt(point.x**2 + point.y**2) 
            if not np.isfinite(dist) or dist > self.LIDAR_MAX_RANGE_FRONT:
                continue  # Skip points beyond max range or invalid
            self.front_lidar_map.append((world_x, world_y))  # Append valid points

    def find_doors(self, points):
            doors_h = self.find_door(points, axis='horizontal')
            doors_v = self.find_door(points, axis='vertical')
            doors = doors_h + doors_v

            # print(f"Detected doors at: {doors}")

            # if self.getTime() - self.plt_timer > 0.5:
            #     self.plt_timer = self.getTime()

            #     plt.figure(figsize=(10, 8))
            #     plt.plot(points[:, 0], points[:, 1], 'b.', label='Map Points')
            #     for i, (door_x, door_y, angle) in enumerate(doors):
            #         plt.plot(door_x, door_y, 'ro', label='Door' if i == 0 else "")
            #         dx = 0.4 * np.cos(angle)
            #         dy = 0.4 * np.sin(angle)
            #         plt.plot([door_x - dx, door_x + dx], [door_y - dy, door_y + dy], 'r-')
            #     plt.title("Door Detection in Lidar Map")
            #     plt.legend()
            #     plt.axis('equal')
            #     plt.grid(True)
                
            #     plt.show()

            return doors

    def find_door(self, points, axis='horizontal', gap_min=0.91, gap_max=0.93, step=0.05, tolerance=0.05):
    # def find_door(self, points, axis='horizontal', gap_min=0.87, gap_max=0.93, step=0.05, tolerance=0.05):
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

    # this should be actually the other way around, we should pick the min distance
    def choose_door(self, doors):
        # min_distance = float('inf')
        max_distance = -1
        chosen_door = None

        for door in doors:
            door_dist = (door[0] - self.pose[0]) ** 2 + (door[1] - self.pose[1]) ** 2
            if door_dist > max_distance:
                max_distance = door_dist
                chosen_door = door
        
        return chosen_door

    def find_first_rough_position(self, qr_data):
        # the only problem is that we find a wrong door that is not the one with the qr code on it
        if self.known_door:
            # i mean since self.known_door is not that accurate, this thing got some minor issues
            x_diff = self.known_door[0] - self.pose[0]
            y_diff = self.known_door[1] - self.pose[1]
            print(f"using self.known_door, x_diff: {x_diff}, y_diff: {y_diff}")

        # this else won't ever happen cause we're starting in a room
        else:
            x_diff = 5.9
            y_diff = 0.1

        print("finding first location roughly")
        if not self.know_rough_location:
            if qr_data in QR_POSITIONS:
                qr_code_pos = QR_POSITIONS[qr_data]
                print(qr_code_pos)
                self.pose[0] = qr_code_pos[0] - x_diff
                self.pose[1] = qr_code_pos[1] - y_diff
                
                self.kf.x[0] = self.pose[0]
                self.kf.x[1] = self.pose[1]
                print("Updated first location: ", self.pose[0], self.pose[1])
                self.know_rough_location = True

    def scan_qr_codes(self):
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        raw = self.camera.getImage()
        image = np.frombuffer(raw, np.uint8).reshape((h, w, 4))[:, :, :3]

        decoded_objects = decode(image)
        if not decoded_objects:
            if self.qr_code_correction:
                self.qr_code_correction = False
                print("going straight and starting out of frame timer")
                self.go_straight = True
                self.out_of_frame_timer = self.getTime()
                # after the qr code goes out of frame we will be going straight and after some amount of time, we will reach the center of the door, so, when we reached the center of the door we can actually update our position
            return

        obj = decoded_objects[0]
        self.current_qr = obj.data.decode('utf-8')
        print(f"scanned qr code data: {self.current_qr}")

        pts = np.array(obj.polygon, np.int32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w_rect, h_rect), angle = rect

        # Reversed signs to correct steering direction
        center_error = -(cx - w / 2) / (w / 2)
        yaw_error = -math.radians(angle)

        print(f'center offset: {center_error}, angle: {yaw_error}')

        # Detect if robot is already centered
        is_centered = abs(center_error) < 0.05
        # is_qr_large = max(w_rect, h_rect) > 0.5 * w
        is_qr_large = max(w_rect, h_rect) > 0.6 * w
        if is_centered and is_qr_large:
            yaw_error = 0.0  # Stop rotating when well aligned

        # Pretty sure yaw error and center both have the same effect, so found out these two yaw and center error are fighting against each other
        # Compute steering (now directionally correct)
        kp_yaw, kp_c = 0.6, 0.8
        # steer = kp_yaw * yaw_error + kp_c * center_error
        steer = kp_c * center_error
        steer = max(-2.0, min(2.0, steer))

        print(f'steer: {steer}')

        # Activate flag
        self.qr_code_correction = True

        # Apply motor commands
        base = 3.0
        left = base - steer
        right = base + steer
        max_vel = self.left_motor.getMaxVelocity()
        left = max(-max_vel, min(max_vel, left))
        right = max(-max_vel, min(max_vel, right))

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def update_kalman_with_qr(self, qr_data):
        if qr_data in QR_POSITIONS:
            true_pos = QR_POSITIONS[qr_data]
            print(f'True position of the qr code: {true_pos}')
            imu_yaw = self.imu.getRollPitchYaw()[2]  # Accurate yaw from IMU
            z_meas = np.array([true_pos[0] - self.kf.x[0], true_pos[1] - self.kf.x[1]])  # Only [dx, dy]
            H = H_jacobian(self.kf.x, true_pos)  # 2x3 Jacobian
            self.kf.update(z_meas, HJacobian=lambda x: H, Hx=lambda x: h(x, true_pos))
            self.kf.x[2] = imu_yaw  # Directly set yaw from IMU
            self.pose = self.kf.x.tolist()
            self.qr_detected = True
            self.prev_left_pos = self.left_encoder.getValue()  # Reset odometry
            self.prev_right_pos = self.right_encoder.getValue()
            print(f"Updated pose with QR {qr_data}: x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, yaw={math.degrees(self.pose[2]):.2f}")

    def go_to_waypoint(self, target_x, target_y, linear_speed=4.0, angular_speed=2.0, threshold=0.1):
        self.called_waypoint = True

        dx = target_x - self.pose[0]
        dy = target_y - self.pose[1]
        distance = math.hypot(dx, dy)
        
        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.pose[2]

        # Normalize yaw error to [-pi, pi]
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        if abs(yaw_error) > 0.1:
            # Rotate in place
            turn_speed = angular_speed if yaw_error > 0 else -angular_speed
            self.left_motor.setVelocity(-turn_speed)
            self.right_motor.setVelocity(turn_speed)
        elif distance > threshold:
            # Move forward
            self.left_motor.setVelocity(linear_speed)
            self.right_motor.setVelocity(linear_speed)
        else:
            # Reached the waypoint
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            self.called_waypoint = False
            return True  # Arrived at goal

        self.called_waypoint = False
        return False  # Still moving

    # when we do this we actually know how far we are from the door_pos, so it's gone be easy to tell our exact location when we know the first rough position
    def align_camera_with_qr_code(self, door_pos):
        """
        Automatically tilts the camera slightly above the QR code to ensure it captures it well.
        :param door_pos: Tuple (x, y) representing the door position.
        """
        # Constants
        CAMERA_HEIGHT = 0.12        # in meters
        QR_CODE_HEIGHT = 1.1        # in meters
        MAX_TILT_UP = -0.7          # in radians (max upward)
        EXTRA_UPWARD_TILT = 1       # in radians (add a bit more upward angle)

        # Get robot pose
        robot_x, robot_y = self.pose[0], self.pose[1]
        dx = door_pos[0] - robot_x
        dy = door_pos[1] - robot_y

        # Horizontal distance
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        dz = QR_CODE_HEIGHT - CAMERA_HEIGHT

        if horizontal_dist < 0.05:
            tilt_angle = MAX_TILT_UP
        else:
            tilt_angle = -math.atan2(dz, horizontal_dist)
            tilt_angle -= EXTRA_UPWARD_TILT  # tilt a bit more up

        # Clamp
        tilt_angle = max(MAX_TILT_UP, tilt_angle)

        # Apply
        self.pitch_motor.setPosition(tilt_angle)
        print(f"Camera tilted to: {tilt_angle:.2f} rad to capture QR code.")
    
    # so this ain't wrong, our finding first rough position is wrong
    def move_from_door(self):
        if self.current_qr in QR_POSITIONS:
            true_pos = QR_POSITIONS[self.current_qr]
            x_diff = true_pos[0] - self.pose[0]
            y_diff = true_pos[1] - self.pose[1] 

            waypoint = [0] * 2
            waypoint[0] = true_pos[0] + x_diff
            waypoint[1] = true_pos[1] + y_diff
            return waypoint

    def go_straight_out_of_the_room(self):
        if self.go_straight:
            speed = 6.0
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(speed)
    
    def reached_door(self):
        if self.current_qr in QR_POSITIONS:
            self.out_of_room = True
            qr_code_pos = QR_POSITIONS[self.current_qr]
            self.pose[0] = qr_code_pos[0]
            # self.pose[1] = qr_code_pos[1] + 0.1
            self.pose[1] = qr_code_pos[1]
            
            self.kf.x[0] = self.pose[0]
            self.kf.x[1] = self.pose[1]

            print(f"Update pose since we reached the door: {self.pose}")

    def find_room_center_bbox(self, points):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        center = (center_x, center_y)

        # plt.figure(figsize=(10, 8))
        # plt.plot(points[:, 0], points[:, 1], 'b.', label='Map Points')
        # plt.plot(center_x, center_y, 'ro', label='Center')
        # plt.title("Center of Room in Lidar Map")
        # plt.legend()
        # plt.axis('equal')
        # plt.grid(True)
        
        # plt.show()

        return center
    
    def rotate(self):
        self.left_motor.setVelocity(-6.0)
        self.right_motor.setVelocity(6.0)

    def go_straight_now(self):
        self.left_motor.setVelocity(6.0)
        self.right_motor.setVelocity(6.0)

    def align_camera_straight(self, neutral_pitch=0.0):
        """
        Aligns the camera to a neutral forward-looking position.
        :param neutral_pitch: Angle in radians that corresponds to straight ahead.
        """
        self.pitch_motor.setPosition(neutral_pitch)
        print(f"Camera aligned to straight ahead ({neutral_pitch:.2f} rad).")
    
    def after_detecting_a_door(self, bbox):
        """
        Aligns the robot to the center of the detected door.
        :param bbox: (x, y, w, h) bounding box of the detected door
        :param image_width: width of the camera image
        """
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        raw = self.camera.getImage()
        image = np.frombuffer(raw, np.uint8).reshape((h, w, 4))[:, :, :3].copy()

        bbox, _, _ = detect_door_white(image, tolerance=0)

        if bbox == None:
            self.need_to_rotate = True
            return False
        else:
            self.need_to_rotate = False

        image_width = self.camera.getWidth()

        x, y, w, h = bbox
        cx = x + w / 2  # center of door in image

        # Compute error (normalized -1..1)
        center_error = -(cx - image_width / 2) / (image_width / 2)

        print(f"[Door Align] door_center: {cx}, center_error: {center_error:.3f}")

        # Detect if robot is already centered
        is_centered = abs(center_error) < 0.05
        is_door_large = w > 0.6 * image_width  # Door takes big chunk of screen

        # if is_centered and is_door_large:
            # print("[Door Align] Robot is centered at door.")
            # self.left_motor.setVelocity(0)
            # self.right_motor.setVelocity(0)
            # return True  # success

        # Steering control (similar to QR)
        kp_c = 0.8
        steer = kp_c * center_error
        steer = max(-2.0, min(2.0, steer))

        print(f"[Door Align] steer: {steer:.3f}")

        # Apply motor commands
        base = 3.0
        left = base - steer
        right = base + steer
        max_vel = self.left_motor.getMaxVelocity()
        left = max(-max_vel, min(max_vel, left))
        right = max(-max_vel, min(max_vel, right))

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

        return False  # still aligning

    def run(self):
        load_points(r"..\Map Points\map_lidar_mavic.txt")
        speed = 6.0
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)
        self.scan_map_timer = self.getTime()
        self.first_timer = self.getTime()
        self.waypoint_timer = self.getTime()
        self.clean_map = self.getTime()
        self.plt_timer = self.getTime()
        self.img_processing = self.getTime()
        self.align_camera_straight()

        # we can put a threshold here that if we couldn't find the qr code going in one direction, then just turn around and go the other direction

        while self.step(self.time_step) != -1:
            self.update_odometry()

            # if self.getTime() - self.clean_map > 3.5:
            if self.getTime() - self.clean_map > 0.5:
                self.map = []
                self.front_lidar_map = []
                self.clean_map = self.getTime()

            # this whole below lines should be only called when we couldn't see a qr code straight up
            if self.getTime() - self.scan_map_timer > 0.5:
                self.update_map()
                self.scan_map_timer = self.getTime()
                # if self.room_center == None:
                self.room_center = self.find_room_center_bbox(np.array(self.map))
                # if not self.target_goal:
                if self.room_centered == False: # in case we didn't already reach the center of the room
                    self.target_goal = self.room_center
                # self.room_centered = False

            if self.target_goal and not self.room_centered and not self.called_waypoint and not self.qr_code_correction and not self.go_straight:
                arrived = self.go_to_waypoint(self.target_goal[0], self.target_goal[1])
                print("Current pose: ", self.pose)
                
                if arrived:
                    print("✅ Reached goal:", self.target_goal)
                    self.room_centered = True
                    self.target_goal = None  # Reset to avoid re-navigation

            if not self.room_centered:
                continue # we should first move to the center of the room then try to find doors and stuffs like that
            
            global bbox

            if self.getTime() - self.img_processing > 0.5 and not self.found_door_img_processing:
                self.rotate()
                self.img_processing = self.getTime()
                w = self.camera.getWidth()
                h = self.camera.getHeight()
                raw = self.camera.getImage()
                image = np.frombuffer(raw, np.uint8).reshape((h, w, 4))[:, :, :3].copy()
                bbox, mask, output = detect_door_white(image, tolerance=0)
                if bbox == None:
                    print("No door detected")
                else:
                    # Detected door
                    # self.update_front_lidar_map()
                    # doors = self.find_doors(np.array(self.front_lidar_map))
                    # self.known_door = self.choose_door(doors)
                    # print(f"after scanning qr code, front lidar senses picked up sth from the front of the robot: {self.known_door}")
                    self.found_door_img_processing = True
                    self.go_straight_now()
                    

            if not self.found_door_img_processing:
                continue

            if self.need_to_rotate:
                self.rotate()

            # if not self.current_qr and not self.need_to_rotate:
            if not self.current_qr:
                self.after_detecting_a_door(bbox)
            
            if self.need_to_rotate:
                continue

            # if self.getTime() - self.first_timer > 5.0:
            if self.getTime() - self.first_timer > 0.5:
                self.first_timer = self.getTime()

                doors = self.find_doors(np.array(self.map))
                
                # if target_goal is already choosen, this will overwrite it
                if not self.target_goal:
                    self.target_goal = self.choose_door(doors)
                    self.known_door = self.target_goal
                if self.target_goal:
                    self.align_camera_with_qr_code(self.target_goal)

            if (self.getTime() - self.waypoint_timer) > 0.5:
                self.waypoint_timer = self.getTime()
                self.scan_qr_codes()

                if self.current_qr:
                    self.update_front_lidar_map()
                    doors = self.find_doors(np.array(self.front_lidar_map))
                    self.known_door = self.choose_door(doors)
                    print(f"after scanning qr code, front lidar senses picked up sth from the front of the robot: {self.known_door}")
                    print("Current pose when front lidar picked up sth: ", self.pose)

                if not self.know_rough_location and self.current_qr:
                    self.find_first_rough_position(self.current_qr)

                if self.know_rough_location and self.current_qr:
                    self.update_kalman_with_qr(self.current_qr)
                    self.target_goal = self.move_from_door()
                    self.called_waypoint = False

            # if self.target_goal and not self.called_waypoint and not self.qr_code_correction and not self.go_straight:
            #     arrived = self.go_to_waypoint(self.target_goal[0], self.target_goal[1])
            #     print("Current pose: ", self.pose)
                
            #     if arrived:
            #         print("✅ Reached goal:", self.target_goal)
            #         self.target_goal = None  # Reset to avoid re-navigation

            self.go_straight_out_of_the_room()
            
            # out of frame = 12.0
            # reached door = 17.0
            
            if hasattr(self, "out_of_frame_timer") and self.getTime() - self.out_of_frame_timer > 4.0 and not self.out_of_room:
                self.reached_door()
                return
    
    def get_best_path(self):
        from path_finder import AStarPlanner

        start_real = (self.pose[0] + 0.15, self.pose[1] + 0.15)
        goal_real = (-2.03, -8.22) # main goal
        # goal_real = (-0.91, -6.7)

        planner = AStarPlanner()
        points = planner.load_points(r"..\Map Points\map_lidar_mavic.txt")
        grid, min_x, min_y = planner.build_grid(points)
        path = planner.a_star(start_real, goal_real)

        GRID_RES = 0.1
        start = planner.real_to_grid(start_real)
        goal = planner.real_to_grid(goal_real)
        # planner.visualize_grid(grid, start, goal, path, min_x, min_y, GRID_RES)
        # planner.draw_path(path, start_real, goal_real)
        
        path_real = [planner.grid_to_real(row, col) for row, col in path]

        print(path_real)
        return path_real

    def plan_path(self):
        checkpoints = self.get_best_path()
        if not checkpoints:
            print("❌ Could not compute path.")
            return

        checkpoint_index = 0
        reached = False

        self.pose_timer = self.getTime()
        
        while self.step(self.time_step) != -1 and checkpoint_index < len(checkpoints):
            self.update_odometry()
            
            target_x, target_y = checkpoints[checkpoint_index]

            if self.getTime() - self.pose_timer > 0.5:
                self.pose_timer = self.getTime()
                print(f"target goal: {target_x}, {target_y}")
                print(f"current pose: {self.pose[0]}, {self.pose[1]}, {self.pose[2]}")
                print(f"checkpoint index: {checkpoint_index}")
                dx = target_x - self.pose[0]
                dy = target_y - self.pose[1]
                distance = math.hypot(dx, dy)
                print(f'distance: {distance}')

            # we can wrap this inside a timer
            # self.go_to_waypoint(target_x, target_y, threshold=0.2)
            self.go_to_waypoint(target_x, target_y, threshold=0.2)

            # pretty sure having large drifts in y values means our yaw is not getting calculated correctly
            if not reached:
                reached = self.go_to_waypoint(target_x, target_y, threshold=0.2)
                if reached:
                    checkpoint_index += 1
                    reached = False

if __name__ == '__main__':
    robot = TurtleBot3()
    robot.run()
    robot.plan_path()