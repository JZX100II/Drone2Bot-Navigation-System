# ok there's only one thing wrong with my map, the whole drone is in the map also, which is quite wrong
# also you gotta give the lidar to fully do a 360, cause most of the time, it can't get the whole thing and some parts are missing
# we can scan qr codes when we first got the full map
# instead of running it again, now we have all the points in map_lidar_30.txt, write sth that visualize those, also make sure kitchen is taken


# ****************************************************************************************************
# increase lidar range so that we can have most parts of the map and not be limited to some parts only

import sys
import math
from controller import Robot
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pyzbar.pyzbar import decode

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

class Mavic(Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.
    MAX_YAW_DISTURBANCE = 5
    MAX_PITCH_DISTURBANCE = -1
    TARGET_PRECISION = 0.5
    LIDAR_MAX_RANGE = 1.0
    # New constants for speed limiting
    MAX_SPEED = 0.5  # Maximum speed in meters/second
    SPEED_GAIN = 0.1  # Proportional gain for speed adjustment
    MIN_DISTANCE = 0.2  # Minimum distance to start slowing down

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        print(f"Time step: {self.time_step} ms")

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        
        # the red axis is the head of lidar
        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()
        
        self.cameraRollMotor = self.getDevice("camera roll")
        self.cameraPitchMotor = self.getDevice("camera pitch")

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0
        self.map = []
        self.qr_codes = []
        self.just_reached = False  # Flag to track if waypoint was just reached
        self.previous_position = [0, 0, 0]  # To calculate velocity
        self.current_velocity = [0, 0, 0]   # To store velocity

    def set_position(self, pos):
        self.current_pose = pos
    
    def update_map(self):
        curr_pose = self.gps.getValues()
        yaw = self.imu.getRollPitchYaw()[2]
        points = self.lidar.getPointCloud()
        for point in points:
            if not np.isfinite(point.x) or not np.isfinite(point.y):
                continue
            
            # ***********************************************************************************************
            # make sure the yaw and stuffs like that is correct since last time this was working with point.z
            world_x = curr_pose[0] + point.x * math.cos(yaw) - point.y * math.sin(yaw)
            world_y = curr_pose[1] + point.x * math.sin(yaw) + point.y * math.cos(yaw)
            dist = np.sqrt(point.x**2 + point.y**2)
            if not np.isfinite(dist) or dist > self.LIDAR_MAX_RANGE:
                continue  # Skip points beyond max range or invalid
            self.map.append((world_x, world_y))  # Append valid points

        with open(f'map_lidar_{self.target_index}.txt', 'w') as f:
            f.write('\n'.join(f'{x},{y}' for x, y in self.map))  # Convert list to string

        print(f"✅ Map saved to map_lidar_{self.target_index}.txt")

    def limit_speed(self, base_inputs):
        """
        Limits the drone's speed based on distance to target and current velocity.
        base_inputs: [front_left, front_right, rear_left, rear_right] motor inputs
        Returns adjusted motor inputs.
        """
        # Get current position and calculate velocity
        current_position = self.gps.getValues()
        dt = self.time_step / 1000.0  # Time step in seconds
        if dt > 0:
            self.current_velocity = [
                (current_position[i] - self.previous_position[i]) / dt
                for i in range(3)
            ]
        self.previous_position = current_position.copy()

        # Calculate horizontal speed (ignore vertical component for now)
        speed = np.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)

        # Calculate distance to target
        distance = np.linalg.norm(
            np.array(self.target_position[:2]) - np.array(self.current_pose[:2])
        )

        # Scale motor inputs based on distance and speed
        speed_factor = 1.0
        if distance < self.TARGET_PRECISION + self.MIN_DISTANCE:
            # Slow down when close to target
            speed_factor = max(0.1, distance / (self.TARGET_PRECISION + self.MIN_DISTANCE))
        if speed > self.MAX_SPEED:
            # Reduce inputs if speed exceeds maximum
            speed_factor = min(speed_factor, self.MAX_SPEED / (speed + 1e-6))

        # Adjust motor inputs
        adjusted_inputs = [
            self.K_VERTICAL_THRUST + (input - self.K_VERTICAL_THRUST) * speed_factor
            for input in base_inputs
        ]

        # Ensure motor inputs stay within safe bounds
        adjusted_inputs = [
            clamp(input, self.K_VERTICAL_THRUST - 10, self.K_VERTICAL_THRUST + 10)
            for input in adjusted_inputs
        ]

        return adjusted_inputs

    def visualize_map(self):
        if not self.map:
            print("⚠️ No points in self.map to visualize.")
            return

        # Extract x and y coordinates from self.map
        x_coords, y_coords = zip(*self.map) if self.map else ([], [])

        # Create a new figure
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, c='blue', s=10, label='LIDAR Points')
        plt.title(f'LIDAR Map (Waypoint {self.target_index})')
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling for x and y axes
        plt.show()
    
    def rotate(self, amount):
        self.yaw_disturbance = amount

    def scan_qr_codes(self):
        self.camera.saveImage('current_frame.png', 100)  # Save camera image
        image = cv2.imread('current_frame.png')
        if image is not None:
            decoded_objects = decode(image)
            curr_pose = self.gps.getValues()  # Current drone position (x, y, z)
            yaw = self.imu.getRollPitchYaw()[2]  # Yaw angle in radians
            image_height, image_width = image.shape[:2]
            
            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8')
                print(f"QR Code detected: {qr_data}")
                # Draw rectangle around QR code (optional visualization)
                pts = obj.polygon
                if len(pts) > 0:
                    pts = pts + [pts[0]]  # Close the polygon
                    pts = np.array(pts, np.int32)
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                    
                    # Approximate QR position in image (center of polygon)
                    qr_center = np.mean(pts[:, 0]), np.mean(pts[:, 1])  # (x_img, y_img)
                    x_img, y_img = qr_center
                    
                    # Camera FOV (assuming 60° horizontal, adjust based on your camera)
                    fov_horizontal = 1.0472  # 60° in radians
                    pixel_per_radian = image_width / fov_horizontal
                    
                    # Calculate angle offset from image center
                    angle_offset = (image_width / 2 - x_img) / pixel_per_radian
                    
                    # Assume QR is 1m ahead (adjust distance as needed)
                    # This distance should be hinted by the lidar also
                    distance = 1.0  # meters
                    # Calculate world coordinates
                    delta_x = distance * math.cos(yaw + angle_offset)
                    delta_y = distance * math.sin(yaw + angle_offset)
                    qr_x = curr_pose[0] + delta_x
                    qr_y = curr_pose[1] + delta_y
                    
                    # Store QR data with position
                    self.qr_codes.append((qr_x, qr_y, qr_data))
                    print(f"QR Position: ({qr_x:.2f}, {qr_y:.2f})")

                cv2.imwrite(f'qr_detection_{self.target_index}.png', image)
        else:
            print("⚠️ Failed to read camera image.")

    def move_to_target(self, waypoints, verbose_movement=True, verbose_target=True):
        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            self.just_reached = False
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        distance = np.linalg.norm(np.array(self.target_position[:2]) - np.array(self.current_pose[:2]))
        if distance < self.TARGET_PRECISION and not self.just_reached:
            # if self.scan_qr_code == False:
            # self.update_map()
            # self.visualize_map()
            
            # if self.scan_qr_code == True:
            self.scan_qr_codes()
            self.just_reached = True
            self.target_index += 1
            
            # circular: 
            # if self.target_index > len(waypoints) - 1:
            #     self.target_index = 0

            # if self.target_index > len(waypoints) - 1:
            #     self.target_index = 0
            #     self.waypoints = self.scan_waypoints
            #     self.target_altitude = 1.4
            #     self.scan_qr_code = True
            #     self.TARGET_PRECISION = 0.1

            self.target_position[0:2] = waypoints[self.target_index]
            self.just_reached = False
            if verbose_target:
                print("Target reached! New target: ", self.target_position[0:2])

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi

        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def run(self):
        starting_time = self.getTime()
        t1 = self.getTime()
        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # route_1
        # self.waypoints = [[-6, -7.6], [-5, -7.6], [-4, -7.6], [-3, -7.6], [-2, -7.6], [-1.2, -7.6], [-0.755, -6], [-1.2, -7.6], [-2.06, -7.6], [-2.1, -8.9], [-1.27, -8.93], [-1.27, -10.0], [-1.27, -8.93], [-2.1, -8.9], [-2.06, -7.6], [-4.7, -7.6], [-4.8, -9.25], [-4.81, -10.7], [-4.8, -7.6], [-6.05, -7.6], [-6.14, -9.0], [-6.77, -9.01], [-6.77, -10.3], [-6.77, -11.3], [-6.77, -9.0], [-6.05, -9.0], [-6.05, -7.6], [-6.11, -7.3], [-8.2, -7.3], [-6.11, -7.3], [-6, -7.6]]
        # you can set some points that not necessarily have qr codes but reaching them is good for scanning
        self.waypoints = [[-1.2, -7.6], [-0.755, -6.9], [-2.0, -7.3], [-2.12, -8.3], [-2.12, -7.3], [-5.0, -7.25], [-5.12, -8.15]]
        self.scan_waypoints = []

        # self.rotate(0.5)
        
        # route_2
        # box and washing machine
        # [-1.2, -7.6], [-0.755, -6], [-1.2, -7.6], [-1.665, -7.6]

        # route_3
        # bedroom
        # [-2.1, -8.9], [-1.34, -9.04], [-1.66, -9.04], [-1.665, -7.6], [-4.7, -7.6]

        # route_4
        # bathroom and tub
        # [-4.7, -7.6], [-4.8, -9.25], [-4.81, -10.7], [-4.8, -7.6]

        # route_5
        # bedroom
        # [-6.05, -7.6], [-6.14, -9.0], [-6.77, -9.01], [-6.77, -10.3], [-6.77, -11.3], [-6.14, -9.0], [-6.05, -7.6]

        # route_6
        # toilet
        # [-6.11, -7.3], [-8.2, -7.3], [-6.11, -7.3], [-6, -7.6]

        # normal waypoints
        # self.target_altitude = 0.2

        # scanning qr codes
        self.target_altitude = 1.2
        self.TARGET_PRECISION = 0.5
        
        while self.step(self.time_step) != -1:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            rollVelocity = self.gyro.getValues()[0]
            pitchVelocity = self.gyro.getValues()[1]

            self.cameraRollMotor.setPosition(-0.115 * rollVelocity)
            self.cameraPitchMotor.setPosition(-0.1 * pitchVelocity)

            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            if altitude > self.target_altitude - 1:
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance = self.move_to_target(
                        self.waypoints, verbose_movement=True)
                    t1 = self.getTime()

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            # Calculate base motor inputs
            base_inputs = [
                self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input,
                self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input,
                self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input,
                self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
            ]

            # Apply speed limiting
            adjusted_inputs = self.limit_speed(base_inputs)

            # Set motor velocities
            self.front_left_motor.setVelocity(adjusted_inputs[0])
            self.front_right_motor.setVelocity(-adjusted_inputs[1])
            self.rear_left_motor.setVelocity(-adjusted_inputs[2])
            self.rear_right_motor.setVelocity(adjusted_inputs[3])

            # if self.target_index >= len(self.waypoints):
            #     map_complete = True

            # self.scan_qr_codes()
            # if map_complete and not hasattr(self, 'qr_scanned'):
            # self.qr_scanned = True  # Flag to avoid re-scanning

robot = Mavic()
robot.run()