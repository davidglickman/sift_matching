import numpy as np
import matplotlib.pyplot as plt

class IMUModel:
    def __init__(self, rate=200.0):
        self.rate = rate
        self.dt = 1.0 / rate

        # Accelerometer noise parameters
        self.accel_noise_density = 100e-6  # g/sqrt(Hz) convert to m/s^2
        self.accel_random_walk = 0.05  # m/s/sqrt(Hz)
        self.accel_bias_stability = 20e-6 * 9.81  # 20µg to m/s^2
        self.accel_bias_repeatability = 5e-3 * 9.81  # mg to m/s^2
        self.accel_bias_over_temp = 3e-3 * 9.81  # mg to m/s^2
        self.accel_nonlinearity = 0.001  # 0.1%
        self.accel_misalignment = 2e-3  # 2mrad

        # Gyroscope noise parameters
        self.gyro_noise_density = 0.005 * np.pi / 180.0  # rad/s/sqrt(Hz)
        self.gyro_random_walk = 0.1 * np.pi / 180.0 / 3600.0  # rad/s/sqrt(hr)
        self.gyro_bias_stability = 2.0 * np.pi / 180.0 / 3600.0  # deg/hr to rad/s
        self.gyro_bias_repeatability = 0.1 * np.pi / 180.0  # deg/sec to rad/s
        self.gyro_bias_over_temp = 0.03 * np.pi / 180.0  # deg/s to rad/s
        self.gyro_nonlinearity = 0.0005  # 0.05%
        self.gyro_misalignment = 2e-3  # 2mrad

        # Initialize biases (can be randomly initialized or set to zero)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def get_imu_measurements(self, ground_truth_pose):
        x, y, z, roll, pitch, yaw, time = ground_truth_pose

        if hasattr(self, 'prev_pose'):
            dx = x - self.prev_pose[0]
            dy = y - self.prev_pose[1]
            dz = z - self.prev_pose[2]

            droll = roll - self.prev_pose[3]
            dpitch = pitch - self.prev_pose[4]
            dyaw = yaw - self.prev_pose[5]

            v = np.array([dx, dy, dz]) / self.dt

            accel_ideal = np.zeros(3)  # No acceleration for now (REPLACE THIS)
            gyro_ideal = np.array([droll, dpitch, dyaw]) / self.dt  # (REPLACE THIS)

        else:
            accel_ideal = np.zeros(3)
            gyro_ideal = np.zeros(3)
            v = np.zeros(3)

        # Add noise and biases
        accel_noise = np.random.normal(0, self.accel_noise_density, 3) + self.accel_random_walk * np.random.normal(0, 1, 3) / np.sqrt(self.dt)
        gyro_noise = np.random.normal(0, self.gyro_noise_density, 3) + self.gyro_random_walk * np.random.normal(0, 1, 3) / np.sqrt(self.dt)

        accel_noisy = accel_ideal + accel_noise + self.accel_bias
        gyro_noisy = gyro_ideal + gyro_noise + self.gyro_bias

        # Add nonlinearity and misalignment
        accel_noisy = accel_noisy * (1 + self.accel_nonlinearity * np.random.normal(0, 1))
        gyro_noisy = gyro_noisy * (1 + self.gyro_nonlinearity * np.random.normal(0, 1))

        R_accel_mis = self._get_misalignment_matrix(self.accel_misalignment)
        R_gyro_mis = self._get_misalignment_matrix(self.gyro_misalignment)

        accel_noisy = R_accel_mis @ accel_noisy
        gyro_noisy = R_gyro_mis @ gyro_noisy

        self.prev_pose = [x, y, z, roll, pitch, yaw]

        return accel_noisy, gyro_noisy, time

    def _get_misalignment_matrix(self, misalignment):
        R = np.eye(3)
        R[0, 1] = -misalignment
        R[1, 0] = misalignment
        return R


class TrajectoryGenerator:
    def __init__(self, duration=3.0, height=100.0, angle_degrees=45.0, rate=200.0):
        self.duration = duration
        self.height = height
        self.angle_rad = np.radians(angle_degrees)
        self.rate = rate
        self.dt = 1.0 / rate
        self.num_samples = int(duration * rate)
        self.ground_truth_trajectory = []

    def generate_trajectory(self):
        for i in range(self.num_samples):
            time = i * self.dt

            x = time * np.cos(self.angle_rad) * 30
            y = time * np.sin(self.angle_rad) * 30
            z = self.height

            roll = 0.0
            pitch = 0.0
            yaw = self.angle_rad

            roll_noisy = np.random.normal(roll, np.radians(1.0))
            pitch_noisy = np.random.normal(pitch, np.radians(1.0))
            yaw_noisy = np.random.normal(yaw, np.radians(1.0))

            self.ground_truth_trajectory.append([x, y, z, roll_noisy, pitch_noisy, yaw_noisy, time])

        return self.ground_truth_trajectory


# Generate trajectory
trajectory_generator = TrajectoryGenerator()
ground_truth_trajectory = trajectory_generator.generate_trajectory()

# Initialize IMU model
imu = IMUModel()

# Lists to store IMU measurements
imu_accel_measurements = []
imu_gyro_measurements = []
imu_times = []

# Simulate IMU measurements
for pose in ground_truth_trajectory:
    accel, gyro, time = imu.get_imu_measurements(pose)
    imu_accel_measurements.append(accel)
    imu_gyro_measurements.append(gyro)
    imu_times.append(time)

# Convert to numpy arrays
imu_accel_measurements = np.array(imu_accel_measurements)
imu_gyro_measurements = np.array(imu_gyro_measurements)

# Extract data for plotting (ground truth)
x = [pose[0] for pose in ground_truth_trajectory]
y = [pose[1] for pose in ground_truth_trajectory]
z = [pose[2] for pose in ground_truth_trajectory]
roll = [np.degrees(pose[3]) for pose in ground_truth_trajectory]
pitch = [np.degrees(pose[4]) for pose in ground_truth_trajectory]
yaw = [np.degrees(pose[5]) for pose in ground_truth_trajectory]
time = [pose[6] for pose in ground_truth_trajectory]

# Plotting
fig = plt.figure(figsize=(16, 12))  # Adjust figure size

# Ground Truth Plots
# 2D Trajectory (x-y plane)
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Ground Truth 2D Trajectory (X-Y)')
plt.grid(True)

# 3D Trajectory
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ground Truth 3D Trajectory')
plt.grid(True)

# Angles vs. Time (Ground Truth)
plt.subplot(2, 2, 3)
plt.plot(time, roll, label='Roll')
plt.plot(time, pitch, label='Pitch')
plt.plot(time, yaw, label='Yaw')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Ground Truth Angles vs. Time')
plt.legend()
plt.grid(True)

# IMU Measurements Plot
plt.subplot(2, 2, 4)

plt.plot(imu_times, imu_accel_measurements[:, 0], label='Accel X')
plt.plot(imu_times, imu_accel_measurements[:, 1], label='Accel Y')
plt.plot(imu_times, imu_accel_measurements[:, 2], label='Accel Z')

plt.plot(imu_times, imu_gyro_measurements[:, 0] * 180.0 / np.pi, label='Gyro X') # to degrees
plt.plot(imu_times, imu_gyro_measurements[:, 1] * 180.0 / np.pi, label='Gyro Y') # to degrees
plt.plot(imu_times, imu_gyro_measurements[:, 2] * 180.0 / np.pi, label='Gyro Z') # to degrees

plt.xlabel('Time (s)')
plt.ylabel('IMU Measurement')
plt.title('IMU Measurements')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()