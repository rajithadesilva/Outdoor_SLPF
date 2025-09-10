import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Helper Functions & Classes ---

def angle_wrap(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Map:
    """A simple rectangular room map."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Walls defined as lines: [x1, y1, x2, y2]
        self.walls = [
            [0, 0, width, 0],       # Bottom
            [width, 0, width, height], # Right
            [width, height, 0, height],# Top
            [0, height, 0, 0]        # Left
        ]

    def plot(self, ax):
        """Plots the map walls."""
        for wall in self.walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', lw=2)
        ax.set_xlim(-1, self.width + 1)
        ax.set_ylim(-1, self.height + 1)
        ax.set_aspect('equal')

def simulate_lidar_scan(pose, map_walls, lidar_angles):
    """
    Simulates a LIDAR scan from a given pose by ray-casting.
    This is a simplified ray-casting implementation.
    """
    x, y, theta = pose
    scan = []

    for angle in lidar_angles:
        ray_angle = theta + angle
        min_dist = float('inf')

        # Check intersection with each wall
        for wall in map_walls:
            x1, y1, x2, y2 = wall
            # Using line intersection formula (from Wikipedia)
            den = (x1 - x2) * np.sin(ray_angle) - (y1 - y2) * np.cos(ray_angle)
            if den == 0:
                continue # Parallel lines

            t = ((x1 - x) * np.sin(ray_angle) - (y1 - y) * np.cos(ray_angle)) / den
            u = -((x1 - x2) * (y1 - y) - (y1 - y2) * (x1 - x)) / den
            
            if 0 < t < 1 and u > 0:
                dist = u
                if dist < min_dist:
                    min_dist = dist
        scan.append(min_dist)
    return np.array(scan)

# --- The Main Particle Filter Logic ---

def particle_filter_lidar(num_particles, true_poses, initial_state, room_map, lidar_angles, motion_noise, measurement_noise):
    """
    Particle filter for a robot with a 2D LIDAR in a known map.
    """
    # 1. Initialization
    # Particles are [x, y, theta]
    particles = np.random.rand(num_particles, 3) * np.array([room_map.width, room_map.height, 2 * np.pi])
    weights = np.ones(num_particles) / num_particles
    
    estimated_poses = []

    for i, true_pose in enumerate(true_poses):
        # --- Get real measurement for this time step ---
        real_lidar_scan = simulate_lidar_scan(true_pose, room_map.walls, lidar_angles)
        # Add noise to the real measurement
        real_lidar_scan += np.random.randn(len(lidar_angles)) * measurement_noise

        # 2. Prediction (Motion Update)
        if i > 0:
            # Simple motion model: move and turn, then add noise
            dx = true_poses[i, 0] - true_poses[i-1, 0]
            dy = true_poses[i, 1] - true_poses[i-1, 1]
            dtheta = angle_wrap(true_poses[i, 2] - true_poses[i-1, 2])
            
            particles[:, 0] += dx + np.random.randn(num_particles) * motion_noise[0]
            particles[:, 1] += dy + np.random.randn(num_particles) * motion_noise[1]
            particles[:, 2] = angle_wrap(particles[:, 2] + dtheta + np.random.randn(num_particles) * motion_noise[2])

        # 3. Update (Measurement Update)
        likelihoods = np.zeros(num_particles)
        for j in range(num_particles):
            # For each particle, simulate a scan
            sim_scan = simulate_lidar_scan(particles[j], room_map.walls, lidar_angles)
            
            # Compare simulated scan to real scan to get a likelihood score
            # A simple approach: use Gaussian probability of the difference
            error = np.linalg.norm(sim_scan - real_lidar_scan)
            likelihoods[j] = np.exp(-0.5 * (error**2) / (measurement_noise**2))

        # Update weights
        weights *= likelihoods
        weights += 1.e-300 # avoid zero weights
        weights /= np.sum(weights)

        # 4. Resampling
        indices = np.random.choice(np.arange(num_particles), num_particles, p=weights)
        particles = particles[indices]
        weights.fill(1.0 / num_particles)

        # Estimate current pose
        estimated_pose = np.mean(particles, axis=0)
        estimated_poses.append(estimated_pose)

    return np.array(estimated_poses), particles

if __name__ == '__main__':
    # --- Simulation Setup ---
    room_map = Map(width=10, height=8)
    num_particles = 1000
    
    # Define a simple LIDAR with 8 beams spanning 180 degrees
    lidar_angles = np.linspace(-np.pi/2, np.pi/2, 8)
    
    # Define a true path for the robot
    t = np.linspace(0, 4, 50)
    true_poses = np.array([
        t + 1,                 # x position
        2 * np.sin(t) + 4,     # y position
        angle_wrap(t/2)      # orientation (theta)
    ]).T

    # --- Run Filter ---
    estimated_poses, final_particles = particle_filter_lidar(
        num_particles=num_particles,
        true_poses=true_poses,
        initial_state=true_poses[0],
        room_map=room_map,
        lidar_angles=lidar_angles,
        motion_noise=[0.05, 0.05, np.deg2rad(2)], # Noise in x, y, theta
        measurement_noise=0.1 # Noise on LIDAR distance readings
    )

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 8))
    room_map.plot(ax)
    
    # Plot final particle cloud
    ax.scatter(final_particles[:, 0], final_particles[:, 1], color='lightcoral', s=10, label='Final Particles')
    
    # Plot paths
    ax.plot(true_poses[:, 0], true_poses[:, 1], 'b-', lw=2, label='True Path')
    ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], 'g--', lw=2, label='Estimated Path')
    
    # Plot final pose estimate
    est_pose = estimated_poses[-1]
    ax.quiver(est_pose[0], est_pose[1], np.cos(est_pose[2]), np.sin(est_pose[2]),
              color='g', scale=15, width=0.005, label='Final Estimated Pose')
    
    plt.title("Particle Filter Localization with 2D LIDAR")
    plt.legend()
    plt.show()
