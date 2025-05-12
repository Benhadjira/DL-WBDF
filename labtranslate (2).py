import numpy as np

def rotation_matrix_diffry(theta):
    theta_rad = np.radians(theta)
    return np.array([
        [np.cos(theta_rad), 0, -np.sin(theta_rad)],
        [0, 1, 0],
        [np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])

def rotation_matrix_chi(alpha):
    alpha_rad = np.radians(alpha)
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
        [0, np.sin(alpha_rad), np.cos(alpha_rad)]
    ])

def rotation_matrix_phi(beta):
    beta_rad = np.radians(beta)
    return np.array([
        [np.cos(beta_rad), -np.sin(beta_rad), 0],
        [np.sin(beta_rad), np.cos(beta_rad), 0],
        [0, 0, 1]
    ])

def calculate_shift(diffry, chi, phi, motion_value, motion_axis):
    # Get the rotation matrices
    R_diffry = rotation_matrix_diffry(diffry)
    R_chi = rotation_matrix_chi(chi)
    R_phi = rotation_matrix_phi(phi)

    # Combine the rotation matrices
    R = R_phi @ R_chi @ R_diffry

    # Define the initial motion vector based on the specified motion axis
    if motion_axis == "smx":
        motion_vector = np.array([motion_value, 0, 0])  # smx is along x-axis in stage coordinates
    elif motion_axis == "smy":
        motion_vector = np.array([0, motion_value, 0])  # smy is along y-axis in stage coordinates
    elif motion_axis == "smz":
        motion_vector = np.array([0, 0, motion_value])  # smz is along z-axis in stage coordinates
    else:
        raise ValueError("Invalid motion axis. Choose from 'smx', 'smy', or 'smz'.")

    # Transform to lab coordinates
    lab_motion = R @ motion_vector

    # Extract dx, dy, dz in lab coordinates
    dx, dy, dz = lab_motion
    return dx, dy, dz

def calculate_detector_shift(dx, dy, dz, pixel_size):
    # Calculate detector shifts in pixels
    shift_y = dx / pixel_size  # Horizontal shift in pixels
    shift_x = dy / pixel_size  # Vertical shift in pixels

    return shift_x, shift_y

def calculate_roll_motion(diffry, chi, phi, scan_motor, motor_value):
    # Get the rotation matrices
    R_diffry = rotation_matrix_diffry(diffry)
    R_chi = rotation_matrix_chi(chi)
    R_phi = rotation_matrix_phi(phi)

    # Combine the rotation matrices
    R = R_phi @ R_chi @ R_diffry

    # Define the initial rotation direction for "roll"
    roll_direction_lab = np.array([1, 0, 0])  # Roll is around the lab x-axis

    # Transform the roll direction to the stage frame
    roll_direction_stage = np.linalg.inv(R) @ roll_direction_lab

    # Determine the motion based on the scan motor
    if scan_motor == "chi":
        roll_motion = motor_value * roll_direction_stage[0]
    elif scan_motor == "phi":
        roll_motion = motor_value * roll_direction_stage[2]
    else:
        raise ValueError("Invalid scan motor. Choose either 'chi' or 'phi'.")

    return roll_motion, roll_direction_stage

# Example usage
diffry = 10  # in degrees
chi = 1.5   # in degrees
phi = 0      # in degrees
motion_value = 1.0  # 1 um in microns
motion_axis = "smz"  # Motion along smz
pixel_size = 0.035  # Effective pixel size in microns

# Calculate lab frame displacement
dx, dy, dz = calculate_shift(diffry, chi, phi, motion_value, motion_axis)

# Calculate detector shift
shift_x, shift_y = calculate_detector_shift(dx, dy, dz, pixel_size)

# Calculate roll motion
diffry_for_roll = 10  # Assuming diffry for roll calculation
scan_motor = "chi"  # Motor to scan for roll
motor_value_for_roll = 5  # Motor step in degrees
roll_motion, roll_direction_stage = calculate_roll_motion(diffry_for_roll, chi, phi, scan_motor, motor_value_for_roll)

print(f"Lab frame displacement due to {motion_axis} motion (in microns):\n")
print(f"dx = {dx:.6f} μm")
print(f"dy = {dy:.6f} μm")
print(f"dz = {dz:.6f} μm")

print(f"\nProposed detector image shift (in pixels):\n")
print(f"Horizontal shift (x) = {shift_x:.3f} pixels")
print(f"Vertical shift (y) = {shift_y:.3f} pixels")

print(f"\nNote: dz = {dz:.6f} μm should be considered for 3D image stacking.")

print(f"\nRoll motion around the lab x-axis for {scan_motor} = {motor_value_for_roll} degrees:\n")
print(f"Roll motion = {roll_motion:.6f} degrees")
print(f"Roll direction in stage coordinates: {roll_direction_stage}")