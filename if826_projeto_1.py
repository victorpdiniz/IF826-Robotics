import numpy as np

# SE2 Translation Matrix

def SE2_xy(xb, yb):
    return np.array([[1, 0, xb],
                     [0, 1, yb],
                     [0, 0, 1]])

# SE2 Rotation Matrix

def SE2_theta(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

# Tests
print('\nSE2 Transformation Matrix/\n')

# 1

R2_P = np.array([0.5, 0.5, 1]).transpose()
R1_T_R2 = SE2_xy(1, 0.25)
R1_P = np.matmul(R1_T_R2, R2_P)
print(f'1/ Coordinates of point P in reference frame R1: (x, y) = {R1_P[:2]}')

# 2

R1_P = np.array([0.5, 0.5, 1]).transpose()
R2_T_R1 = np.linalg.inv(R1_T_R2)
R2_P = np.matmul(R2_T_R1, R1_P)
print(f'2/ Coordinates of point P in reference frame R2: (x, y) = {R2_P[:2]}')

# 3

R2_P = np.array([0.5, 0.5, 1]).transpose()
R1_T_R2 = np.matmul(SE2_xy(1, 0.25), SE2_theta(np.pi/4))
R1_P = np.matmul(R1_T_R2, R2_P)
print(f'3/ Coordinates of point P in reference frame R1: (x, y) = {R1_P[:2]}')

# 4

R1_P = np.array([0.5, 0.5, 1]).transpose()
R2_T_R1 = np.linalg.inv(R1_T_R2)
R2_P = np.matmul(R2_T_R1, R1_P)
print(f'4/ Coordinates of point P in reference frame R2: (x, y) = {R2_P[:2]}')

# Forward Kinematics

def fk(theta1, theta2):
    a1 = a2 = 1
    O_T_A = np.matmul(SE2_theta(theta1), SE2_xy(a1, 0))
    A_T_B = np.matmul(SE2_theta(theta2), SE2_xy(a2, 0))
    O_T_B = np.matmul(O_T_A, A_T_B)
    return O_T_B[:2, 2]

# Inverse Kinematics

def ik(x, y):
    a1 = a2 = 1
    theta2 = np.arccos((x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2))
    theta3 = np.arctan((a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2)))
    theta1 = np.arctan(y / x) - theta3
    return theta1, theta2

# Tests FK
print('\nfk/\n')

# 1
theta1, theta2 = 0, np.pi/2
print(f'Manipulator position for configuration (theta1, theta2) = {theta1, theta2}: (x, y) = {fk(theta1, theta2)}')

# 2
theta1, theta2 = np.pi/2, np.pi/2
print(f'Manipulator position for configuration (theta1, theta2) = {theta1, theta2}: (x, y) = {fk(theta1, theta2)}')

# 3
theta1, theta2 = np.pi/2, -np.pi/2
print(f'Manipulator position for configuration (theta1, theta2) = {theta1, theta2}: (x, y) = {fk(theta1, theta2)}')

# 4
theta1, theta2 = -np.pi, np.pi
print(f'Manipulator position for configuration (theta1, theta2) = {theta1, theta2}: (x, y) = {fk(theta1, theta2)}')

# Tests IK
print('\nik/\n')

# 1
x, y = 1, 1
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 2
x, y = 1, -1
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 3
x, y = -1, 1
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 4
x, y = -1, -1
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 5
x, y = 2, 1
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 6
x, y = 2, 0
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 7
x, y = 0, 2
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')

# 8
x, y = -2, 0
print(f'Arm configuration for position (x, y) = {(x, y)}: (theta1, theta2) = {ik(x, y)}')