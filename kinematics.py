import numpy as np
from transformations import SE2

def forward_kinematics(theta1, theta2, a1, a2):
    """
    Compute the forward kinematics for a 2-DOF planar robotic arm

    Parameters:
    :param theta1: The angle of the first joint
    :type theta1: float
    :param theta2: The angle of the second joint
    :type theta2: float
    :param a1: length of the first link
    :type a1: float
    :param a2: length of the second link
    :type a2: float
    :return: end effector position
    :rtype: ndarray

    This function returns a 2-element array containing the (x, y) coordinates of the end effector.
    """

    tAB = np.matmul(SE2.rotation(theta1), SE2.translation(a1, 0))
    tBC = np.matmul(SE2.rotation(theta2), SE2.translation(a2, 0))
    tAC = np.matmul(tAB, tBC)
    x, y = tAC[:2, 2]

    return np.array([x, y])

def inverse_kinematics(x, y, a1, a2):
    """
    Compute the inverse kinematics for a 2-DOF planar robotic arm

    :param x: x coordinate of the end effector
    :type x: float
    :param y: y coordinate of the end effector
    :type y: float
    :param a1: length of the first link
    :type a1: float
    :param a2: length of the second link
    :type a2: float
    :return: joint configurations
    :rtype: ndarray

    This function returns a 2D array containing the possible pairs of joint angles (theta1, theta2)
    that achieve the given end effector position. Each row corresponds to a different solution. If
    the position is unreachable, an empty array is returned.
    """

    if x**2 + y**2 > (a1 + a2)**2:
        return np.array([])
    elif x**2 + y**2 == (a1 + a2)**2:
        theta2 = 0
        theta1 = np.arctan2(y, x)
        return np.array([[theta1, theta2]])
    else:
        theta2a = np.arccos((x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2))
        theta2b = -theta2a

        theta3 = np.arctan((a2 * np.sin(theta2a)) / (a1 + a2 * np.cos(theta2a)))

        theta1a = np.arctan2(y, x) - theta3
        theta1b = np.arctan2(y, x) + theta3

        return np.array([[theta1a, theta2a], [theta1b, theta2b]])


def main():
    ################################ Inverse and Forward Kinematics ################################

    a1 = 1 # Length of the first link of a 2-DOF planar robotic arm
    a2 = 1 # Length of the second link of a 2-DOF planar robotic arm

    # Forward Kinematics
    configurations = [[0, np.pi/2], [np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], [-np.pi, np.pi]]

    print('Forward Kinematics\n', '-' * 105)

    for i, c in enumerate(configurations):
        c = np.round(c, 2)
        p = np.round(forward_kinematics(c[0], c[1], a1, a2), 2)
        print(f'{i}/ End effector position for configuration [theta1, theta2] = {np.round(c, 2)} -> [x, y] = {p}\n')

    # Inverse Kinematics
    positions = [[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 1], [2, 0], [0, 2], [-2, 0]]

    print('Inverse Kinematics\n', '-' * 105)

    for i, p in enumerate(positions):
        p = np.round(p, 2)
        c = np.round(inverse_kinematics(p[0], p[1], a1, a2), 2)

        if len(c) == 0:
            print(f'{i}/ Position [x, y] = {p} is unreachable\n')
        elif len(c) == 1:
            print(f'{i}/ Joint configuration for position [x, y] = {p} -> [theta1, theta2] = {c[0]}\n')
        else:
            print(f'{i}/ Joint configuration for position [x, y] = {p} -> [theta1, theta2] = {c[0]} or {c[1]}\n')


if __name__ == '__main__':
    main()
