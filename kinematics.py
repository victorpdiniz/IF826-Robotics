import numpy as np
from transformations import SE2


class Kinematics2DOF:
    @staticmethod
    def forward_kinematics(theta1, theta2, link1_length, link2_length):
        base_to_joint1 = np.matmul(SE2.rotation(theta1), SE2.translation(link1_length, 0))
        joint1_to_joint2 = np.matmul(SE2.rotation(theta2), SE2.translation(link2_length, 0))
        base_to_end_effector = np.matmul(base_to_joint1, joint1_to_joint2)
        x, y = base_to_end_effector[:2, 2]
        return np.array([x, y])

    @staticmethod
    def inverse_kinematics(x, y, link1_length, link2_length):
        if x**2 + y**2 > (link1_length + link2_length)**2:
            return np.array([])
        elif x**2 + y**2 == (link1_length + link2_length)**2:
            theta2 = 0
            theta1 = np.arctan2(y, x)
            return np.array([[theta1, theta2]])
        else:
            theta2_1 = np.arccos((x**2 + y**2 - link1_length**2 - link2_length**2)
                                 / (2 * link1_length * link2_length))
            theta2_2 = -theta2_1

            theta3 = np.arctan((link2_length * np.sin(theta2_1))
                               / (link1_length + link2_length * np.cos(theta2_1)))

            theta1_1 = np.arctan2(y, x) - theta3
            theta1_2 = np.arctan2(y, x) + theta3

            return np.array([[theta1_1, theta2_1], [theta1_2, theta2_2]])


def main():
    ################################ Inverse and Forward Kinematics ################################
    link1_length = link2_length = 1

    # Forward Kinematics
    configurations = [[0, np.pi/2], [np.pi/2, np.pi/2], [np.pi/2, -np.pi/2], [-np.pi, np.pi]]

    print('Forward Kinematics')
    print('-' * 150)

    for i, configuration in enumerate(configurations):
        theta1, theta2 = configuration
        position = Kinematics2DOF.forward_kinematics(theta1, theta2, link1_length, link2_length)
        print(f'{i}/ End effector position for configuration [theta1, theta2] = {configuration}',
              f'-> [x, y] = {position}\n')

    # Inverse Kinematics
    positions = [[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 1], [2, 0], [0, 2], [-2, 0]]

    print('Inverse Kinematics')
    print('-' * 150)

    for i, position in enumerate(positions):
        x, y = position
        configuration = Kinematics2DOF.inverse_kinematics(x, y, link1_length, link2_length)
        print(f'{i}/ Joint configuration for position [x, y] = {position} -> [theta1, theta2] =')
        print(configuration, '\n')


if __name__ == '__main__':
    main()
