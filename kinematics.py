import numpy as np
from transformations import SE2


class Kinematics2DOF:
    @staticmethod
    def forward_kinematics(q1, q2, a1, a2):
        O_T_A = np.matmul(SE2.rotation(q1), SE2.translation(a1, 0))
        A_T_B = np.matmul(SE2.rotation(q2), SE2.translation(a2, 0))
        O_T_B = np.matmul(O_T_A, A_T_B)
        return O_T_B[:2, 2]

    @staticmethod
    def inverse_kinematics(x, y, a1, a2):
        if x**2 + y**2 > a1**2 + a2**2 + 2 * a1 * a2:
            return None
        elif x**2 + y**2 == a1**2 + a2**2 + 2 * a1 * a2:
            q2 = 0
            q1 = np.arctan2(y, x)
            return np.array([[q1, q2]])
        else:
            q2_1 = np.arccos((x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2))
            q2_2 = -q2_1
            q3 = np.arctan((a2 * np.sin(q2_1)) / a1 + a2 * np.cos(q2_1))
            q1_1 = np.arctan2(y, x) - q3
            q1_2 = np.arctan2(y, x) + q3
            return np.array([[q1_1, q2_1], [q1_2, q2_2]])


def main():
    ################################### Inverse and Forward Kinematics ###################################

    a1 = a2 = 1

    # Forward Kinematics
    configurations = [[0, np.pi / 2], [np.pi / 2, np.pi / 2], [np.pi / 2, -np.pi / 2], [-np.pi, np.pi]]

    print('Forward Kinematics')
    print('-' * 150)

    for i, configuration in enumerate(configurations):
        print(f'{i + 1}/ End effector position for configuration [theta1, theta2] = {configuration} ->',
              f'[x, y] = {Kinematics2DOF.forward_kinematics(configuration[0], configuration[1], a1, a2)}\n')

    # Inverse Kinematics
    positions = [[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 1], [2, 0], [0, 2], [-2, 0]]

    print('Inverse Kinematics')
    print('-' * 150)

    for i, position in enumerate(positions):
        print(f'{i + 1}/ Joint configuration for position [x, y] = {position} ->',
              f'[theta1, theta2] =\n{Kinematics2DOF.inverse_kinematics(position[0], position[1], a1, a2)}\n')


if __name__ == '__main__':
    main()
