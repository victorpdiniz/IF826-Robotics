import numpy as np


class SE2:
    @staticmethod
    def translation(tx, ty):
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0,  1]])

    @staticmethod
    def rotation(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [            0,              0, 1]])

    @staticmethod
    def transformation(tx, ty, theta):
        return np.array([[np.cos(theta), -np.sin(theta), tx],
                         [np.sin(theta),  np.cos(theta), ty],
                         [            0,              0, 1]])


def main():
    ############################## SE2 Transformations #############################

    # 1
    R2_P = np.array([0.5, 0.5, 1]).transpose()
    R1_T_R2 = SE2.translation(1, 0.25)
    R1_P = np.matmul(R1_T_R2, R2_P)
    print(f'1/ Coordinates of point P in reference frame R1 -> [x, y] = {R1_P[:2]}')

    # 2
    R1_P = np.array([0.5, 0.5, 1]).transpose()
    R2_T_R1 = np.linalg.inv(R1_T_R2)
    R2_P = np.matmul(R2_T_R1, R1_P)
    print(f'2/ Coordinates of point P in reference frame R2 -> [x, y] = {R2_P[:2]}')

    # 3
    R2_P = np.array([0.5, 0.5, 1]).transpose()
    R1_T_R2 = np.matmul(SE2.translation(1, 0.25), SE2.rotation(np.pi/4))
    R1_P = np.matmul(R1_T_R2, R2_P)
    print(f'3/ Coordinates of point P in reference frame R1 -> [x, y] = {R1_P[:2]}')

    # 4
    R1_P = np.array([0.5, 0.5, 1]).transpose()
    R2_T_R1 = np.linalg.inv(R1_T_R2)
    R2_P = np.matmul(R2_T_R1, R1_P)
    print(f'4/ Coordinates of point P in reference frame R2 -> [x, y] = {R2_P[:2]}')


if __name__ == '__main__':
    main()
