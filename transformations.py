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
    ###################################### SE2 Transformations #####################################
    print('SE2 Transformations')
    print('-' * 150)

    # 1
    point_in_frame_R2 = np.array([0.5, 0.5, 1]).transpose()
    transform_R1_to_R2 = SE2.translation(1, 0.25)
    point_in_frame_R1 = np.matmul(transform_R1_to_R2, point_in_frame_R2)
    print(f'1/ Coordinates of point P in reference frame R1 -> [x, y] = {point_in_frame_R1[:2]}\n')

    # 2
    point_in_frame_R1 = np.array([0.5, 0.5, 1]).transpose()
    transform_R2_to_R1 = np.linalg.inv(transform_R1_to_R2)
    point_in_frame_R2 = np.matmul(transform_R2_to_R1, point_in_frame_R1)
    print(f'2/ Coordinates of point P in reference frame R2 -> [x, y] = {point_in_frame_R2[:2]}\n')

    # 3
    point_in_frame_R2 = np.array([0.5, 0.5, 1]).transpose()
    transform_R1_to_R2 = np.matmul(SE2.translation(1, 0.25), SE2.rotation(np.pi / 4))
    point_in_frame_R1 = np.matmul(transform_R1_to_R2, point_in_frame_R2)
    print(f'3/ Coordinates of point P in reference frame R1 -> [x, y] = {point_in_frame_R1[:2]}\n')

    # 4
    point_in_frame_R1 = np.array([0.5, 0.5, 1]).transpose()
    transform_R2_to_R1 = np.linalg.inv(transform_R1_to_R2)
    point_in_frame_R2 = np.matmul(transform_R2_to_R1, point_in_frame_R1)
    print(f'4/ Coordinates of point P in reference frame R2 -> [x, y] = {point_in_frame_R2[:2]}\n')


if __name__ == '__main__':
    main()
