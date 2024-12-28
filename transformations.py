import numpy as np


class SE2:
    """
    A class used to represent SE(2) transformations.
    """

    @staticmethod
    def translation(tx, ty):
        """
        Create a 2D translation matrix.

        :param tx: Translation along the x-axis
        :type tx: float
        :param ty: Translation along the y-axis
        :type ty: float
        :return: A 3x3 homogeneous transformation matrix
        :rtype: ndarray(3, 3)
        """
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0,  1]])

    @staticmethod
    def rotation(theta):
        """
        Create a 2D rotation matrix.

        :param theta: Rotation angle
        :type theta: float
        :return: A 3x3 homogeneous transformation matrix
        :rtype: ndarray(3, 3)
        """
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [            0,              0, 1]])

    @staticmethod
    def transformation(tx, ty, theta):
        """
        Create a 2D transformation matrix.

        :param tx: Translation along the x-axis
        :type tx: float
        :param ty: Translation along the y-axis
        :type ty: float
        :param theta: Rotation angle
        :type theta: float
        :return: A 3x3 homogeneous transformation matrix
        :rtype: ndarray(3, 3)

        This transformation matrix is the result of a translation followed by a rotation.
        """
        return np.array([[np.cos(theta), -np.sin(theta), tx],
                         [np.sin(theta),  np.cos(theta), ty],
                         [            0,              0, 1]])


def main():
    ###################################### SE2 Transformations #####################################

    print('SE2 Transformations\n', '-' * 75)

    # 1
    pR2 = np.array([0.5, 0.5, 1]).T
    tR1R2 = SE2.translation(1, 0.25)
    pR1 = np.matmul(tR1R2, pR2)
    print(f'1/ Coordinates of point P in reference frame R1 -> [x, y] = {np.round(pR1[:2], 2)}\n')

    # 2
    pR1 = np.array([0.5, 0.5, 1]).T
    tR2R1 = np.linalg.inv(tR1R2)
    pR2 = np.matmul(tR2R1, pR1)
    print(f'2/ Coordinates of point P in reference frame R2 -> [x, y] = {np.round(pR2[:2], 2)}\n')

    # 3
    pR2 = np.array([0.5, 0.5, 1]).T
    tR1R2 = np.matmul(SE2.translation(1, 0.25), SE2.rotation(np.pi / 4))
    pR1 = np.matmul(tR1R2, pR2)
    print(f'3/ Coordinates of point P in reference frame R1 -> [x, y] = {np.round(pR1[:2], 2)}\n')

    # 4
    pR1 = np.array([0.5, 0.5, 1]).T
    tR2R1 = np.linalg.inv(tR1R2)
    pR2 = np.matmul(tR2R1, pR1)
    print(f'4/ Coordinates of point P in reference frame R2 -> [x, y] = {np.round(pR2[:2], 2)}\n')


if __name__ == '__main__':
    main()
