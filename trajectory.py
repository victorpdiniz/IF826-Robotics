import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb


def trapezoidal_helper(s0, sf, v, a, tf):
    """
    Compute the velocity of the linear segment of a trapezoidal profile for a given set of
    parameters

    :param s0: initial value
    :type s0: float
    :param sf: final value
    :type sf: float
    :param v: maximum joint velocity
    :type v: float
    :param a: maximum joint acceleration
    :type a: float
    :return: velocity of the linear segment
    :rtype: float

    This function raises an error if there are no feasible velocities for the given parameters.
    """

    v1 = (a * tf + np.sqrt(a**2 * tf**2 - 4 * a * (sf - s0))) / 2
    v2 = (a * tf - np.sqrt(a**2 * tf**2 - 4 * a * (sf - s0))) / 2

    if 0 < v1 <= v:
        return v1
    elif 0 < v2 <= v:
        return v2
    else:
        raise ValueError('There is no feasible velocity for the given parameters.')


def trapezoidal_function(s0, sf, v, a):
    """
    Trapezoidal profile as a function

    :param s0: initial value
    :type s0: float
    :param sf: final value
    :type sf: float
    :param v: absolute value of the velocity of the linear segment
    :type v: float
    :param a: absolute value of the acceleration of the ramp segments
    :type a: float
    :return: trapezoidal profile function
    :rtype: callable

    Returns a function which computes the specific trapezoidal profile as described by the given
    parameters.
    """

    if v <= 0:
        raise ValueError('The absolute value of the velocity of the linear segment must be positive.')
    if a <= 0:
        raise ValueError('The absolute value of the acceleration of the ramp segments must be positive.')

    # If the initial and final values are the same, the function is a constant function that is
    # equal to the initial value. Here we consider the acceleration and end time to be zero.
    if s0 == sf:
        func = lambda t: s0
        func.ta = 0
        func.tf = 0
        return func
    else:
        # If the trajectory is ascending, the velocity and acceleration must be positive. If the
        # trajectory is descending, the velocity and acceleration must be negative.
        v = v * np.sign(sf - s0)
        a = a * np.sign(sf - s0)

        ta = v / a
        tf = ta + (sf - s0) / v

        # Not every value of v is feasible. The velocity must be greater than the average velocity and
        # less than twice the average velocity.
        # Ref: https://www.mathworks.com/help/robotics/ug/design-a-trajectory-with-velocity-limits-using-a-trapezoidal-velocity-profile.html
        if abs(v) < (abs(sf - s0) / tf):
            raise ValueError("The value of the maximum velocity is too small. It's not possible to generate a feasible trajectory.")
        elif abs(v) > (2 * abs(sf - s0) / tf):
            raise ValueError("The value of the maximum velocity is too big. It's not possible to generate a feasible trajectory.")

        def trap_func(t):
            if t < 0:
                return s0
            elif t <= ta:
                # acceleration
                return s0 + a / 2 * t**2
            elif t <= (tf - ta):
                # linear motion
                return s0 + v * t - v**2 / (2 * a)
            elif t <= tf:
                # deceleration
                return s0 + (2 * a * v * tf - 2 * v**2 - a**2 * (t - tf)**2) / (2 * a)
            else:
                return sf

        # Return the function, but add some computed parameters as attributes as a way of returning
        # extra values without a tuple return
        func = trap_func
        func.ta = ta
        func.tf = tf

        return func


def joint_trajectory(q1i, q2i, q1f, q2f, v, a, ts):
    """
    Compute a joint-space trajectory

    :param q1i: initial q1 joint coordinate
    :type q1i: float
    :param q1f: final q1 joint coordinate
    :type q1f: float
    :param q2i: initial q2 joint coordinate
    :type q2i: float
    :param q2f: final q2 joint coordinate
    :type q2f: float
    :param v: maximum joint velocity
    :type v: float
    :param a: maximum joint acceleration
    :type a: float
    :param ts: sample time
    :type ts: float
    :return: trajectory
    :rtype: ndarray(n, 2)

    Returns a vector of joint coordinates that describe a trajectory from the initial configuration
    to the final configuration at evenly spaced time steps of duration ts.
    """
    q1_trap_func = trapezoidal_function(q1i, q1f, v, a)
    q2_trap_func = trapezoidal_function(q2i, q2f, v, a)

    tf1 = q1_trap_func.tf
    tf2 = q2_trap_func.tf
    tf = max(tf1, tf2)

    if tf1 != 0 and tf2 != 0 and tf1 != tf2:
        if tf == tf1:
            v2 = trapezoidal_helper(q2i, q2f, v, a, tf)
            q2_trap_func = trapezoidal_function(q2i, q2f, v2, a)
        elif tf == tf2:
            v1 = trapezoidal_helper(q1i, q1f, v, a, tf)
            q1_trap_func = trapezoidal_function(q1i, q1f, v1, a)

    if tf != 0:
        t = np.arange(0, tf + ts, ts)
    else:
        t = np.arange(0, 1, ts)

    q1_traj = np.array([q1_trap_func(ti) for ti in t])
    q2_traj = np.array([q2_trap_func(ti) for ti in t])
    traj = np.vstack((q1_traj, q2_traj)).T

    return traj


def euclidean_trajectory(x0, y0, xf, yf, v, a, ts):
    """
    Compute a euclidean-space trajectory

    :param x0: initial x coordinate
    :type x0: float
    :param y0: initial y coordinate
    :type y0: float
    :param xf: final x coordinate
    :type xf: float
    :param yf: final y coordinate
    :type yf: float
    :param v: maximum euclidean velocity
    :type v: float
    :param a: maximum euclidean acceleration
    :type a: float
    :param ts: sample time
    :type ts: float
    :return: trajectory
    :rtype: ndarray(n, 2)

    Returns a vector of joint configurations that describe a trajectory from the initial
    configuration to the final configuration at evenly spaced time steps of duration ts.
    """
    pass


def main():
    ###################################### Trajectory Planning #####################################
    robot = rtb.models.DH.Planar2()

    # Joint Trajectory Planning
    q1i, q2i, q1f, q2f = 0, 0, np.pi/4, np.pi/6
    v, a, ts = 1, 1, 0.01

    trajectory = joint_trajectory(q1i, q2i, q1f, q2f, v, a, ts)
    robot.plot(trajectory, dt=ts, backend='pyplot', movie= 'joint_trajectory_planning.gif')


if __name__ == '__main__':
    main()