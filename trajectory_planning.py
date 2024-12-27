import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from kinematics import Kinematics2DOF


def helper_function(s_init, s_final, a_max, end_time):
    v1 = (a_max * end_time + np.sqrt(a_max**2 * end_time**2 - 2 * a_max * (s_final - s_init))) / 2
    v2 = (a_max * end_time - np.sqrt(a_max**2 * end_time**2 - 2 * a_max * (s_final - s_init))) / 2
    return v1, v2


def trapezoidal_velocity_profile(t, a, v, accel_time, end_time):
    if 0 <= t < accel_time:
        return a * t**2 / 2
    elif accel_time <= t < end_time - accel_time:
        return v * t - v**2 / (2 * a)
    elif end_time - accel_time <= t <= end_time:
        return (2 * a * v * end_time - 2 * v**2 - a**2 * (t - end_time)**2) / (2 * a)
    else:
        return 0


class TrajectoryPlanning:
    @staticmethod
    def trapezoidal_velocity_profile_trajectory(s_init, s_final, a, v, ts):
        delta_s = abs(s_final - s_init)

        accel_time = v / a
        end_time = accel_time + delta_s / v

        # The trajectory is unfeasible with the given parameters
        if end_time <= delta_s / v or end_time > 2 * delta_s / v:
            return np.array([])

        t = np.arange(0, end_time + ts, ts)
        direction = 1 if s_final >= s_init else -1
        trajectory = [s_init + direction * trapezoidal_velocity_profile(ti, a, v, accel_time, end_time) for ti in t]

        return np.array(trajectory)

    @staticmethod
    def joint_trajectory(q1_init, q2_init, q1_final, q2_final, a, v, ts):
        q1_trajectory = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q1_init, q1_final,
                                                                                   a, v, ts)
        q2_trajectory = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q2_init, q2_final,
                                                                                   a, v, ts)

        # The trajectory is unfeasible with the given parameters
        if len(q1_trajectory) == 0 or len(q2_trajectory) == 0:
            return np.array([])

        # The trajectories must have the same length. If not, we need to calculate a new trajectory
        # for the shorter one with a new v value.
        if len(q1_trajectory) != len(q2_trajectory):
            if max(len(q1_trajectory), len(q2_trajectory)) == len(q1_trajectory):
                end_time = len(q1_trajectory) * ts
                v = TrajectoryPlanning.helper_function(q2_init, q2_final, a, end_time)
                traj1 = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q2_init, q2_final,
                                                                                   v[0], a, ts)
                traj2 = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q2_init, q2_final,
                                                                                   v[1], a, ts)
            else:
                end_time = len(q2_trajectory) * ts
                v = TrajectoryPlanning.helper_function(q1_init, q1_final, a, end_time)
                traj1 = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q1_init, q1_final,
                                                                                   v[0], a, ts)
                traj2 = TrajectoryPlanning.trapezoidal_velocity_profile_trajectory(q1_init, q1_final,
                                                                                   v[1], a, ts)

        return np.vstack((q1_trajectory, q2_trajectory)).T

    @staticmethod
    def euclidean_trajectory(x_init, y_init, x_final, y_final, a, v, ts, link1_length, link2_length):
        pass


def main():
    ###################################### Trajectory Planning #####################################
    robot = rtb.models.DH.Planar2()

    # Joint Trajectory Planning
    q1_init, q2_init, q1_final, q2_final = 0, 0, np.pi/2, np.pi/2
    a, v, ts = 1, 1, 0.01

    trajectory = TrajectoryPlanning.joint_trajectory(q1_init, q2_init, q1_final, q2_final, a, v, ts)
    robot.plot(trajectory, dt=ts, backend='pyplot', movie= 'joint_trajectory_planning.gif')

    # # Euclidean Trajectory Planning
    # x_init, y_init, x_final, y_final = 0, 0, 1, 1
    # a, v, ts = 1, 1, 0.01
    # link1_length = link2_length = 1

    # trajectory = TrajectoryPlanning.euclidean_trajectory(x_init, y_init, x_final, y_final, a, v, ts,
    #                                                      link1_length, link2_length)
    # robot.plot(trajectory, dt=ts, backend='pyplot', movie='euclidean_trajectory_planning.gif')


if __name__ == '__main__':
    main()
