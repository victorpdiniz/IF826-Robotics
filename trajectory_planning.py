import numpy as np
from kinematics import Kinematics2DOF

def s(t, a, v, acceleration_time, endTime):
    if 0 <= t < acceleration_time:
        return a * t**2 / 2
    elif acceleration_time <= t < endTime - acceleration_time:
        return v * t - v**2 / (2 * a)
    elif endTime - acceleration_time <= t <= endTime:
        return v * t - a * (t - endTime)**2 / 2
    else:
        return 0

class TrajectoryPlanning:
    @staticmethod
    def trapezoidal_velocity_trajectory(s_init, s_final, v_max, a_max, ts):
        delta_s = abs(s_final - s_init)  

        acceleration_time = v_max / a_max

        endTime = 2 * acceleration_time + delta_s / v_max

        s_acceleration = a_max * acceleration_time**2 / 2

        if endTime < delta_s / v_max or endTime > 2 * delta_s / v_max or 2 * s_acceleration < delta_s:
            return None

        t = np.arange(0, endTime + ts, ts)

        direction = 1 if s_final >= s_init else -1

        trajectory = [s_init + direction * s(t_i, a_max, v_max, acceleration_time, endTime) for t_i in t]

        return np.array(trajectory)

            
           
    @staticmethod
    def joint_trajectory(q1_init, q2_init, q1_final, q2_final, v_max, a_max, ts):
        q1_trajectory = TrajectoryPlanning.trapezoidal_velocity_trajectory(q1_init, q1_final, v_max, a_max, ts)
        q2_trajectory = TrajectoryPlanning.trapezoidal_velocity_trajectory(q2_init, q2_final, v_max, a_max, ts)
        return np.vstack((q1_trajectory, q2_trajectory)).T

    @staticmethod
    def euclidean_trajectory(x_init ,y_init ,x_final ,y_final ,v_max ,a_max ,ts):
        delta_x = x_final - x_init
        delta_y = y_final - y_init
        distance = np.sqrt(delta_x**2 + delta_y**2)
        trajectory = TrajectoryPlanning.trapezoidal_velocity_trajectory(0, distance, v_max, a_max, ts)
        
        if trajectory is None:
            return None

        x_trajectory = x_init + trajectory * delta_x / distance
        y_trajectory = y_init + trajectory * delta_y / distance

        return np.vstack((x_trajectory, y_trajectory)).T
        


def main():
    q_joint = TrajectoryPlanning.joint_trajectory(0, 0, np.pi/4, np.pi/3, v_max=1.0, a_max=0.5, ts=0.01)
    if q_joint is not None:
        print("joint:")
        print(q_joint)

    q_euclid = TrajectoryPlanning.euclidean_trajectory(0, 0, 1, 1, v_max=1.0, a_max=0.5, ts=0.01)
    if q_euclid is not None:
        print("euclidean:")
        print(q_euclid)


if __name__ == '__main__':
    main()