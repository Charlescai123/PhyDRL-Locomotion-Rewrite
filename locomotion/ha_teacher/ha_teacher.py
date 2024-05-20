import multiprocessing

import matlab
import matlab.engine
from omegaconf import DictConfig
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv
from locomotion.robots.motors import MotorCommand
from locomotion.robots.motors import MotorControlMode
import multiprocessing as mp
import numpy as np
# import cvxpy as cp
from typing import Tuple, Any
import copy
import enum
import time


class HACActionMode(enum.Enum):
    """The state of a leg during locomotion."""
    MPC = 0
    PHYDRL = 1
    SIMPLEX = 2


class HATeacher:
    def __init__(self,
                 robot: Any,
                 mat_engine: Any):

        self._robot = robot

        # HAC Configure
        self.chi = 0.25
        self.epsilon = 0.6
        self.dwell_step_max = 100
        self.teacher_enable = True
        self.continual_learn = True
        self.teacher_learn = True
        self.p_mat = np.array([[122.164786064669, 0, 0, 0, 2.48716597374493, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 480.62107526958, 0, 0, 0, 0, 0, 155.295455907449],
                               [2.48716597374493, 0, 0, 0, 3.21760325418695, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 155.295455907449, 0, 0, 0, 0, 0, 156.306807893237]])

        # HAC Runtime
        self._dwell_time = 0
        self._last_action_mode = None
        self.action_mode_list = []

        # Multiprocessing
        self._lock = mp.Lock()
        self._state_trig = mp.RawArray('d', 12 * 1)
        # self.shared_arr = mp.Array('d', 12 * 1)
        self.state_update_flag = mp.Value('b', False)
        self.mat_process = mp.Process(target=self.update_feedback_gain,
                                      args=(self.state_update_flag, self._state_trig, self._lock))
        self.mat_process.daemon = True  # Set to be daemon process

        self._F_kp = np.diag((0., 0., 100., 100., 100., 0.))
        self._F_kd = np.diag((40., 30., 10., 10., 10., 30.))

        # self._F_kp = np.array([[0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0],
        #                        [0, 0, 128, 0, 0, 0],
        #                        [0, 0, 0, 83, -25, -2],
        #                        [0, 0, 0, -33, 80, 2],
        #                        [0, 0, 0, 1, 0, 80]])
        #
        # self._F_kd = np.array([[39, 0, 0, 0, 0, 0],
        #                        [0, 35, 0, 0, 0, 0],
        #                        [0, 0, 35, 0, 0, 0],
        #                        [0, 0, 0, 37, -1, -9],
        #                        [0, 0, 0, -1, 37, 9],
        #                        [0, 0, 0, 0, 0, 40]])
        self.mat_engine = mat_engine
        self.mat_process.start()
        # self.mat_process.join()

        # Matlab Engine
        # self.cvx_setup()
        # self.matlab_engine_launch()

    def matlab_engine_launch(self, path="./locomotion/ha_teacher"):
        self.mat_engine = matlab.engine.start_matlab()
        self.mat_engine.cd(path)
        print("Matlab current working directory is ---->>>", self.mat_engine.pwd())

    def update_feedback_gain(self, update_flag, state_trig, lock):
        while True:
            print("123")
            print("LMI process monitoring...")
            print(f"update_flag: {update_flag}")
            print(f"update_flag value: {update_flag.value}")
            print(f"state_trig: {state_trig}")
            if update_flag.value == 1 and state_trig is not None:
                # if lock.acquire():
                roll, pitch, yaw = state_trig[3], state_trig[4], state_trig[5]
                print("Obtained new state, updating the feedback gain using matlab engine")
                self._F_kp, self._F_kd = self.mat_engine.feedback_law2(roll, pitch, yaw, nargout=2)
                update_flag.value = 0
                print("Feedback gain is updated now ---->>>")
                # lock.release()
            time.sleep(0.005)

    def feedback_law(self, roll, pitch, yaw):
        # roll = matlab.double(roll)
        # pitch = matlab.double(pitch)
        # yaw = matlab.double(yaw)
        # self._F_kp, self._F_kd = np.array(self.mat_engine.feedback_law2(roll, pitch, yaw, nargout=2))
        return self._F_kp, self._F_kd

    def safety_value(self, states, p_mat=None):
        if p_mat is None:
            p_mat = self.p_mat

        states_shrink = states[2:]
        safety_val = np.squeeze(states_shrink.transpose() @ p_mat @ states_shrink)

        print(f"states: {states}")
        print(f"safety val: {safety_val}")
        return safety_val

    def get_hac_action(self, states):

        safety_val = self.safety_value(states)

        # Inside Safety Envelope (bounded by epsilon)
        if safety_val <= self.epsilon:

            # HAC Dwell-Time
            if self._last_action_mode == HACActionMode.SIMPLEX:

                if self._dwell_time < self.dwell_step_max:
                    print(f"current dwell time: {self._dwell_time}")
                    self._dwell_time += 1

                # Switch back to HPC (If Simplex enabled)
                elif self.continual_learn:
                    self._dwell_time = 0
                    self._state_trig = None
                    self.state_update_flag.value = False
                    s_action = time.time()
                    if self._robot.controller.ddpg_agent is not None:
                        self._last_action_mode = HACActionMode.PHYDRL
                    else:
                        self._last_action_mode = HACActionMode.MPC
                    e_action = time.time()
                    print(f"get action duration: {e_action - s_action}")
                    print(f"Simplex control switch back to {self._last_action_mode} control")

            else:
                self._dwell_time = 0
                s_action = time.time()
                if self._robot.controller.ddpg_agent is not None:
                    self._last_action_mode = HACActionMode.PHYDRL
                else:
                    self._last_action_mode = HACActionMode.MPC
                e_action = time.time()
                print(f"get action duration: {e_action - s_action}")

        # Outside Safety Envelope (bounded by epsilon)
        else:
            if self._last_action_mode != HACActionMode.SIMPLEX:
                print(f"Safety value {safety_val} is "
                      f"outside epsilon range: {self.epsilon}, switch to simplex")
                if self._state_trig is None:
                    self._state_trig = mp.RawArray('d', 12 * 1)
                    arr = states.reshape(12, 1)
                else:
                    arr = np.frombuffer(self._state_trig, dtype=np.float64).reshape(12, 1)
                    arr = states.reshape(12, 1)
                self.state_update_flag.value = 1
                self._last_action_mode = HACActionMode.SIMPLEX

        # Append for record
        self.action_mode_list.append(self._last_action_mode)

        # Get action
        if self._last_action_mode == HACActionMode.PHYDRL:
            action, qp_sol = self._robot.controller.get_action(phydrl=True)
            return action, qp_sol

        elif self._last_action_mode == HACActionMode.MPC:
            action, qp_sol = self._robot.controller.get_action(phydrl=False)
            return action, qp_sol

        elif self._last_action_mode == HACActionMode.SIMPLEX:
            # Swing action
            s = time.time()
            swing_action = self._robot.controller.swing_leg_controller.get_action()
            e_swing = time.time()

            # Stance action
            rpy = np.asarray(states[3:6])
            s = time.time()
            F_kp, F_kd = self.feedback_law(rpy[0], rpy[1], rpy[2])
            e = time.time()
            print(f"F_kp is: {F_kp}")
            print(f"F_kd is: {F_kd}")
            print(f"LMI time duration: {e - s}")
            state_trig = np.frombuffer(self._state_trig, dtype=np.float64).reshape(12, 1)
            stance_action, qp_sol = self._robot.controller.stance_leg_controller.get_hac_action(
                chi=self.chi,
                state_trig=state_trig,
                F_kp=F_kp,
                F_kd=F_kd
            )

            e_stance = time.time()
            print(f"swing_action time: {e_swing - s}")
            print(f"stance_action time: {e_stance - e_swing}")
            print(f"total get_action time: {e_stance - s}")

            actions = []
            for joint_id in range(self._robot.num_motors):
                if joint_id in swing_action:
                    actions.append(swing_action[joint_id])
                else:
                    assert joint_id in stance_action
                    actions.append(stance_action[joint_id])

            vectorized_action = MotorCommand(
                desired_position=[action.desired_position for action in actions],
                kp=[action.kp for action in actions],
                desired_velocity=[action.desired_velocity for action in actions],
                kd=[action.kd for action in actions],
                desired_torque=[
                    action.desired_torque for action in actions
                ])

            return vectorized_action, dict(qp_sol=qp_sol)

        else:
            raise RuntimeError(f"Unsupported HAC action {self._last_action_mode}")

    @property
    def feedback_gain(self):
        return self._F_kp, self._F_kd

    @property
    def last_action_mode(self):
        return self._last_action_mode

    @property
    def dwell_time(self):
        return self._dwell_time

    @last_action_mode.setter
    def last_action_mode(self, action_mode: HACActionMode):
        self._last_action_mode = action_mode


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])

    ha_teacher = HATeacher()
    K = ha_teacher.feedback_law(0, 0, 0)
    print(K)
