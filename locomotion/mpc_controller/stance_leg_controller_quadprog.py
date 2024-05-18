# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Sequence, Tuple
import pybullet as p
import numpy as np
import time

from locomotion.robots.motors import MotorCommand
from locomotion.gait_generator import gait_generator as gait_generator_lib
# from mpc_controller import leg_controller
from locomotion.mpc_controller import qp_torque_optimizer
# from config.variables import FORCE_DIMENSION, MAX_DDQ, MIN_DDQ, QP_KP, QP_KD
# from config.variables import QP_FRICTION_COEFF
from config.locomotion.controllers.stance_params import StanceControllerParams


# _FORCE_DIMENSION = 3
# KP = np.array((0., 0., 100., 100., 100., 0.))
# KD = np.array((40., 30., 10., 10., 10., 30.))
# MAX_DDQ = np.array((10., 10., 10., 20., 20., 20.))
# MIN_DDQ = -MAX_DDQ


class TorqueStanceLegController:
    """A torque based stance leg controller framework.

    Takes in high level robot like walking speed and turning speed, and
    generates necessary the torques for stance legs.
    """

    def __init__(
            self,
            robot: Any,
            gait_generator: Any,
            state_estimator: Any,
            stance_params: StanceControllerParams,
            desired_speed: Tuple[float, float] = (0, 0),
            desired_twisting_speed: float = 0,
            desired_com_height: float = 0.24,
            body_mass: float = 110 / 9.8,
            body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            num_legs: int = 4,
            # friction_coeffs: Sequence[float] = tuple([QP_FRICTION_COEFF] * 4),
    ):
        """Initializes the class.

        Tracks the desired position/velocity of the robot by computing proper joint
        torques using MPC module.

        Args:
          robot: A robot instance.
          gait_generator: Used to query the locomotion phase and leg states.
          state_estimator: Estimate the robot states (e.g. CoM velocity).
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_body_height: The standing height of the robot.
          body_mass: The total mass of the robot.
          body_inertia: The inertia matrix in the body principle frame. We assume
            the body principle coordinate frame has x-forward and z-up.
          num_legs: The number of legs used for force planning.
          friction_coeffs: The friction coeffs on the contact surfaces.
        """
        self._robot = robot
        self._params = stance_params
        self._gait_generator = gait_generator
        self._state_estimator = state_estimator
        self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
        self.desired_twisting_speed = desired_twisting_speed

        self._desired_body_height = desired_com_height
        self._body_mass = body_mass
        self._body_inertia = body_inertia
        self._num_legs = num_legs

        self._friction_coeffs = np.array(tuple([self._params.friction_coeff] * 4))

        self._qp_torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            robot_mass=self._body_mass,
            robot_inertia=self._body_inertia
        )

    def reset(self, current_time):
        del current_time

    def update(self, current_time):
        del current_time

    def _estimate_robot_height(self, contacts):
        if np.sum(contacts) == 0:
            # All foot in air, no way to estimate
            return self._desired_body_height
        else:
            # base_orientation = self._robot.GetBaseOrientation()
            base_orientation = self._robot.base_orientation_quaternion
            rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
                base_orientation)
            rot_mat = np.array(rot_mat).reshape((3, 3))

            # foot_positions = self._robot.GetFootPositionsInBaseFrame()
            foot_positions = self._robot.foot_positions_in_body_frame
            foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T
            # pylint: disable=unsubscriptable-object
            useful_heights = contacts * (-foot_positions_world_frame[:, 2])
            return np.sum(useful_heights) / np.sum(contacts)

    def get_action(self):
        """Computes the torque for stance legs."""
        print("----------------------------------- Stance Control Quadprog -----------------------------------")

        s = time.time()

        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_generator.desired_leg_states],
            dtype=np.int32)

        robot_com_position = np.array(self._state_estimator.com_position_in_ground_frame)
        # robot_com_position = np.array(
        #     (0., 0., self._estimate_robot_height(contacts)))

        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame

        # robot_com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw())
        robot_com_roll_pitch_yaw = np.array(
            p.getEulerFromQuaternion(self._state_estimator.com_orientation_quaternion_in_ground_frame))

        robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting

        # robot_com_roll_pitch_yaw_rate = self._robot.GetBaseRollPitchYawRate()
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))

        e1 = time.time()

        # Desired q and dq
        desired_com_position = np.array((0., 0., self._desired_body_height),
                                        dtype=np.float64)
        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)

        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)

        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)

        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack(
            (desired_com_velocity, desired_com_angular_velocity))

        # Desired ddq
        QP_KP = self._params.qp_kp
        QP_KD = self._params.qp_kd
        MIN_DDQ = self._params.min_ddq
        MAX_DDQ = self._params.max_ddq

        ss = time.time()
        desired_ddq = QP_KP * (desired_q - robot_q) + QP_KD * (desired_dq - robot_dq)
        desired_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)
        ee = time.time()
        print(f"desired_ddq calc time: {ee - ss}")

        e2 = time.time()

        # Calculate needed contact forces
        # contact_forces = qp_torque_optimizer.compute_contact_force(
        #     self._robot, desired_ddq, friction_coef=QP_FRICTION_COEFF, contacts=contacts)
        foot_positions = self._robot.foot_positions_in_body_frame
        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=desired_ddq,
            contacts=contacts,
            acc_weights=self._params.acc_weights,
            reg_weight=self._params.reg_weight
        )
        e3 = time.time()

        print(f"foot_positions_in_body_frame: {foot_positions}")
        print(f"desired_ddp: {desired_ddq}")
        print(f"contact_forces: {contact_forces}")
        print("//////////////////////////////////////////////////")

        action = {}
        for leg_id, force in enumerate(contact_forces):
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)
            for joint_id, torque in motor_torques.items():
                action[joint_id] = MotorCommand(desired_position=0,
                                                kp=0,
                                                desired_velocity=0,
                                                kd=0,
                                                desired_torque=torque)
        # print("After IK: {}".format(time.time() - start_time))

        print("-----------------------------------------------------------------------------------------------")

        e4 = time.time()
        print(f"stance part 1 time: {e1 - s}")
        print(f"stance part 2 time: {e2 - e1}")
        print(f"stance part 3 time: {e3 - e2}")
        print(f"stance part 4 time: {e4 - e3}")

        return action, contact_forces

    def get_final_action(self, current_step, states_vector, drl_action=None):

        contacts = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_generator.desired_leg_states],
            dtype=np.float64)

        return self.get_drl_action(current_step, drl_action)  # original mpc
        # if len(np.nonzero(contacts)[0]) < 4:
        # return self.get_action_our(current_step, states_vector, drl_action)
        # else:
        # return self.get_action(drl_action)

    def get_drl_action(self, current_step, drl_action=None):
        """Computes the torque for stance legs."""
        s = time.time()
        # Actual q and dq
        contacts = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT))
             for leg_state in self._gait_generator.desired_leg_states],
            dtype=np.float64)

        # this is relative positions of the leg to the base
        # foot_positions = self._robot.GetFootPositionsInBaseFrame()
        foot_positions = self._robot.foot_positions_in_body_frame

        robot_com_height = self._estimate_robot_height(contacts)
        robot_com_velocity = self._state_estimator.com_velocity_in_body_frame

        # robot_com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw())
        robot_com_roll_pitch_yaw = np.array(self._robot.base_orientation_rpy)

        robot_com_roll_pitch_yaw[2] = 0.  # To prevent yaw drifting  why??????

        # robot_com_roll_pitch_yaw_rate = self._robot.GetBaseRollPitchYawRate()
        robot_com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        # robot_q = np.hstack(([0., 0., robot_com_height], robot_com_roll_pitch_yaw))

        robot_q = np.hstack((self._state_estimator.estimate_robot_x_y_z(), robot_com_roll_pitch_yaw))
        robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))
        # Desired q and dq

        # print(robot_dq)
        # print(robot_q)

        sastate = np.hstack(
            (self._state_estimator.estimate_robot_x_y_z(),
             robot_com_roll_pitch_yaw,
             robot_com_velocity,
             robot_com_roll_pitch_yaw_rate)
        )

        # np.savetxt('myfile.txt', robot_q, fmt='%.2f')

        # with open("games.txt", "a") as text_file:
        # print(robot_q[0])
        # text_file.write(robot_q[0] + "\n")

        # with open("phydrl10vp1.txt", 'a') as myfile:
        #     myfile.write(', '.join(map(str, sastate)) + '\n')

        # print(robot_dq)
        # print(robot_q)

        # current_time = current_step * 0.002
        reference_vx = 1.0
        # reference_p_x = reference_vx * current_time
        reference_p_x = 0
        desired_com_position = np.array((reference_p_x, 0., self._desired_body_height), dtype=np.float64)
        # desired_com_position = np.array((0., 0., self._desired_body_height), dtype=np.float64)

        desired_com_velocity = np.array(
            (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
        desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
        desired_com_angular_velocity = np.array(
            (0., 0., self.desired_twisting_speed), dtype=np.float64)

        desired_q = np.hstack((desired_com_position, desired_com_roll_pitch_yaw))
        desired_dq = np.hstack(
            (desired_com_velocity, desired_com_angular_velocity))
        # Desired ddq

        # print(desired_dq)
        # print(desired_q)

        # print(KP)
        KP = self._robot.stance_params.qp_kp
        KD = self._robot.stance_params.qp_kd
        desired_ddq = KP * (desired_q - robot_q) + KD * (desired_dq - robot_dq)

        # desired_ddq = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # print(desired_ddq)

        # sys_sta = np.hstack((desired_q - robot_q,   desired_dq - robot_dq))  #1
        # sys_sta = np.hstack((desired_dq - robot_dq, desired_q - robot_q)) #2

        # F = np.array([[-9.85733029e+01, -7.33928840e-01, -5.39730576e+02, 1.12574255e-02,
        #               -1.68112514e+00, -8.25854425e-04, -4.84707618e+02, -7.46038513e+01,
        #               1.57651354e+01, 1.95353054e+00, -4.05542712e+01, -6.29620208e-02],
        #              [-8.62214768e+01, -1.21398935e+00, -5.48319011e+02, -1.62316156e-02,
        #               5.47610672e-02, -4.26575220e-05, 5.46928876e+01, -1.21190759e+02,
        #               -9.48617185e-01, 3.14723556e+00, 4.98731397e+00, -1.00339414e-01],
        #              [9.48955055e+02, -4.20695748e+01, -4.79076177e+04, 5.54561147e-01,
        #               2.96897008e+00, -3.61574959e-02, 4.52210096e+03, -4.19632528e+03,
        #              -1.32601670e+02, 1.12436390e+02, 4.13661495e+02, -3.56382810e+00],
        #               [1.79243200e+02, -2.63900786e+00, -3.27571956e+03, -1.28993992e+00,
        #               1.03743758e-01, 4.52116647e-03, 2.88005230e+02, -2.60303632e+02,
        #              -8.62891884e+00, 6.97317625e+00, 2.64829476e+01, -2.21041035e-01],
        #              [1.86198782e+03, -5.69702178e+00, -1.10568625e+04, 2.83083845e-02,
        #              -3.90302915e-01, -3.38790662e-03, 8.36077697e+02, -5.67812798e+02,
        #              -2.76332649e+01, 1.57548882e+01, 7.77952740e+01, -4.96166917e-01],
        #              [-5.61496850e+00, 8.07524602e-02, 1.00290230e+02, 5.66607925e-03,
        #              1.23894955e-03, -1.00595011e+00, -7.46926314e+00, 7.97819272e+00,
        #               2.21338194e-01, -2.14289001e-01, -6.97152597e-01, -1.65604852e-02]]) * -0.001

        # test_desired_ddq = F @ sys_sta

        # desired_ddq = F @ sys_sta

        # add residual actions from DRL
        if drl_action is not None:
            # print(drl_action)

            drl_action *= 1  # 1 0.8

            # drl_action *= 20  # 1 0.8
            desired_ddq += drl_action

            # desired_ddq += 0

        # print(desired_ddq)

        MIN_DDQ = self._robot.stance_params.min_ddq
        MAX_DDQ = self._robot.stance_params.max_ddq
        desired_ddq = np.clip(desired_ddq, MIN_DDQ, MAX_DDQ)

        # print(desired_ddq)

        e1 = time.time()
        contact_forces = self._qp_torque_optimizer.compute_contact_force(
            foot_positions=foot_positions,
            desired_acc=desired_ddq,
            contacts=contacts,
            acc_weights=self._params.acc_weights,
            reg_weight=self._params.reg_weight
        )
        # contact_forces = self._qp_torque_optimizer.compute_contact_force(
        #     foot_positions, desired_ddq, contacts=contacts)
        e2 = time.time()

        print(f"part 1: {e1 - s}")
        print(f"part 2: {e2 - e1}")

        action = {}

        for leg_id, force in enumerate(contact_forces):
            # print("MPC force", force)
            # print("DRL force", force)
            # force *= 0.5
            # While "Lose Contact" is useful in simulation, in real environment it's
            # susceptible to sensor noise. Disabling for now.
            # if self._gait_generator.leg_state[
            #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
            #   force = (0, 0, 0)
            # motor_torques = self._robot.MapContactForceToJointTorques(leg_id, force)
            motor_torques = self._robot.map_contact_force_to_joint_torques(leg_id, force)

            for joint_id, torque in motor_torques.items():
                # action[joint_id] = (0, 0, 0, 0, torque)
                action[joint_id] = MotorCommand(desired_position=0,
                                                kp=0,
                                                desired_velocity=0,
                                                kd=0,
                                                desired_torque=torque)

        difference_q = desired_q - robot_q
        difference_dq = desired_dq - robot_dq

        return action, contact_forces, difference_q, difference_dq
