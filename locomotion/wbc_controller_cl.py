"""A phydrl based controller framework."""
from absl import logging

import enum
import ml_collections
import numpy as np
import os
import pickle
import pybullet
from datetime import datetime
from pybullet_utils import bullet_client
import threading
import multiprocessing
import time
from typing import Tuple, Any
import copy

from envs.worlds import plane_world, abstract_world

from config.locomotion.robots.pose import Pose
from config.locomotion.robots.a1_params import A1Params
from config.locomotion.gait_scheduler import crawl, trot
from config.locomotion.gait_scheduler import flytrot
from config.locomotion.robots.a1_robot_params import A1RobotParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams

from agents.phydrl.policies.ddpg import DDPGAgent
from config.a1_phydrl_params import A1PhyDRLParams
# from locomotion.robot.a1 import A1
from agents.phydrl.replay_buffers.replay_memory import ReplayMemory
from locomotion.mpc_controller import swing_leg_controller
from locomotion.gait_scheduler import offset_gait_scheduler
from locomotion.state_estimator import com_velocity_estimator

from locomotion.mpc_controller import stance_leg_controller_mpc
from locomotion.mpc_controller import stance_leg_controller_quadprog

from locomotion.robots.motors import MotorCommand
from locomotion.ha_teacher import ha_teacher


class ControllerMode(enum.Enum):
    DOWN = 1
    STAND = 2
    WALK = 3
    TERMINATE = 4


class GaitType(enum.Enum):
    CRAWL = 1
    TROT = 2
    FLYTROT = 3


class WholeBodyController(object):
    """Whole Body Controller for the robot's entire locomotion.

    The actual effect of this controller depends on the composition of each
    individual subcomponent.

    """

    def __init__(
            self,
            robot: Any = None,
            mat_engine: Any = None,
            ddpg_agent: DDPGAgent = None,
            desired_speed: Tuple[float, float] = [0., 0.],
            desired_twisting_speed: float = 0.,
            desired_com_height: float = 0.,
            mpc_body_mass: float = 110 / 9.8,
            mpc_body_inertia: Tuple[float, float, float, float, float, float, float, float, float] = (
                    0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447),
            swing_params: SwingControllerParams = None,
            stance_params: StanceControllerParams = None,
            logdir: str = 'logs/',
    ):
        """Initializes the class.

        Args:
          robot: A robot instance. (Provides sensor input and kinematics)
          ddpg_agent: An agent instance used to get PhyDRL action (WBC will get mpc action if None)
          desired_speed: desired CoM speed in x-y plane.
          desired_twisting_speed: desired CoM rotating speed in z direction.
          desired_com_height: The standing height of CoM of the robot.
          mpc_body_mass: The total mass of the robot.
          mpc_body_inertia: The inertia matrix in the body principle frame. We assume the body principle
                            coordinate frame has x-forward and z-up.
          swing_params: Parameters for swing leg controller.
          stance_params: Parameters for stance leg controller.
        """

        self._robot = robot
        self._ddpg_agent = ddpg_agent
        self._gait_scheduler = None
        self._velocity_estimator = None
        self._swing_params = swing_params
        self._stance_params = stance_params

        self._ha_teacher = ha_teacher.HATeacher(
            robot=robot,
            mat_engine=mat_engine
        )  # HA Teacher
        self._replay_buffer = ReplayMemory(size=5e4)
        self._max_episode_steps = 5000
        self._total_steps = 50000
        self._minibatch_size = 128
        self._logs = []
        self._logdir = logdir

        self._desired_speed = desired_speed
        self._desired_twisting_speed = desired_twisting_speed

        self._desired_com_height = desired_com_height
        self._mpc_body_mass = mpc_body_mass
        self._mpc_body_inertia = mpc_body_inertia

        self._desired_mode = None
        self._is_control = False

        self._reset_time = None
        self._setup_controllers()

        # self._reset_time = self._clock()
        self._time_since_reset = 0

        self._mode = ControllerMode.WALK
        self.set_controller_mode(ControllerMode.WALK)
        self._gait = None
        self._desired_gait = GaitType.TROT

        # self._handle_gait_switch()

        self._control_thread = None
        # self._control_thread = threading.Thread(target=self.run)
        # self.run_thread.start()
        # self.run_thread = multiprocessing.Process(target=self.run)
        # self.run_thread.start()

        vx = self._desired_speed[0]
        vy = self._desired_speed[1]
        wz = self._desired_twisting_speed
        pz = self._desired_com_height
        self.set_point = np.array([0., 0., pz,  # p
                                   0., 0., 0.,  # rpy
                                   vx, vy, 0.,  # v
                                   0., 0., wz])  # rpy_dot

        self.reset_controllers()

    def _setup_controllers(self):
        print("Setting up the whole body controller...")
        self._clock = lambda: self._robot.time_since_reset

        # Gait Generator
        print("Setting up the gait generator")
        init_gait_phase = self._robot.robot_params.init_gait_phase
        gait_params = self._robot.robot_params.gait_params
        self._gait_scheduler = offset_gait_scheduler.OffsetGaitScheduler(
            robot=self._robot,
            init_phase=init_gait_phase,
            gait_parameters=gait_params
        )

        # State Estimator
        print("Setting up the state estimator")
        window_size = self._robot.robot_params.window_size
        ground_normal_window_size = self._robot.robot_params.ground_normal_window_size
        self._velocity_estimator = com_velocity_estimator.COMVelocityEstimator(
            robot=self._robot,
            velocity_window_size=window_size,
            ground_normal_window_size=ground_normal_window_size
        )

        # Swing Leg Controller
        print("Setting up the swing leg controller")
        self._swing_controller = \
            swing_leg_controller.RaibertSwingLegController(
                robot=self._robot,
                gait_scheduler=self._gait_scheduler,
                state_estimator=self._velocity_estimator,
                desired_speed=self._desired_speed,
                desired_twisting_speed=self._desired_twisting_speed,
                desired_com_height=self._desired_com_height,
                swing_params=self._swing_params
            )

        # Stance Leg Controller
        print("Setting up the stance leg controller")
        if self._stance_params.objective_function == 'acceleration':
            self._stance_controller = \
                stance_leg_controller_quadprog.TorqueStanceLegController(
                    robot=self._robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_params
                )

        elif self._stance_params.objective_function == 'state':
            self._stance_controller = \
                stance_leg_controller_mpc.TorqueStanceLegController(
                    robot=self._robot,
                    gait_scheduler=self._gait_scheduler,
                    state_estimator=self._velocity_estimator,
                    desired_speed=self._desired_speed,
                    desired_twisting_speed=self._desired_twisting_speed,
                    desired_com_height=self._desired_com_height,
                    body_mass=self._mpc_body_mass,
                    body_inertia=self._mpc_body_inertia,
                    stance_params=self._stance_params
                )

        else:
            raise RuntimeError("Unspecified objective function for stance controller")

        print("Whole body controller settle down!")

    @property
    def ddpg_agent(self):
        return self._ddpg_agent

    @property
    def control_thread(self):
        return self._control_thread

    def stop_thread(self):
        self._is_control = False
        # self._control_thread.join()

    def start_thread(self):
        self._is_control = True
        self._control_thread = threading.Thread(target=self.run)
        self._control_thread.start()

    @property
    def swing_leg_controller(self):
        return self._swing_controller

    @property
    def stance_leg_controller(self):
        return self._stance_controller

    @property
    def gait_scheduler(self):
        return self._gait_scheduler

    @property
    def state_estimator(self):
        return self._velocity_estimator

    @property
    def time_since_reset(self):
        return self._time_since_reset

    # def reset_robot(self):
    #     # self._robot.reset(hard_reset=False)
    #     if self._show_gui and not self._use_real_robot:
    #         self.pybullet_client.configureDebugVisualizer(
    #             self.pybullet_client.COV_ENABLE_RENDERING, 1)

    def reset_controllers(self):
        print("Reset robot whole body controller...")
        self._reset_time = self._clock()
        self._time_since_reset = 0
        self._gait_scheduler.reset()
        self._velocity_estimator.reset(self._time_since_reset)
        self._swing_controller.reset(self._time_since_reset)
        self._stance_controller.reset(self._time_since_reset)
        self._is_control = False

    def update(self):
        self._time_since_reset = self._clock() - self._reset_time
        # print(f"self._time_since_reset: {self._time_since_reset}")

        self._gait_scheduler.update()
        self._velocity_estimator.update(self._gait_scheduler.desired_leg_states)
        self._swing_controller.update(self._time_since_reset)
        # future_contact_estimate = self._gait_scheduler.get_estimated_contact_states(
        #     stance_leg_controller_mpc.PLANNING_HORIZON_STEPS,
        #     stance_leg_controller_mpc.PLANNING_TIMESTEP)
        # self._stance_controller.update(self._time_since_reset, future_contact_estimate)
        self._stance_controller.update(self._time_since_reset)

    def get_action(self, phydrl=False, drl_action=None, simplex=False):
        """Returns the control outputs (e.g. positions/torques) for all motors."""
        drl_nominal_ddq = None
        # Get PhyDRL action (Inference)
        if phydrl is True and drl_action is None:
            # print(f"Getting action from PhyDRL model {self._ddpg_agent.params.model_path}")

            # State vector
            state_vector = self.state_vector

            # Tracking error
            tracking_error = state_vector - self.set_point
            print(f"states_vector: {state_vector}")
            # print(f"set_points: {set_points}")
            print(f"tracking_error: {tracking_error}")

            # Observation
            observation = tracking_error

            s_drl = time.time()
            drl_action = self._ddpg_agent.get_action(observation, mode='test')

            if simplex:
                drl_nominal_ddq = copy.deepcopy(drl_action)

            # drl_action_magnitude = np.array([2, 2, 3, 4, 4, 2])
            drl_action_magnitude = np.array([2, 2, 3, 4, 4, 2])
            drl_action *= drl_action_magnitude

            e_drl = time.time()
            print(f"get drl action time: {e_drl - s_drl}")

        # Action delay
        if self._ddpg_agent is not None and self._ddpg_agent.params.add_action_delay:
            print("add action delay...")
            drl_action = self._ddpg_agent.get_delayed_action(drl_action)

        s = time.time()
        swing_action = self._swing_controller.get_action()
        e_swing = time.time()
        stance_action, qp_sol, phy_nominal_ddq = self._stance_controller.get_action(drl_action=drl_action)
        e_stance = time.time()
        # print(f"swing_action time: {e_swing - s}")
        # print(f"stance_action time: {e_stance - e_swing}")
        # print(f"total get_action time: {e_stance - s}")

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

        return vectorized_action, dict(qp_sol=qp_sol), drl_nominal_ddq, phy_nominal_ddq

    def _get_stand_action(self):
        return MotorCommand(
            # desired_position=self._robot.motor_group.init_positions,
            desired_position=Pose.STANDING_POSE,
            kp=self._robot.motor_group.kps,
            desired_velocity=0,
            kd=self._robot.motor_group.kds,
            desired_torque=0)

    def _handle_mode_switch(self):
        print("Entering _handle_mode_switch")
        if self._mode == self._desired_mode:
            return
        self._mode = self._desired_mode
        if self._desired_mode == ControllerMode.DOWN:
            logging.info("Entering joint damping mode.")
            self._flush_logging()

        elif self._desired_mode == ControllerMode.STAND:
            logging.info("Standing up.")
            # self.reset_robot()  # Reset the robot if in standing pose

        else:
            logging.info("Walking.")
            # self.reset_controllers()
            self._start_logging()

    def _start_logging(self):
        self._logs = []

    def _update_logging(self, action, qp_sol):
        frame = dict(
            timestamp=self._time_since_reset,
            tracking_error=self._stance_controller.tracking_error,
            desired_speed=(self._swing_controller.desired_speed,
                           self._swing_controller.desired_twisting_speed),
            desired_com_height=self._desired_com_height,
            # step_counter=self._robot.step_counter,
            # action_counter=self._robot.action_counter,
            base_position=self._robot.base_position,
            base_rpy=self._robot.base_orientation_rpy,
            base_vel=self._robot.motor_velocities,
            base_linear_vel_in_body_frame=self._velocity_estimator.com_velocity_in_body_frame,
            base_angular_vel_in_body_frame=self._robot.base_angular_velocity_in_body_frame,
            motor_angles=self._robot.motor_angles,
            motor_vels=self._robot.motor_velocities,
            motor_torques=self._robot.motor_torques,
            foot_contacts=self._robot.foot_contacts,
            swing_action=self._swing_controller.swing_action,
            stance_action=self._stance_controller.stance_action,
            stance_ddq=self._stance_controller.stance_ddq,
            stance_ddq_limit=self._stance_controller.stance_ddq_limit,
            desired_ground_reaction_forces=self._stance_controller.ground_reaction_forces,
            gait_scheduler_phase=self._gait_scheduler.current_phase.copy(),
            leg_states=self._gait_scheduler.leg_states,
            ground_orientation=self._velocity_estimator.ground_orientation_in_world_frame,
        )
        # print(f"ground_reaction_forces: {self._stance_controller.ground_reaction_forces}")
        self._logs.append(frame)

    def _flush_logging(self):
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
        filename = 'log_{}.pkl'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        pickle.dump(self._logs, open(os.path.join(self._logdir, filename), 'wb'))
        logging.info("Data logged to: {}".format(os.path.join(self._logdir, filename)))

    def _handle_gait_switch(self):
        print("Entering _handle_gait_switch")
        if self._gait == self._desired_gait:
            return
        if self._desired_gait == GaitType.CRAWL:
            logging.info("Switched to Crawling gait.")
            self._gait_config = crawl.get_config()

        elif self._desired_gait == GaitType.TROT:
            logging.info("Switched  to Trotting gait.")
            self._gait_config = trot.get_config()

        else:
            logging.info("Switched to Fly-Trotting gait.")
            self._gait_config = flytrot.get_config()

        self._gait = self._desired_gait
        self._gait_scheduler.gait_params = self._gait_config.gait_parameters
        self._swing_controller.foot_lift_height = self._gait_config.foot_clearance_max
        self._swing_controller.foot_landing_clearance = \
            self._gait_config.foot_clearance_land

    def run(self):
        logging.info("Low level thread started...")
        curr_time = time.time()
        print(f"control_thread: {self.control_thread}")
        dwell_steps = 0
        global_steps = 0

        while self._is_control:

            self._handle_mode_switch()
            # self._handle_gait_switch()
            self.update()

            print(f"vx: {self._stance_controller.desired_speed}")
            print(f"mode is: {self.mode}")
            # time.sleep(1)

            if self._mode == ControllerMode.DOWN:
                pass
                # time.sleep(0.1)

            elif self._mode == ControllerMode.STAND:
                action = self._get_stand_action()
                self._robot.step(action)
                # time.sleep(0.001)

            elif self._mode == ControllerMode.WALK:
                action = None
                curr_state = self.stance_leg_controller.tracking_error
                reward_list = []
                critic_loss_list = []
                ep_steps = 0

                for _ in range(self._max_episode_steps):

                    # Simplex Enable
                    if self._ha_teacher.teacher_enable:
                        action, qp_sol, drl_nominal_ddq = self._ha_teacher.get_hac_action(states=curr_state)
                    # Without Simplex
                    else:
                        print("Simplex non implemented for continual learning")

                    ss = time.time()
                    self._robot.step(action)
                    next_state = self.stance_leg_controller.tracking_error  # Current state
                    reward = get_lyapunov_reward(s=curr_state, s_next=next_state)
                    self._replay_buffer.add((curr_state, drl_nominal_ddq, reward, next_state, False))

                    reward_list.append(reward)

                    if self._replay_buffer.get_size() > self._minibatch_size:
                        minibatch = self._replay_buffer.sample(self._minibatch_size)
                        critic_loss = self.ddpg_agent.optimize(minibatch)
                    else:
                        critic_loss = 100

                    critic_loss_list.append(critic_loss)
                    curr_state = copy.deepcopy(next_state)

                    global_steps += 1
                    ep_steps += 1
                    ee = time.time()
                    s_log = time.time()
                    self._update_logging()
                    e_log = time.time()
                    print(f"log update time: {e_log - s_log}")
                    print(f"step duration: {ee - ss}")

            else:
                logging.info("Running loop terminated, exiting...")
                break

                mean_reward = np.mean(reward_list)
                mean_critic_loss = np.mean(critic_loss_list)
                print(f"Training at {global_steps}, average_reward{mean_reward}, average_critic{mean_critic_loss}")

            self.ddpg_agent.save_weights(self._logdir + '/online_cl_model')

            with open(self._logdir + '/buffer.pickle', 'wb') as f:
                pickle.dump(self._replay_buffer, f)

            final_time = time.time()
            duration = final_time - curr_time
            curr_time = final_time
            # time.sleep(0.0001)
            print(f"running times in this loop:{duration}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def set_controller_mode(self, mode):
        self._desired_mode = mode

    def set_gait(self, gait):
        self._desired_gait = gait

    @property
    def is_safe(self):
        if self.mode != ControllerMode.WALK:
            return True
        rot_mat = np.array(
            self._robot.pybullet_client.getMatrixFromQuaternion(
                self._velocity_estimator.com_orientation_quaternion_in_ground_frame)).reshape(
            (3, 3))
        up_vec = rot_mat[2, 2]  # Check the body tilting
        base_height = self._robot.base_position[2]  # Check the body height
        print("============================ checking robot safety =============================")
        print(f"up_vec is: {up_vec}")
        print(f"base_height is: {base_height}")
        print("================================================================================")

        return up_vec > 0.85 and base_height > self._robot.robot_params.safe_height

    @property
    def mode(self):
        return self._mode

    def set_desired_speed(self, desired_lin_speed_ratio,
                          desired_rot_speed_ratio):
        # desired_lin_speed = (
        #     self._gait_config.max_forward_speed * desired_lin_speed_ratio[0],
        #     self._gait_config.max_side_speed * desired_lin_speed_ratio[1],
        #     0,
        # )
        # desired_rot_speed = \
        #     self._gait_config.max_rot_speed * desired_rot_speed_ratio

        if len(desired_lin_speed_ratio) == 3:
            desired_lin_speed = desired_lin_speed_ratio
            desired_lin_speed[2] = 0
        else:
            desired_lin_speed = (
                desired_lin_speed_ratio[0],
                desired_lin_speed_ratio[1],
                0)

        desired_rot_speed = desired_rot_speed_ratio

        # print(f"set desired_lin_speed: {desired_lin_speed}")
        # print(f"set desired_rot_speed: {desired_rot_speed}")

        self._swing_controller.desired_speed = desired_lin_speed
        self._swing_controller.desired_twisting_speed = desired_rot_speed
        self._stance_controller.desired_speed = desired_lin_speed
        self._stance_controller.desired_twisting_speed = desired_rot_speed

    def set_gait_parameters(self, gait_parameters):
        raise NotImplementedError()

    def set_qp_weight(self, qp_weight):
        raise NotImplementedError()

    def set_mpc_mass(self, mpc_mass):
        raise NotImplementedError()

    def set_mpc_inertia(self, mpc_inertia):
        raise NotImplementedError()

    def set_mpc_foot_friction(self, mpc_foot_friction):
        raise NotImplementedError()

    def set_foot_landing_clearance(self, foot_landing_clearance):
        raise NotImplementedError()

    def dump_logs(self):
        self._flush_logging()
    @property
    def state_vector(self):
        com_position = self.state_estimator.com_position_in_ground_frame
        com_velocity = self.state_estimator.com_velocity_in_body_frame
        com_roll_pitch_yaw = np.array(
            self._robot.pybullet_client.getEulerFromQuaternion(
                self.state_estimator.com_orientation_quaternion_in_ground_frame))
        com_roll_pitch_yaw_rate = self._robot.base_angular_velocity_in_body_frame

        state_vector = np.hstack((com_position, com_roll_pitch_yaw, com_velocity, com_roll_pitch_yaw_rate))
        return state_vector

def get_lyapunov_reward(s, s_next):
    p_mat4 = np.array([[118.83570123101, -1.41466936361059e-14, -7.74878980369276e-16, -4.28620267366142e-13,
                        1.77104974116305, 1.0914658030531e-13, -8.47570932463836e-15, -2.52234339728665e-16,
                        -2.90202953929494e-15, -2.91177290574822e-15],
                       [-1.41466936361059e-14, 2.71120800069087e-06, -1.69789213802264e-14, -1.04804330946118e-13,
                        2.30100584844932e-16, 1.6420103779497e-14, -1.21896753338876e-15, 8.33443687830191e-07,
                        -3.89467348967968e-15, -1.17332842761011e-14],
                       [-7.74878980369276e-16, -1.69789213802264e-14, 2.77811891813838e-06, 1.23697571395491e-13,
                        2.25562936581415e-16, -2.3008164506072e-15, 2.2505744167202e-16, -5.23705946818313e-15,
                        8.55050158007754e-07, 1.46599620833702e-14],
                       [-4.28620267366142e-13, -1.04804330946118e-13, 1.23697571395491e-13, 129.692650109888,
                        -4.4876591346318e-14, 1.15093245097422e-13, 7.88823216885879e-14, -4.55269055462651e-14,
                        5.36345985228269e-14, 2.68789780217039],
                       [1.77104974116305, 2.30100584844932e-16, 2.25562936581415e-16, -4.4876591346318e-14,
                        3.06811961466463, 1.43633345827227e-14, -1.67237093395989e-15, -3.25317676205998e-16,
                        3.84673358731058e-17, -2.13353051325487e-15],
                       [1.0914658030531e-13, 1.6420103779497e-14, -2.3008164506072e-15, 1.15093245097422e-13,
                        1.43633345827227e-14, 7.68528406239351e-07, -4.34499130427616e-15, 2.62606155587371e-15,
                        -2.53613901411584e-16, 8.22206623381904e-15],
                       [-8.47570932463836e-15, -1.21896753338876e-15, 2.2505744167202e-16, 7.88823216885879e-14,
                        -1.67237093395989e-15, -4.34499130427616e-15, 6.88192800383425e-07, 1.99167451325745e-17,
                        -1.33451029121985e-15, 6.92211632759464e-15],
                       [-2.52234339728665e-16, 8.33443687830191e-07, -5.23705946818313e-15, -4.55269055462651e-14,
                        -3.25317676205998e-16, 2.62606155587371e-15, 1.99167451325745e-17, 3.67579589406552e-07,
                        -2.63427397571634e-15, -5.25797263071599e-15],
                       [-2.90202953929494e-15, -3.89467348967968e-15, 8.55050158007754e-07, 5.36345985228269e-14,
                        3.84673358731058e-17, -2.53613901411584e-16, -1.33451029121985e-15, -2.63427397571634e-15,
                        3.75439766169053e-07, 6.43418363102829e-15],
                       [-2.91177290574822e-15, -1.17332842761011e-14, 1.46599620833702e-14, 2.68789780217039,
                        -2.13353051325487e-15, 8.22206623381904e-15, 6.92211632759464e-15, -5.25797263071599e-15,
                        6.43418363102829e-15, 0.268777685660629]])
    s_new = s[2:]
    s_next_new = s_next[2:]
    ly_reward_curr = s_new.T @ p_mat4 @ s_new
    ly_reward_next = s_next_new.T @ p_mat4 @ s_next_new
    reward = ly_reward_curr - ly_reward_next  # multiply scaler to decrease

    return reward