from turtle import rt

from torch import float32

import pickle
from natsort import natsorted
from os.path import join, exists
from os import listdir

from yarr.envs.env import Env
from yarr.utils.transition import Transition
from yarr.utils.observation_type import ObservationElement

import sys
sys.path.append('/home/albert/Desktop/Work/UR5/ur5-env')
sys.path.append('/home/albert/Desktop/Work/C2FARM/ARM/arm/')

from ur_env import base, scene
from ur_env.cameras.realsense import RealSense
from ur_env.robot.arm import ArmActionMode, TCPPosition
from ur_env.robot.gripper import GripperActionMode, Continuous, Discrete
from dashboard_client import DashboardClient

from robot_basetask import BaseTask
from robot_pick_and_lift import PickAndLift
import pyrealsense2 as rs


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface



from typing import Type, List


import numpy as np



ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']









class RobotEnv(Env):
    def __init__(
        self, rl_env,
        rtde_control: RTDEControlInterface = None,
        rtde_receive: RTDEReceiveInterface = None,
        dashboard_client: DashboardClient = None,
        arm_action_mode: ArmActionMode = None,
        gripper_action_mode: GripperActionMode = None,
        realsense: RealSense = None,
        episode_lenth: int = 10,
        task_name='PickAndLift',
        reward_scale=100.0    
    ):

        super(RobotEnv).__init__()
        self.reward_scale = reward_scale


        

        self.reward_scale = reward_scale
        
        self._active_task_id = 0
        self.episode_index = 0
        self.episode_lenth = episode_lenth
        # self._recorded_images = []
        self.camera_intrinsics = np.array([[450, 0, 424],
                                           [0, 450, 240],
                                           [0, 0, 1]], dtype=np.float32)
        self.step_n = 0
        self.rl_env = rl_env
        self.my_obs = {
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }
        self.start_pos = self.my_obs
        self.previous_obs = self.start_pos
        self._dataset_root = '/home/albert/Desktop/myDemo'



    def launch(self):
        # intr = self.scene.realsense.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        pass
        # Connect to robot
        # !! Activate gripper + set_speed + set_force
        

    def shutdown(self):
        # Close connection
        pass
    
    def extract_obs(self, obs, t=None, prev_action=None):
        # low_dim_names = ['arm/ActualTCPPose', 
        #                  'gripper/pose',
        #                  'gripper/is_closed', 
        #                  'gripper/object_detected', 
        #                 ]


        
        # low_dim_state = np.concatenate([obs[k] if k=='arm/ActualTCPPose' else [obs[k]] for k in low_dim_names])

        
        return {
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }

    def extract_action(self, action):
        return {'arm': action[:-1], 'gripper': action[-1]}

    def reset(self):
        self.step_n = 0
        
        
        # self.task.reset()
        #self.scene.step(self.extract_action(self.extract_obs(self.start_pos)['low_dim_state'][:7]))
        # obs = self.scene.get_observation()

        

        self.previous_obs = {
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }
        return {
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }
        
    def step(self, action):
        #self.scene.step(self.extract_action(action))
        print(action.action)
        reward = 1
        terminal = False
        self.step_n += 1
        self.previous_obs = {
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }
        return Transition({
                    'low_dim_state': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32),
                    'gripper_pose': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                    'front_rgb': np.ones((3, 128, 128), dtype=np.uint8),
                    'front_point_cloud': np.ones((3, 128, 128), dtype=np.float32),
                    'front_camera_intrinsics': self.camera_intrinsics,
                    'front_camera_extrinsics': np.array([[1,0,0,0], 
                                                         [0,1,0,0], 
                                                         [0,0,1,0],
                                                         [0,0,0,1]], dtype=np.float32)
                  }, reward * self.reward_scale, terminal)
    
    def observation_space(self):
        return (1,)

    @property
    def observation_elements(self):
        elements = []
        elements.append(ObservationElement('low_dim_state', (11, ), np.float32))
        self.low_dim_state_len = 11
        elements.append(ObservationElement('gripper_pose', (7, ), np.float32))
        elements.append(ObservationElement('front_rgb', (3, 128, 128), np.uint8))
        elements.append(ObservationElement('front_point_cloud', (3, 128, 128), np.float32))
        elements.append(ObservationElement('front_camera_intrinsics', (3, 3), np.float32))
        elements.append(ObservationElement('front_camera_extrinsics', (4, 4), np.float32))

        return elements

    @property
    def action_shape(self):
        return (7,)

    @property
    def env(self):
        # return self.rl_env.env
        return self


    def get_demos(self, task_name: str, amount: int,
                  variation_number=0,
                  image_paths=False,
                  random_selection: bool = True,
                  from_episode_number: int = 0):
        if self._dataset_root is None or len(self._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        task_path = join(self._dataset_root, task_name)
        
        

        examples_path = join(task_path, 'episodes')
        examples = listdir(examples_path)



        if random_selection:
            selected_examples = np.random.choice(examples, amount, replace=False)
        else:
            selected_examples = natsorted(examples)[from_episode_number:from_episode_number+amount]

        demos = []
           
        for example in selected_examples:
            path = join(examples_path, example)
            obs_path = join(path, 'obs.pkl')
            
            with open(obs_path, 'rb') as f:
                obs = pickle.load(f)
            
            demos.append(obs)

        return demos