from tabnanny import check
from robot_basetask import BaseTask

from ur_env.scene import Scene


class PickAndLift(BaseTask):
    def __init__(
        self,
        scene: Scene
    ):
        super(PickAndLift, self).__init__()
        self.scene = scene
        self._success = False
        self._treshold_z = 0 # TODO



    def check_success(self):
        gripper_obs = self.scene.gripper.get_observation()
        arm_pose = self.scene.arm.get_observation()['ActualTCPPose']

        if arm_pose[2] > self._treshold_z and gripper_obs["object_detected"]:
            self._success = True

        
        return self._success

    def step(self):
        if self.check_success(): 
            rew = 1
            terminal = True
        else:
            rew = 0
            terminal = False

        return rew, terminal

    def reset(self):
        self._success = False    
