import gym
from gym import spaces
import numpy as np

import jpype
 
class MarioEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render=True):
        super(MarioEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(10)
        # left run jump, left jump, left run, left, jump, nothing, right, right run, right jump, right run jump, 

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 16, 16), dtype=np.uint8) # 4 frames stacked

        self.render = render

        # Connect JVM
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")
        jpype.addClassPath("/home/michael/Documents/pcg/CMPUT652FinalProject/Mario-AI-Framework/src")
        self.main = jpype.JClass('PythonController')
        self.obs = np.zeros([1, 16, 16])



    def step(self, action):
        # Convert discrete action to boolean array actions for [Left, Right, Action, Run, Jump]
        # One 'step' equals 4 frames of holding the action (frame-skipping)

        if action == 0:
            act = [False, False, False, False, False] # None
        elif action == 1:
            act = [False, False, False, False, True] # Jump
        elif action == 2:
            act = [True, False, False, False, False] # Left
        elif action == 3:
            act = [True, False, False, True, False] # Left Run
        elif action == 4:
            act = [True, False, False, False, True] # Left Jump
        elif action == 5:
            act = [True, False, False, True, True] # Left Run Jump
        elif action == 6:
            act = [False, True, False, False, False] # Right
        elif action == 7:
            act = [False, True, False, True, False] # Right Run
        elif action == 8:
            act = [False, True, False, False, True] # Right Jump
        elif action == 9:
            act = [False, True, False, True, True] # Right Run Jump

        # For now, just step the game 4 times with the same input, in future might want to change the reward calculation to only calculate once
        # Stack every 4th skipped frame into the observation space
        reward = 0
        for i in range(4): 
            result = self.main.step(act)
            reward += result.reward

        # self.obs[0] = self.obs[1]
        # self.obs[1] = self.obs[2]
        # self.obs[2] = self.obs[3]
        # self.obs[3] = self._get_single_obs(result)
        self.obs[0] = self._get_single_obs(result)

        done = result.done
        info = {}

        if done:
            self.main.close()

        return self.obs, reward, done, info
        
    def reset(self):
        
        

        # reset Java game environment
        result = self.main.reset(self.render)

        # fill observation with first frame data
        single_obs = self._get_single_obs(result)
        
        self.obs = np.zeros([1, 16, 16])
        self.obs[0] = single_obs
        # self.obs[0] = single_obs
        # self.obs[1] = single_obs
        # self.obs[2] = single_obs
        # self.obs[3] = single_obs

        return self.obs  
    def render(self, mode="human"):
        return
    def close (self):
        self.main.close()
        return

    def _get_single_obs(self, result):

        obs = np.zeros([16, 16])
        for i in range(result.observation.length):
            for j in range(result.observation[i].length):
                obs[i][j] = result.observation[i][j]

        return obs
