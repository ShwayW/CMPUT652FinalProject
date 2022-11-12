import gym
from gym import spaces
import numpy as np
import jpype
 
class MarioEnv(gym.Env):
    """Custom Gym environment based on Mario-AI-Framework written in Java"""

    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render=True):
        super(MarioEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        # left run jump, left jump, left run, left, jump, nothing, right, right run, right jump, right run jump, 

        # Using image as an input in the form (width, height, channel)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(16, 16, 1), dtype=np.uint8)

        self.obs = np.zeros([1, 16, 16]) # note that this is in channel-first because its easier to do operations on
        self.render = render

        # Connect JVM
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")
        jpype.addClassPath("/home/michael/Documents/pcg/CMPUT652FinalProject/Mario-AI-Framework/src") # TODO: Edit this

        self.main = jpype.JClass('PythonController')


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

        # TODO: For now, just step the game 4 times with the same input, in future might want to change the reward calculation to only calculate once
        # Stack every 4th skipped frame into the observation space
        reward = 0
        for i in range(4): 
            result = self.main.step(act)
            # reward += result.reward

        reward = result.reward

        obs = self._get_single_obs(result) # NOTE: Frame-stacking will be done outside of the environment through a Gym-wrapper
        self.obs = np.moveaxis(obs, -1, 0) # Switch from channel-first to channel-last
        self.obs = self.obs.reshape([16, 16, 1])

        done = result.done # episode terminates if mario dies, completes the level, or runs out of time
        info = {} # Can place debugging info here

        try: # TODO: This is just a band-aid fix and sometimes doesnt close everything properly. Should look into it more
            if done:
                self.main.close()
        except:
            pass
        
        return self.obs, reward, done, info
        
    def reset(self):
        # Reset the level and all necessary variables

        # reset Java game environment
        result = self.main.reset(self.render)

        # get initial observation        
        obs = self._get_single_obs(result)
        self.obs = np.moveaxis(obs, -1, 0)
        

        return self.obs.reshape([16, 16, 1])  

    def render(self, mode="human"):
        return

    def close (self):
        # TODO: Might need to edit this code on the Java side for proper cleanup of JVM etc.
        self.main.close()
        return

    def _get_single_obs(self, result):
        # Get observation from a single frame. Each value represents a tile which is a collection of pixels visible on the screen
        # Wrappers will be used to stretch this to 84x84 input and then stack 4 frames to 4x84x84

        obs = np.zeros([16, 16], dtype=np.uint8)
        for i in range(result.observation.length):
            for j in range(result.observation[i].length):
                obs[i][j] = result.observation[i][j]

        return obs
