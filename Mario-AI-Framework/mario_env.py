import gym
from gym import spaces
import numpy as np
import jpype
import cv2
 
class MarioEnv(gym.Env):
    """Custom Gym environment based on Mario-AI-Framework written in Java"""

    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render=True, level = "./levels/original/lvl-1.txt", horizons=False, starts=False, timer = 20, sticky=False, paths=False, skip=4, max_timestep=165):
        super(MarioEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        # left run jump, left jump, left run, left, jump, nothing, right, right run, right jump, right run jump, 

        # Using image as an input in the form (width, height, channel)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 84, 1), dtype=np.uint8)

        self.obs = np.zeros([1, 84, 84]) # note that this is in channel-first because its easier to do operations on, remember to convert it later
        self.render = render
        self.done = False # terminal condition met?
        self.kills = 0 # used to keep track of # of kills
        self.level = level # current mario level
        self.horizons = horizons # use non-variable (fixed) horizon?
        self.max_timestep = max_timestep # if fixed-horizons, end on what timestep
        self.starts = starts # randomize starting point?
        self.timer = timer # max game time
        self.sticky = sticky # use sticky actions?
        self.last_action = [False, False, False, False, False] # last action taken
        self.last_x = 0 # used for calculating reward function\
        self.paths = paths # save path data
        self.first_run = True
        self.skip = skip # subsampling rate

        # Connect JVM
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")
        jpype.addClassPath("/home/mikeg/Documents/CMPUT652FinalProject/CMPUT652FinalProject/Mario-AI-Framework/src") # TODO: *** Edit this to point to your own java build, relative pathing not working correctly ***

        self.main = jpype.JClass('PythonController')



    def step(self, action):
        # Take a step into the environment using the given action
        # One 'step' equals 4 frames of holding the action (frame-skipping)
        
        # Convert discrete action to boolean array actions for [Left, Right, Action, Run, Jump]
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
        else: act = [False, False, False, False, False]        

        # Apply sticky action of with 20% chance to keep previous action
        if self.sticky:
            real_action = act if np.random.random() > 0.20 else self.last_action
            self.last_action = act # even if desired action isn't inputted, we set this as the next last action
        else:
            real_action = act

        # Step the environment 4 times
        for i in range(self.skip): 
            result = self.main.step(real_action) 
            

        info = result.info # Grab 'info' dict from the step result - currently holds [vx, d, f, c, x, g] (see java code for more details on these in-game variables)

        # Calculate reward (see https://pypi.org/project/gym-super-mario-bros/) for details on vanilla reward
        vx = (info[4] - self.last_x)/4 # reward component for +ve x displacement between steps
        c = -4 # penalty for game clock == number of frames skipped
        d = info[1] # 0 if mario is alive, -15 if mario is dead # TODO: replace this with 0 or 1 
        f = info[2] # 15 if flag_get, 0 otherwise # TODO: replace this with 0 or 1

        reward = -15 if d == -15 else 15 if f == 15 else vx + c

        # reward goomba kill
        if info[5] > self.kills:
            self.kills +=1
            reward = 15

        self.last_x = info[4] # set new last_x for next step

        reward = np.clip(reward, -15, 15) # Clip reward between -15 and + 15

        self.obs = self._get_single_obs(result) # NOTE: Frame-stacking will be done outside of the environment through a Gym-wrapper


        # Calculate if episode should terminate. NOTE: SB3 handles vectorized environments strange and discards the last observation value of an episode, BUT usually this observation tells us if we died or completed the level. See https://github.com/hill-a/stable-baselines/issues/400 
        # So, let us set our terminal condition to 1 step AFTER the intended termination so that it is not the one being discarded

        done = False 

        # If we want to have fixed horizons (fixed timestep)
        if self.horizons:
            if self.timestep == self.max_timestep:
                done = True
                
                if self.render:
                    self.main.close()
            else:
                if self.done == True:
                    self.obs = self.obs * 0 # Clear screen and prep to blit it with win/loss screen

                    if info[2] > 0:
                        self.obs = (self.obs+1) * 85 # reserved for win screen
                        reward = 15
                    elif info[1] < 0:
                        self.obs = (self.obs+1) * 185 # reserved for loss screen
                        reward = -15
                     
        else:
            if self.done == True:
                done = True
                
                if self.render:
                    self.main.close()
            else:
                done = False

        self.done = result.done # episode terminates if mario dies, completes the level, or runs out of time, note the difference in when self.done and done are called

        # TODO: Check if this is needed (python interprets the java arrays strangely)
        a = []
        for i in info:
            a.append(i)
        # print(info)
                
        self.timestep += 1

        return self.obs, reward, done, {"vx": a[0], "d":a[1], "f":a[2], "c":a[3], "x":a[4], "g": a[5] }
        
    def reset(self):
        # Reset the level and all necessary variables

        if self.paths and not self.first_run:
            print("For now, we can only generate one trajectory at a time.") # Due to the way the reset function gets called a 2nd time even if only 1 episode is ran, current data logging will wipe the desired data
            return np.zeros([84, 84, 1])
        self.first_run = False


        # reset Java game environment
        self.done = False

        if self.starts: # randomize starting position by modifying a temp level.txt file
            with open(self.level, 'r') as f:
                data = f.readlines()

            ground = data[13]
            floor = data[14]

            ground = ground.replace('M', '-')

            valid_ground = [i for i, ltr in enumerate(ground) if ltr == '-']

            valid_g = []

            # Allow spawn points that are only above ground and 3 tiles away from an enemy
            for g in range(len(valid_ground)): 
                if valid_ground[g] > 3 and valid_ground[g] < len(ground)-3:
                    if ground[valid_ground[g]-3] == '-' and ground[valid_ground[g]-2] == '-' and ground[valid_ground[g]-1] == '-' and ground[valid_ground[g]+1] == '-' and ground[valid_ground[g]+2] == '-' and ground[valid_ground[g]+3] == '-':
                        valid_g.append(valid_ground[g])

            valid_floor = [i for i, ltr in enumerate(floor) if ltr == 'X']

            valids = np.intersect1d(valid_g, valid_floor)
            choice = np.random.randint(len(valids))
            choice = valids[choice]

            data[13] = ground[:choice] + 'M' + ground[choice+1:]

            with open('temp-lvl.txt', 'w') as f:
                f.writelines(data)
            
            self.level = 'temp-lvl.txt'


            
        result = self.main.reset(self.render, self.level, self.timer)
        self.last_x = result.info[4] # set new last_x for next step

        # get initial observation        
        self.obs = self._get_single_obs(result) # NOTE: Frame-stacking will be done outside of the environment through a Gym-wrapper

        self.kills = 0
        self.timestep = 0

        return self.obs

    def render(self, mode="human"):
        return

    def close (self):
        # TODO: Might need to edit this code on the Java side for proper cleanup of JVM etc.
        if self.render:
            self.main.close()
        return

    def _get_single_obs(self, result):
        # Get observation from a single frame. Each value represents a tile which is a collection of pixels visible on the screen
        # Wrappers will be used to stretch this to 84x84 input and then stack 4 frames to 4x84x84

        obs = np.zeros([16, 16], dtype=np.uint8)
        for i in range(result.observation.length):
            for j in range(result.observation[i].length):
                obs[i][j] = result.observation[i][j]



        # TODO: do this process for other levels aside from 1-1...
        # Manually replace tiles in python which is easier than in Java...
        # about 10 important tiles so try to spread by 25
        
        # self.obs[self.obs == 0] = 0 # sky 
        obs[obs == 17] = 25 # ground
        obs[obs == 18] = 50 # stair block

        obs[obs == 56] = 75 # flag pole   
        obs[obs == 55] = 75 # flag top

        obs[obs == 34] = 100 # pipe
        obs[obs == 35] = 100
        obs[obs == 36] = 100
        obs[obs == 37] = 100

        obs[obs == 22] = 125 # break block
        obs[obs == 24] = 150 # coint block

        obs[obs == 2] = 175 # enemy 
        obs[obs == 22] = 125 # break block
        obs[obs == 24] = 150 # coint block
        
        obs[obs == 2] = 175 # enemy 

        obs[obs == 12] = 200 # mushroom
        obs[obs == 30] = 225 # coin

        obs[obs == 99] = 255 # mario
        obs[obs == 97] = 245 # mario left

        obs = np.moveaxis(obs, -1, 0) # Swap x and y for readability

        obs = cv2.resize(obs, [80, 80], interpolation=cv2.INTER_AREA)
        obs = np.pad(obs, [[2,2],[2,2]])
        obs = obs.reshape([84, 84, 1])

        return obs
