from typing import Optional
import numpy as np
import gymnasium as gym

class SimpleEnv(gym.Env):
    def __init__(self,
                 Ad, Bd_HVAC, Bd_dist,
                 Cd, Dd_HVAC, Dd_dist,
                 COP, Price, u_dist):
        self._state = np.array([[24, 25.5]], dtype=np.float32)
        self._current_indoor_temperature = self._state[0,0]
        self._current_timestep = 0
        self.Ad = Ad
        self.Bd_HVAC = Bd_HVAC
        self.Bd_dist = Bd_dist
        self.Cd = Cd
        self.Dd_HVAC = Dd_HVAC
        self.Dd_dist = Dd_dist
        self._COP = COP
        self._Price = Price
        self._u_dist = u_dist

        self.observation_space = gym.spaces.Dict({"Tin": gym.spaces.Box(0, 50, dtype=float),
                                                  "curret_timestep": gym.spaces.Box(0, 300, dtype=int)})
        self.action_space = gym.spaces.Box(-2, 0, dtype=float)

    def _get_obs(self):
        return {"Tin": self._current_indoor_temperature,
                "curret_timestep": self._current_timestep}
    
    def _get_info(self):
        return {"curret_timestep": self._current_timestep}
    
    def reset(self, seed: Optional[int] = None):
        super().reset()
        self._state = np.array([[24, 25.5]], dtype=np.float32)
        self._current_indoor_temperature = self._state[0,0]
        self._current_timestep = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def compute_x(self, x0, u0_HVAC, u0_dist, 
                Ad, Bd_HVAC, Bd_dist):
        return np.dot(Ad,x0) + np.dot(Bd_HVAC,u0_HVAC) + np.dot(Bd_dist,u0_dist)

    def compute_y(self, x0,u0_HVAC, u0_dist, 
                Cd, Dd_HVAC, Dd_dist):
        return np.dot(Cd,x0) + np.dot(Dd_HVAC,u0_HVAC) + np.dot(Dd_dist,u0_dist)

    def step(self, action):
        i = self._current_timestep
        self._state = self.compute_x(self._state.T,
                                np.array([action*1000]),
                                self._u_dist[i:i+1,:].T,
                                self.Ad, self.Bd_HVAC, self.Bd_dist).T
        
        self._current_indoor_temperature = self.compute_y(self._state.T,
                                                     np.array([action*1000]),
                                                     self._u_dist[i:i+1,:].T,
                                                     self.Cd, self.Dd_HVAC, self.Dd_dist)[0]
        if self._current_indoor_temperature > 24.:
            penalty = self._current_indoor_temperature - 24.
        else:
            penalty = self._current_indoor_temperature * 0.

        # reward =  - penalty
        reward = (action/self._COP[i]/6 * self._Price[i])*100 - 10*penalty

        self._current_timestep += 1
        observation = self._get_obs()
        info = self._get_info()
        if i == self._u_dist.shape[0]-1:
            terminated =True
        else: terminated =False
        truncated = False
        
        return observation, reward, terminated, truncated, info

# Electricity rates
def func_price(i):
    # $/kWh
    hour = (i//6)%24
    if hour<7:
        return 0.082
    elif (hour>=7)&(hour<11):
        return 0.113
    elif (hour>=11)&(hour<17):
        return 0.17
    elif (hour>=17)&(hour<19):
        return 0.113
    else:
        return 0.082

# HVAC COP
def func_COP(To):
    return -2/25 * To + 6