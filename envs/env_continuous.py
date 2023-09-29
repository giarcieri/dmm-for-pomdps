import gym
import numpy as np
from gym.spaces import Box


class ContinuousEnv(gym.Env):
    """
    POMDP benchmarking environment with continuous states, observations and actions.
    """
    def __init__(
            self,
            return_states=False,
            power=1,
            min_std_next_state=0.02,
    ):
        self.return_states = return_states

        self.state_space = Box(low=np.NINF, high=np.PINF, shape=(1,), dtype=np.float32)
        if return_states:
            self.observation_space = self.state_space
        else:
            self.observation_space = Box(low=np.NINF, high=np.PINF, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=np.zeros(1), high=np.ones(1), shape=(1,), dtype=np.float32)

        self.power = power
        self.min_std_next_state = min_std_next_state

    def reset(self):
        self.t = 0
        self.state = np.array([1.], dtype=np.float32)
        if self.return_states:
            return self.state
        else:
            self.obs = self.observation_generating_process(self.state)
            return self.obs
        
    def step(self, action):
        self.t += 1
        reward = self.reward_function(self.state, action)
        self.state = self.transition_model(self.state, action)
        if self.return_states:
            return self.state, reward, False, {}
        else:
            self.obs = self.observation_generating_process(self.state)
            return self.obs, reward, False, {'state': self.state}
    
    def transition_model(self, state, action):
        return self.transition_deterioration(state)*(1-action**self.power) + self.transition_replacement(state)*action**self.power
    
    def transition_deterioration_mean(self, state):
        return np.maximum(0,state-np.exp(-state*5)*0.5-0.1, dtype=np.float32)
    
    def transition_deterioration(self, state):
        next_state_mean = self.transition_deterioration_mean(state)
        next_state_std = (np.maximum(0, state, dtype=np.float32)-np.maximum(0, next_state_mean, dtype=np.float32))*0.5 + self.min_std_next_state
        return next_state_mean + next_state_std*np.random.default_rng().standard_normal(size=len(state), dtype=np.float32)
    
    def transition_replacement(self, state):
        next_state_mean = 0.96*np.ones(len(state), dtype=np.float32)
        next_state_std = self.min_std_next_state
        return next_state_mean + next_state_std*np.random.default_rng().standard_normal(size=len(state), dtype=np.float32)
    
    def observation_generating_process(self, state):
        std_obs = np.exp(state)*0.05
        return state + std_obs*np.random.default_rng().standard_normal(size=len(state), dtype=np.float32)
    
    def reward_function(self, state, action):
        failure_prob = (1-np.clip(a=state, a_min=0., a_max=1., dtype=np.float32))
        failure_cost = -1000*failure_prob
        maintenance_cost = -500*action**2
        return failure_cost + maintenance_cost