import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple, MultiBinary

class VaryingObservationSpace(gym.Space):
    """
    The observation space that varies between 3 or 5 depending on whether a Visual-inspection 
    or an Ultrasonic-inspection was taken. 

    The first dimension is the inspection action, 
    the second dimension is the observation.
    """
    def __init__(self):
        self.available_observations = np.array([
            [0, None],
            [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]
        ])

    def sample(self):
        idx = np.random.randint(len(self.available_observations))
        return self.available_observations[idx]

    def contains(self, x):
        return x in self.available_observations.tolist()

class DiscreteEnv(gym.Env):
    """
    POMDP benchmarking environment with 5 states, 4 maintenance actions and 3 inspection actions 
    for a total of 12 actions, and 3 or 5 observations depending on the inspection chosen.
    Deatils in Papakonstantinou et al. (2018) and Corotis et al. (2005).

    A simpler version with continuous permanent monitoring is also implemented, 
    where a visual inpection is performed at every timestep.
    """
    def __init__(
            self,
            monitoring='non-permanent', # 'permanent' or 'non-permanent'
            return_belief=False,
            reward_belief=False,
            seed=42,
    ):
        self.monitoring = monitoring
        self.return_belief = return_belief
        self.reward_belief = reward_belief
        self.random_generator = np.random.default_rng(seed)

        self.maintenance_actions = {
            0: 'Do-nothing',
            1: 'Cleaning and repainting the corrosion surfaces',
            2: 'Repainting and strengthening the girders',
            3: 'Extensive repair'
        }

        self.inspection_actions = {
            0: 'No-inspection',
            1: 'Visual-inspection',
            2: 'Ultrasonic-inspection'
        }

        if monitoring == 'permanent':
            self.action_space = Discrete(4) #TODO: maybe action as OneHotEncoded?
            self.n_actions = 4
        elif monitoring == 'non-permanent':
            self.action_space = MultiDiscrete([4,3])
            self.n_actions = 12
        else:
            raise NotImplementedError
        
        self.n_states = 5
        self.state_space = Discrete(5)

        if return_belief:
            self.observation_space = Box(low=np.zeros(5), high=np.ones(5), shape=(5,), dtype=np.float32)
        elif monitoring == 'permanent':
            self.n_obs = 3
            self.observation_space = Discrete(3)
        elif monitoring == 'non-permanent':
            # since the observation space should be fixed in the RL context
            # and not varying (3 or 5) as in the original refernce, 
            # we set a binary variable for the first dimension, 
            # whether it is a Visual-inspection or a Ultrasonic-inspection, 
            # and 3 or 5 possible observations for the second dimension.
            self.observation_space = VaryingObservationSpace()
        else:
            raise NotImplementedError

        # deterioration matrix (action 'Do-nothing')
        self.P1 = np.array([ 
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.00, 0.70, 0.17, 0.05, 0.08],
            [0.00, 0.00, 0.75, 0.15, 0.10],
            [0.00, 0.00, 0.00, 0.60, 0.40],
            [0.00, 0.00, 0.00, 0.00, 1.00]
        ])

        # transition matrix for action 'Cleaning and repainting the corrosion surfaces'
        self.P2 = np.array([
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.00, 0.80, 0.10, 0.02, 0.08],
            [0.00, 0.00, 0.80, 0.10, 0.10],
            [0.00, 0.00, 0.00, 0.60, 0.40],
            [0.00, 0.00, 0.00, 0.00, 1.00]
        ])

        # transition matrix for action 'Repainting and strengthening the girders'
        self.P3 = np.array([
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.19, 0.65, 0.08, 0.02, 0.06],
            [0.10, 0.20, 0.56, 0.08, 0.06],
            [0.00, 0.10, 0.25, 0.55, 0.10],
            [0.00, 0.00, 0.00, 0.00, 1.00]
        ])

        # transition matrix for action 'Extensive repair'
        self.P4 = np.array([
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.80, 0.13, 0.02, 0.00, 0.05],
            [0.80, 0.13, 0.02, 0.00, 0.05]
        ])

        self.transition_matrices= np.array([self.P1, self.P2, self.P3, self.P4])

        # Observation matrix for 'Visual-inspection'
        self.O2 = np.array([
            [0.80, 0.20, 0.00],
            [0.20, 0.60, 0.20],
            [0.05, 0.70, 0.25],
            [0.00, 0.30, 0.70],
            [0.00, 0.00, 1.00]
        ])

        # Observation matrix for 'Ultrasonic-inspection'
        self.O3 = np.array([
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.05, 0.90, 0.05, 0.00, 0.00],
            [0.00, 0.05, 0.90, 0.05, 0.00],
            [0.00, 0.00, 0.05, 0.95, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00]
        ])

        # Cost matrix for maintenance actions (A x S)
        self.C_m = np.array([
            [ 0 ,  0  ,  0  , -300, -2000],
            [-5 , -8  , -15 , -320, -2050],
            [-25, -80 , -100, -450, -2500],
            [-40, -120, -550, -800, -4000]
        ])

        # Cost matrix per inspection actions (A x S)
        self.C_ins = np.array([
            [ 0 ,  0 ,  0 ,  0 ,  0 ],
            [-4 , -4 , -4 , -4 , -4 ],
            [-18, -18, -18, -18, -18]
        ])

    def reset(self):
        self.state = 0 #np.array([0])
        self.belief = np.array([1., 0., 0., 0., 0.])
        self.t = 0
        if self.monitoring == 'permanent':
            self.obs = self.observation_generating_process(None, self.state)
        elif self.monitoring == 'non-permanent':
            # Since the belief would be initialized with certainty on 0 in the POMDP context,
            # but here the DMM has to predict the belief, so it will never be exactly that,
            # we assume that the previous inspections action was 2, 
            # hopefully to get a belief closer to the correct initialization.
            self.obs = np.array([2, 0]) #TODO: check if correct

        if self.return_belief:
            return self.belief
        else:
            return self.obs
        
    def step(self, action):
        self.t += 1
        reward = self.generate_cost(action, self.state, self.belief)
        self.state = self.transition_process(action, self.state)
        self.obs = self.observation_generating_process(action, self.state)
        self.belief = self.update_belief(self.belief, self.obs, action)
        if self.return_belief:
            return self.belief, reward, False, {'obs': self.obs, 'state': self.state, 'timestep': self.t}
        else:
            return self.obs, reward, False, {'belief': self.belief, 'state': self.state, 'timestep': self.t}

    def generate_cost(self, action, state, belief):
        if self.monitoring == 'permanent':
            inspection_action = 1
            maintenance_action = action
        elif self.monitoring == 'non-permanent':
            maintenance_action, inspection_action = action
        if self.reward_belief:
            inspection_cost = np.sum(self.C_ins[inspection_action]*belief)
            maintenance_cost = np.sum(self.C_m[maintenance_action]*belief)
        else:
            inspection_cost = self.C_ins[inspection_action, state]
            maintenance_cost = self.C_m[maintenance_action, state]
        return inspection_cost + maintenance_cost

    def transition_process(self, action, state):
        if self.monitoring == 'permanent':
            transition_probs = self.transition_matrices[action, state]
        elif self.monitoring == 'non-permanent':
            maintenance_action, _ = action
            transition_probs = self.transition_matrices[maintenance_action, state]
        next_state = self.random_generator.choice(len(transition_probs), p=transition_probs)
        return next_state

    def observation_generating_process(self, action, state):
        if self.monitoring == 'permanent':
            obs_probs = self.O2[state]
            obs = self.random_generator.choice(len(obs_probs), p=obs_probs)
        elif self.monitoring == 'non-permanent':
            _, inspection_action = action
            if inspection_action == 0:
                obs = np.array([inspection_action, None])
            elif inspection_action == 1:
                obs_probs = self.O2[state]
                _obs = self.random_generator.choice(len(obs_probs), p=obs_probs)
                obs = np.array([inspection_action, _obs])
            elif inspection_action == 2:
                obs_probs = self.O3[state]
                _obs = self.random_generator.choice(len(obs_probs), p=obs_probs)
                obs = np.array([inspection_action, _obs])
        return obs
    
    def update_belief(self, belief, obs, action):
        if self.monitoring == 'permanent':
            inspection_action = 1
            maintenance_action = action
        elif self.monitoring == 'non-permanent':
            maintenance_action, inspection_action = action
            assert inspection_action == obs[0]
            obs = obs[1]
        new_belief = np.zeros(belief.shape)
        total_prob = 0
        if inspection_action == 0:
            return belief
        for next_state in np.arange(self.n_states):
            if inspection_action == 1:
                obs_prob = self.O2[next_state, obs]
            elif inspection_action == 2:
                obs_prob = self.O3[next_state, obs]
            transition_prob = 0
            for state in np.arange(self.n_states):
                transition_prob += self.transition_matrices[maintenance_action, 
                                                            state, next_state] * belief[state]
            new_belief[next_state] = obs_prob * transition_prob
            total_prob += new_belief[next_state]
        new_belief /= total_prob
        return new_belief
    
    def update_belief_parallelized(self, belief, obs, action):
        # Belief state computation
        new_belief_prior = self.transition_matrices[action].T @ belief

        obs_probs = self.O2[:, obs] # likelihood of observation

        # Bayes' rule
        new_belief = obs_probs * new_belief_prior # likelihood * prior
        new_belief /= np.sum(new_belief) # normalize
        return new_belief
    
class SimpleDiscreteEnv(gym.Env):
    """Simple Env with 4 discrete states, 4 discrete obs, and 3 actions"""
    def __init__(self, seed=42, return_belief=False, reward_belief=False,):
        self.random_generator = np.random.default_rng(seed)
        self.n_states = 4
        self.n_obs = 4
        self.n_actions = 3
        self.initial_observation = 0
        self.return_belief = return_belief
        self.reward_belief = reward_belief
        self.state_space = Discrete(self.n_states)
        if return_belief:
            self.observation_space = Box(low=np.zeros(self.n_states), 
                                         high=np.ones(self.n_states), 
                                         shape=(self.n_states,), 
                                         dtype=np.float32)
        else:
            self.observation_space = Discrete(self.n_obs)
        self.action_space = Discrete(self.n_actions)
        self.transition_tables = np.array([
             [# Action 0: do-nothing
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [# Action 1: minor repair
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.2, 0.1, 0.0]
            ],
            [# Action 2: major repair (replacement)
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ]
        ])

        self.observation_tables = np.array([
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
        ])
        
        # Costs (negative rewards)
        self.state_action_reward = np.array([
            [0, -20, -150],
            [0, -25, -150],
            [0, -30, -150],
            [0, -40, -150],
        ])

        self.reset()

    def reset(self):
        self.t = 0
        self.state = 0
        self.observation = self.initial_observation
        self.belief = np.array([1, 0, 0, 0])
        if self.return_belief:
            return self.belief
        else:
            return self.observation

    def step(self, action):
        # actions: [do_nothing, minor repair, replacement] = [0, 1, 2]

        self.t += 1
        
        if self.observation == 3:
            action = 2 # force replacement
        
        next_deterioration_state = self.random_generator.choice(
            np.arange(self.n_states), p=self.transition_tables[action][self.state]
        )

        reward = self.state_action_reward[self.state][action]
        self.state = next_deterioration_state

        self.observation = self.random_generator.choice(
            np.arange(self.n_states), p=self.observation_tables[self.state]
        )

        # Belief state computation
        self.belief = self.transition_tables[action].T @ self.belief

        state_probs = self.observation_tables[:, self.observation] # likelihood of observation

        # Bayes' rule
        self.belief = state_probs * self.belief # likelihood * prior
        self.belief /= np.sum(self.belief) # normalize

        if self.return_belief:
            return self.belief, reward, False, {'obs': self.observation, 'state': self.state, 'timestep': self.t}
        else:
            return self.observation, reward, False, {'belief': self.belief, 'state': self.state, 'timestep': self.t}

    


    