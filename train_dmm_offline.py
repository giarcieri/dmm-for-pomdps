from envs.env_discrete import DiscreteEnv
from dmms.dmm import * 
from tqdm import tqdm

def epsilon_random_action(environment, epsilon=0.):
    "epsilon probability to pick a random action and (1-epsilon) to pick action 0"
    if np.random.rand() < epsilon:
        return environment.action_space.sample()
    else:
        return 0
    
n_batch = 400
epsilon_prob = 0.2
obs_seq = []
state_seq = []
belief_seq = []
action_seq = []
print('Collecting sequences')
env = DiscreteEnv(monitoring='permanent', return_belief=False, reward_belief=False)
for _ in tqdm(range(n_batch)):
    obs_list = []
    state_list = []
    belief_list = []
    action_list = []
    lenght = 100
    obs = env.reset()
    for _ in range(lenght):
        obs_list.append([obs])
        state_list.append([env.state])
        belief_list.append(env.belief)
        action = epsilon_random_action(env, epsilon_prob)
        action_list.append([action])
        obs, reward, done, info = env.step(action)
    obs_seq.append(obs_list)
    state_seq.append(state_list)
    belief_seq.append(belief_list)
    action_seq.append(action_list)
obs_seq, state_seq, belief_seq, action_seq = torch.Tensor(obs_seq), \
    torch.Tensor(state_seq), torch.Tensor(belief_seq), torch.Tensor(action_seq)

print('DMM training')
dmm = DMM(
    x_prob_dim=3, 
    num_particles=1, 
    adam_params={'lr': 1e-3, 'lrd': 1., 'betas' : (0.95, 0.999)}
)
epoch_nll_list = dmm.train(x_seq=obs_seq, a_seq=action_seq, num_epochs=800, mini_batch_size=20,)
print('NLL:', epoch_nll_list)
print('Ground truth observation prob for hidden state 0 [0.80, 0.20, 0.00],\nLearned:', dmm.dmm.emitter(torch.Tensor([0])))
b_t_1 = torch.Tensor([1., 0., 0., 0., 0.])
a_t_1 = torch.Tensor([0])
print('Learned propagated belief from initialization:', dmm.dmm.trans(b_t_1, a_t_1))