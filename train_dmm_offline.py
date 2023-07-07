from envs.env_discrete import DiscreteEnv
from dmms.dmm import * 
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser(description='dmm')
parser.add_argument('-n', '--n_batch', type=int, default=500)
parser.add_argument('-e', '--eps', type=float, default=0.2)
parser.add_argument('-l', '--length', type=int, default=100)
parser.add_argument('-ae', '--annealing_epochs', type=int, default=1500)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-lrd', '--learning-rate-decay', type=float, default=1.)
parser.add_argument('-b1', '--beta1', type=float, default=0.95)
parser.add_argument('-b2', '--beta2', type=float, default=0.999)
parser.add_argument('-eh', '--emitter_hidden_dim', type=int, default=100)
parser.add_argument('-th', '--transition_hidden_dim', type=int, default=100)
parser.add_argument('-ih', '--inference_hidden_dim', type=int, default=100)
parser.add_argument('-ne', '--num_epochs', type=int, default=1500)
parser.add_argument('-bs', '--mini_batch_size', type=int, default=20)
parser.add_argument('-af', '--minimum_annealing_factor', type=float, default=0.)
parser.add_argument('-uc', '--use-cuda', type=int, default=0)
args = parser.parse_args()

print(args)

def epsilon_random_action(environment, epsilon=0.):
    "epsilon probability to pick a random action and (1-epsilon) to pick action 0"
    if np.random.rand() < epsilon:
        return environment.action_space.sample()
    else:
        return 0
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]
    
n_batch = int(args.n_batch)
epsilon_prob = float(args.eps)
length = int(args.length)
annealing_epochs = int(args.annealing_epochs)
lr = float(args.learning_rate)
lrd = float(args.learning_rate_decay)
beta1 = float(args.beta1)
beta2 = float(args.beta2)
emitter_hidden_dim = int(args.emitter_hidden_dim)
transition_hidden_dim = int(args.transition_hidden_dim)
inference_hidden_dim = int(args.inference_hidden_dim)
num_epochs = int(args.num_epochs)
mini_batch_size = int(args.mini_batch_size)
minimum_annealing_factor = float(args.minimum_annealing_factor)
use_cuda = bool(args.use_cuda)

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
    obs = env.reset()
    for _ in range(length):
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
    
action_seq = to_categorical(np.array(action_seq).astype(np.int32), 4).squeeze()
obs_seq = to_categorical(np.array(obs_seq).astype(np.int32), 3).squeeze()

if use_cuda:
    obs_seq, action_seq = obs_seq.cuda(), action_seq.cuda()

print('DMM training')
start_time = time.time()
dmm = DMM(
    mode='discrete',
    x_prob_dim=3, 
    x_dim=3,
    a_dim = 4,
    b_dim=5,
    z_dim=5,
    num_particles=1, 
    annealing_epochs=annealing_epochs,
    adam_params={'lr': lr, 'lrd': lrd, 'betas' : (beta1, beta2)},
    emitter_hidden_dim=emitter_hidden_dim,
    transition_hidden_dim=transition_hidden_dim,
    inference_hidden_dim=inference_hidden_dim,
    minimum_annealing_factor=minimum_annealing_factor,
    use_cuda = use_cuda
)
epoch_nll_list = dmm.train(x_seq=obs_seq, a_seq=action_seq, num_epochs=num_epochs, mini_batch_size=mini_batch_size,)
train_time = time.time()-start_time
print(f"Training time: {round(train_time/3600, 1)}h")
print('Best NLL:', np.max(epoch_nll_list))
plt.plot(epoch_nll_list)
plt.savefig('images/nll');
state = torch.tensor(to_categorical([0], 5))
if use_cuda:
    state = state.cuda()
print('Ground truth observation prob for hidden state 0 [0.80, 0.20, 0.00],\nLearned:', dmm.dmm.emitter(state))
b_t_1 = torch.tensor([[1., 0., 0., 0., 0.]])
a_t_1 = torch.tensor(to_categorical([0], 4))
if use_cuda:
    b_t_1, a_t_1 = b_t_1.cuda(), a_t_1.cuda()
print('Learned propagated belief from initialization:', dmm.dmm.trans(b_t_1, a_t_1))