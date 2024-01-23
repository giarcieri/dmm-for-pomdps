from envs.env_discrete import DiscreteEnv, SimpleDiscreteEnv
from dmms.dmm import * 
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy

parser = argparse.ArgumentParser(description='dmm')
parser.add_argument('-n', '--n_batch', type=int, default=500)
parser.add_argument('-e', '--eps', type=float, default=0.2)
parser.add_argument('-l', '--length', type=int, default=100)
parser.add_argument('-ae', '--annealing_epochs', type=int, default=1500)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=1.)
parser.add_argument('-b1', '--beta1', type=float, default=0.95)
parser.add_argument('-b2', '--beta2', type=float, default=0.999)
parser.add_argument('-eh', '--emitter_hidden_dim', type=int, default=100)
parser.add_argument('-th', '--transition_hidden_dim', type=int, default=100)
parser.add_argument('-ih', '--inference_hidden_dim', type=int, default=100)
parser.add_argument('-ne', '--num_epochs', type=int, default=1500)
parser.add_argument('-bs', '--mini_batch_size', type=int, default=20)
parser.add_argument('-af', '--minimum_annealing_factor', type=float, default=0.)
parser.add_argument('-uc', '--use_cuda', type=int, default=0)
parser.add_argument('-env', '--simple_env', type=int, default=0)
parser.add_argument('-ug', '--use_gate', type=int, default=1)
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
simple_env = bool(args.simple_env)
use_gate = bool(args.use_gate)

obs_seq = []
state_seq = []
belief_seq = []
action_seq = []
print('Collecting sequences')
if simple_env:
    env = SimpleDiscreteEnv(seed=42, return_belief=False, reward_belief=False)
else:
    env = DiscreteEnv(seed=42, monitoring='permanent', return_belief=False, reward_belief=False)
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
    
action_seq = to_categorical(np.array(action_seq).astype(np.int32), env.n_actions).squeeze()
obs_seq = to_categorical(np.array(obs_seq).astype(np.int32), env.n_obs).squeeze()

if use_cuda:
    obs_seq, action_seq = obs_seq.cuda(), action_seq.cuda()

print('DMM training')
start_time = time.time()
dmm = DMM(
    mode='discrete',
    x_prob_dim=env.n_obs, 
    x_dim=env.n_obs,
    a_dim=env.n_actions,
    b_dim=env.n_states,
    z_dim=env.n_states,
    num_particles=1, 
    annealing_epochs=annealing_epochs,
    adam_params={'lr': lr, 'lrd': lrd, 'betas' : (beta1, beta2)},
    emitter_hidden_dim=emitter_hidden_dim,
    transition_hidden_dim=transition_hidden_dim,
    inference_hidden_dim=inference_hidden_dim,
    minimum_annealing_factor=minimum_annealing_factor,
    use_cuda=use_cuda,
    use_gate=use_gate
)
epoch_nll_list = dmm.train(x_seq=obs_seq, a_seq=action_seq, num_epochs=num_epochs, mini_batch_size=mini_batch_size,)
train_time = time.time()-start_time
print(f"Training time: {round(train_time/3600, 1)}h")
print('Best NLL:', np.min(epoch_nll_list))
plt.plot(epoch_nll_list)
plt.savefig('images/nll');
states = torch.arange(env.n_states).reshape(env.n_states, 1)
states = to_categorical(states, env.n_states)
if simple_env:
    true_matrix = np.array([
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
    ])
else:
    true_matrix = np.array([
        [0.80, 0.20, 0.00],
        [0.20, 0.60, 0.20],
        [0.05, 0.70, 0.25],
        [0.00, 0.30, 0.70],
        [0.00, 0.00, 1.00]
    ])
if use_cuda:
    states = states.cuda()
print(f'Ground truth observation matrix: \n{true_matrix},\nLearned:', torch.round(dmm.dmm.emitter(states), decimals=2))
b_t_1 = states
a_t_1 = torch.zeros((env.n_states,), dtype=torch.int).reshape(-1, 1)
a_t_1 = to_categorical(a_t_1, env.n_actions)
if use_cuda:
    b_t_1, a_t_1 = b_t_1.cuda(), a_t_1.cuda()
print('Learned propagated beliefs for action 0:', torch.round(dmm.dmm.trans(b_t_1, a_t_1), decimals=2))

predictive = pyro.infer.Predictive(dmm.dmm.generative_model, guide=dmm.dmm.inference_model, num_samples=500)
svi_samples = predictive(x_batch=obs_seq[:1], a_batch=action_seq[:1], annealing_factor=1.0,)
for i in range(50):
    z_t = f'z_{i}'
    print(f'pred z_{i} {svi_samples[z_t].float().mean(0)} \
          \npred z_{i} rounded {svi_samples[z_t].float().mean(0).round()} \
          \ntrue z_{i} {np.array(state_seq)[0, i]} \
          \ntrue b_{i} {np.array(belief_seq)[0, i].round(2)} \
          \ntrue b_{i} rounded {np.array(belief_seq)[0, i].argmax()} \
          \nobs_{i} {obs_seq[0, i]} \
          \na_{i} {action_seq[0, i]}\n')

predictive = pyro.infer.Predictive(dmm.dmm.generative_model, guide=dmm.dmm.inference_model, num_samples=100)
svi_samples = predictive(x_batch=obs_seq[:100], a_batch=action_seq[:100], annealing_factor=1.0,)
pred_z_samples = np.empty((100, length))
pred_z_samples_round = np.empty((100, length))
for i in range(length):
    z_t = f'z_{i}'
    pred_z_t = svi_samples[z_t].float().mean(0)[:, 0].cpu()
    pred_z_samples[:, i] = pred_z_t
    pred_z_samples_round[:, i] = pred_z_t.round()
pred_z_samples = torch.tensor(pred_z_samples).flatten()
pred_z_samples_round = torch.tensor(pred_z_samples_round).flatten()
state_seq = torch.tensor(state_seq)[:100].flatten()
belief_seq = torch.tensor(belief_seq)[:100].argmax(-1).flatten()

mca = MulticlassAccuracy(num_classes=env.n_states, average=None)
print(f"Accuracy rounded samples pred_z-state {mca(pred_z_samples_round, state_seq)}")
print(f"Accuracy rounded samples true_b-state {mca(belief_seq, state_seq)}")
print(f"Accuracy rounded samples pred_z-true_b {mca(pred_z_samples_round, belief_seq)}")
cm = MulticlassConfusionMatrix(num_classes=env.n_states)
print(f"Confusion matrix pred_z-state {cm(pred_z_samples_round, state_seq)}")
print(f"Confusion matrix true_b-state {cm(belief_seq, state_seq)}")
print(f"Confusion matrix pred_z-true_b {cm(pred_z_samples_round, belief_seq)}")