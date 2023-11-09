from envs.env_continuous import ContinuousEnv
from dmms.dmm import * 
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser(description='dmm')
parser.add_argument('-n', '--n_batch', type=int, default=500)
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
parser.add_argument('-pw', '--power', type=int, default=1)
parser.add_argument('-uc', '--use-cuda', type=int, default=0)
parser.add_argument('-el', '--elbo', type=str, default='gaussian')
args = parser.parse_args()

print(args)

n_batch = int(args.n_batch)
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
power = int(args.power)
use_cuda = bool(args.use_cuda)
elbo = str(args.elbo)

obs_seq = []
state_seq = []
action_seq = []
print('Collecting sequences')
env = ContinuousEnv(power=power)
for _ in tqdm(range(n_batch)):
    obs_list = []
    state_list = []
    action_list = []
    obs = env.reset()
    for _ in range(length):
        obs_list.append(obs)
        state_list.append(env.state)
        action = env.action_space.sample()
        action_list.append(action)
        obs, reward, done, info = env.step(action)
    obs_seq.append(obs_list)
    state_seq.append(state_list)
    action_seq.append(action_list)
    
action_seq = torch.tensor(action_seq)
obs_seq = torch.tensor(obs_seq)

if use_cuda:
    obs_seq, action_seq = obs_seq.cuda(), action_seq.cuda()

print('DMM training')
start_time = time.time()

dmm = DMM(
    mode='continuous',
    x_prob_dim=None, 
    x_dim=1,
    a_dim = 1,
    b_dim=1,
    z_dim=1,
    num_particles=1, 
    annealing_epochs=annealing_epochs,
    adam_params={'lr': lr, 'lrd': lrd, 'betas' : (beta1, beta2)},
    emitter_hidden_dim=emitter_hidden_dim,
    transition_hidden_dim=transition_hidden_dim,
    inference_hidden_dim=inference_hidden_dim,
    minimum_annealing_factor=minimum_annealing_factor,
    use_cuda=use_cuda,
    elbo=elbo
)
epoch_nll_list = dmm.train(x_seq=obs_seq, a_seq=action_seq, num_epochs=num_epochs, mini_batch_size=mini_batch_size,)
train_time = time.time()-start_time
print(f"Training time: {round(train_time/3600, 1)}h")
print('Best NLL:', np.min(epoch_nll_list))
plt.plot(epoch_nll_list)
plt.ylim(-10, 10)
plt.savefig('images/nll_continuous');

dmm.dmm.eval()
states = torch.range(0, 1, 0.25).reshape(-1,1)
if use_cuda:
    states = states.cuda()
with torch.no_grad():
    print(f"Emission means for", states, dmm.dmm.emitter(states)[0])
    print(f"Emission std for", torch.exp(states)*0.05, dmm.dmm.emitter(states)[1])

a_t_1 = torch.zeros(len(states)).reshape(-1,1)
if use_cuda:
    a_t_1 = a_t_1.cuda()
states_np = np.linspace(0, 1, 5).reshape(-1,1)
true_means = np.maximum(0,states_np-np.exp(-states_np*5)*0.5-0.1, dtype=np.float32)
true_std = (np.maximum(0, states_np, dtype=np.float32)-np.maximum(0, true_means, dtype=np.float32))*0.5 + 0.02
obs = torch.tensor([0.5]*len(states)).reshape(-1,1)
if use_cuda:
    obs = obs.cuda()
with torch.no_grad():
    print(f"Next state means for action {a_t_1}", true_means, dmm.dmm.trans(states, a_t_1)[0])
    print(f"Next state std for action {a_t_1}", true_std, dmm.dmm.trans(states, a_t_1)[1])
    print(f"Next state inferred for obs 0.5", states, dmm.dmm.inference(states, obs))

predictive = pyro.infer.Predictive(dmm.dmm.generative_model, guide=dmm.dmm.inference_model, num_samples=500)
svi_samples = predictive(x_batch=obs_seq[:1], a_batch=action_seq[:1], annealing_factor=1.0,)
for i in range(50):
    z_t = f'z_{i}'
    print(f'pred z_{i} {svi_samples[z_t].mean(0)}\ntrue z_{i} {np.array(state_seq)[0, i]}\nobs_{i} {obs_seq[0, i]}\na_{i} {action_seq[0, i]}\n')