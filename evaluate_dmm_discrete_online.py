import sys
import argparse
import multiprocessing
import pickle
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.nn import CrossEntropyLoss

from envs.env_discrete import DiscreteEnv, SimpleDiscreteEnv
from dmms.dmm import * 

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, timesteps, obs_dim, act_dim, size):
        self.obs_buf = torch.zeros((size, timesteps, obs_dim))
        self.act_buf = torch.zeros((size, timesteps, act_dim))
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs_list, act_list):
        length_list = len(obs_list)
        self.obs_buf[self.ptr:self.ptr+length_list] = obs_list
        self.act_buf[self.ptr:self.ptr+length_list] = act_list
        self.ptr = (self.ptr+length_list) % self.max_size
        self.size = min(self.size+length_list, self.max_size)

    def sample_batch(self, batch_size):
        idxs = torch.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs])
        return batch
    
    def get_buffers(self):
        return self.obs_buf[:self.size], self.act_buf[:self.size]

def policy():
    return 0

def epsilon_random_action(environment, epsilon=0.):
    "epsilon probability to pick a random action and (1-epsilon) to pick action 0"
    if np.random.rand() < epsilon:
        return environment.action_space.sample()
    else:
        return policy()
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]

def collect_one_episode(seed, dmm, simple_env, length, epsilon_prob, use_cuda):
    if simple_env:
        env = SimpleDiscreteEnv(seed=seed, return_belief=False, reward_belief=False)
    else:
        env = DiscreteEnv(seed=seed, monitoring='permanent', return_belief=False, reward_belief=False)
    obs_list = []
    state_list = []
    belief_true_list = []
    belief_predicted_list = []
    action_list = []
    obs = env.reset()
    prior_belief = torch.eye(1, env.n_states).reshape(1,-1)
    if use_cuda:
        prior_belief = prior_belief.cuda()
    for _ in range(length):
        obs_input = to_categorical(obs, env.n_obs).reshape(1,-1)
        if use_cuda:
            obs_input = obs_input.cuda()
        with torch.no_grad():
            pred_belief = dmm.dmm.inference(prior_belief, obs_input)
        obs_list.append([obs])
        state_list.append([env.state])
        belief_true_list.append(env.belief)
        belief_predicted_list.append(pred_belief)
        action = epsilon_random_action(env, epsilon_prob)
        action_list.append([action])
        obs, reward, done, info = env.step(action)
        prior_belief = pred_belief
    return obs_list, state_list, belief_true_list, belief_predicted_list, action_list

def evaluate_dmm(states, belief_true, belief_pred, use_cuda):
    states = states.reshape(-1)
    belief_true = belief_true.reshape(-1, belief_true.size(-1))
    belief_pred = belief_pred.reshape(-1, belief_pred.size(-1))

    loss = nn.CrossEntropyLoss()
    loss_b_pred_states = loss(belief_pred, states)
    loss_b_true_states = loss(belief_true, states)
    loss_b_pred_b_true = loss(belief_pred, belief_true)

    if use_cuda:
        mca = MulticlassAccuracy(num_classes=belief_true.shape[-1], average=None).to("cuda")
    else:
        mca = MulticlassAccuracy(num_classes=belief_true.shape[-1], average=None)
    belief_pred_rounded = belief_pred.argmax(-1)
    belief_true_rounded = belief_true.argmax(-1)
    mca_b_pred_states = mca(belief_pred_rounded, states)
    mca_b_true_states = mca(belief_true_rounded, states)
    mca_b_pred_b_true = mca(belief_pred_rounded, belief_true_rounded)

    if use_cuda:
        cm = MulticlassConfusionMatrix(num_classes=belief_true.shape[-1]).to("cuda")
    else:
        cm = MulticlassConfusionMatrix(num_classes=belief_true.shape[-1])
    cm_b_pred_states = cm(belief_pred_rounded, states)
    cm_b_true_states = cm(belief_true_rounded, states)
    cm_b_pred_b_true = cm(belief_pred_rounded, belief_true_rounded)
    return {"loss_b_pred_states": loss_b_pred_states, "loss_b_true_states": loss_b_true_states, \
            "loss_b_pred_b_true": loss_b_pred_b_true, "mca_b_pred_states": mca_b_pred_states, \
            "mca_b_true_states": mca_b_true_states, "mca_b_pred_b_true": mca_b_pred_b_true, \
            "cm_b_pred_states": cm_b_pred_states, "cm_b_true_states": cm_b_true_states, "cm_b_pred_b_true": cm_b_pred_b_true}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--number_evaluations', type=int, default=5)
    parser.add_argument('-nt', '--number_trials_per_evaluations', type=int, default=500)
    parser.add_argument('-e', '--eps', type=float, default=0.2)
    parser.add_argument('-l', '--length', type=int, default=100)
    parser.add_argument('-ae', '--annealing_epochs', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=1.)
    parser.add_argument('-b1', '--beta1', type=float, default=0.95)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-eh', '--emitter_hidden_dim', type=int, default=100)
    parser.add_argument('-th', '--transition_hidden_dim', type=int, default=100)
    parser.add_argument('-ih', '--inference_hidden_dim', type=int, default=100)
    parser.add_argument('-ep', '--num_epochs', type=int, default=1000)
    parser.add_argument('-bs', '--mini_batch_size', type=int, default=20)
    parser.add_argument('-af', '--minimum_annealing_factor', type=float, default=0.)
    parser.add_argument('-uc', '--use_cuda', type=int, default=0)
    parser.add_argument('-env', '--simple_env', type=int, default=0)
    parser.add_argument('-ug', '--use_gate', type=int, default=1)
    parser.add_argument('-tb', '--train_last_batch', type=int, default=0)
    parser.add_argument('-se', '--save_evaluation_results', type=int, default=0)
    parser.add_argument('-wo', '--workers', type=int, default=8)
    args = parser.parse_args() 

    use_cuda = bool(args.use_cuda)
    simple_env = bool(args.simple_env)

    if simple_env:
        env = SimpleDiscreteEnv(return_belief=False, reward_belief=False)
    else:
        env = DiscreteEnv(monitoring='permanent', return_belief=False, reward_belief=False)

    buffer_size = int(args.number_evaluations)*int(args.number_trials_per_evaluations)

    buffer = ReplayBuffer(timesteps=args.length, obs_dim=env.n_obs, act_dim=env.n_actions, size=buffer_size)

    dmm = DMM(
        mode='discrete',
        x_prob_dim=env.n_obs, 
        x_dim=env.n_obs,
        a_dim=env.n_actions,
        b_dim=env.n_states,
        z_dim=env.n_states,
        num_particles=1, 
        annealing_epochs=int(args.annealing_epochs),
        adam_params={'lr': float(args.learning_rate), 'lrd': float(args.learning_rate_decay), 'betas' : (float(args.beta1), float(args.beta2))},
        emitter_hidden_dim=int(args.emitter_hidden_dim),
        transition_hidden_dim=int(args.transition_hidden_dim),
        inference_hidden_dim=int(args.inference_hidden_dim),
        minimum_annealing_factor=float(args.minimum_annealing_factor),
        use_cuda=use_cuda,
        use_gate=bool(args.use_gate)
    )

    start_seed = 0
    
    n_workers = args.workers
    print(f"Using {n_workers} workers")

    epoch_nll_list_all = []

    for evaluation in tqdm(range(int(args.number_evaluations))):
        collect_one_episode_partial = partial(
            collect_one_episode, 
            dmm=dmm,
            simple_env=simple_env, 
            length=int(args.length), 
            epsilon_prob=float(args.eps), 
            use_cuda=use_cuda
        )
        seeds = range(start_seed, start_seed+int(args.number_trials_per_evaluations))
        start_seed += int(args.number_trials_per_evaluations)
        with Pool(n_workers) as pool:
            obs_list, state_list, belief_true_list, belief_predicted_list, action_list = zip(*pool.map( # consider imap and/or unordered
                collect_one_episode_partial, seeds
            ))
        action_seq = to_categorical(torch.as_tensor(action_list), env.n_actions).squeeze()
        obs_seq = to_categorical(torch.as_tensor(obs_list), env.n_obs).squeeze()
        state_list, belief_true_list = torch.as_tensor(np.asarray(state_list)), torch.as_tensor(np.asarray(belief_true_list))
        belief_predicted_list = torch.stack([torch.stack(belief_predicted_sublist).squeeze() for belief_predicted_sublist in belief_predicted_list])
        if use_cuda:
            state_list, belief_true_list, obs_seq, action_seq = state_list.cuda(), belief_true_list.cuda(), obs_seq.cuda(), action_seq.cuda()
        buffer.store(obs_seq, action_seq)
        evaluation_results = evaluate_dmm(state_list, belief_true_list, belief_predicted_list, use_cuda)
        for key, value in evaluation_results.items():
            print(f"\n{key}: {value}")
        if bool(args.save_evaluation_results):
            with open(f'results/evaluation_results_{time.strftime("%d-%m-%Y")}.pkl', 'wb') as f:
                pickle.dump(evaluation_results, f)
        states_dummy = torch.arange(env.n_states).reshape(env.n_states, 1)
        states_dummy = to_categorical(states_dummy, env.n_states)
        if use_cuda:
            states_dummy = states_dummy.cuda()
        with torch.no_grad():
            print(f'Ground truth observation matrix:', torch.round(dmm.dmm.emitter(states_dummy), decimals=2))
        if evaluation < int(args.number_evaluations) - 1:
            if bool(args.train_last_batch):
                epoch_nll_list = dmm.train(
                    x_seq=obs_seq, 
                    a_seq=action_seq, 
                    num_epochs=int(args.num_epochs), 
                    mini_batch_size=int(args.mini_batch_size),
                )
            else:
                obs_buf, act_buf = buffer.get_buffers()
                if use_cuda:
                    obs_buf, act_buf = obs_buf.cuda(), act_buf.cuda()
                epoch_nll_list = dmm.train(
                    x_seq=obs_buf, 
                    a_seq=act_buf, 
                    num_epochs=int(args.num_epochs), 
                    mini_batch_size=int(args.mini_batch_size),
                )
            epoch_nll_list_all.append(epoch_nll_list)
            print('Best NLL:', np.min(epoch_nll_list))
            plt.plot(np.array(epoch_nll_list_all).flatten())
            plt.savefig('images/nll_discrete_online');
        if use_cuda:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        use_cuda = bool(sys.argv[16])
    except:
        use_cuda = False
    if use_cuda:
        torch.multiprocessing.set_start_method('spawn')
    main()



