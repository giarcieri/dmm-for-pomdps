import sys
import argparse
import pickle
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial
from torch.nn import MSELoss
from tqdm import tqdm
from pyro.infer import SVI

from envs.env_continuous import ContinuousEnv
from dmms.dmm import * 

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
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
    
def policy(env):
    return env.action_space.sample()

def collect_one_episode(seed, dmm, length, use_cuda):
    env = ContinuousEnv(seed=seed, power=1)
    obs_list = []
    state_list = []
    state_dist_list = []
    belief_pred_list = []
    action_list = []
    obs = torch.tensor(env.reset()).reshape(1, -1)
    prior_belief_mean = torch.ones(1)
    prior_belief_std = torch.zeros(1) + 1e-5 
    prior_belief = torch.cat([prior_belief_mean, prior_belief_std], -1).reshape(1, -1)
    if use_cuda:
        prior_belief = prior_belief.cuda()
        obs = obs.cuda()
    for t in range(length):
        if use_cuda:
            obs = obs.cuda()
        if t == 0:
            with torch.no_grad():
                z_loc, z_scale = dmm.dmm.inference(prior_belief, obs)
                pred_belief = torch.cat([z_loc, z_scale], -1)
        else:
            if use_cuda:
                action = action.cuda()
            with torch.no_grad():
                z_loc_tilde, z_scale_tilde = dmm.dmm.trans(prior_belief, action) 
                pred_belief_tilde = torch.cat([z_loc_tilde, z_scale_tilde], -1)
                z_loc, z_scale = dmm.dmm.inference(pred_belief_tilde, obs) 
                pred_belief = torch.cat([z_loc, z_scale], -1)
        obs_list.append(obs)
        state_list.append(env.state)
        state_dist_list.append([env.state_mean.squeeze(), env.state_std.squeeze()])
        belief_pred_list.append(pred_belief.squeeze())
        action = policy(env)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        action = torch.tensor(action).reshape(1, -1)
        obs = torch.tensor(obs).reshape(1, -1)
        prior_belief = pred_belief
    return obs_list, state_list, state_dist_list, belief_pred_list, action_list

def kl_gaussian(mean_p, std_p, mean_q, std_q):
    """Calculate KL(p,q) between two gaussian distributions."""
    kl = torch.log(std_q/std_p) + (std_p**2 + (mean_p - mean_q)**2)/(2*std_q**2) - 0.5
    return torch.mean(kl)

def evaluate_dmm(states, state_dist, belief_pred, obs):
    states = states.reshape(-1,)
    obs = obs.reshape(-1,)
    belief_pred = belief_pred.reshape(-1, belief_pred.size(-1))
    state_dist = state_dist.reshape(-1, state_dist.size(-1))
    belief_pred_mean = belief_pred[:, 0]
    belief_pred_std = belief_pred[:, 1]
    state_mean = state_dist[:, 0]
    state_std = state_dist[:, 1]

    z = (states - belief_pred_mean)/belief_pred_std
    print("standard", z.mean(), z.std())

    loss = MSELoss()
    loss_b_pred_states = loss(belief_pred_mean, states)
    loss_obs_states = loss(obs, states)

    kl = kl_gaussian(state_mean, state_std, belief_pred_mean, belief_pred_std)
    return {"loss_obs_states": loss_obs_states, "loss_b_pred_states": loss_b_pred_states, "kl_state_b_pred": kl}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--number_evaluations', type=int, default=5)
    parser.add_argument('-nt', '--number_trials_per_evaluations', type=int, default=500)
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
    parser.add_argument('-pw', '--power', type=int, default=1)
    parser.add_argument('-uc', '--use_cuda', type=int, default=0)
    parser.add_argument('-el', '--elbo', type=str, default='gaussian')
    parser.add_argument('-tb', '--train_last_batch', type=int, default=0)
    parser.add_argument('-se', '--save_evaluation_results', type=int, default=0)
    parser.add_argument('-wo', '--workers', type=int, default=8)
    args = parser.parse_args()

    print(args)

    dmm = DMM(
        mode='continuous',
        x_prob_dim=None, 
        x_dim=1,
        a_dim=1,
        b_dim=2,
        z_dim=1,
        num_particles=1, 
        annealing_epochs=int(args.annealing_epochs),
        adam_params={'lr': float(args.learning_rate), 'lrd': float(args.learning_rate_decay), 'betas' : (float(args.beta1), float(args.beta2))},
        emitter_hidden_dim=int(args.emitter_hidden_dim),
        transition_hidden_dim=int(args.transition_hidden_dim),
        inference_hidden_dim=int(args.inference_hidden_dim),
        minimum_annealing_factor=float(args.minimum_annealing_factor),
        use_cuda=bool(args.use_cuda),
        elbo=str(args.elbo)
    )

    use_cuda = bool(args.use_cuda)
    buffer_size = int(args.number_evaluations)*int(args.number_trials_per_evaluations)

    buffer = ReplayBuffer(timesteps=args.length, obs_dim=1, act_dim=1, size=buffer_size)


    start_seed = 0
    
    n_workers = args.workers
    print(f"Using {n_workers} workers")

    all_belief_pred = torch.zeros((buffer_size, args.length, 2))
    all_true_states = torch.zeros((buffer_size, args.length, 1))
    ptr = 0

    epoch_nll_list_all = []

    for evaluation in tqdm(range(int(args.number_evaluations))):
        dmm.dmm.eval()

        # Collect new trials
        collect_one_episode_partial = partial(
            collect_one_episode, 
            dmm=dmm,
            length=int(args.length), 
            use_cuda=use_cuda
        )
        seeds = range(start_seed, start_seed+int(args.number_trials_per_evaluations))
        start_seed += int(args.number_trials_per_evaluations)
        with Pool(n_workers) as pool:
            obs_list, state_list, state_dist_list, belief_pred_list, action_list = zip(*pool.map( # consider imap and/or unordered
                collect_one_episode_partial, seeds
            ))
        pool.close()
        pool.join()
        obs_list = torch.tensor(obs_list).unsqueeze(-1)
        state_list = torch.tensor(state_list)
        state_dist_list = torch.tensor(np.asarray(state_dist_list))
        belief_pred_list = torch.stack([torch.stack(belief_predicted_sublist).squeeze() for belief_predicted_sublist in belief_pred_list])
        action_list = torch.tensor(action_list)
        if use_cuda:
            obs_list = obs_list.cuda()
            state_list = state_list.cuda()
            state_dist_list = state_dist_list.cuda()
            belief_pred_list = belief_pred_list.cuda()
            action_list = action_list.cuda()
        buffer.store(obs_list, action_list)
        all_belief_pred[ptr:ptr+args.number_trials_per_evaluations] = belief_pred_list
        all_true_states[ptr:ptr+args.number_trials_per_evaluations] = state_list
        ptr += args.number_trials_per_evaluations

        # Evaluate dmm
        evaluation_results = evaluate_dmm(state_list, state_dist_list, belief_pred_list, obs_list)
        for key, value in evaluation_results.items():
            print(f"\n{key}: {value}")
        if bool(args.save_evaluation_results):
            with open(f'results/evaluation_results_{time.strftime("%d-%m-%Y")}.pkl', 'wb') as f:
                pickle.dump(evaluation_results, f)
        states_dummy = torch.range(0, 1, 0.25).reshape(-1,1)
        if use_cuda:
            states_dummy = states_dummy.cuda()
        with torch.no_grad():
            print(f"Emission means for", states_dummy, dmm.dmm.emitter(states_dummy)[0])
            print(f"Emission std for", torch.exp(states_dummy)*0.05, dmm.dmm.emitter(states_dummy)[1])
        obs_dummy = torch.tensor([0.5]*len(states_dummy)).reshape(-1,1)
        std_dummy = torch.zeros((5,1)) + 1e-5
        a_dummy = torch.zeros(len(states_dummy)).reshape(-1,1) + 0.1
        if use_cuda:
            std_dummy = std_dummy.cuda()
            obs_dummy = obs_dummy.cuda()
            a_dummy = a_dummy.cuda()
        b_dummy = torch.cat([states_dummy, std_dummy], -1)
        with torch.no_grad():
            print(f"Next state means for action {a_dummy}", dmm.dmm.trans(b_dummy, a_dummy)[0])
            print(f"Next state std for action {a_dummy}", dmm.dmm.trans(b_dummy, a_dummy)[1])
            print(f"Next state inferred for obs 0.5", dmm.dmm.inference(b_dummy, obs_dummy))

        # Training
        if evaluation < int(args.number_evaluations) - 1:
            dmm.dmm.train()
            if bool(args.train_last_batch):
                epoch_nll_list = dmm.train(
                    x_seq=obs_list, 
                    a_seq=action_list, 
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
            plt.ylim(-15, 60)
            plt.savefig('images/nll_continuous_online');
        if use_cuda:
            torch.cuda.empty_cache()
    #with open(f'all_belief_pred.pkl', 'wb') as f:
    #    pickle.dump(all_belief_pred, f)
    #with open(f'all_true_states.pkl', 'wb') as f:
    #    pickle.dump(all_true_states, f)

if __name__ == "__main__":
    try:
        use_cuda = bool(int(sys.argv[32]))
        print("Use cuda", use_cuda)
    except:
        use_cuda = False
    if use_cuda:
        torch.multiprocessing.set_start_method('spawn')

    main()