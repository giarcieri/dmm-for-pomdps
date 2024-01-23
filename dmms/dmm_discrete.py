import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes).to(y.device)[y]

class EmissionNet(nn.Module):
    """
    Parameterizes the observation probability vector `p(x_t | z_t, a_{t-1})` with a softmax activation
    """

    def __init__(self, z_dim, hidden_dim, x_prob_dim, a_dim=None): 
        super().__init__()
        # in some problems, the action may impact the observation generation and should thus be passed as input
        if a_dim:
            inp_dim = z_dim + a_dim
        else:
            inp_dim = z_dim
        # initialize the three linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(inp_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_emission = nn.Linear(hidden_dim, x_prob_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z_t, a_t_1=None):
        """
        Given the latent z at a particular time step t 
        (and the previous action if affects the observation generation),
        we return the vector of probabilities `ps` that parameterizes the categorical distribution `p(x_t|z_t)`
        """
        if a_t_1:
            inp = torch.cat([z_t, a_t_1], -1) 
        else:
            inp = z_t
        h1 = self.relu(self.lin_input_to_hidden(inp))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = self.softmax(self.lin_hidden_to_emission(h2))
        return ps
    
class TransitionNet_gate(nn.Module):
    """
    Parameterizes the transition probability vector `\tilde{b}_t = p(z_t | b_{t-1}, a_{t-1})` 
    with a softmax activation
    """

    def __init__(self, b_dim, a_dim, transition_hidden_dim):
        super().__init__()
        # input dimension
        input_dim = b_dim + a_dim 
        # initialize the six linear transformations used in the neural network
        self.gate_input_to_hidden = nn.Linear(input_dim, transition_hidden_dim)
        self.gate_hidden_to_gate = nn.Linear(transition_hidden_dim, b_dim)
        self.non_lin_part_input_to_hidden = nn.Linear(input_dim, transition_hidden_dim)
        self.non_lin_part_hidden_to_z_next = nn.Linear(transition_hidden_dim, b_dim)
        self.lin_part_input_to_z_next = nn.Linear(input_dim, b_dim)
        # modify the default initialization of lin_part_input_to_z_next
        # so that it's starts out as the identity function
        self.lin_part_input_to_z_next.weight.data = torch.eye(b_dim, input_dim)
        self.lin_part_input_to_z_next.bias.data = torch.zeros(b_dim)
        # activation function for the logits
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, b_t_1, a_t_1): #TODO: try not to use the gate function but as in dmm_continuous
        """
        Given the probability `b_{t-1}` over the latent `z_{t-1}` and the action `a_{t-1}`
        we return the probability vector `\tilde{b}_{t-1}` that parameterizes the
        Categorical distribution `\tilde{b}_t = p(z_t | \tilde{b}_{t-1}, a_{t-1})`
        """
        # compute the gating function
        input = torch.cat([b_t_1, a_t_1], -1) 
        _gate = self.relu(self.gate_input_to_hidden(input))
        gate = torch.sigmoid(self.gate_hidden_to_gate(_gate))
        # compute the non linear part of the gating function
        _non_lin_logits = self.relu(self.non_lin_part_input_to_hidden(input))
        non_lin_logits = self.non_lin_part_hidden_to_z_next(_non_lin_logits)
        # assemble the probability distribution used to sample z_t, which mixes a linear transformation
        # of the input with the non linear transformation modulated by the gating function
        logits = (1 - gate) * self.lin_part_input_to_z_next(input) + gate * non_lin_logits
        # return the state transition probability vector 
        # which approximates the probability distribution over the next state
        return self.softmax(logits)
    
class TransitionNet_no_gate(nn.Module):
    """
    Parameterizes the transition probability vector `\tilde{b}_t = p(z_t | b_{t-1}, a_{t-1})` 
    with a softmax activation
    """

    def __init__(self, b_dim, a_dim, transition_hidden_dim):
        super().__init__()
        # input dimension
        input_dim = b_dim + a_dim 
        # initialize the linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(input_dim, transition_hidden_dim)
        self.lin_hidden_to_b_next = nn.Linear(transition_hidden_dim, b_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, b_t_1, a_t_1): 
        """
        Given the probability `b_{t-1}` over the latent `z_{t-1}` and the action `a_{t-1}`
        we return the probability vector `\tilde{b}_{t-1}` that parameterizes the
        Categorical distribution `\tilde{b}_t = p(z_t | \tilde{b}_{t-1}, a_{t-1})`
        """
        input = torch.cat([b_t_1, a_t_1], -1) 
        _b_t = self.relu(self.lin_input_to_hidden(input))
        b_t = self.softmax(self.lin_hidden_to_b_next(_b_t))
        return b_t
    
class InferenceNet(nn.Module):
    """
    Parameterizes `b_t = q(z_t | b_{t-1}, a_{t-1}, x_{t})`, which is the basic building block
    of the guide (i.e. the variational distribution).
    We share the parameters with the TransitionNet to first propagate the belief without the observation,
    namely computing `\tilde{b}_t`, and subsequently inferring the belief network `b_t` with `x_t`.
    """

    def __init__(self, b_dim, x_dim, hidden_dim):
        super().__init__()
        # input of the inference network, namely the proagated belief and the observation
        input_q_dim = b_dim + x_dim
        # initialize the linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(input_q_dim, hidden_dim)
        self.lin_hidden_to_b_next = nn.Linear(hidden_dim, b_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, b_tilde_t, x_t):
        """
        Given the belief `b_{t-1}` and the action `a_{t-1}` at at a particular time step t-1, 
        we first propagate the belief to the next time step by predicting the probability distribution
        `\tilde{b}_t` over the latent `z_t` with the TransitionNet.
        Then, given the current observation x at time t, 
        the network belief `b_t` over z at time t is returned,
        which parameterizes `b_t = q(z_t | b_{t-1}, a_{t-1}, x_{t})`.

        Here it is assumed that `\tilde{b}_t` has already been computed and passed as input.
        """
        input_q = torch.cat([b_tilde_t, x_t], -1) 
        _b_t = self.relu(self.lin_input_to_hidden(input_q))
        b_t = self.softmax(self.lin_hidden_to_b_next(_b_t))
        return b_t
    
class DMM_discrete(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    based on discrete states and actions.
    """

    def __init__(
        self,
        z_dim=1,
        x_dim=1,
        x_prob_dim=5,
        b_dim=4,
        a_dim=1,
        use_action_emitter=False,
        emitter_hidden_dim=100,
        transition_hidden_dim=100,
        inference_hidden_dim=100,
        use_cuda=False,
        use_gate=True,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        if use_action_emitter:
            self.emitter = EmissionNet(z_dim, emitter_hidden_dim, x_prob_dim, a_dim)
        else:
            self.emitter = EmissionNet(z_dim, emitter_hidden_dim, x_prob_dim)
        if use_gate:
            self.trans = TransitionNet_gate(b_dim, a_dim, transition_hidden_dim)
        else:
            self.trans = TransitionNet_no_gate(b_dim, a_dim, transition_hidden_dim)
        self.inference = InferenceNet(b_dim, x_dim, inference_hidden_dim)

        # The initial belief is always [1., 0., 0., ...]
        self.b_tilde_0 = torch.eye(1, b_dim).squeeze()

        self.use_cuda = use_cuda
        self.use_action_emitter = use_action_emitter
        self.z_dim = z_dim
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the generative model p(x_{0:T} | z_{0:T}) p(z_{0:T} | a_{0:T-1}) 
    def generative_model(
            self,
            x_batch, # shape (Batches x Timesteps x Dimensions)
            a_batch, # shape (Batches x Timesteps x Dimensions) 
            annealing_factor=1.0,
    ):        
        # this is the number of time steps we need to process in the mini-batch
        T_max = x_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # initialize the probability distribution over the latent
        b_tilde_0 = self.b_tilde_0.expand(x_batch.size(0), self.b_tilde_0.size(0))
        if self.use_cuda:
            b_tilde_0 = b_tilde_0.cuda()

        #b_tilde_prev = b_tilde_0

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(x_batch)):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov 
            # so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(T_max)):
                # the next chunk of code samples \tilde{b}_t = p(z_t | \tilde{b}_{t-1}, a_{t-1})

                # for the first time step t=0, we do not have actions and sample directly from
                # the initialized distribution, 
                # else we compute the propagated belief  \tilde{b}_t = p(z_t | \tilde{b}_{t-1}, a_{t-1})
                if t == 0:
                    b_tilde_t = b_tilde_0
                else:
                    b_tilde_t = self.trans(b_tilde_prev, a_batch[:, t-1]) #TODO: if this doesn't work, maybe passing z_prev as input?
                # track variables
                #b_tilde_t = pyro.param("b_tilde_%d" % t, b_tilde_t)


                # then sample z_t according to dist.Categorical(b_tilde_t)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Categorical(b_tilde_t).to_event(1)
                    )
                #z_t = z_t[:, None].float()
                z_t = to_categorical(z_t, self.z_dim)
                # compute the probabilities that parameterize the Categorical distribution
                # for the observation likelihood
                if self.use_action_emitter:
                    emission_probs_t = self.emitter(z_t, a_batch[:, t-1]) 
                else:
                    emission_probs_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # Categorical distribution p(x_t|z_t)
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.Categorical(emission_probs_t)
                    .to_event(1),
                    obs=x_batch[:, t].squeeze().argmax(-1), 
                )
                # the latent sampled at this time step will be conditioned upon
                # in the next time step by carring the belief variable
                b_tilde_prev = b_tilde_t

    # the guide q(z_{0:T} | x_{0:T},  a_{0:T-1}) (i.e. the variational distribution)
    def inference_model(
            self,
            x_batch,
            a_batch,
            annealing_factor=1.0,
    ):
        # this is the number of time steps we need to process in the mini-batch
        T_max = x_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # We initialize the belief with b_tilde_0, which can be seen as the prior belief
        # when no observation has been observed yet. 
        b_tilde_0 = self.b_tilde_0.expand(x_batch.size(0), self.b_tilde_0.size(0)) 
        if self.use_cuda:
            b_tilde_0 = b_tilde_0.cuda()

        #b_prev = b_tilde_0

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(x_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO 
            # can use multiple samples from the guide at each z
            for t in pyro.markov(range(T_max)):
                # We take into account the possibility that the observations could not be acquired 
                # at every timestep, in which case the propagated belief `\tilde{b}_t` is returned.
                if t == 0:
                    # No action at the first t, b_tilde_0 coincides with the belief b_0
                    b_t = self.inference(b_tilde_0, x_batch[:, t])
                elif x_batch[:, t] is not None: #TODO: this is wrong, it is always True, fix it for non-permanent monitoring
                    # we have acquired an observation, so we first propgate the belief and
                    # then use the observation to reduce the uncertainty and infer b_t, namely
                    # the distribution q(z_t | b_{t-1}, x_{t}, a_{t-1})
                    b_tilde_t = self.trans(b_prev, a_batch[:, t-1]) #TODO: if sharing parameters doesn't work,
                    b_t = self.inference(b_tilde_t, x_batch[:, t]) # maybe using one inference nn only like in the original DMM?
                else:
                    # We did not acquire a new observation  
                    # so our new belief is just the propagated b_tilde
                    b_t = b_tilde_t = self.trans(b_prev, a_batch[:, t-1]) 
                # track variables
                #b_t = pyro.param("b_%d" % t, b_t)

                z_dist = dist.Categorical(b_t)

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                            "z_%d" % t,
                            z_dist.to_event(1)
                        )
                # the latent sampled at this time step will be conditioned upon in the next time step
                # by carring the belief variable
                b_prev = b_t