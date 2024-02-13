import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive

class EmissionNet(nn.Module):
    """
    Parameterizes the observation probability vector `p(x_t | z_t, a_{t-1})` 
    """

    def __init__(self, z_dim, hidden_dim, x_dim, a_dim=None): 
        super().__init__()
        # in some problems, the action may impact the observation generation and should thus be passed as input
        if a_dim:
            inp_dim = z_dim + a_dim
        else:
            inp_dim = z_dim
        # initialize the three linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(inp_dim, hidden_dim)
        self.lin_hidden_to_mean = nn.Linear(hidden_dim, x_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, x_dim)
        # initialize non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t, a_t_1=None): #TODO: maybe passing b_t instead of z_t?
        """
        Given the latent z at a particular time step t 
        (and the previous action if affects the observation generation),
        we return the vector of probabilities `ps` that parameterizes the categorical distribution `p(x_t|z_t)`
        """
        if a_t_1:
            inp = torch.cat([z_t, a_t_1], -1) 
        else:
            inp = z_t
        hidden = self.relu(self.lin_input_to_hidden(inp))
        loc = self.lin_hidden_to_mean(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden)) + 1e-5
        return loc, scale
    
class TransitionNet(nn.Module):
    """
    Parameterizes the transition probability vector `\tilde{b}_t = p(z_t | b_{t-1}, a_{t-1})` 
    """

    def __init__(self, b_dim, a_dim, transition_hidden_dim):
        super().__init__()
        # input dimension
        input_dim = b_dim + a_dim 
        # output dimension
        out_dim = int(b_dim/2)
        # initialize the three linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(input_dim, transition_hidden_dim)
        self.lin_hidden_to_b_next_mean = nn.Linear(transition_hidden_dim, out_dim)
        self.lin_hidden_to_b_next_scale = nn.Linear(transition_hidden_dim, out_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, b_t_1, a_t_1): 
        """
        Given the probability `b_{t-1}` over the latent `z_{t-1}` and the action `a_{t-1}`
        we return the loc and scale  that parameterizes the Normal distribution 
        `\tilde{b}_t = p(z_t | \tilde{b}_{t-1}, a_{t-1})`
        """
        # compute the gating function
        input = torch.cat([b_t_1, a_t_1], -1) 
        hidden = self.relu(self.lin_input_to_hidden(input))
        loc = self.lin_hidden_to_b_next_mean(hidden)
        scale = self.softplus(self.lin_hidden_to_b_next_scale(hidden)) + 1e-5
        return loc, scale
    
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
        # output dimension
        out_dim = int(b_dim/2)
        # initialize the three linear transformations used in the neural network
        self.lin_input_to_hidden = nn.Linear(input_q_dim, hidden_dim)
        self.lin_hidden_to_b_next_mean = nn.Linear(hidden_dim, out_dim)
        self.lin_hidden_to_b_next_scale = nn.Linear(hidden_dim, out_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

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
        hidden = self.relu(self.lin_input_to_hidden(input_q))
        b_t_loc = self.lin_hidden_to_b_next_mean(hidden)
        b_t_scale = self.softplus(self.lin_hidden_to_b_next_scale(hidden)) + 1e-5
        return b_t_loc, b_t_scale
    
class DMM_continuous(nn.Module): 
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    based on continuous variables.
    """

    def __init__(
        self,
        z_dim=1,
        x_dim=1,
        b_dim=2,
        a_dim=1,
        use_action_emitter=False,
        emitter_hidden_dim=100,
        transition_hidden_dim=100,
        inference_hidden_dim=100,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        if use_action_emitter:
            self.emitter = EmissionNet(z_dim, emitter_hidden_dim, x_dim, a_dim)
        else:
            self.emitter = EmissionNet(z_dim, emitter_hidden_dim, x_dim)
        self.trans = TransitionNet(b_dim, a_dim, transition_hidden_dim)
        self.inference = InferenceNet(b_dim, x_dim, inference_hidden_dim)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [
            affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # The initial belief is always 1
        self.b_tilde_0_mean = torch.ones(z_dim) 
        self.b_tilde_0_std = torch.zeros(z_dim) + 1e-5
        self.b_tilde_0 = torch.cat([self.b_tilde_0_mean, self.b_tilde_0_std], -1) 

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
        # initialize std as well if used
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

                if t == 0:
                    z_loc, z_scale = b_tilde_0[:, 0, None], b_tilde_0[:, 1, None]
                else:
                    z_loc, z_scale = self.trans(b_tilde_prev, a_batch[:, t-1]) 

                # track variables
                #b_tilde_t = pyro.deterministic("b_tilde_%d" % t, z_loc)

                # then sample z_t according to Normal(z_loc, z_scale)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale)
                        .to_event(1)
                    )

                # compute the emissions
                if self.use_action_emitter:
                    x_loc, x_scale = self.emitter(z_t, a_batch[:, t-1])
                else:
                    x_loc, x_scale = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # Normal distribution p(x_t|z_t)
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.Normal(x_loc, x_scale)
                    .to_event(1),
                    obs=x_batch[:, t], 
                )

                # the latent sampled at this time step will be conditioned upon
                # in the next time step by carring the belief variable
                b_tilde_prev = torch.cat([z_loc, z_scale], -1) #TODO: try inverse_softplus z_scale
                #b_tilde_t_param = pyro.param("b_tilde_%d" % t, b_tilde_prev.clone().detach())

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
                    z_loc, z_scale = self.inference(b_tilde_0, x_batch[:, t])
                    #b_tilde_t_param = pyro.param("b_tilde_%d" % t, torch.cat([z_loc, z_scale], -1).clone().detach())
                elif x_batch[:, t] is not None: #TODO: this is wrong, it is always True, fix it for non-permanent monitoring
                    # we have acquired an observation, so we first propgate the belief and
                    # then use the observation to reduce the uncertainty and infer b_t, namely
                    # the distribution q(z_t | b_{t-1}, x_{t}, a_{t-1})
                    #TODO: if sharing parameters doesn't work, maybe using one inference nn only like in the original DMM?
                    z_loc_tilde_t, z_scale_tilde_t = self.trans(b_prev, a_batch[:, t-1]) 
                    b_tilde_t = torch.cat([z_loc_tilde_t, z_scale_tilde_t], -1) #TODO: try inverse_softplus z_scale
                    #b_tilde_t_param = pyro.param("b_tilde_%d" % t, b_tilde_t.clone().detach())
                    z_loc, z_scale = self.inference(b_tilde_t, x_batch[:, t]) 
                else:
                    # We did not acquire a new observation  
                    # so our new belief is just the propagated b_tilde
                    z_loc, z_scale = self.trans(b_prev, a_batch[:, t-1])
                # track variables
                #b_t = pyro.deterministic("b_%d" % t, b_t)

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(
                        dist.Normal(z_loc, z_scale), self.iafs
                    )
                    assert z_dist.event_shape == (self.b_tilde_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(x_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (
                        len(x_batch),
                        int(self.b_tilde_0.size(0)/2),
                    )

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample(
                            "z_%d" % t, z_dist
                        )
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample(
                            "z_%d" % t,
                            z_dist.to_event(1),
                        )

                # the latent sampled at this time step will be conditioned upon in the next time step
                # by carring the belief variable
                b_prev = torch.cat([z_loc, z_scale], -1) #TODO: try inverse_softplus z_scale