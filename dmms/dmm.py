from .dmm_discrete import *
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam
import logging 
from tqdm import tqdm

class DMM():
    """
    This encapsulates the Deep Markov Model
    both based on dicrete and continuous variables and the training functions
    """

    def __init__(
        self,
        mode='discrete', # 'discrete' or 'continuous'
        z_dim=1,
        x_dim=1,
        x_prob_dim=5,
        b_dim=5,
        a_dim=1,
        use_action_emitter=False,
        emitter_hidden_dim=100,
        transition_hidden_dim=100,
        inference_hidden_dim=100,
        #num_iafs=0, #needed for continuous
        #iaf_dim=50, #needed for continuous
        use_cuda=False,
        adam_params={'lr': 1e-3, 'lrd': 1., 'betas' : (0.9, 0.999)},
        num_particles=1,
        filename=None,
        annealing_epochs=0,
        minimum_annealing_factor=0.2,
        keep_logs=False
    ):
        # initialize the dmm with dicrete or with continuous variables
        if mode == 'discrete':
            self.dmm = DMM_discrete(z_dim, x_dim, x_prob_dim, b_dim, a_dim, use_action_emitter, emitter_hidden_dim, 
                                    transition_hidden_dim, inference_hidden_dim, use_cuda)
        elif mode == 'continuous':
            raise NotImplementedError
        
        self.adam = ClippedAdam(adam_params)
        elbo = Trace_ELBO(num_particles=num_particles)
        self.svi = SVI(self.dmm.generative_model, self.dmm.inference_model, self.adam, loss=elbo)

        self.annealing_epochs = annealing_epochs
        self.minimum_annealing_factor = minimum_annealing_factor
        self.keep_logs = keep_logs

        # logging
        if keep_logs:
            logging.basicConfig(
                level=logging.DEBUG, format="%(message)s", filename=filename, filemode="w"
            )
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            logging.getLogger("").addHandler(console)

    def save_checkpoint(self, path_save_model, path_save_opt): # paths could be passed in init
        if self.keep_logs:
            logging.info("saving model to %s..." % path_save_model)
            logging.info("saving optimizer states to %s..." % path_save_opt)
        torch.save(self.dmm.state_dict(), path_save_model)
        self.adam.save(path_save_opt)

    # loads the model and optimizer states from disk
    def load_checkpoint(self, path_load_model, path_load_opt):
        if self.keep_logs:
            logging.info("loading model from %s..." % path_load_model)
            logging.info("loading optimizer states from %s..." % path_load_opt)
        self.dmm.load_state_dict(torch.load(path_load_model))
        self.adam.load(path_load_opt)

    def train(
            self, 
            x_seq,
            a_seq,
            num_epochs,
            mini_batch_size=20,
            load_model=False, 
            path_load_model=None, 
            path_load_opt=None,
            save_model=False,
            path_save_model=None, 
            path_save_opt=None
    ):
        if load_model:
            self.load_checkpoint(path_load_model, path_load_opt)
        epoch_nll_list = []
        for epoch in tqdm(range(num_epochs)):
            # accumulator for our estimate of the negative log likelihood for this epoch
            epoch_nll = 0.0
            # prepare mini-batch subsampling indices for this epoch
            N_train_data = len(x_seq)
            max_T = x_seq.shape[1]
            shuffled_indices = torch.randperm(N_train_data)
            N_mini_batches = int(
                N_train_data / mini_batch_size
                + int(N_train_data % mini_batch_size > 0)
            )

            # process each mini-batch; this is where we take gradient steps
            for which_mini_batch in range(N_mini_batches):
                # Compute mini-batch indices
                mini_batch_start = which_mini_batch * mini_batch_size
                mini_batch_end = np.min(
                    [(which_mini_batch + 1) * mini_batch_size, N_train_data]
                )
                mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
                # select mini-batch from data
                x_batch, a_batch = x_seq[mini_batch_indices, :], a_seq[mini_batch_indices, :]
                # compute annealing factor
                if self.annealing_epochs > 0 and epoch < self.annealing_epochs:
                    # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                    min_af = self.minimum_annealing_factor
                    annealing_factor = min_af + (1.0 - min_af) * (
                        float(which_mini_batch + epoch * N_mini_batches + 1)
                        / float(self.annealing_epochs * N_mini_batches)
                    )
                else:
                    # by default the KL annealing factor is unity
                    annealing_factor = 1.0
                epoch_nll += self.svi.step(x_batch, a_batch, annealing_factor)/(mini_batch_size*max_T)
            epoch_nll_list.append(epoch_nll)
            if save_model:
                self.save_checkpoint(path_save_model, path_save_opt)
        return epoch_nll_list
        
    def evaluation(self, x_batch, a_batch, n_eval_samples=1):
        def rep(x):
            return torch.repeat_interleave(x, n_eval_samples, dim=0)
        seq_lengths = len(x_batch)
        x_batch = rep(x_batch)
        a_batch = rep(a_batch)
        loss = self.svi.evaluate_loss(x_batch, a_batch) / seq_lengths
        return loss


