import os
import random
# python data types
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from torch.distributions import transforms
# model-specific imports
from lotka_volterra import LotkaVolterra

import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

DTYPE = torch.float32
NP_DTYPE = np.float32


class Model():

    def __init__(self, network_params, p, dt, obs, params, priors, features):
        self.p = p
        self.obs = obs
        self.time_index = np.int32(self.obs['times'] / dt)
        # print(features['feature_init'])
        # print(obs)
        self.SDE = LotkaVolterra(network_params, p, dt, obs, params, priors, features)
        self.opt = torch.optim.Adam(self.SDE.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=50, eta_min=0)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.95)
    
    def train_step(self, step):
        self.opt.zero_grad()
        mean_loss, kl, n_obs_logprob = self.SDE()
        mean_loss.backward()
        nn.utils.clip_grad_norm_(self.SDE.parameters(), 4e3, norm_type=1)
        self.opt.step()
        self.scheduler.step()
        return mean_loss.item(), kl.item(), n_obs_logprob.item()

    # train the model
    def train(self, niter, path):
        '''
        trains model
        :params niter: number of iterations
        :params PATH: path to tensorboard output
        '''
        print("Training model...")
        writer = SummaryWriter()
        # log_dir = f'{path}/{datetime.now().strftime("%y-%m-%d-%H:%M:%S")}'
        # writer = tf.summary.create_file_writer(log_dir)
        for i in range(niter):
            loss, kl, n_obs_logprob = self.train_step(i)
            # if i == 0:
            #     writer.add_graph(self.SDE, torch.zeros(1))
                # print(self.SDE._diff_bridge._rnn_cell.model[0].b.grad.mean())
            if i % 10 == 1:
                with torch.no_grad():
                    writer.add_scalar('Loss/mean_loss', loss, i)
                    writer.add_scalar('Loss/kl', kl, i)
                    writer.add_scalar('Loss/n_obs_logprob', n_obs_logprob, i)
                    writer.add_scalar('Loss/sde_loss', loss - kl - n_obs_logprob, i)
                    # writer.add_scalar('params/c1/c1_mean', self.SDE.c1_mean.item(), i)
                    # writer.add_scalar('params/c1/c1_std', F.softplus(self.SDE.c1_std_o).item(), i)
                    # writer.add_scalar('params/c2/c2_mean', self.SDE.c2_mean.item(), i)
                    # writer.add_scalar('params/c2/c2_std', F.softplus(self.SDE.c2_std_o).item(), i)
                    # writer.add_scalar('params/c3/c3_mean', self.SDE.c3_mean.item(), i)
                    # writer.add_scalar('params/c3/c3_std', F.softplus(self.SDE.c3_std_o).item(), i)
                    # writer.add_scalar('params/theta/theta_1_mean', self.SDE.c1_mean.exp().item(), i)
                    # writer.add_scalar('params/theta/theta_2_mean', self.SDE.c2_mean.exp().item(), i)
                    # writer.add_scalar('params/theta/theta_3_mean', self.SDE.c3_mean.exp().item(), i)
            if i % 25 == 1:
                with torch.no_grad():
                    vi_paths, _, _ = self.SDE._diff_bridge(Test=True)
                    theta1, _, theta2, _, theta3, _ \
                        = self.SDE.sample_squeeze()
                    writer.add_histogram('vi/theta1', theta1, i)
                    writer.add_histogram('vi/theta2', theta2, i)
                    writer.add_histogram('vi/theta3', theta3, i)
                    x1_path = vi_paths[:, 0, :]
                    x2_path = vi_paths[:, 1, :]
                    for p_i in range(self.p):
                        plt.plot(x1_path[p_i, :], 'b', alpha=1/2, linewidth=0.5)
                        plt.plot(x2_path[p_i, :], 'g', alpha=1/2, linewidth=0.5)
                    plt.ylim(0, self.obs['obs'].max()+100)
                    plt.plot(self.time_index, self.obs['obs'][0, :], 'bx')
                    plt.plot(self.time_index, self.obs['obs'][1, :], 'gx')
                    plt.savefig(f"figs/test_{i:0>5d}.png")
                    plt.cla()


        writer.close()


if __name__ == "__main__":
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from lotka_volterra_data_augmentation import *

    lotka_volterra = Model(network_params=NETWORK_PARAMS, p=P,
                            dt=DT, obs=obs, params=params, priors=PRIORS, features=features)
    # desired number of iterations. currently no implementation of a
    # convergence criteria.
    lotka_volterra.train(5000, PATH_TO_TENSORBOARD_OUTPUT)
