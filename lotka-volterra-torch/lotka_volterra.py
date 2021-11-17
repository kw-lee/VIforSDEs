import os
import numpy as np
import torch
import pyro
import torch.nn.functional as F
import torch.distributions as D
import math
import torch.nn as nn

DTYPE = torch.float32
NP_DTYPE = np.float32
ALMOSTZERO = 1e-4

class Layer(nn.Module):
    def __init__(self, input_dim, output_dim, p, name=None):
        super(Layer, self).__init__()
        self.name = name
        self.register_parameter(name='w', param=nn.Parameter(torch.empty(1, input_dim, output_dim), requires_grad=True))
        self.register_parameter(name='b', param=nn.Parameter(torch.full([1, 1, output_dim], 0.1), requires_grad=True))
        self.p = p
        
        with torch.no_grad():
            nn.init.trunc_normal_(self.w, mean=0, std=0.1, a=-0.2, b=0.2)

    def forward(self, x):
        W = torch.tile(self.w, [self.p, 1, 1])
        B = torch.tile(self.b, [self.p, 1, 1])
        y = torch.matmul(x, W) + B
        return F.relu(y)


# the rnn cell called by diff_bridge
class RNNCell(nn.Module): 
    def __init__(self, network_params, p, no_input):
        super(RNNCell, self).__init__()

        model = []
        self.no_input = no_input
        # model.append(Layer(self.no_input, network_params['hidden_layer_width'], p, f'hidden_layer_0'))
        model.append(nn.Linear(self.no_input, network_params['hidden_layer_width']))
        model.append(nn.ReLU())
        for i in range(1, network_params['num_hidden_layers']):
            # model.append(Layer(network_params['hidden_layer_width'], 
            #     network_params['hidden_layer_width'], p, f'hidden_layer_{i}'))
            model.append(nn.Linear(network_params['hidden_layer_width'], network_params['hidden_layer_width']))
            model.append(nn.ReLU())
        # model.append(Layer(network_params['hidden_layer_width'], 5, p, 'output_layer'))
        model.append(nn.Linear(network_params['hidden_layer_width'], 5))

        self.network_params = network_params
        self.model = nn.Sequential(*model)
        self.p = p
    
    def forward(self, inp, eps_identity=ALMOSTZERO):
        '''
        rnn cell for supplying Gaussian state transitions
        :param eps: eps * identity added to diffusion matrix to control numerical stability
        '''
        output = self.model(inp)

        mu, sigma_11, sigma_21, sigma_22 = torch.split(output, [2, 1, 1, 1], 2)

        # reshaping sigma matrix to lower-triangular cholesky factor
        zeros = torch.zeros_like(sigma_11)
        sigma_11 = F.softplus(sigma_11)
        sigma_22 = F.softplus(sigma_22)
        sigma_chol = torch.concat(
                [torch.concat([sigma_11, zeros], 2),
                 torch.concat([sigma_21, sigma_22], 2)], 1
            )
        # sigma = torch.linalg.cholesky(torch.matmul(sigma_chol, sigma_chol.permute([0, 2, 1])) + eps_identity * torch.tile(
            # torch.eye(2).unsqueeze(0), [self.p, 1, 1]))

        return mu, sigma_chol

class DiffBridge(nn.Module):
    def __init__(self, network_params, p, dt, obs, features):
        super(DiffBridge, self).__init__()
        
        self.network_params = network_params
        self.p = p
        self.obs = obs
        self.features = features
        self.dt = dt
        self.no_input = self.features['feature_init'].shape[2] + 2
        self._rnn_cell = RNNCell(self.network_params, self.p, self.no_input)
        self.len_time = int(self.obs['T'] / self.dt)
        path_dist = D.MultivariateNormal(
            loc=torch.zeros(self.p * self.len_time, 2), 
            scale_tril=torch.eye(2).unsqueeze(0).tile([self.p * self.len_time, 1, 1])
        )
        self.path_seed = path_dist.sample().unsqueeze(2).reshape(self.len_time, self.p, 2, 1)
    
    # functions to return p simulations of a diffison bridge
    def _path_sampler(self, inp, mu_nn, sigma_nn, i, Test=False):
        '''
        sample new state using learned Gaussian state transitions
        :param inp: current state of system (p, 2)
        :param mu_nn: drift vector from RNN (p, 2)
        :param sigma_nn: diffusion matrix from RNN as cholesky factor (p, 2, 2)

        return out (p, 1, 2)
        '''
        if Test:
            out_dist = D.TransformedDistribution(
                        base_distribution=D.MultivariateNormal(
                        loc=inp + self.dt * mu_nn,
                        scale_tril=math.sqrt(self.dt) * sigma_nn),
                transforms=pyro.distributions.transforms.SoftplusTransform())
            out = out_dist.rsample().unsqueeze(1)
        else:
            out = inp + self.dt * mu_nn + \
                math.sqrt(self.dt) * torch.matmul(sigma_nn, self.path_seed[i, ...]).squeeze()
            out = F.softplus(out).unsqueeze(1)
        return out

    def forward(self, Test=False):
        inp = torch.concat([self.obs['obs_init'], self.features['feature_init']], 2)
        pred_mu, pred_sigma = self._rnn_cell(inp)

        mu_store = [pred_mu]
        sigma_store = [pred_sigma.unsqueeze(1)]

        output = self._path_sampler(inp[:, 0, 0:2], pred_mu.squeeze(), pred_sigma, 0, Test)
        path_store = [inp[:, :, :2]]
        path_store.append(output)

        for i in range(self.len_time - 1):
            x1_next_vec = torch.full([self.p, 1, 1], self.features['x1_store'][i])
            x2_next_vec = torch.full([self.p, 1, 1], self.features['x2_store'][i])

            inp = torch.concat([
                    output, # (50, 1, 2)
                    torch.tile(torch.concat(
                        [(inp[0, 0, 2] + self.dt).reshape(1), 
                         self.features['tn_store'][i].reshape(1), 
                         self.features['x1_store'][i].reshape(1), 
                         self.features['x2_store'][i].reshape(1)], 
                    ).reshape(1, 1, 4), [self.p, 1, 1]), # (50, 1, 4)
                    torch.concat([x1_next_vec, x2_next_vec], 2) # (50, 1, 2)
                ], 2
            )

            pred_mu, pred_sigma = self._rnn_cell(inp)
            mu_store.append(pred_mu)
            sigma_store.append(pred_sigma.unsqueeze(1))
            output = self._path_sampler(inp[:, 0, 0:2], pred_mu.squeeze(), pred_sigma, i+1, Test)
            path_store.append(output)
        
        path_store = torch.concat(path_store, 1)
        mu_store = torch.concat(mu_store, 1)
        sigma_store = torch.concat(sigma_store, 1)
        
        return path_store.permute([0, 2, 1]), mu_store.reshape(-1, 2), sigma_store.reshape(-1, 2, 2)

class LotkaVolterra(nn.Module):
    def __init__(self, network_params, p, dt, obs, params, priors, features):
        super(LotkaVolterra, self).__init__()

        self.network_params = network_params
        self.p = p
        self.obs = obs

        pt_features = {}
        for key in features.keys():
            pt_features[key] = torch.from_numpy(features[key])
        self.features = pt_features
        self.obs['obs'] = torch.from_numpy(self.obs['obs'])
        self.dt = dt
        self.time_index = torch.from_numpy(np.int32(self.obs['times'] / self.dt))
        self.dim2 = int(self.obs['T'] / self.dt)

        # diff bridge
        self._diff_bridge = DiffBridge(self.network_params, self.p, self.dt, self.obs, self.features)
        
        # variational parameters
        self.register_parameter(name='c1_mean', param=nn.Parameter(torch.tensor(params['c1_mean']).reshape(1)))
        self.register_parameter(name='c1_std_o', param=nn.Parameter(params['c1_std'].clone().detach()))

        self.register_parameter(name='c2_mean', param=nn.Parameter(torch.tensor(params['c2_mean']).reshape(1)))
        self.register_parameter(name='c2_std_o', param=nn.Parameter(params['c2_std'].clone().detach()))

        self.register_parameter(name='c3_mean', param=nn.Parameter(torch.tensor(params['c3_mean']).reshape(1)))
        self.register_parameter(name='c3_std_o', param=nn.Parameter(params['c3_std'].clone().detach()))
        
        c_seed_dist = D.Normal(loc=0.0, scale=1.0)
        self.c1_seed = c_seed_dist.sample([p, 1])
        self.c2_seed = c_seed_dist.sample([p, 1])
        self.c3_seed = c_seed_dist.sample([p, 1])
        # prior distribution
        self.prior_dist = D.MultivariateNormal(
            loc=torch.tensor([priors['c1_mean'], priors['c2_mean'], priors['c3_mean']]),
            scale_tril=torch.diag(torch.tensor([priors['c1_std'], priors['c2_std'], priors['c3_std']]))
        )

    def get_params(self):
        """get parameters
        """
        pass

    # functions for mu (alpha) and sigma (beta) - LV
    def alpha(self, x1, x2, c1_strech, c2_strech, c3_strech):
        '''
        returns drift vector for approx p(x)
        '''
        a = torch.concat([c1_strech * x1 - c2_strech * x1 * x2,
                          c2_strech * x1 * x2 - c3_strech * x2], 1)
        return a
    
    def beta(self, x1, x2, c1_strech, c2_strech, c3_strech):
        '''
        returns diffusion matrix for approx p(x)
        '''
        a = torch.sqrt(c1_strech * x1 + c2_strech * x1 * x2).unsqueeze(1)
        b = (-c2_strech * x1 * x2).unsqueeze(1) / a
        c = torch.sqrt((c3_strech * x2 + c2_strech * x1 * x2).unsqueeze(1) - torch.square(b))
        # a += ALMOSTZERO
        # c += ALMOSTZERO
        zeros = torch.zeros_like(a)
        beta_chol = torch.concat([torch.concat([a, zeros], 2), torch.concat([b, c], 2)], 1)
        return beta_chol

    def sample_squeeze(self):
        '''
        reshape param sample for use in ELBO
        '''
        theta1 = (F.softplus(self.c1_std_o) * self.c1_seed + self.c1_mean).exp()
        theta2 = (F.softplus(self.c2_std_o) * self.c2_seed + self.c2_mean).exp()
        theta3 = (F.softplus(self.c3_std_o) * self.c3_seed + self.c3_mean).exp()
        return theta1, torch.tile(theta1, [1, self.dim2]).reshape(-1, 1), \
               theta2, torch.tile(theta2, [1, self.dim2]).reshape(-1, 1), \
               theta3, torch.tile(theta3, [1, self.dim2]).reshape(-1, 1)

    def forward(self, inp=None):
        return self.ELBO()

    # ELBO loss function
    def ELBO(self):
        '''
        calculate ELBO under SDE model
        :param obs: observations of SDE
        :param vi_paths: diffusion paths produced by generative VI approx
        :param vi_mu: drift vectors produced by generative VI approx
        :param vi_sigma: diffusion matrices produced by generative VI approx
        :param params: current params of model
        :param p: number of samples used for monte-carlo estimate
        :param dt: discretization used
        '''
        
        vi_paths, vi_mu, vi_sigma = self._diff_bridge(Test=True)
        theta1, theta1_strech, theta2, theta2_strech, theta3, theta3_strech \
            = self.sample_squeeze()
        # observations
        obs_logprob_store = []

        for i in range(len(self.time_index)):
            obs_dist = D.MultivariateNormal(vi_paths[:, :, self.time_index[i]], scale_tril=(torch.eye(2) * math.sqrt(self.obs['tau'])).unsqueeze(0).tile([self.p, 1, 1]))
            obs_loglik = obs_dist.log_prob(torch.tile(self.obs['obs'][:, i].unsqueeze(0), [self.p, 1]))
            obs_logprob_store.append(obs_loglik.unsqueeze(1))

        obs_logprob_store = torch.concat(obs_logprob_store, 1)
        obs_logprob = torch.sum(obs_logprob_store, 1)

        x1_path = vi_paths[:, 0, :] # (p, times)
        x2_path = vi_paths[:, 1, :] # (p, times)

        x_path_diff = vi_paths[:, :, 1:] - vi_paths[:, :, :-1] # (p, 2, times-1)
        # (p * (times-1), 2)
        x_diff = torch.concat([x_path_diff[:, 0, :].reshape(-1, 1),
                               x_path_diff[:, 1, :].reshape(-1, 1)], 1)

        # (p * (times-1), 2)
        x_path_mean = torch.concat(
            [vi_paths[:, 0, :-1].reshape(-1, 1), vi_paths[:, 1, :-1].reshape(-1, 1)], 1)
        # (p * (times-1), 2)
        x_path_eval = torch.concat(
            [vi_paths[:, 0, 1:].reshape(-1, 1), vi_paths[:, 1, 1:].reshape(-1, 1)], 1)

        # (p * (times - 1))
        x1_head = x1_path[:, :-1].reshape(-1, 1)
        x2_head = x2_path[:, :-1].reshape(-1, 1)

        # (p * (times-1), 2)
        gen_path_dist = D.TransformedDistribution(
                base_distribution=D.MultivariateNormal(
                    loc=x_path_mean + self.dt * vi_mu,
                    scale_tril=math.sqrt(self.dt) * vi_sigma),
                transforms=pyro.distributions.transforms.SoftplusTransform()
        )

        # alpha: (p * (times - 1), 2)
        alpha_eval = self.alpha(x1_head, x2_head, theta1_strech, theta2_strech, theta3_strech)
        # beta: (p * (times - 1), 2, 2)
        beta_eval = self.beta(x1_head, x2_head, theta1_strech, theta2_strech, theta3_strech)

        # (p * (times-1), 2)
        sde_path_dist = D.MultivariateNormal(
                loc=self.dt * alpha_eval,
                scale_tril=math.sqrt(self.dt) * beta_eval)
        
        # (p * (times - 1))
        gen_logprob = gen_path_dist.log_prob(x_path_eval)
        # gen_logprob = gen_path_dist.log_prob(x_diff)
        
        # (p * (times-1))
        sde_logprob = sde_path_dist.log_prob(x_diff)

        sum_eval = (gen_logprob - sde_logprob).reshape(self.p, -1).sum(1)
        # sum_eval = -sde_logprob.reshape(self.p, -1).sum(1)

        c_cat = torch.log(torch.concat([theta1, theta2, theta3], 1))

        gen_dist = D.MultivariateNormal(
                loc=torch.concat([self.c1_mean, self.c2_mean, self.c3_mean], axis=0),
                scale_tril=torch.diag(torch.concat([F.softplus(self.c1_std_o), F.softplus(self.c2_std_o), F.softplus(self.c3_std_o)], axis=0))
        )

        prior_loglik = self.prior_dist.log_prob(c_cat.squeeze())
        gen_loglik = gen_dist.log_prob(c_cat.squeeze())

        kl = gen_loglik - prior_loglik

        mean_kl = kl.mean(0)
        mean_n_obs_logprob = -obs_logprob.mean(0)
        mean_sum_eval = sum_eval.mean(0)
        # mean_sum_eval = 0.0
        mean_loss = mean_sum_eval + mean_n_obs_logprob + mean_kl

        return mean_loss, mean_kl, mean_n_obs_logprob