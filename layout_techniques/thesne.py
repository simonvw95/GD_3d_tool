# Copyright (c) 2016 Paulo Eduardo Rauber

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#                 ^
#                / \
#                 |
#                 |
#
# License included because this module is a heavily modified version based on
# Paulo Rauber's implementation of dynamic t-SNE.
# (https://github.com/paulorauber/thesne)

import networkx as nx
import numpy as np
import torch
from sklearn.utils import check_random_state
import copy

epsilon = 1e-16
floath = np.float32
dtype = torch.float32


class SigmaTooLowException(Exception):
    pass


class NaNException(Exception):
    pass


def tsnet(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=10, final_lr=4, lr_switch=None, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250,
         initial_l_kl=None, final_l_kl=None, l_kl_switch=None,
         initial_l_c=None, final_l_c=None, l_c_switch=None,
         initial_l_r=None, final_l_r=None, l_r_switch=None,
         r_eps=1, random_state=None,
         autostop=False, window_size=10, verbose=1):
    random_state = check_random_state(random_state)
    N = X.shape[0]
    X_shared = torch.tensor(np.asarray(X),dtype=dtype)
    sigma_shared = torch.tensor(np.ones(N),dtype=dtype)
    if Y is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
    Y_shared = torch.tensor(np.asarray(Y),dtype=dtype,requires_grad=True)


    sigma_shared = find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose)


    Y = find_Y(
        X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
        initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
        final_momentum, momentum_switch,
        initial_l_kl, final_l_kl, l_kl_switch,
        initial_l_c, final_l_c, l_c_switch,
        initial_l_r, final_l_r, l_r_switch,
        r_eps, autostop, window_size, verbose
    )
    return Y

def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, verbose=0):
    target = np.log(perplexity)
    sigmin_shared = torch.tensor(np.full(N,np.sqrt(epsilon),dtype=floath))
    sigmax_shared = torch.tensor(np.full(N,np.inf,dtype=floath))

    for i in range(sigma_iters):
        e,sigmin_shared,sigmax_shared = update_intervals(X_shared,sigma_shared,sigmin_shared,sigmax_shared,target)
        sigma_shared = update_sigma(sigma_shared,sigmax_shared,sigmin_shared)
        if verbose:
            print('Finding sigmas... Iteration {0}/{1}: Perplexities in [{2:.4f}, {3:.4f}].'.format(i + 1, sigma_iters,
                                                                                                    np.exp(e.min()),                                                                                  np.exp(e.max())),
                  end='\r')

        if np.any(np.isnan(np.exp(e.numpy()))):
            raise SigmaTooLowException('Invalid sigmas. The perplexity is probably too low.')

    if verbose:
        print('\nDone. Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()), np.exp(e.max())))
    return sigma_shared


def update_sigma(sigma, sigmax,sigmin):

    upsigma = torch.where(torch.isinf(sigmax),sigma*2,(sigmin+sigmax)/2.)
    return upsigma

def update_intervals(X_shared,sigma_shared,sigmin,sigmax,target):
    P = torch.maximum(p_ij_conditional_var(X_shared,sigma_shared),torch.tensor(epsilon))
    entropy = -torch.sum(P*torch.log(P),axis=1)
    upmin = torch.where(torch.lt(entropy,target),sigma_shared,sigmin)
    upmax = torch.where(torch.gt(entropy,target),sigma_shared,sigmax)
    return entropy, upmin,upmax

def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch,
           initial_l_kl, final_l_kl, l_kl_switch,
           initial_l_c, final_l_c, l_c_switch,
           initial_l_r, final_l_r, l_r_switch,
           r_eps, autostop=False, window_size=10, verbose=0):
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr_shared = initial_lr
    momentum_shared = initial_momentum

    initial_l_kl = np.array(initial_l_kl, dtype=floath)
    final_l_kl = np.array(final_l_kl, dtype=floath)
    initial_l_c = np.array(initial_l_c, dtype=floath)
    final_l_c = np.array(final_l_c, dtype=floath)
    initial_l_r = np.array(initial_l_r, dtype=floath)
    final_l_r = np.array(final_l_r, dtype=floath)

    l_kl_shared = initial_l_kl
    l_c_shared = initial_l_c
    l_r_shared = initial_l_r

    Yv_shared = torch.tensor(np.zeros((N,output_dims),dtype=floath))
    stepsize_over_time = np.zeros(n_epochs)

    def is_converged(epoch, stepsize_over_time, tol=1e-8):
        if epoch > window_size:
            max_stepsize = stepsize_over_time[epoch - window_size:epoch].max()
            return max_stepsize < tol
        return False

    converged = False
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared = final_lr
        if epoch == momentum_switch:
            momentum_shared = final_momentum
        if epoch == l_kl_switch:
            l_kl_shared = final_l_kl
        if epoch == l_c_switch:
            l_c_shared = final_l_c
        if epoch == l_r_switch:
            s1 = epoch
            l_r_shared = final_l_r

        dY_norm,Yv_shared = update_Yv(Yv_shared,N,lr_shared,Y_shared,momentum_shared,X_shared,sigma_shared,l_kl_shared,l_c_shared,l_r_shared,r_eps)
        stepsize_over_time[epoch] = dY_norm

        #Y_shared += Yv_shared
        Y_shared2 = Y_shared + Yv_shared
        Y_shared = Y_shared2

        c = torch.sum(cost_var(X_shared,Y_shared,sigma_shared,l_kl_shared,l_c_shared,l_r_shared,r_eps))
        if np.isnan(float(c)):
            raise NaNException('Encountered NaN for cost.')

        if verbose:
            if autostop and epoch >= window_size:
                dlast_period = stepsize_over_time[epoch - window_size:epoch]
                max_stepsize = dlast_period.max()
                print('Epoch: {0}. Cost: {1:.6f}. Max step size of last {2}: {3:.2e}'.format(epoch + 1, float(c), window_size, max_stepsize), end='\r')
            else:
                print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)), end='\r')

        # Switch phases if we're converged. Or exit if we're already in the last phase.
        if autostop and is_converged(epoch, stepsize_over_time, tol=autostop):
            if epoch < lr_switch:
                lr_switch = epoch + 1
                momentum_switch = epoch + 1
                l_kl_switch = epoch + 1
                l_c_switch = epoch + 1
                l_r_switch = epoch + 1
                #print('\nAuto-switching at epoch {0}'.format(epoch))
            elif epoch > lr_switch + window_size:
                #print('\nAuto-stopping at epoch {0}'.format(epoch))
                converged = True
                break

    if not converged:
        print('\nWarning: Did not converge!')
    return np.array(Y_shared.detach().numpy())


def update_Yv(Yv,N,lr,Y,momentum,X,sigma,l_kl,l_c,l_r,r_eps):
    step_size = torch.sum(torch.sum(Yv ** 2, axis=1) ** 0.5) / (
                N * lr * torch.max(torch.max(Y, axis=0)[0] - torch.min(Y, axis=0)[0]))

    costs = cost_var(X, Y, sigma, l_kl, l_c, l_r, r_eps)
    cost = torch.sum(costs)

    grad_y = torch.autograd.grad(cost,Y)[0]

    grad_y = torch.nan_to_num(grad_y)

    momentum = torch.tensor(momentum,dtype=dtype)
    lr = torch.tensor(lr,dtype=dtype)


    Yv = momentum * Yv - lr * grad_y
    return step_size, Yv

def cost_var(X, Y, sigma, l_kl, l_c, l_r, r_eps):
    N= X.shape[0]
    l_sum = l_kl + l_c + l_r


    p_ij_conditional = p_ij_conditional_var(X,sigma)
    p_ij = p_ij_sym_var(p_ij_conditional)
    q_ij = q_ij_student_t_var(Y)

    p_ij_safe = torch.maximum(p_ij,torch.tensor(epsilon))
    q_ij_safe = torch.maximum(q_ij,torch.tensor(epsilon))


    kl = torch.sum(p_ij*torch.log(p_ij_safe/q_ij_safe),axis=1)
    compression = (1/(2*N))*torch.sum(Y**2,axis=1)
    repulsion = -(1/(2*N**2)) * torch.sum(torch.log(euclidian_var(Y)+r_eps).fill_diagonal_(0),axis=1)
    cost = (l_kl/l_sum) * kl + (l_c/l_sum)*compression + (l_r/l_sum) * repulsion
    return cost

def p_ij_sym_var(p_ij_conditional):
    return (p_ij_conditional + p_ij_conditional.T) / (2 * p_ij_conditional.shape[0])

def q_ij_student_t_var(Y):
    sqdistance = sqeuclidian_var(Y)
    one_over = (1/(sqdistance+1)).fill_diagonal_(0)
    return one_over /one_over.sum()

def sqeuclidian_var(X):
    N = X.shape[0]
    ss = (X**2).sum(axis=1)
    value = ss.reshape((N,1)) + ss.reshape((1,N))-2*X.matmul(X.T)
    return value

def euclidian_var(X):
    return torch.maximum(sqeuclidian_var(X),torch.tensor(epsilon))**0.5

def p_ij_conditional_var(X,sigma):
    N= X.shape[0]
    sqdistance = X**2
    esqdistance = torch.exp(-sqdistance / ((2*(sigma**2)).reshape((N,1))))
    esqdistance_zd = esqdistance.fill_diagonal_(0)
    row_sum = torch.sum(esqdistance_zd,axis=1).reshape((N,1))
    return esqdistance_zd/row_sum
