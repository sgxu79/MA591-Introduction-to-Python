# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:23:36 2019

@author: 15306
"""

import numpy as np
from scipy.stats import norm
from rpy2.robjects.packages import importr, data
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri
from matplotlib import animation
pandas2ri.activate()

def asymm_laplace (y, mu = 0, sigma = 1, p = 0.5, log = False):
    dens = p * (1 - p) / sigma * np.exp(-(y - mu) * (p - (y < mu)) / sigma)
    if log:
        return(np.log(dens))
    else:
        return(dens)

def tanh (X, grad = False):
    
    if grad:
        return 1-np.tanh(X)**2
    else:
        return np.tanh(X)
        

def forward (X, weight, bias):
    """
    Forward propagation
    """
    out = X @ weight + bias 
    
    return out

def neural_net (X, W_1, b_1, W_2, b_2, W_3, b_3):
    """
    Three layer feedforward neural network
    """
    W_1 = np.reshape(W_1, (1,J))
    W_2 = np.reshape(W_2, (J,K), 'F')
    W_3 = np.reshape(W_3, (K,1))
    hidden_one = tanh(forward(X, W_1, b_1))
    hidden_two = tanh(forward(hidden_one, W_2, b_2))
    output = forward(hidden_two, W_3, b_3)
    return output

def nn_plot (param):
    W_1 = param[0:J]
    b_1 = param[J:(2*J)]
    W_2 = param[(2*J):(2*J+J*K)]
    b_2 = param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = param[2*J+J*K+2*K]
    nn = neural_net(X,W_1,b_1,W_2,b_2,W_3,b_3)
    q_s = (nn+1)*(max_y-min_y)/2+min_y
    plt.plot(X_s,y_s,'b.')
    plt.plot(X_s,q_s,'-')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.show()

def loglik (param, posterior = True):
    """
    Observed log-likelihood 
    """
    W_1 = param[0:J]
    b_1 = param[J:(2*J)]
    W_2 = param[(2*J):(2*J+J*K)]
    b_2 = param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = param[2*J+J*K+2*K]
    nn = neural_net(X, W_1, b_1, W_2, b_2, W_3, b_3)
    diff = y - nn
    if posterior:
        loglik = np.sum(asymm_laplace(diff,0,sigma,tau,log = True)) + \
        np.sum(norm.logpdf(W_1)) + np.sum(norm.logpdf(b_1)) + \
        np.sum(norm.logpdf(W_2)) + np.sum(norm.logpdf(b_2)) + \
        np.sum(norm.logpdf(W_3)) + np.sum(norm.logpdf(b_3)) 
    else: 
        loglik = np.sum(asymm_laplace(diff,0,sigma,tau,log = True))
    return loglik


def loglik_grad (param):
    """
    Empirical gradient
    """
    W_1 = param[0:J]
    b_1 = param[J:(2*J)]
    W_2 = param[(2*J):(2*J+J*K)]
    b_2 = param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = param[2*J+J*K+2*K]
    W_1 = np.reshape(W_1, (1,J))
    W_2 = np.reshape(W_2, (J,K), 'F')
    W_3 = np.reshape(W_3, (K,1))
    layer_one = forward(X, W_1, b_1)
    layer_two = forward(tanh(layer_one), W_2, b_2)
    nn = forward(tanh(layer_two), W_3, b_3)
    diff = y - nn
    grad_param = np.empty(len(param))
    grad_param[0:J] = 1/ sigma * np.sum((tau - (diff < 0)) * X * tanh(layer_two,True) @ (W_3 * np.transpose(W_2)) * tanh(layer_one,True),axis=0) - W_1
    grad_param[J:(2*J)] = 1 / sigma * np.sum((tau - (diff < 0)) * tanh(layer_two,True) @ (W_3 * np.transpose(W_2)) * tanh(layer_one,True),axis=0) - b_1
    grad_param[(2*J):(2*J+J*K)] = (1 / sigma * np.transpose(tanh(layer_one)) @ ((tau-(diff<0)) * tanh(layer_two,True) * np.transpose(W_3)) - W_2).flatten(order='F')
    grad_param[(2*J+J*K):(2*J+J*K+K)] = 1 / sigma * np.reshape(W_3 * np.transpose(tanh(layer_two,True)) @ (tau - (diff < 0)),(K,)) - b_2
    grad_param[(2*J+J*K+K):(2*J+J*K+2*K)] = 1 / sigma * np.sum((tau - (diff < 0))*tanh(layer_two),axis=0) - np.transpose(W_3)
    grad_param[2*J+J*K+2*K] = 1 / sigma * np.sum(tau - (diff < 0)) - b_3
    
    return grad_param
    
    
def find_start_ep (param):
    """
    Find a good starting value for the step size
    """
    q = param
    epsilon = epsilon_ = 0.01
    a_min = 0.25
    a_cross = 0.5
    a_max=  0.75
    d = 2.
    p = np.random.normal(size = len(q))
    current_E = loglik(q) - np.sum(p ** 2) / 2
    p = p + epsilon * loglik_grad(q) / 2
    q = q + epsilon * p
    p = p + epsilon * loglik_grad(q) / 2   
    proposed_E = loglik(q) - np.sum(p ** 2) / 2
    diff_E = proposed_E - current_E
    direction = 2 * (diff_E > np.log(a_cross)) - 1
    
    while direction*diff_E > direction * np.log(a_cross):
        epsilon = epsilon_
        epsilon_ = d ** direction * epsilon
        current_E = loglik(q) - np.sum(p ** 2) / 2
        p = p + epsilon_ * loglik_grad(q) / 2
        q = q + epsilon_ * p
        p = p + epsilon_ * loglik_grad(q) / 2   
        proposed_E = loglik(q) - np.sum(p ** 2) / 2     
        diff_E = proposed_E - current_E
        
    ep = np.sort((epsilon, epsilon_))
    epsilon, epsilon_ = ep
    counter = 0
    
    while ((diff_E > np.log(a_max)) | (diff_E < np.log(a_min))) & (counter < 100):
        
        epsilon_m = (epsilon + epsilon_) / 2
        current_E = loglik(q) - np.sum(p ** 2) / 2
        p = p + epsilon * loglik_grad(q) / 2
        q = q + epsilon * p
        p = p + epsilon * loglik_grad(q) / 2   
        proposed_E = loglik(q) - np.sum(p ** 2) / 2     
        diff_E = proposed_E - current_E
        
        if np.abs(diff_E) >= 1e5:
            epsilon = ep[0]
            break
        if diff_E > np.log(a_max):
            epsilon = epsilon_m
        elif diff_E < np.log(a_min):
            epsilon_ = epsilon_m
        else:
            epsilon = epsilon_m
            break
        counter += 1
    return epsilon
    
def hmc_samp (n_iter, n_adapt, inits, lam, delta):
    """
    Hamiltonian Monte Carlo sampler with leap frog integrator
    """
    print_step = np.int(n_iter/10)
    samps = np.empty((n_iter, len(inits)))
    log_ep = np.empty((n_iter,))
    lp_ = np.empty((n_iter,))
    log_ep_ = np.empty((n_adapt,))
    samps[0,] = inits
    lp_[0] = loglik(inits)
    log_ep[0] = np.log(find_start_ep(inits))
    mu = np.log(10*np.exp(log_ep[0]))
    H = 0
    gamma = 0.05
    t = 10
    kappa = 0.75
    for it in range(1,n_iter):
        current_q = samps[it-1,]
        epsilon = np.exp(log_ep[it-1])
        q = current_q
        p = np.random.normal(size = len(q))
        current_p = p
        n_step = np.minimum(500,np.int(np.maximum(1,np.round(lam/epsilon))))
        for l in range(0,n_step):
            p = p + epsilon * loglik_grad(q) / 2
            q = q + epsilon * p
            p = p + epsilon * loglik_grad(q) / 2
        p = -p
        current_E = loglik(current_q) - np.sum(current_p ** 2) / 2
        proposed_E = loglik(q) - np.sum(p ** 2) / 2
        diff_K = proposed_E - current_E
        alpha = np.exp(min(0,diff_K))
        if np.log(np.random.uniform()) < diff_K:
            current_q = q
        lp_[it] = loglik(current_q, False)
        samps[it,] = current_q
        if it <= n_adapt-1:
            H = (1 - 1 / (it + t)) * H + 1 / (it + t) * (delta - alpha)
            log_ep[it] = mu - np.sqrt(it) / gamma*H
            log_ep_[it] = it ** (-kappa) * log_ep[it] + (1 - it ** (-kappa)) * log_ep_[it-1]
        else:
            log_ep[it] = log_ep_[n_adapt-1]
        if (it + 1) % print_step == 0:
            print(it+1,"iterations completed.")
        if (it + 1) == n_iter:
            print("HMC sampling finished.")
    return samps, log_ep, lp_

def create_inits ():
    """
    Generate initial values
    """
    W_1 = np.random.normal(size = J)
    b_1 = np.random.normal(size = J)
    W_2 = np.random.normal(size = J*K)
    b_2 = np.random.normal(size = K)
    W_3 = np.random.normal(size = K)
    b_3 = np.random.normal(size = 1)
    inits = np.concatenate((W_1,b_1,W_2,b_2, W_3, b_3))
    inits = inits + np.random.uniform(-1,1,len(inits))
    return inits


X = np.sort(np.random.uniform(-1,1,300))
tau_x = np.random.uniform(size = 300)
y = 0.5*norm.ppf(tau_x)+5*(tau_x-0.5)*X**2
plt.plot(X,y,'.')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data from simulation')
for i in np.linspace(0.05,0.95,19):
    q = 0.5*norm.ppf(i)+5*(i-0.5)*X**2
    plt.plot(X,q,'r-')
X = np.reshape(X,(len(X),1))
y = np.reshape(y,(len(y),1))

max_X = np.max(X)
min_X = np.min(X)
max_y = np.max(y)
min_y = np.min(y)

X_s = X
y_s = y

X = (X - min_X)/(max_X - min_X)
y = -1 + (y - min_y)*2/(max_y - min_y)


sigma = 0.1

tau = 0.9

J = 10

K = 10

def init():
    plt.plot(X_s, y_s, 'k.')
    line.set_data([], [])
    return line,

def animate(i):
    plt_param = samps[i,]
    W_1 = plt_param[0:J]
    b_1 = plt_param[J:(2*J)]
    W_2 = plt_param[(2*J):(2*J+J*K)]
    b_2 = plt_param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = plt_param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = plt_param[2*J+J*K+2*K]
    nn = neural_net(X,W_1,b_1,W_2,b_2,W_3,b_3)
    q_plt = (nn+1)*(max_y-min_y)/2+min_y
    line.set_data(X_s, q_plt)
    return line,

fig = plt.figure()
ax = plt.axes(xlim=(min_X, max_X), ylim=(min_y, max_y))
line, = ax.plot([], [])
plt.xlabel('X')
plt.ylabel('y')

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=100, blit=True)

q_true = 0.5*norm.ppf(0.9)+5*(0.9-0.5)*X_s**2
plt.plot(X_s,q_true,'r-')
anim.save('591Sim.mp4')

samps, log_ep, lp_ = hmc_samp(200,100,create_inits(),2,0.65)

est_samp = samps[100:200,]
est_q = np.empty((100,len(X)))
for i in range(0,100):
    cur_param = est_samp[i,]
    W_1 = cur_param[0:J]
    b_1 = cur_param[J:(2*J)]
    W_2 = cur_param[(2*J):(2*J+J*K)]
    b_2 = cur_param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = cur_param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = cur_param[2*J+J*K+2*K]
    nn = neural_net(X,W_1,b_1,W_2,b_2,W_3,b_3)
    est_q[i,] = np.reshape((nn+1)*(max_y-min_y)/2+min_y,(len(X),))

est_q = np.mean(est_q, axis = 0)

plt.plot(X_s,y_s,'k.')
plt.plot(X_s,q_true,'r-',label='True')
plt.plot(X_s,est_q,'b-',label='Estimate')
plt.legend()

utils = importr("utils")
utils.install_packages("MASS")
MASS = importr("MASS")
motor = data(MASS).fetch('mcycle')['mcycle']
motor = pandas2ri.ri2py(motor)

plt.plot('times', 'accel', '.', data = motor)
plt.xlabel('Time')
plt.ylabel('Acceleration')

X = np.array(motor['times'])
X = np.reshape(X,(len(X),1))
y = np.array(motor['accel'])
y = np.reshape(y,(len(y),1))

max_X = np.max(X)
min_X = np.min(X)
max_y = np.max(y)
min_y = np.min(y)

X_s = X
y_s = y

X = (X - min_X)/(max_X - min_X)
y = -1 + (y - min_y)*2/(max_y - min_y)

sigma = 0.06

tau = 0.3

fig = plt.figure()
ax = plt.axes(xlim=(min_X, max_X), ylim=(min_y, max_y))
line, = ax.plot([], [])
plt.xlabel('X')
plt.ylabel('y')

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=100, blit=True)


anim.save('591App.mp4')

samps, log_ep, lp_ = hmc_samp(200,100,create_inits(),2,0.65)

est_samp = samps[100:200,]
est_q = np.empty((100,len(X)))
for i in range(0,100):
    cur_param = est_samp[i,]
    W_1 = cur_param[0:J]
    b_1 = cur_param[J:(2*J)]
    W_2 = cur_param[(2*J):(2*J+J*K)]
    b_2 = cur_param[(2*J+J*K):(2*J+J*K+K)]
    W_3 = cur_param[(2*J+J*K+K):(2*J+J*K+2*K)]
    b_3 = cur_param[2*J+J*K+2*K]
    nn = neural_net(X,W_1,b_1,W_2,b_2,W_3,b_3)
    est_q[i,] = np.reshape((nn+1)*(max_y-min_y)/2+min_y,(len(X),))

est_q = np.mean(est_q, axis = 0)

plt.plot(X_s,y_s,'k.')
plt.plot(X_s,est_q,'b-',label='Estimate')
plt.legend()