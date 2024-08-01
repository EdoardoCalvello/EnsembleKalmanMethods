#!/opt/local/bin/python3 --

import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize

#########################################################
#########################################################
#############          1D Experiment        ############# 
#########################################################
#########################################################

EKI_transport = np.load('./results/EKI_transport0.00025_bayes_opt.npz')['Bayes_sol']
EKI_transport_one = np.load('./results/EKI_transport1_bayes_opt.npz')['Bayes_sol']
EKI_opt = np.load('./results/EKI_transport0.00025_bayes_opt.npz')['opt_sol']
EKI_post = np.load('./results/EKI_post0.00025.npz')['Bayes_sol']

w = 2
gamma = 1


def G(u):

    return (7/12)*u**3 - (7/2)*u**2 + 8*u


def prior(u):

    return (1/np.sqrt(np.pi))*np.exp(-(u+2)**2)

def likelihood(u):

    return (1/np.sqrt(2*np.pi*gamma))*np.exp(-0.5*(G(u) - w)**2/gamma)

def priorXlikelihood(u):

    return (1/(np.pi*(np.sqrt(2))))*np.exp(-(u+2)**2)*np.exp(-0.5*(G(u) - w)**2/gamma)

normalization_constant, error = quad(priorXlikelihood, -1, 1, epsabs=1e-5, epsrel=1e-5)

def posterior(u):

    return priorXlikelihood(u)/normalization_constant

def negative_posterior(u):

    return -posterior(u)

maximum = minimize(negative_posterior, 0.0, bounds=[(-1, 1)])
# Get the maximum value and the corresponding u value
MAP = maximum.x[0]
print(MAP)

def least_squares_func(u):

    return 0.5*(w - G(u))**2

least_squares_min = minimize(least_squares_func, 0.0, bounds=[(-1, 1)])
minimizer = least_squares_min.x[0]


x_min = -4
x_max = 1

x = np.linspace(x_min, x_max, num=1000)

fig, axs = plt.subplots(3, 1, figsize=(16, 15))

axs[0].plot(x, prior(x), 'k--', linewidth=4, label='Prior')
axs[0].plot(x, likelihood(x), 'k:', linewidth=4, label='Likelihood')
axs[0].plot(x, posterior(x), 'k-', linewidth=4, label='Posterior')
axs[0].legend(fontsize=20)
axs[0].set_xlabel('Position', fontsize=25)
axs[0].set_title('True Posterior', fontsize=30)
axs[0].set_xlim([x_min, x_max])
axs[0].tick_params(axis='both', which='major', labelsize=20)  # Adjust tick labels size


#axs[1].hist(EKI_transport[0,:], weights=np.zeros_like(EKI_transport[0,:]) + 1. / EKI_transport[0,:].size, bins=30, color='lightgreen', label='EKI (Transport)')
#axs[1].hist(EKI_opt[0,:], weights=np.zeros_like(EKI_opt[0,:]) + 1. / EKI_opt[0,:].size, bins=30, color='darkgreen', label='EKI (Iteration to Infinity)')
axs[1].set_ylim([0, 10])
axs[1].hist(EKI_transport_one[0,:], density=True, bins=50, color='gray', label=r'EKI (Transport), $\Delta t=1$')
axs[1].hist(EKI_transport[0,:], density=True, bins=30, color='green', label=r'EKI (Transport), $\Delta t=2.5\cdot 10^{-4}$')
axs[1].hist(EKI_opt[0,:], density=True, bins=30, color='lightgreen', label='EKI (Iteration to Infinity)')
axs[1].plot(x, posterior(x), 'k-', linewidth=4, label='Posterior')
axs[1].axvline(minimizer, color='black', linestyle='--', linewidth=2, label='Minimizer of Least Squares Loss')
axs[1].legend(fontsize=20)
axs[1].set_xlabel('Position', fontsize=25)
#axs[1].set_ylabel('Relative Frequency', fontsize=25)
axs[1].set_title('EKI Transport and Iteration to Infinity', fontsize=30)
axs[1].set_xlim([x_min, x_max])
axs[1].tick_params(axis='both', which='major', labelsize=20)  # Adjust tick labels size

#axs[2].hist(EKI_post[0,:], weights=np.zeros_like(EKI_post[0,:]) + 1. / EKI_post[0,:].size, bins=30, color='green', label='EKI (Posterior Inflation)')
axs[2].hist(EKI_post[0,:], density=True, bins=30, color='green', label='EKI (Posterior Inflation)')
axs[2].plot(x, posterior(x), 'k-', linewidth=4, label='Posterior')
axs[2].legend(fontsize=20)
axs[2].set_xlabel('Position', fontsize=25)
#axs[2].set_ylabel('Relative Frequency', fontsize=25)
axs[2].set_title('EKI Posterior Inflation', fontsize=30)
axs[2].set_xlim([x_min, x_max])
axs[2].tick_params(axis='both', which='major', labelsize=20)  # Adjust tick labels size

plt.tight_layout()
fig.savefig('./plots/EKI_1D.png')



'''
#########################################################
#########################################################
#############         L96 Experiment        ############# 
#########################################################
#########################################################

EKI_transport = np.load('./results/EKI_transport0.05_L96_bayes_opt.npz')['Bayes_sol']

import pdb; pdb.set_trace()
'''