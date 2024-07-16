#!/opt/local/bin/python3 --

import sys # for exit
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
from L96_multiscale import L96M
from DATools_ import DATools
from time import perf_counter as timer
from joblib import dump, load

l96m = L96M()
#l96m = L96M(J=64)

################################################################################
# constants section ############################################################
################################################################################
EPS = 1e-14
PRINTLINE = '-'*42
DEBUG = False

T_conv = 50 # converging integration time
T_learn = 300 # time to gather training data for GP

T_dyn = 50
TAU = 0.001

gamma_small = 0.001
sigma_small = 0.001

gamma_large = 0.1
sigma_large = 0.1

dt = 0.001 # maximum step size
dt_conv = 0.01 # maximum step size for converging to attractor

################################################################################
# IC section ###################################################################
################################################################################
z0 = np.empty(l96m.K + l96m.K * l96m.J)

z0[:l96m.K] = np.random.rand(l96m.K) * 15 - 5
for k_ in range(0,l96m.K):
  z0[l96m.K + k_*l96m.J : l96m.K + (k_+1)*l96m.J] = z0[k_]


################################################################################
# main section #################################################################
################################################################################


# full L96M integration (converging to attractor)
sol_conv = scint.solve_ivp(
    l96m,
    [0,T_conv],
    z0,
    method = 'RK45',
    max_step = dt_conv)
z0_conv = sol_conv.y[:,-1]
print("Number of steps (converging):", len(sol_conv.t), flush=True)
del sol_conv

# full L96M integration (for learning)
sol_learn = scint.solve_ivp(
    l96m,
    [0,T_learn],
    z0_conv,
    method = 'RK45',
    max_step = dt)
print("Number of steps (full, learn):", len(sol_learn.t), flush=True)
#initial condition for dynamics
z0 = sol_learn.y[:,-1]

# scatter plots for full L96M integration
da_tools = DATools(mem_thrsh = 800)
l96m.set_stencil()
da_tools.set_pairs(l96m.gather_pairs(sol_learn.y))
#GET m
da_tools.learn_gpr()

#set m
da_tools = DATools(mem_thrsh = 800)
gp = load('closure.joblib')
da_tools.set_gpr(gp)
l96m.set_stencil()
l96m.set_predictor(da_tools.get_gpr())


################################################################################
# Run Single Scale Dynamics with gamma/sigma small##############################
################################################################################

states = np.zeros((l96m.K, np.arange(0,T_dyn,TAU).shape[0]))
its = states.shape[1]
obs = np.zeros((6, np.arange(0,T_dyn,TAU).shape[0]))
states[:,0] = z0[:l96m.K]

#First generate the true states and the true data
for n in range(its-1):

    Psi_state = scint.solve_ivp(
    l96m.regressed,
    [0,TAU],
    states[:,n],
    method = 'RK45',
    max_step = TAU)
    
    states[:,n+1] = Psi_state.y[:,-1] + np.random.normal(0,sigma_small,size=(Psi_state.y).shape[0])

    obs[:,n+1] = np.array([states[0,n+1],states[1,n+1],states[3,n+1],states[4,n+1],states[6,n+1],states[7,n+1]]) \
        + np.random.normal(0,gamma_small,size=6)

#save the x and y in a single npy file with labels states and observations respectively

np.savez('simulation_data_singlescale_001.npz', states=states, observations=obs)

################################################################################
# Run Single Scale Dynamics with gamma/sigma large##############################
################################################################################

states = np.zeros((l96m.K, np.arange(0,T_dyn,TAU).shape[0]))
its = states.shape[1]
obs = np.zeros((6, np.arange(0,T_dyn,TAU).shape[0]))
states[:,0] = z0[:l96m.K]

#First generate the true states and the true data
for n in range(its-1):

    Psi_state = scint.solve_ivp(
    l96m.regressed,
    [0,TAU],
    states[:,n],
    method = 'RK45',
    max_step = TAU)
    
    states[:,n+1] = Psi_state.y[:,-1] + np.random.normal(0,sigma_large,size=(Psi_state.y).shape[0])

    obs[:,n+1] = np.array([states[0,n+1],states[1,n+1],states[3,n+1],states[4,n+1],states[6,n+1],states[7,n+1]]) \
        + np.random.normal(0,gamma_large,size=6)

#save the x and y in a single npy file with labels states and observations respectively

np.savez('simulation_data_singlescale_1.npz', states=states, observations=obs)

################################################################################
# Run Multi-Scale Dynamics #####################################################
################################################################################

states = np.zeros((l96m.K + l96m.K * l96m.J, np.arange(0,T_dyn,TAU).shape[0]))
its = states.shape[1]
obs = np.zeros((6, np.arange(0,T_dyn,TAU).shape[0]))
states[:,0] = z0

#First generate the true states and the true data
for n in range(its-1):

    Psi_state = scint.solve_ivp(
    l96m.regressed,
    [0,TAU],
    states[:,n],
    method = 'RK45',
    max_step = TAU)
    
    states[:,n+1] = Psi_state.y[:,-1] + np.random.normal(0,sigma_small,size=(Psi_state.y).shape[0])

    obs[:,n+1] = np.array([states[0,n+1],states[1,n+1],states[3,n+1],states[4,n+1],states[6,n+1],states[7,n+1]]) \
        + np.random.normal(0,gamma_small,size=6)

#save the x and y in a single npy file with labels states and observations respectively

np.savez('simulation_data_multiscale_001.npz', states=states, observations=obs)


################################################################################
# Generate Data for Inversion Experiment #######################################
################################################################################
    
#data generation
ITS = np.arange(0,T_dyn,TAU).shape[0]

x = np.zeros((l96m.K, np.arange(0,T_dyn,TAU).shape[0]))
x[:,0] = z0[:l96m.K]

#First generate the true data
for n in range(ITS-1):

    Psix = scint.solve_ivp(
    l96m.regressed,
    [0,TAU],
    x[:,n],
    method = 'RK45',
    max_step = TAU)
    
    x[:,n+1] = Psix.y[:,-1]

mm = (1/9)*((1/ITS)*x.sum(axis=1)).sum(axis=0)
mv = (1/9)*(np.var(x,axis=1)).sum(axis=0)

w = np.array([mm,mv]) + np.random.normal(0,gamma_small,size=2)

np.savez('simulation_data_singlescale_inversion.npz', states=x, observation=w)