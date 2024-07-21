import sys
sys.path.append('../')
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
from Lorenz96.L96_multiscale import L96M
from Lorenz96.DATools_ import DATools
from Data_Assimilation.DA_Filtering import EnKF
from joblib import load
from time import perf_counter as timer

#initialize Lorenz `96 dynamical system
l96m = L96M()
#set the learned closure m
da_tools = DATools(mem_thrsh = 800)
gp = load('../Lorenz96/closure.joblib')
da_tools.set_gpr(gp)
l96m.set_stencil()
l96m.set_predictor(da_tools.get_gpr())

#Define forward model
class Psi:

    def __init__(self, TAU):

        self.TAU = TAU
    
    def forward(self, v):

        v = scint.solve_ivp(
        l96m.regressed,
        [0,self.TAU],
        v,
        method = 'RK45',
        max_step = self.TAU)

        return v.y[:,-1]


class observation_operator:

    def __init__(self):

        self.H = np.zeros((6,l96m.K))
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.H[2,3] = 1
        self.H[3,4] = 1
        self.H[4,6] = 1
        self.H[5,7] = 1

    def forward(self,v):

        return self.H @ v
    

#Load the observations and underlying true states
truth = np.load('../Lorenz96/simulation_data_singlescale_1.npz')
true_states = truth['states']
true_observations = truth['observations'][:,:10000]


##############################################################################
##############################################################################
##########         Run Experiments with 100 ensemble members        ##########
##############################################################################
##############################################################################

#Set the inter-observation time
TAU = 0.001
#Set the model and observation noise
sigma = 0.001
gamma = 0.001
#Set the ensemble size
J=100
#initialize the data assimilation method
enkf = EnKF(Psi(TAU), 
            observation_operator(), 
            sigma, 
            gamma, 
            ensemble_size = J)

#set initial condition for the data assimilation scheme
initial_condition = np.random.normal(10,10,size=(l96m.K,J))
#Run the data assimilation scheme
predicted_states =  enkf.run(true_observations, initial_condition)
np.savez('./results/EnKF'+'%s' %J + '_predicted_observations_singlescale_1.npz', prediction=predicted_states)

##############################################################################
##############################################################################
##########        Run Experiments with 1000 ensemble members        ##########
##############################################################################
##############################################################################

#Set the ensemble size
J=1000
#initialize the data assimilation method
enkf = EnKF(Psi(TAU), 
            observation_operator(), 
            sigma, 
            gamma, 
            ensemble_size = J)

#set initial condition for the data assimilation scheme
initial_condition = np.random.normal(10,10,size=(l96m.K,J))
#Run the data assimilation scheme
predicted_states =  enkf.run(true_observations, initial_condition)
np.savez('./results/EnKF'+'%s' %J + '_predicted_observations_singlescale_1.npz', prediction=predicted_states)
