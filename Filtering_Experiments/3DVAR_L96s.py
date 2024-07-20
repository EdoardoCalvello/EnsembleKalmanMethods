import sys
sys.path.append('../')
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
from Lorenz96.L96_multiscale import L96M
from Lorenz96.DATools_ import DATools
from Data_Assimilation.DA_Filtering import ThreeDVAR
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

#Define the gain matrix 

K = np.zeros((l96m.K, 6))
K[0,0] = 1
K[1,1] = 1
K[3,2] = 1
K[4,3] = 1
K[6,4] = 1
K[7,5] = 1


##############################################################################
##############################################################################
##########      Run Experiments with inter-observation time 0.001   ##########
##############################################################################
##############################################################################

#Set the inter-observation time
TAU = 0.001
#initialize the data assimilation method
threeDVAR = ThreeDVAR(Psi(TAU), observation_operator(), K)

#Load the observations and underlying true states

truth = np.load('../Lorenz96/simulation_data_singlescale_001.npz')
true_states = truth['states']
true_observations = truth['observations']

#set initial condition for the data assimilation scheme
initial_condition = 10*np.ones(l96m.K)
#Run the data assimilation scheme
predicted_states =  threeDVAR.run(true_observations, initial_condition)
np.savez('./results/3DVAR_TAU%s' %TAU + '_predicted_observations_singlescale_001.npz', prediction=predicted_states)


### Experiment with larger noise level ###


#Load the observations and underlying true states
truth = np.load('../Lorenz96/simulation_data_singlescale_1.npz')
true_states = truth['states']
true_observations = truth['observations']
#Run the data assimilation scheme
predicted_states =  threeDVAR.run(true_observations, initial_condition)
np.savez('./results/3DVAR_TAU%s' %TAU + '_predicted_observations_singlescale_1.npz', prediction=predicted_states)



##############################################################################
##############################################################################
##########      Run Experiments with inter-observation time 0.5     ##########
##############################################################################
##############################################################################

#Set the inter-observation time
TAU = 0.5
#initialize the data assimilation method
threeDVAR = ThreeDVAR(Psi(TAU), observation_operator(), K)

#Load the observations and underlying true states

truth = np.load('../Lorenz96/simulation_data_singlescale_001.npz')
true_states = truth['states']
true_observations = truth['observations']

#set initial condition for the data assimilation scheme
initial_condition = 10*np.ones(l96m.K)
#Run the data assimilation scheme
predicted_states =  threeDVAR.run(true_observations, initial_condition)
np.savez('./results/3DVAR_TAU%s' %TAU + '_predicted_observations_singlescale_001.npz', prediction=predicted_states)

##############################################################################
##############################################################################
##########      Run Experiments with inter-observation time 0.5     ##########
##############################################################################
##############################################################################

#Set the inter-observation time
TAU = 1
#initialize the data assimilation method
threeDVAR = ThreeDVAR(Psi(TAU), observation_operator(), K)

#Load the observations and underlying true states

truth = np.load('../Lorenz96/simulation_data_singlescale_001.npz')
true_states = truth['states']
true_observations = truth['observations']

#set initial condition for the data assimilation scheme
initial_condition = 10*np.ones(l96m.K)
#Run the data assimilation scheme
predicted_states =  threeDVAR.run(true_observations, initial_condition)
np.savez('./results/3DVAR_TAU%s' %TAU + '_predicted_observations_singlescale_001.npz', prediction=predicted_states)