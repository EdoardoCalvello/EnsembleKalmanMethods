import sys
sys.path.append('../')
import numpy as np
import scipy.integrate as scint
from Data_Assimilation.DA_Inversion import EKI_transport, EKI_post
from joblib import load
from time import perf_counter as timer


#Define forward model
class G:

    def forward(self, u):

        return (7/12)*u**3 - (7/2)*u**2 + 8*u
    
class G_R:

    def forward(self, u):

        return np.concatenate((G().forward(u),u), axis=0)
    
#Experiment Parameters

#Load the observation
w=2
observation = np.array([w])
observation_R = np.array([[w],[-2]])
#Set the noise std dev
gamma = 1
gamma_R = np.array([[gamma**2,0],[0,0.5]])

#Data Assimilation Scheme Parameters

#Set the ensemble size
J=2000
#set initial condition for the data assimilation scheme
initial_condition = np.random.normal(-2,np.sqrt(0.5),size=(1,J))

##############################################################################
##############################################################################
###############      Run Experiments with EKI transport      ################# 
##############################################################################
##############################################################################

#Set dt
dt = 0.00025
#initialize the data assimilation method
eki = EKI_transport(G(), 
            gamma,
            dt, 
            ensemble_size = J,
            optimization=True,
            its=100000)

#Run the data assimilation scheme
predicted_states =  eki.run(observation, initial_condition)
np.savez('./results/EKI_transport'+'%s' %dt + '_bayes_opt.npz', Bayes_sol=predicted_states[...,0], opt_sol=predicted_states[...,1])

##############################################################################
##############################################################################
############     Run Experiments with EKI posterior inflation      ########### 
##############################################################################
##############################################################################

#Set dt
dt = 0.00025
#Set the ensemble size
J=2000
#initialize the data assimilation method
eki = EKI_post(G_R(), 
            gamma_R,
            dt, 
            ensemble_size = J,
            its=100000)

#Run the data assimilation scheme
predicted_states =  eki.run(observation_R, initial_condition)
np.savez('./results/EKI_post'+'%s' %dt + '.npz', Bayes_sol=predicted_states)