import sys
sys.path.append('../')
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
from Lorenz96.L96_multiscale import L96M
from Lorenz96.DATools_ import DATools
from Data_Assimilation.DA_Inversion import EKI_transport, EKI_post
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
class G:

    def __init__(self, TAU, t, its):

        self.TAU = TAU
        self.t = t
        self.its = its

    def forward(self, u):
    
        J = u.shape[1]
        Gu = np.zeros((2,J))

        for j in range(J):

            l96m.set_F(u[0,j])
            #set ic
            x_hat = np.zeros((l96m.K, np.arange(0,self.t,self.TAU).shape[0]))
            x_hat[:,0] = np.random.normal(0,40,size=l96m.K)

            for k in range(self.its-1):

                Psi = scint.solve_ivp(
                l96m.regressed,
                [0,self.TAU],
                x_hat[:,k],
                method = 'RK45',
                max_step = self.TAU)
            
                x_hat[:,k+1] = Psi.y[:,-1]


            mm = (1/9)*((1/(self.its-1000))*x_hat[:,1000:].sum(axis=1)).sum(axis=0)
            mv = (1/9)*(np.var(x_hat[:,1000:],axis=1)).sum(axis=0)

            Gu[:,j] = np.array([mm,mv])

        return Gu
    
class G_R:

    def __init__(self, TAU, t, its):

        self.TAU = TAU
        self.t = t
        self.its = its

    def forward(self, u):

        G_func = G(self.TAU, self.t, self.its)
        Gu = G_func.forward(u)

        return np.concatenate((Gu,u), axis=0)
    

#Load the observations and underlying true states
observation = np.load('../Lorenz96/simulation_data_singlescale_inversion.npz')['observation']


##############################################################################
##############################################################################
###############      Run Experiments with EKI transport      ################# 
##############################################################################
##############################################################################

#Forward model parameters
TAU = 0.01
t = 20
integration_its = np.arange(0,t,TAU).shape[0]

#Data Assimilation Scheme Parameters

#Set dt
dt = 0.05
gamma = 0.1
J = 30
initial_condition = np.random.normal(0,10,size=(1,J))

#initialize the data assimilation method
eki = EKI_transport(G(TAU,t,integration_its), 
            gamma,
            dt, 
            ensemble_size = J,
            optimization=True,
            its=40)

#Run the data assimilation scheme
predicted_states =  eki.run(observation, initial_condition)
np.savez('./results/EKI_transport'+'%s' %dt + '_L96_bayes_opt.npz', Bayes_sol=predicted_states[...,0], opt_sol=predicted_states[...,1])

##############################################################################
##############################################################################
#########      Run Experiments with EKI transport one step     ############### 
##############################################################################
##############################################################################

#Forward model parameters
TAU = 0.01
t = 20
integration_its = np.arange(0,t,TAU).shape[0]

#Data Assimilation Scheme Parameters

#Set dt
dt = 1
gamma = 0.1
J = 30
initial_condition = np.random.normal(0,10,size=(1,J))

#initialize the data assimilation method
eki = EKI_transport(G(TAU,t,integration_its), 
            gamma,
            dt, 
            ensemble_size = J)

#Run the data assimilation scheme
predicted_states =  eki.run(observation, initial_condition)
np.savez('./results/EKI_transport'+'%s' %dt + '_L96_bayes_one.npz', Bayes_sol=predicted_states)


##############################################################################
##############################################################################
###############      Run Experiments with EKI posterior     ################## 
##############################################################################
##############################################################################


#Forward model parameters
TAU = 0.01
t = 20
integration_its = np.arange(0,t,TAU).shape[0]

#Data Assimilation Scheme Parameters

#Set dt
dt = 0.05
gamma = 0.1
J = 30
initial_condition = np.random.normal(0,10,size=(1,J))

gamma_R = np.array([[gamma**2,0,0],[0,gamma**2,0],[0,0,10**2]])
observation_R = np.array([[observation[0]],[observation[1]],[0]])

#initialize the data assimilation method
eki = EKI_post(G_R(TAU,t,integration_its), 
            gamma_R,
            dt, 
            ensemble_size = J,
            its=20)

#Run the data assimilation scheme
predicted_states =  eki.run(observation_R, initial_condition)
np.savez('./results/EKI_post'+'%s' %dt + '_L96.npz', Bayes_sol=predicted_states)