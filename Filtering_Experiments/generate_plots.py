#!/opt/local/bin/python3 --
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump,load
from sklearn.metrics import mean_squared_error

##############################################################################
##############################################################################
##################         Plot of the function m     ########################
##############################################################################
##############################################################################

gp = load('../Lorenz96/closure.joblib')

range_min = -40
range_max = 40
mesh = np.linspace(range_min, range_max, 10000)
mean, std = gp.predict(mesh[:,np.newaxis], return_std = True)


fig, ax = plt.subplots(figsize=(20,8))
ax.set_xlim(range_min, range_max)
ax.plot(mesh, mean, '.', ms = 4, color='black')
#make ticks and tick labels larger
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('$v$', fontsize=30)
plt.ylabel('$m(v)$', fontsize=30)
#make tick numbers larger
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Function $m$', fontsize=30)
fig.savefig('./plots/m.png')

##############################################################################
##############################################################################
########    3DVAR Experiments with inter-observation time 0.001     ##########
##############################################################################
##############################################################################

### CHECK


sigma = 0.001
gamma = 0.001
tau = 0.001
T = 20

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.001_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.001_predicted_observations_singlescale_001.png')

####
sigma = 0.1
gamma = 0.1
tau = 0.001
T = 20

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_1.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.001_predicted_observations_singlescale_1.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.001_predicted_observations_singlescale_1.png')


####################################################################
####################################################################
#########       3DVAR with inter-observation time 0.5      #########
####################################################################
####################################################################

sigma = 0.001
gamma = 0.001
tau = 0.5
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',s=150, color='darkorange', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.5_predicted_observations_singlescale_001.png')

####################################################################
####################################################################
#########       3DVAR with inter-observation time 1      #########
####################################################################
####################################################################

sigma = 0.001
gamma = 0.001
tau = 1
T = 30

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.5_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/0.5)]
v_3DVAR1 = np.load('./results/3DVAR_TAU1_predicted_observations_singlescale_001.npz')['prediction'][:,:int(T/tau)]

fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,::2], label = r'3DVAR, $\tau=5\cdot 10^{-1}$',s=150, color='darkorange', marker='o')
ax.scatter(np.arange(0,T,1), v_3DVAR1[2,:], label = r'3DVAR, $\tau=10^0$',s=150, color='black', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=20, loc='lower left')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)


fig.savefig('./plots/3DVAR_TAU1_predicted_observations_singlescale_001.png')

##############################################################################
##############################################################################
##################            Multiscale Experiment          #################
##############################################################################
##############################################################################
k=2
j=1
sigma = 0.001
gamma = 0.001
tau = 0.001
T = 20

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_multiscale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.001_predicted_observations_multiscale_001.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,tau), v_3DVAR[2,:], label = '3DVAR',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 multiscale', fontsize=30)

fig.savefig('./plots/3DVAR_TAU0.001_predicted_observations_multiscale_001.png')

#plot of fast variable coupled with slow variable

fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,tau), x[9 + (k)*8+j,:], lw = 0.6, alpha = 0.6, color='gray')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('Dynamics of one slow variable and associated fast variable of the Lorenz 96 multiscale model', fontsize=30)

fig.savefig('./plots/fast.png')

tau = 1
T = 20

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_multiscale_001.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.001_predicted_observations_multiscale_001.npz')['prediction'][:,:int(T/0.001)]
v_3DVAR1 = np.load('./results/3DVAR_TAU1_predicted_observations_multiscale_001.npz')['prediction'][:,:int(T/tau)]

fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.scatter(np.arange(0,T,tau), v_3DVAR[2,::1000], label = r'3DVAR, $\tau=10^{-3}$',s=150, color='darkorange', marker='o')
ax.scatter(np.arange(0,T,1), v_3DVAR1[2,:], label = r'3DVAR, $\tau=10^0$',s=150, color='black', marker='o')
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=25, loc='lower right')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('3DVAR on singlescale for true model given by Lorenz 96 multiscale', fontsize=30)


fig.savefig('./plots/3DVAR_TAU1_predicted_observations_multiscale_001.png')

##############################################################################
##############################################################################
##################            Noisy 3DVAR Experiment        ##################
##############################################################################
##############################################################################

sigma = 0.1
gamma = 0.1
tau = 0.001
T = 20

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_1.npz')['states'][:,:tot_len]
v_3DVARnn = np.load('./results/3DVAR_TAU0.001_predicted_observations_singlescale_1.npz')['prediction'][:,:int(T/tau)]
v_3DVAR = np.load('./results/noisy3DVAR_TAU0.001_predicted_observations_singlescale_1.npz')['prediction'][:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
#ax.plot(np.arange(0,T,tau), v_3DVARnn[2,:], label = '3DVAR',linewidth=3)
ax.plot(np.arange(0,T,tau), v_3DVAR[2,:], label = 'Noisy 3DVAR',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('Noisy 3DVAR on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/noisy3DVAR_TAU0.001_predicted_observations_singlescale_1.png')

MSE_3DVAR = 0
MSE_noisy3DVAR = 0
for t in range(3000, int(T/tau), 1):
    MSE_3DVAR += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], v_3DVARnn[:,t])
    MSE_noisy3DVAR += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], v_3DVAR[:,t])
    
print('3DVARs MSE is %s' %MSE_3DVAR)
print('Noisy 3DVARs MSE is %s' %MSE_noisy3DVAR)

##############################################################################
##############################################################################
##################            EnKF Experiment               ##################
##############################################################################
##############################################################################

sigma = 0.1
gamma = 0.1
tau = 0.001
T = 10

tot_len = int(T/0.001)
skip_len = int(tau/0.001)
x = np.load('../Lorenz96/simulation_data_singlescale_1.npz')['states'][:,:tot_len]
v_3DVAR = np.load('./results/3DVAR_TAU0.001_predicted_observations_singlescale_1.npz')['prediction'][:,:int(T/tau)]
v_EnKF = np.load('./results/EnKF100_predicted_observations_singlescale_1.npz')['prediction'][:,:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.set_ylim(-12, 16)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,0.001), v_3DVAR[2,:], label = '3DVAR',linewidth=3)
ax.plot(np.arange(0,T,tau), np.mean(v_EnKF, axis=1)[2,:], label = 'EnKF, J=100',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=25, loc='lower right')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('EnKF on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/EnKF100_TAU0.001_predicted_observations_singlescale_1.png')

MSE_3DVAR = 0
MSE_EnKF = 0
vEnKF= np.mean(v_EnKF, axis=1)
for t in range(3000, int(T/tau), 1):
    MSE_3DVAR += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], v_3DVAR[:,t])
    MSE_EnKF += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], vEnKF[:,t])
    
print('3DVARs MSE is %s' %MSE_3DVAR)
print('EnKFs100 MSE is %s' %MSE_EnKF)

v_EnKF = np.load('./results/EnKF500_predicted_observations_singlescale_1.npz')['prediction'][:,:,:int(T/tau)]
fig, ax = plt.subplots(figsize=(20,8))

ax.set_xlim(0, T)
ax.set_ylim(-12, 16)
ax.plot(np.arange(0,T,0.001), x[2,:], label = 'dns',linewidth=3)
ax.plot(np.arange(0,T,0.001), v_3DVAR[2,:], label = '3DVAR',linewidth=3)
ax.plot(np.arange(0,T,tau), np.mean(v_EnKF, axis=1)[2,:], label = 'EnKF, J=500',linewidth=3)
ax.set_xlabel('$t$', fontsize=30),ax.set_ylabel('$v$', fontsize=30)
ax.legend(fontsize=25, loc='lower right')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('EnKF on singlescale for true model given by Lorenz 96 singlescale', fontsize=30)

fig.savefig('./plots/EnKF500_TAU0.001_predicted_observations_singlescale_1.png')

MSE_3DVAR = 0
MSE_EnKF = 0
vEnKF= np.mean(v_EnKF, axis=1)
for t in range(3000, int(T/tau), 1):
    MSE_3DVAR += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], v_3DVAR[:,t])
    MSE_EnKF += (1/((T/tau) - 3000))*mean_squared_error(x[:,t], vEnKF[:,t])
    
print('3DVARs MSE is %s' %MSE_3DVAR)
print('EnKFs500 MSE is %s' %MSE_EnKF)