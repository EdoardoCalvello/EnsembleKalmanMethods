import numpy as np

class ThreeDVAR:

    def __init__(self, forward_operator, observation_operator, gain, noisy3DVAR=False, sigma=None, gamma=None):
        """
        Initialize the 3DVAR class.
        
        Parameters:
        - observations: list, the list of observations
        - forward_operator: function, the forward operator
        - observation_operator: function, the observation operator
        - gain: matrix, the gain
        - noisy3DVAR: bool, whether to use noisy 3DVAR or not (default is False)
        """

        self.forward_operator = forward_operator
        self.observation_function = observation_operator
        self.K = gain
        self.noisy3DVAR = noisy3DVAR
        self.sigma = sigma
        self.gamma = gamma
        

    def forecast(self, v):
        """
        Perform the forecast step.
        
        Parameters:
        - v: np.array, state vector
        
        Returns:
        - v_hat: np.array, simulated state vector
        """
        v_hat = self.forward_operator.forward(v)
        return v_hat
    
    def analysis(self, v_hat, y_observed):
        """
        Perform the analysis step
        
        Parameters:
        - v_hat: np.array, simulated state vector
        - y_observed: np.array, observation vector
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """
        if self.noisy3DVAR:
            v_hat = v_hat + np.random.normal(0,self.sigma,size=v_hat.shape)
            y_hat = self.observation_function.forward(v_hat) + np.random.normal(0,self.gamma,size=y_observed.shape)
        else:
            y_hat = self.observation_function.forward(v_hat)
        # Innovation
        innovation = y_observed - y_hat
        
        # Analysis step
        v = v_hat + self.K @ innovation
        
        return v
    
    def run(self, observations, ic):
        """
        Run the 3DVAR data assimilation method for the given number of iterations.
        
        Parameters:
        - x_background: np.array, background state vector
        - y_observed: np.array, observed state vector
        - n_iter: int, number of iterations
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """
        its = observations.shape[1]
        predicted_states = np.zeros((ic.shape[0], its+1))
        predicted_states[:,0] = ic

        for n in range(its-1):
            y_observed = observations[:,n+1]
            predicted_states[:,n+1] = self.analysis(self.forecast(predicted_states[:,n]), y_observed)
        return predicted_states


class EnKF:

    def __init__(self, forward_operator, observation_operator, sigma, gamma, ensemble_size=100):
        """
        Initialize the 3DVAR class.
        
        Parameters:
        - observations: list, the list of observations
        - forward_operator: function, the forward operator
        - observation_operator: function, the observation operator
        - gain: matrix, the gain
        - noisy3DVAR: bool, whether to use noisy 3DVAR or not (default is False)
        """

        self.forward_operator = forward_operator
        self.observation_function = observation_operator
        self.sigma = sigma
        self.gamma = gamma
        self.J = ensemble_size

    def KalmanGain_matmul(self, v, h, innovation):

        Chh = np.cov(h)
        first_dim, second_dim = v.shape[0], h.shape[0]
        Cvh = np.cov(v, h)[:first_dim, -second_dim:]
        Gamma = self.gamma * np.eye(second_dim)
        return Cvh@np.linalg.solve((Chh + Gamma), innovation)

    def forecast(self, v):
        """
        Perform the forecast step.
        
        Parameters:
        - v: np.array, state vector
        
        Returns:
        - v_hat: np.array, simulated state vector
        """
        Psi_v = np.zeros(v.shape)

        for j in range(self.J):
            Psi_v[:,j] = self.forward_operator.forward(v[:,j])
        
        v_hat = Psi_v + np.random.normal(0,self.sigma,size=Psi_v.shape)

        return v_hat
    
    def analysis(self, v_hat, y_observed):
        """
        Perform the analysis step
        
        Parameters:
        - v_hat: np.array, simulated state vector
        - y_observed: np.array, observation vector
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """


        h_vhat = self.observation_function.forward(v_hat)
        y_hat = h_vhat + np.random.normal(0,self.gamma,size=h_vhat.shape)

        # Innovation
        innovation = np.tile(y_observed.reshape(-1,1), (1,self.J)) - y_hat

        # Analysis step
        v = v_hat + self.KalmanGain_matmul(v_hat, y_hat, innovation)
        
        return v
    
    def run(self, observations, ic):
        """
        Run the 3DVAR data assimilation method for the given number of iterations.
        
        Parameters:
        - x_background: np.array, background state vector
        - y_observed: np.array, observed state vector
        - n_iter: int, number of iterations
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """
        its = observations.shape[1]
        predicted_states = np.zeros((ic.shape[0],self.J,its+1))
        predicted_states[:,:,0] = ic

        for n in range(its-1):
            y_observed = observations[:,n+1]
            predicted_states[:,:,n+1] = self.analysis(self.forecast(predicted_states[:,:,n]), y_observed)
        return predicted_states


