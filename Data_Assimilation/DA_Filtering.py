import numpy as np

class ThreeDVAR:

    def __init__(self, forward_operator, observation_operator, gain, noisy3DVAR=False):
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
        
        # Innovation
        innovation = y_observed - self.observation_function.forward(v_hat)
        
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

        for n in range(its):
            y_observed = observations[:,n]
            predicted_states[:,n+1] = self.analysis(self.forecast(predicted_states[:,n]), y_observed)
        return predicted_states


class EnKF:

    def __init__(self, forward_operator, observation_operator, ensemble_size, B, R, H, noisyEnKF=False):
        """
        Initialize the EnKF class.
        
        Parameters:
        - forward_operator: function, the forward operator
        - observation_operator: function, the observation operator
        - ensemble_size: int, number of ensemble members
        - B: np.array, background error covariance matrix
        - R: np.array, observation error covariance matrix
        - H: np.array, observation operator matrix
        """
        self.forward_operator = forward_operator
        self.observation_operator = observation_operator
        self.ensemble_size = ensemble_size

    def prediction(self, ensemble):
        """
        Perform one iteration of the forecast step.
        
        Parameters:
        - ensemble: np.array, ensemble of state vectors
        
        Returns:
        - ensemble_forecast: np.array, forecast ensemble of state vectors
        """
        ensemble_forecast = np.array([self.forward_operator(x) for x in ensemble])
        return ensemble_forecast


    def analysis(self, ensemble_forecast, y_observed):
        """
        Perform one iteration of the EnKF data assimilation method.
        
        Parameters:
        - ensemble_forecast: np.array, forecast ensemble of state vectors
        - y_observed: np.array, observed state vector
        
        Returns:
        - ensemble_analysis: np.array, analysis ensemble of state vectors
        """
        
        # Compute ensemble mean
        ensemble_mean = np.mean(ensemble_forecast, axis=0)
        
        # Compute ensemble perturbations
        ensemble_perturbations = ensemble_forecast - ensemble_mean
        
        # Compute observation perturbations
        observation_perturbations = y_observed - np.array([self.observation_operator(x) for x in ensemble_forecast])
        
        # Compute Kalman gain
        K = np.linalg.solve((self.H @ self.B @ self.H.T + self.R).T, self.H @ self.B.T).T
        
        # Update ensemble
        ensemble_analysis = ensemble_forecast + K @ observation_perturbations
        
        return ensemble_analysis
    
    def run(self, ensemble, y_observed, n_iter):

        """
        Run the EnKF data assimilation method for a given number of iterations.
        
        Parameters:
        - ensemble: np.array, ensemble of state vectors
        - y_observed: np.array, observed state vector
        - n_iter: int, number of iterations
        
        Returns:
        - ensemble_analysis: np.array, analysis ensemble of state vectors
        """
        ensemble_analysis = ensemble
        for _ in range(n_iter):
            ensemble_analysis = self.analysis(ensemble_analysis, y_observed)
        return ensemble_analysis


