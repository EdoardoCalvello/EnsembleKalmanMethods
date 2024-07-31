import numpy as np

class EKI_transport:

    def __init__(self, forward_operator, gamma, dt, ensemble_size=100, optimization=False, its=None):
        """
        Initialize the 3DVAR class.
        
        Parameters:
        - observations: list, the list of observations
        - forward_operator: function, the forward operator
        - gain: matrix, the gain
        - noisy3DVAR: bool, whether to use noisy 3DVAR or not (default is False)
        """

        self.forward_operator = forward_operator
        self.gamma = gamma
        self.J = ensemble_size
        self.dt = dt
        self.optimization = optimization

        if self.optimization:
            if its is None:
                raise ValueError("Number of iterations must be provided if optimization is True")
            self.its = its
        else:
            self.its = int(1/self.dt)

    def KalmanGain_matmul(self, u, G, innovation):

        CGG = np.cov(G)
        first_dim, second_dim = u.shape[0], G.shape[0]
        CuG = np.cov(u, G)[:first_dim, -second_dim:]
        Gamma = (self.gamma **2) * np.eye(second_dim)
        return self.dt*CuG@np.linalg.solve((self.dt*CGG + Gamma), innovation)

    def forecast(self, u):
        """
        Perform the forecast step.
        
        Parameters:
        - v: np.array, state vector
        
        Returns:
        - v_hat: np.array, simulated state vector
        """

        u_hat = u

        return u_hat
    
    def analysis(self, u_hat, w_observed):
        """
        Perform the analysis step
        
        Parameters:
        - v_hat: np.array, simulated state vector
        - y_observed: np.array, observation vector
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """

        Gu_hat = self.forward_operator.forward(u_hat)
        w_hat = Gu_hat + np.random.normal(0,np.sqrt((self.gamma**2)/self.dt),size=Gu_hat.shape)

        # Innovation
        innovation = np.tile(w_observed.reshape(-1,1), (1,self.J)) - w_hat

        # Analysis step
        u = u_hat + self.KalmanGain_matmul(u_hat, Gu_hat, innovation)
        
        return u
    
    def run(self, observation, ic):
        """
        Run the 3DVAR data assimilation method for the given number of iterations.
        
        Parameters:
        - x_background: np.array, background state vector
        - y_observed: np.array, observed state vector
        - n_iter: int, number of iterations
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """
        EKI_ensemble = ic

        if self.optimization:

            EKI_sol = np.zeros((ic.shape[0],self.J,2))

            for n in range(self.its-1):
                EKI_ensemble = self.analysis(self.forecast(EKI_ensemble), observation)
                if int((n+1)*self.dt) == 1:
                    EKI_sol[...,0] = EKI_ensemble
            EKI_sol[...,1] = EKI_ensemble

            return EKI_sol
        
        else:

            EKI_sol = np.zeros((ic.shape[0],self.J))

            for n in range(self.its-1):
                EKI_ensemble = self.analysis(self.forecast(EKI_ensemble), observation)
            EKI_sol = EKI_ensemble

            return EKI_sol


class EKI_post:

    def __init__(self, forward_operator, gamma_R, dt, ensemble_size=100, its=100):
        """
        Initialize the 3DVAR class.
        
        Parameters:
        - observations: list, the list of observations
        - forward_operator: function, the forward operator
        - gain: matrix, the gain
        - noisy3DVAR: bool, whether to use noisy 3DVAR or not (default is False)
        """

        self.forward_operator = forward_operator
        self.gamma_R = gamma_R
        self.J = ensemble_size
        self.dt = dt
        self.its = its

    def KalmanGain_matmul(self, u, G, innovation):

        CGG = np.cov(G)
        first_dim, second_dim = u.shape[0], G.shape[0]
        CuG = np.cov(u, G)[:first_dim, -second_dim:]
        return CuG@np.linalg.solve((self.dt*CGG + self.gamma_R), innovation)

    def forecast(self, u):
        """
        Perform the forecast step.
        
        Parameters:
        - v: np.array, state vector
        
        Returns:
        - v_hat: np.array, simulated state vector
        """

        C_n = np.cov(u).reshape(u.shape[0],-1)
        u_hat = u + np.random.multivariate_normal(np.zeros(u.shape[0]),(self.dt/(1-self.dt))*C_n, size=self.J).T

        return u_hat
    
    def analysis(self, u_hat, w_R):
        """
        Perform the analysis step
        
        Parameters:
        - v_hat: np.array, simulated state vector
        - y_observed: np.array, observation vector
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """

        GuR_hat = self.forward_operator.forward(u_hat)
        y_hat = self.dt*GuR_hat + np.random.multivariate_normal(np.zeros(self.gamma_R.shape[0]),self.gamma_R*self.dt, size=self.J).T

        # Innovation
        innovation = self.dt*np.tile(w_R.reshape(-1,1), (1,self.J)) - y_hat

        # Analysis step
        u = u_hat + self.KalmanGain_matmul(u_hat, GuR_hat, innovation)
        
        return u
    
    def run(self, observation_R, ic):
        """
        Run the 3DVAR data assimilation method for the given number of iterations.
        
        Parameters:
        - x_background: np.array, background state vector
        - y_observed: np.array, observed state vector
        - n_iter: int, number of iterations
        
        Returns:
        - x_analysis: np.array, analysis state vector
        """
        EKI_ensemble = ic

        for n in range(self.its-1):
            EKI_ensemble = self.analysis(self.forecast(EKI_ensemble), observation_R)

        return EKI_ensemble
