import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilter:
    def __init__(self):
        self.v = 0
        self.prev_time = 0
        # Initial State
        self.x = np.matrix([[0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[50., 0.],
                            [0., 50.]])

        # Next State Function
        self.F = np.matrix([[1., 1.],
                            [0., 1.]])

        # Measurement Function
        self.H = np.matrix([[1., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[0.5]])

        # Identity Matrix
        self.I = np.matrix([[1., 0.],
                            [0., 1.]])
    def predict(self,t):
        # Calculate dt.
        dt = t - self.prev_time
        # Put dt into the state transition matrix.
        self.F[0,1] = dt
        self.x = self.F*self.x
        self.P =  self.F*self.P* np.transpose(self.F)
        return self.x[0,0]

    def measure_and_update(self,measurements,t):
        dt = t - self.prev_time
        self.F[0,1] = dt
        Z = np.matrix(measurements)
        y = Z.transpose() - self.H*self.x
        S = self.H*self.P*self.H.transpose() + self.R 
        K =  self.P*self.H.transpose()* np.linalg.inv(S)
        self.x = self.x+K*y
        self.P = (self.I-K*self.H)*self.P
        self.P[0, 0] += 0.2
        self.P[1, 1] += 0.2
        self.v = self.x[1,0]
        self.prev_time = t
        return


sim_run(options,KalmanFilter)
