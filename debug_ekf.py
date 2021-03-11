from ekf import ekf, Constants
import math
import numpy as np
import matplotlib.pyplot as plt
import tf
import random

def ekf_estimation2(xEst, PEst, z, u):
    """
    :param xEst: The estimated state of the rover
    :param PEst: Covariance of the state
    :param z: measurements with noise
    :param z_available: which measurements are available? An array of indices into z.
    :param u: Control input with noise
    :return: xEst and PEst
    """

    #  Predict
    xPred = E.motion_model(xEst, u)  # here we predict the state based on previous state.
    jF = E.jacobF(xPred, u)  # here we get the jacobian of the model function, F. 

    PPred = np.matmul(np.matmul(jF, PEst),jF.T) + E.Q    #Here we calculate the covariance of the prediction xPred. Great. 

    #  Update

    # find out which measurements are not None
    z_available = np.array([i for i in range(z.size) if z[0,i] is not None])

    jH = E.jacobH(xPred, z_available) # Here we get the jacobian of the state -> measurement function, which in this case isn't really a jacobian but a constant. 
    zPred = E.observation_model(xPred, z_available)   # Here we predict the measurements based on our xPred. 
    y = (z[0,z_available] - zPred.T).astype(float) # Seems to be the difference between the original measurement and the predicted measurement. 

    S = np.matmul(np.matmul(jH, PPred), jH.T) + E.R[z_available[:, None], (z_available)]  #   The covariance of the prediction xPred, i.e. the model's guess of the state, is transformed into the covariance of the measurement prediction, plus the covariance of error in measurement. 
    K = np.matmul(np.matmul(PPred, jH.T), np.linalg.inv(S)) #This is just the Kalman gain. 

    tmp = np.matmul(K, y.T) # and we multiply by the difference between z observed and the predicted z, per the equation to yield the updated estimate. Not sure why that's not just done in the line below.  

    xEst = xPred + tmp  # completing the formula
    PEst = np.matmul((np.eye(len(xEst)) - np.matmul(K, jH)), PPred) # formula for the covariance update, but factorised for simplicity.  

    xEst[2] = (xEst[2]) % (2 * math.pi)
    return xEst, PEst   

prevx1 = np.array([[ 2.01792803],
            [-4.79473828],
            [6.27778496],
            [16.97710831]])

realprevx = np.array([[2.032589389230107],
[-4.797843125165141],
[0.017453292519938657],
[16.97710831]])

prevpx1 = np.array([[1.08403297*10**-4, -2.93400923*10**-5, -6.52683264*10**-6,  1.67437305*10**-4],
[-2.93400924*10**-5,  5.95682439*10**-5,  1.46251648*10**-5, -4.75650167*10**-5],
[-6.52683264*10**-6,  1.46251648*10**-5, 1.15576674*10**-1, -9.11047095*10**-6],
[ 1.67437305*10**-4, -4.75650167*10**-5, -9.11047095*10**-6,  1.63052221]])

z = np.array([[2.938364956549898, -5.829428574276752, 5.43208417760029, 5.729651456309737, 7.7681136978978405, 15.777265101279973, None, None, None]])
u = np.array([[17.2,1.74532925]])


E = ekf(10, Constants.R, Constants.Q)
print(ekf_estimation2(prevx1, prevpx1, z,u))
print(ekf_estimation2(realprevx, prevpx1, z,u))

