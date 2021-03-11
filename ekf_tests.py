# EKF with AR tag data.  
import math
import numpy as np
import matplotlib.pyplot as plt
import tf
import random
# EKF with AR tag data.  
import math
import numpy as np
import matplotlib.pyplot as plt
import tf
import random


class Constants:
    arx0 = 0
    ary0 = 0
    arx1 = 10
    ary1 = 0
    arx2 = 10
    ary2 = 10
    P_AR_OBSERVED = 0.7
    ERROR_AR = 0.01
    ERROR_AR_ANGLE = np.deg2rad(360)
    Q = np.diag([
            0.3,  # variance of location on x-axis
            0.3,  # variance of location on y-axis
            np.deg2rad(25.0),  # variance of yaw angle
            0.5  # variance of velocity
        ]) ** 2  # predict state covariance
    
    R = np.diag([1.3, 1.3, 0.5, ERROR_AR, ERROR_AR, ERROR_AR, ERROR_AR_ANGLE,ERROR_AR_ANGLE,ERROR_AR_ANGLE]) ** 2 # Assuming that the SD is in Metres? 


class ekf:
    def __init__(self, hz,R,Q):
        assert(hz != 0)
        self.R = R
        self.Q = Q
        self.dt = 1.0/hz    # window of time to which a set of measurements applies
        self.hz = hz
        #self.GPSVisualiser = GPSVisualiser()
        #self.odomPublisher = odomPublisher()

    def motion_model(self, x, u):
        """ Predict then next state of the system given the past state and a model
        :param x: [x, y, yaw, v],
        :param u: [[v]
                [yawrate]]
        :return:
        """ 
        assert(x.shape == (4,1))
        assert(u.shape == (1,2))

        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])
              
        # dimensions are 4 x 2. 
        # so u.T is 2 x 1. Makes sense, ends up w 4x1.
        B = np.array([[self.dt * math.cos(x[2, 0]), 0], # time interval by cosine of yaw. Essentially the horizontal distance component of the velocity vector, when the velocity is 1 at the current orientation.  
                      [self.dt * math.sin(x[2, 0]), 0], # vertical distance component, but why in this matrix form?
                      [0.0, self.dt],   # time interval
                      [1.0, 0.0]])  # to set velocity to the new velocity? 
        x1 = np.matmul(F, x)    # keep position and yaw, but remove velocity. 
        x2 = np.matmul(B, u.T)  # Get the change in x and y, and the change in yaw, as well as the new velocity.
        # updates the x, y and yaw according to the motion model
        retval = x1+x2
        # make sure that the yaw is between 0 - 2pi
        retval[2] = retval[2]%(2*math.pi)
        #retval[2] = retval[2] if retval[2] >= 0 else 2 * math.pi + retval[2]
        return retval

    def observation_model(self, x, z_available):
        # what comes in from the observation? a 4d vector somehow, of which we need the first three elements.
        ar0Theta = np.arctan2(Constants.arx0 - x[0,0], Constants.ary0 - x[1,0])
        ar0Theta = (ar0Theta if ar0Theta >= 0 else 2 * math.pi + ar0Theta)
        ar1Theta = np.arctan2(Constants.arx1 - x[0,0], Constants.ary1 - x[1,0])
        ar1Theta = (ar1Theta if ar1Theta >= 0 else 2 * math.pi + ar1Theta)
        ar2Theta = np.arctan2(Constants.arx2 - x[0,0], Constants.ary2 - x[1,0])
        ar2Theta =  (ar2Theta if ar2Theta >= 0 else 2 * math.pi + ar2Theta)

        z = np.array([[x[0,0], x[1,0], x[2,0], 
            math.sqrt((x[0,0] - Constants.arx0)**2 + (x[1,0] - Constants.ary0)**2), 
            math.sqrt((x[0,0] - Constants.arx1)**2 + (x[1,0] - Constants.ary1)**2),
            math.sqrt((x[0,0] - Constants.arx2)**2 + (x[1,0] - Constants.ary2)**2),
            ar0Theta - x[2,0] if ar0Theta - x[2,0] >= 0 else ar0Theta - x[2,0] + 2*math.pi,
            ar1Theta - x[2,0] if ar1Theta - x[2,0] >= 0 else ar1Theta - x[2,0] + 2*math.pi,
            ar2Theta - x[2,0] if ar2Theta - x[2,0] >= 0 else ar2Theta - x[2,0] + 2*math.pi
            ]])

        
        return z[:, z_available].T
    
    def jacobF(self, x, u):
        """
        Jacobian of Motion Model
        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw) 
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
            [0.0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF

    def jacobH(self, x, z_available):
        # Jacobian of Observation Model
        """
        x,y,yaw,d0,d1,d2,a0,a1,a2
        [1,0,0,0]
        [0,1,0,0]
        [0,0,1,0]
        ((x-d0x)^2 + (y-d0y)^2)^1/2
        ((x-d1x)^2 + (y-d1y)^2)^1/2
        ((x-d2x)^2 + (y-d2y)^2)^1/2
        (np.arctan2(Constants.arx0 - x, Constants.ary0 - y) - yaw
        So the derivative in terms of x is: 
        """
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [(x[0,0]-Constants.arx0)/math.sqrt((x[0,0]-Constants.arx0)**2 + (x[1,0]-Constants.ary0)**2),(x[1,0]-Constants.ary0)/math.sqrt((x[0,0]-Constants.arx0)**2 + (x[1,0]-Constants.ary0)**2),0,0], #AR tag 0 d 
            [(x[0,0]-Constants.arx1)/math.sqrt((x[0,0]-Constants.arx1)**2 + (x[1,0]-Constants.ary1)**2),(x[1,0]-Constants.ary1)/math.sqrt((x[0,0]-Constants.arx1)**2 + (x[1,0]-Constants.ary1)**2),0,0], #AR tag 1 d
            [(x[0,0]-Constants.arx2)/math.sqrt((x[0,0]-Constants.arx2)**2 + (x[1,0]-Constants.ary2)**2),(x[1,0]-Constants.ary2)/math.sqrt((x[0,0]-Constants.arx2)**2 + (x[1,0]-Constants.ary2)**2),0,0], #AR tag 2 d

            [-(Constants.ary0 - x[1,0])/((Constants.ary0 - x[1,0])**2 + (Constants.arx0 - x[0,0])**2),(Constants.arx0 - x[0,0])/((Constants.ary0 - x[1,0])**2 + (Constants.arx0 - x[0,0])**2), 0, -1],
            [-(Constants.ary1 - x[1,0])/((Constants.ary1 - x[1,0])**2 + (Constants.arx1 - x[0,0])**2),(Constants.arx1 - x[0,0])/((Constants.ary1 - x[1,0])**2 + (Constants.arx1 - x[0,0])**2), 0, -1],
            [-(Constants.ary2 - x[1,0])/((Constants.ary2 - x[1,0])**2 + (Constants.arx2 - x[0,0])**2),(Constants.arx2 - x[0,0])/((Constants.ary2 - x[1,0])**2 + (Constants.arx2 - x[0,0])**2), 0, -1]
        ])

        return jH[z_available]

    def ekf_estimation(self, xEst, PEst, z, u):
        """
        :param xEst: The estimated state of the rover
        :param PEst: Covariance of the state
        :param z: measurements with noise
        :param z_available: which measurements are available? An array of indices into z.
        :param u: Control input with noise
        :return: xEst and PEst
        """

        #  Predict
        xPred = self.motion_model(xEst, u)  # here we predict the state based on previous state.
        jF = self.jacobF(xPred, u)  # here we get the jacobian of the model function, F. 

        PPred = np.matmul(np.matmul(jF, PEst),jF.T) + self.Q    #Here we calculate the covariance of the prediction xPred. Great. 

        #  Update

        # find out which measurements are not None
        z_available = np.array([i for i in range(z.size) if z[0,i] is not None])

        jH = self.jacobH(xPred, z_available) # Here we get the jacobian of the state -> measurement function, which in this case isn't really a jacobian but a constant. 
        zPred = self.observation_model(xPred, z_available)   # Here we predict the measurements based on our xPred. 
        y = (z[0,z_available] - zPred.T).astype(float) # Seems to be the difference between the original measurement and the predicted measurement. 

        S = np.matmul(np.matmul(jH, PPred), jH.T) + self.R[z_available[:, None], (z_available)]  #   The covariance of the prediction xPred, i.e. the model's guess of the state, is transformed into the covariance of the measurement prediction, plus the covariance of error in measurement. 
        K = np.matmul(np.matmul(PPred, jH.T), np.linalg.inv(S)) #This is just the Kalman gain. 

        tmp = np.matmul(K, y.T) # and we multiply by the difference between z observed and the predicted z, per the equation to yield the updated estimate. Not sure why that's not just done in the line below.  

        xEst = xPred + tmp  # completing the formula
        PEst = np.matmul((np.eye(len(xEst)) - np.matmul(K, jH)), PPred) # formula for the covariance update, but factorised for simplicity.  

        # gross boundConstants.ary checkup on the yaw to keep it positive, gotta figure out a better way
        xEst[2] = (xEst[2]) % (2 * math.pi)
        #xEst[2] = xEst[2] if xEst[2] >= 0 else 2 * math.pi + xEst[2]
        return xEst, PEst   

    def ekf_estimation2(self, xEst, PEst, z, u):
        """
        :param xEst: The estimated state of the rover
        :param PEst: Covariance of the state
        :param z: measurements with noise
        :param z_available: which measurements are available? An array of indices into z.
        :param u: Control input with noise
        :return: xEst and PEst
        """

        #  Predict
        xPred = self.motion_model(xEst, u)  # here we predict the state based on previous state.
        jF = self.jacobF(xPred, u)  # here we get the jacobian of the model function, F. 

        PPred = np.matmul(np.matmul(jF, PEst),jF.T) + self.Q    #Here we calculate the covariance of the prediction xPred. Great. 

        #  Update

        # find out which measurements are not None
        z_available = np.array([i for i in range(z.size) if z[0,i] is not None])

        jH = self.jacobH(xPred, z_available) # Here we get the jacobian of the state -> measurement function, which in this case isn't really a jacobian but a constant. 
        zPred = self.observation_model(xPred, z_available)   # Here we predict the measurements based on our xPred. 
        y = (z[0,z_available] - zPred.T).astype(float) # Seems to be the difference between the original measurement and the predicted measurement. 

        S = np.matmul(np.matmul(jH, PPred), jH.T) + self.R[z_available[:, None], (z_available)]  #   The covariance of the prediction xPred, i.e. the model's guess of the state, is transformed into the covariance of the measurement prediction, plus the covariance of error in measurement. 
        K = np.matmul(np.matmul(PPred, jH.T), np.linalg.inv(S)) #This is just the Kalman gain. 

        tmp = np.matmul(K, y.T) # and we multiply by the difference between z observed and the predicted z, per the equation to yield the updated estimate. Not sure why that's not just done in the line below.  

        xEst = xPred + tmp  # completing the formula
        PEst = np.matmul((np.eye(len(xEst)) - np.matmul(K, jH)), PPred) # formula for the covariance update, but factorised for simplicity.  

        # gross boundConstants.ary checkup on the yaw to keep it positive, gotta figure out a better way
        xEst[2] = (xEst[2]) % (2 * math.pi)
        #xEst[2] = xEst[2] if xEst[2] >= 0 else 2 * math.pi + xEst[2]
        return xEst, PEst   

    def runFilter(self, visualizeGps=False):
        x1 = np.zeros((4, 1))
        x2 = np.zeros((4, 1))
        x_real = np.zeros((4, 1))
        p1 = np.eye(4)
        p2 = np.eye(4)
        rate = 10

        estimate_data = ([],[])
        true_data = [[],[]]
        gps_data = [[], []]
        u = np.array([[10,np.deg2rad(0)/self.dt]])
        for _ in range(50):
            x_real, x_update = self.simulateFakeDataMeasurement(x_real[0],x_real[1],x_real[2], u) 
            print(x_real[2])
            z = x_update.T
            
            
            x1, p1 = self.ekf_estimation2(x1, p1, z, u)
            #x2, p2 = self.ekf_estimation2(x2,p2,z,u)
            
            #print("\n")
            #print(x1)
            #print(x2)
            
            u = np.array([[u[0,0] + 0.2,np.deg2rad(10)/self.dt]])

            estimate_data[0].append(x1[0,0])
            estimate_data[1].append(x1[1,0])
            true_data[0].append(x_real[0])
            true_data[1].append(x_real[1])
            gps_data[0].append(x_update[0])
            gps_data[1].append(x_update[1])

        return estimate_data, true_data, gps_data

    def simulateFakeDataMeasurement(self,x,y,theta,u):
        """
        I want the rover to move in circles.
        """
        x_pos_real = x + u[0,0] * self.dt * math.cos(theta) + np.random.normal(0, 0.3)
        x_pos = x_pos_real + np.random.normal(0, 1.3)# previous x plus a movement plus noise (from measurement)
        y_pos_real = y + u[0,0] * self.dt * math.sin(theta) + np.random.normal(0, 0.3)
        y_pos = y_pos_real + np.random.normal(0, 1.3)# previous y plus a movement plus noise (from measurement)
        theta_pos_real = (theta + u[0,1] * self.dt) % (2 * math.pi)# increases by 10 degrees 
        theta_pos = (theta_pos_real + np.random.normal(0,0.5)) % (2 * math.pi)

        ar0d_real = None
        ar0d = None
        ar0A_real = None
        ar0A = None
        if random.random() < Constants.P_AR_OBSERVED:
            ar0d_real = math.sqrt((x_pos_real - Constants.arx0)**2 + (y_pos_real - Constants.ary0)**2)
            ar0d = ar0d_real + np.random.normal(0, Constants.ERROR_AR)

            temp_angle = np.arctan2(Constants.arx0 - x_pos_real[0], Constants.ary0 - y_pos_real[0])
            #ar0A_real = temp_angle if temp_angle >= 0 else temp_angle + 2*math.pi

            #ar0A = ar0A_real + np.random.normal(0, Constants.ERROR_AR_ANGLE)
        
        ar1d_real = None
        ar1d = None
        ar1A_real = None
        ar1A = None
        if random.random() < Constants.P_AR_OBSERVED:
            ar1d_real = math.sqrt((x_pos_real - Constants.arx1)**2 + (y_pos_real - Constants.ary1)**2)
            ar1d = ar1d_real + np.random.normal(0, Constants.ERROR_AR)

            temp_angle = np.arctan2(Constants.arx1 - x_pos_real[0], Constants.ary1 - y_pos_real[0])
            #ar1A_real = temp_angle if temp_angle >= 0 else temp_angle + 2*math.pi

            #ar1A = ar1A_real + np.random.normal(0, Constants.ERROR_AR_ANGLE)

        ar2d_real = None
        ar2d = None
        ar2A_real = None
        ar2A = None
        if random.random() < Constants.P_AR_OBSERVED:
            ar2d_real = math.sqrt((x_pos_real - Constants.arx2)**2 + (y_pos_real - Constants.ary2)**2)
            ar2d = ar2d_real + np.random.normal(0, Constants.ERROR_AR)

            temp_angle = np.arctan2(Constants.arx2 - x_pos_real[0], Constants.ary2 - y_pos_real[0])
            #ar2A_real = temp_angle if temp_angle >= 0 else temp_angle + 2*math.pi

            #ar2A = ar2A_real + np.random.normal(0, Constants.ERROR_AR_ANGLE)


        return np.array([x_pos_real, y_pos_real, theta_pos_real, [ar0d_real], [ar1d_real], [ar2d_real], [ar0A_real], [ar1A_real], [ar2A_real]]), np.array([x_pos, y_pos, theta_pos, [ar0d], [ar1d], [ar2d], [ar0A], [ar1A], [ar2A]])   

if __name__ == '__main__':
    error = []
    #for i in range(0, 100, 5):
    Constants.P_AR_OBSERVED = 100/100
    EKF = ekf(10, Constants.R, Constants.Q)
    est_data, true_data, gps_data = EKF.runFilter()
    plt.figure()
    plt.plot(gps_data[0], gps_data[1])
    plt.plot(true_data[0], true_data[1])
    plt.title("True position (orange) and GPS measurement (blue)")
    plt.figure()
    plt.plot(est_data[0], est_data[1])
    plt.plot(true_data[0], true_data[1])
    plt.title("True position (orange) and Kalman filter position estimate (blue)")

    #print("Average error per time-step with position and AR tag distances: ")



    # compute some form of the RMSE for the kalman estimate and for the GPS only estimate. 
    avg_error_kalman = (np.average(((np.array(est_data[0]) - np.squeeze(np.array(true_data[0])).astype(float))**2)) + np.average(((np.array(est_data[1]) - np.squeeze(np.array(true_data[1])).astype(float))**2)))/2

    avg_error_gps = (np.average(((np.array(gps_data[0]).astype(float) - np.array(true_data[0]).astype(float))**2)) + np.average(((np.array(gps_data[1]).astype(float) - np.array(true_data[1]).astype(float))**2)))/2
    error.append(avg_error_kalman)

    #print("Kalman, no AR: ", avg_error_kalman)
    #print("Pose only: ", avg_error_gps)
