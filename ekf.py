import math
import numpy as np
import matplotlib.pyplot as plt
import tf
import random

# 1. comment this out to use the layout from the tag_layouts.py file
NORTH = math.pi
EAST = -math.pi / 2
SOUTH = 0
WEST = 3 * math.pi / 2

layout_field_test = {
    0: [(NORTH, 5, 5), (EAST, 5, 5), (SOUTH, 5, 5), (WEST, 5, 5)],
    1: [(EAST, 0, 10), (SOUTH, 0, 10)],
    2: [(SOUTH, 9, 10)],
    3: [(EAST, 4, 2)],
    4: []
}
# end commented block 

# 2. UNCOMMENT the 2 lines below to get the tag_layouts.py file layout
# from tag_layouts import layout_field_test
# ar_dict = layout_field_test


ar_dict = {}
for i in layout_field_test.keys():
    if layout_field_test[i]:
        ar_dict[i] = (layout_field_test[i][0][1], layout_field_test[i][0][2])


arx0 = 5
ary0 = 5
arx1 = 0
ary1 = 10
arx2 = 9
ary2 = 10
P_AR_OBSERVED = 0.7
ERROR_AR = 0.1
class ekf:
    def __init__(self, hz):

        self.Q = np.diag([
            0.3,  # variance of location on x-axis | this is really the SD, and same for below, no?
            0.3,  # variance of location on y-axis
            0,#np.deg2rad(25.0),  # variance of yaw angle
            0.5  # variance of velocity
        ]) ** 2  # predict state covariance

        self.R = np.diag([1.3, 1.3, 0.5] + [ERROR_AR for _ in ar_dict.values()])

        #self.R = np.diag([1.3, 1.3, 0.5, ERROR_AR, ERROR_AR, ERROR_AR]) ** 2 # Assuming that the SD is in Metres? 

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
        return x1+x2

    def simple_motion_model(self, x, u):
        """ Predict then next state of the system given the past state and a model
        :param x: [x, y, yaw, v],
        :param u: [[v]
                [yawrate]]
        :return:
        """
        retval = np.array([
            [x[0,0] + math.sin(x[2,0])* u[0,0] * self.dt],
            [x[1,0] + math.cos(x[2,0])* u[0,0] * self.dt],
            [x[2,0] + u[0,1]*self.dt],
            [u[0,0]]
        ])
        return retval
    
    def observation_model(self, x, z_available):
        # what comes in from the observation? a 4d vector somehow, of which we need the first three elements.
        """
        z = np.array([[x[0,0], x[1,0], x[2,0], 
            math.sqrt((x[0,0] - arx0)**2 + (x[1,0] - ary0)**2), 
            math.sqrt((x[0,0] - arx1)**2 + (x[1,0] - ary1)**2),
            math.sqrt((x[0,0] - arx2)**2 + (x[1,0] - ary2)**2)]])
        """
        z0 =  np.array([[x[0,0], x[1,0], x[2,0]]])
        zar = np.array([[math.sqrt((x[0,0] - arx)**2 + (x[1,0] - ary)**2) \
             for _,(arx,ary) in sorted(zip(ar_dict.keys(),ar_dict.values()))]])
        
        z = np.concatenate([z0, zar], axis=1)
        return z[:, z_available].T
    
    def jacobF(self, x, u):
        """
        Jacobian of Motion Model
        motion model
        x_{t+1} = x_t+v*dt*sin(yaw)
        y_{t+1} = y_t+v*dt*cos(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = v*dt*cos(yaw)
        dx/dv = dt*sin(yaw)
        dy/dyaw = -v*dt*sin(yaw)
        dy/dv = dt*cos(yaw) 
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
            [0.0, 1.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
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
        
        
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [(x[0,0]-arx0)/math.sqrt((x[0,0]-arx0)**2 + (x[1,0]-ary0)**2),(x[1,0]-ary0)/math.sqrt((x[0,0]-arx0)**2 + (x[1,0]-ary0)**2),0,0], #AR tag 0 d 
            [(x[0,0]-arx1)/math.sqrt((x[0,0]-arx1)**2 + (x[1,0]-ary1)**2),(x[1,0]-ary1)/math.sqrt((x[0,0]-arx1)**2 + (x[1,0]-ary1)**2),0,0], #AR tag 1 d
            [(x[0,0]-arx2)/math.sqrt((x[0,0]-arx2)**2 + (x[1,0]-ary2)**2),(x[1,0]-ary2)/math.sqrt((x[0,0]-arx2)**2 + (x[1,0]-ary2)**2),0,0] #AR tag 2 d

        ])
        """
        jH0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        jHAR = np.array([
            [(x[0,0]-arx)/math.sqrt((x[0,0]-arx)**2 + (x[1,0]-ary)**2), \
                (x[1,0]-ary)/math.sqrt((x[0,0]-arx)**2 + (x[1,0]-ary)**2),0,0] \
                for _,(arx,ary) in sorted(zip(ar_dict.keys(),ar_dict.values()))
        ])
        jH = np.concatenate([jH0,jHAR])
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
        xPred = self.simple_motion_model(xEst, u)  # here we predict the state based on previous state.
        #make sure the predicted theta is bounded
        xPred[2] = xPred[2] % (math.pi * 2)

        jF = self.jacobF(xPred, u)  # here we get the jacobian of the model function, F. 

        PPred = np.matmul(np.matmul(jF, PEst),jF.T) + self.Q    #Here we calculate the covariance of the prediction xPred. Great. 

        #  Update

        # find out which measurements are not None
        z_available = np.array([i for i in range(z.size) if z[0,i] is not None])

        jH = self.jacobH(xPred, z_available) # Here we get the jacobian of the state -> measurement function, which in this case isn't really a jacobian but a constant. 
        zPred = self.observation_model(xPred, z_available)   # Here we predict the measurements based on our xPred. 
        
        # make sure the measurement thetas are bounded properly between 0 and 2pi
        zPred[2,0] = zPred[2,0] % (2 * math.pi)    
        z[0,2] = z[0,2] % (2 * math.pi)

        y = (z[0,z_available] - zPred.T).astype(np.float) # Seems to be the difference between the original measurement and the predicted measurement. 
        y[0,2] = (y[0,2] + math.pi) % (2*math.pi) - math.pi # make sure to take the smallest distance for theta
        S = np.matmul(np.matmul(jH, PPred), jH.T) + self.R[z_available[:, None], (z_available)]  #   The covariance of the prediction xPred, i.e. the model's guess of the state, is transformed into the covariance of the measurement prediction, plus the covariance of error in measurement. 
        K = np.matmul(np.matmul(PPred, jH.T), np.linalg.inv(S)) #This is just the Kalman gain. 

        tmp = np.matmul(K, y.T) # and we multiply by the difference between z observed and the predicted z, per the equation to yield the updated estimate. Not sure why that's not just done in the line below. 
        
        xEst = xPred + tmp  # completing the formula
        xEst[2] = xEst[2] % (math.pi * 2)
        PEst = np.matmul((np.eye(len(xEst)) - np.matmul(K, jH)), PPred) # formula for the covariance update, but factorised for simplicity.  

        return xEst, PEst   

    def runFilter(self, visualizeGps=False):
        x = np.zeros((4, 1))
        x_real = np.zeros((4, 1))
        p = np.eye(4)
        rate = 10

        estimate_data = ([],[])
        true_data = [[],[]]
        gps_data = [[], []]
        u = np.array([[1000,np.deg2rad(10)/self.dt]])
        for _ in range(1000):
            x_real, x_update = self.simulateFakeDataMeasurement(x_real[0],x_real[1],x_real[2], u)
            z = x_update.T

            x, p = self.ekf_estimation(x, p, z, u)

            u = np.array([[u[0,0]+0.2,np.deg2rad(10)/self.dt]])

            estimate_data[0].append(x[0,0])
            estimate_data[1].append(x[1,0])
            true_data[0].append(x_real[0])
            true_data[1].append(x_real[1])
            gps_data[0].append(x_update[0])
            gps_data[1].append(x_update[1])

        return estimate_data, true_data, gps_data

    def simulateFakeDataMeasurement(self,x,y,theta,u):
        """
        I want the rover to move in circles.
        """
        x_pos_real = x + u[0,0] * self.dt * math.sin(theta) + np.random.normal(0, 0.3)
        x_pos = x_pos_real + np.random.normal(0, 1.3)# previous x plus a movement plus noise (from measurement)
        y_pos_real = y + u[0,0] * self.dt * math.cos(theta) + np.random.normal(0, 0.3)
        y_pos = y_pos_real + np.random.normal(0, 1.3)# previous y plus a movement plus noise (from measurement)
        
        unbounded_theta_pos_real = (theta + u[0,1] * self.dt)
        unbounded_theta_pos = unbounded_theta_pos_real + np.random.normal(0,0.5)

        theta_pos_real = unbounded_theta_pos_real % (2*math.pi) # increases by 10 degrees 
        theta_pos = unbounded_theta_pos % (2*math.pi)

        ar0d_real = None
        ar0d = None
        if random.random() < P_AR_OBSERVED:
            ar0d_real = math.sqrt((x_pos_real - arx0)**2 + (y_pos_real - ary0)**2)
            ar0d = ar0d_real + np.random.normal(0, ERROR_AR)
        
        ar1d_real = None
        ar1d = None
        if random.random() < P_AR_OBSERVED:
            ar1d_real = math.sqrt((x_pos_real - arx1)**2 + (y_pos_real - ary1)**2)
            ar1d = ar1d_real + np.random.normal(0, ERROR_AR)

        ar2d_real = None
        ar2d = None
        if random.random() < P_AR_OBSERVED:
            ar2d_real = math.sqrt((x_pos_real - arx2)**2 + (y_pos_real - ary2)**2)
            ar2d = ar2d_real + np.random.normal(0, ERROR_AR)
        

        return np.array([x_pos_real, y_pos_real, theta_pos_real, [ar0d_real], [ar1d_real], [ar2d_real]]), np.array([x_pos, y_pos, theta_pos, [ar0d], [ar1d], [ar2d]])   

if __name__ == '__main__':
    error = []
    #for i in range(0, 100, 5):
    i = 100
    P_AR_OBSERVED = i / 100
    EKF = ekf(10)
    est_data, true_data, gps_data = EKF.runFilter()
    plt.figure()
    plt.plot(gps_data[0], gps_data[1])
    plt.plot(true_data[0], true_data[1])
    plt.title("True position (orange) and GPS measurement (blue)")
    plt.figure()
    plt.plot(est_data[0], est_data[1])
    plt.plot(true_data[0], true_data[1])
    plt.title("True position (orange) and Kalman filter position estimate (blue)")

    print("Average error per time-step with position and AR tag distances: ")



        # compute some form of the RMSE for the kalman estimate and for the GPS only estimate. 
    avg_error_kalman = (np.average(((np.array(est_data[0]) - np.squeeze(np.array(true_data[0])).astype(np.float))**2)) + np.average(((np.array(est_data[1]) - np.squeeze(np.array(true_data[1])).astype(np.float))**2)))/2

    avg_error_gps = (np.average(((np.array(gps_data[0]).astype(np.float) - np.array(true_data[0]).astype(np.float))**2)) + np.average(((np.array(gps_data[1]).astype(np.float) - np.array(true_data[1]).astype(np.float))**2)))/2
    error.append(avg_error_kalman)
    print("Kalman, + AR: ", avg_error_kalman)
    print("Pose only: ", avg_error_gps)