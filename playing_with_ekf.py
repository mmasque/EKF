#! /usr/bin/env python
"""
Extended kalman filter (EKF) localization
Based on ekf code by: Atsushi Sakai (@Atsushi_twi)
Adapted for NovaRover by Jack McRobbie, Cheston Chow and Peter Shi
"""
import tools
import ros
import math
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

from common.msg import DriveCmd
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry

#import rospy
import tf
from sensor_msgs.msg import Imu

# imports for the GPSVisualiser
#import sys
#import std_msgs.msg
#from sensor_msgs.msg import PointCloud2, PointField
#from sensor_msgs import point_cloud2

from std_msgs.msg import Header


class ekf:
    def __init__(self, hz):

        self.Q = np.diag([
            0.3,  # variance of location on x-axis
            0.3,  # variance of location on y-axis
            np.deg2rad(25.0),  # variance of yaw angle
            0.5  # variance of velocity
        ]) ** 2  # predict state covariance

        self.R = np.diag([1.3, 1.3, 0.05]) ** 2

        self.dt = 1.0/hz
        self.hz = hz
        self.GPSVisualiser = GPSVisualiser()
        self.odomPublisher = odomPublisher()

    def motion_model(self, x, u):
        """
        :param x: [x, y, yaw, v]
        :param u: [[v]
                [yawrate]]
        :return:
        """
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                      [self.dt * math.sin(x[2, 0]), 0],
                      [0.0, self.dt],
                      [1.0, 0.0]])
        x1 = np.matmul(F, x)
        x2 = np.matmul(B, u.T)
        # updates the x, y and yaw according to the motion model
        return x1+x2

    def observation_model(self, x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        z = np.matmul(H, x)

        return z

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

    def jacobH(self, x):
        # Jacobian of Observation Model
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        return jH

    def ekf_estimation(self, xEst, PEst, z, u):
        """
        :param xEst: The estimated state of the rover
        :param PEst: Covariance of the state
        :param z: GPS with noise
        :param u: Control input with noise
        :return: xEst and PEst
        """

        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacobF(xPred, u)
        
        
        PPred = np.matmul(np.matmul(jF, PEst),jF.T) + self.Q

        #  Update
        jH = self.jacobH(xPred)
        zPred = self.observation_model(xPred)
        y = z - zPred.T

        S = np.matmul(np.matmul(jH, PPred), jH.T) + self.R
        K = np.matmul(np.matmul(PPred, jH.T), np.linalg.inv(S))

        tmp = np.matmul(K, y.T)

        xEst = xPred + tmp
        PEst = np.matmul((np.eye(len(xEst)) - np.matmul(K, jH)), PPred)

        return xEst, PEst

    def runFilter(self, visualizeGps=False):
        x = np.zeros((4, 1))
        p = np.eye(4)
        rate = rospy.Rate(self.hz)
        sensors = ekfSensorManager()
        rospy.loginfo("Running at %.2f hz...", self.hz)

        while not rospy.is_shutdown():
            rospy.loginfo("Running...")

            x_pos = sensors.gps[0][0]
            y_pos = sensors.gps[0][1]
            theta = sensors.yaw

            self.odomPublisher.setState(x[0], x[1], x[2], x[3])
            self.odomPublisher.publishOdom()

            z = np.array([x_pos, y_pos, theta])
            u = sensors.u

            x, p = self.ekf_estimation(x, p, z, u)

            # send location data to the GPSVisualiser for it to publish to the point cloud
            if visualizeGps:
                self.GPSVisualiser.publish_GPS_to_point_cloud([x_pos, y_pos])
            rate.sleep()


class ekfSensorManager:
    def __init__(self):
        self.gpsTopicName = "/fix"  # Or something like that
        self.gpsType = NavSatFix

        self.drivecmdName = "/cmd_vel"  # Or something like that
        self.driveCommandType = Twist

        self.imuTopicName = '/imu'
        self.imuTopicType = Imu

        self.gpsSub = rospy.Subscriber(
            self.gpsTopicName, self.gpsType, self.gpsCallback)

        self.driveCmdSub = rospy.Subscriber(
            self.drivecmdName, self.driveCommandType, self.driveCmdCallback)

        self.imuSub = rospy.Subscriber(
            self.imuTopicName, self.imuTopicType, self.imuCallback
        )
        self. u = np.zeros([1, 2])
        self.gps = np.zeros([1, 2])

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.firstGpsRecieved = False

        self.originGps = [0.0, 0.0]

    def imuCallback(self, imu):
        quart = imu.orientation
        [yaw, pitch, roll] = tools.quaternion_to_euler(quart.x,
                                                       quart.y,
                                                       quart.z,
                                                       quart.w)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        return

    def driveCmdCallback(self, data):
        self.u[0, 0] = data.linear.x
        self.u[0, 1] = data.angular.z
        return

    def gpsCallback(self, data):
        latitude = data.latitude
        longitude = data.longitude
        x, y = self.latlong2pos(latitude, longitude)
        self.gps[0][0] = x
        self.gps[0][1] = y
        return

    def latlong2pos(self, lat, longitude):

        if self.firstGpsRecieved is False:
            self.firstGpsRecieved = True
            self.originGps[0] = lat
            self.originGps[1] = longitude
            return 0.0, 0.0
        else:
            '''
            Using Y = north, X = east, z = up coordinates.
            Computing x y distance from starting point.
            '''
            dlong = longitude-self.originGps[1]
            dlat = lat - self.originGps[0]
            earthRadius = 6378100
            x = 2.0*earthRadius*math.sin(math.radians(dlong/2.0))
            y = 2.0*earthRadius*math.sin(math.radians(dlat/2.0))
            return x, y


class odomPublisher:
    def __init__(self, topicName="/ekf_odom"):
        self.topicName = topicName
        self.topicType = Odometry
        self.odomPublisher = rospy.Publisher(
            self.topicName, self.topicType, queue_size=50)
        self.odomBroadcaster = tf.TransformBroadcaster()

    def setState(self, x, y, theta, v):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

    def publishOdom(self):
        time = rospy.Time.now()
        odomQuaternion = tf.transformations.quaternion_from_euler(
            0.0, 0.0, self.theta)
        self.odomBroadcaster.sendTransform((self.x, self.y, 0.0),
                                           odomQuaternion, time, "odom", "base_link")
        odom = Odometry()
        odom.header.stamp = time
        odom.header.frame_id = "base_link"
        odom.pose.pose = Pose(Point(self.x, self.y, 0.0), Quaternion(*odomQuaternion)
                              )
        odom.child_frame_id = "odom"
        vx = self.v * math.sin(self.theta)
        vy = self.v * math.cos(self.theta)
        odom.twist.twist = Twist(Vector3(vx, vy, 0.0), Vector3(0.0, 0.0, 0.0))
        self.odomPublisher.publish(odom)


class GPSVisualiser:

    def __init__(self):
        self.pcl2Name = "/gps_clouds"
        self.pcl2Type = PointCloud2
        self.points = []
        self.pcl2_pub = rospy.Publisher(
            self.pcl2Name, self.pcl2Type, queue_size=10)

    def publish_GPS_to_point_cloud(self, cloud_point):
        # header
        # make it 2d (even if height will be 1)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  ]
        new_points = [float(cloud_point[0]), float(cloud_point[1]), 0.0]
        self.points.append(new_points)
        header = Header()
        header.frame_id = "/base_link"
        pc2 = point_cloud2.create_cloud(header, fields, self.points)
        self.pcl2_pub.publish(pc2)


if __name__ == '__main__':
    rospy.init_node('EKF', anonymous=False)
    runnable = ekf(10)
    runnable.runFilter(True)