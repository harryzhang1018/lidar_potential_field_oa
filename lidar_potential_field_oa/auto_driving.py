# BSD 3-Clause License
#
# Copyright (c) 2022 University of Wisconsin - Madison
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#
###############################################################################
## Author: Harry Zhang
###############################################################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from chrono_ros_interfaces.msg import DriverInputs as VehicleInput
from chrono_ros_interfaces.msg import Body
from nav_msgs.msg import Path
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
import csv 
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from keras_core.models import load_model
class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # update frequency of this node
        self.freq = 10.0

        # READ IN SHARE DIRECTORY LOCATION
        package_share_directory = get_package_share_directory('lidar_potential_field_oa')
        # initialize control inputs
        self.steering = 0.0
        self.throttle = 0.0
        self.braking = 0.0
        # initialize control inputs
        self.steering_flw = 0.0
        self.throttle_flw = 0.8
        # initialize vehicle state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        
        # initialize second vehicle state
        self.x_1 = 0.0
        self.y_1 = 0.0
        self.theta_1 = 0.0
        self.v_1 = 0.0

        # data that will be used by this class
        self.state = Body()
        self.path = Path()
        self.go = False
        self.vehicle_cmd = VehicleInput()
        self.lidar_data = LaserScan()
        self.tracking_model = load_model('/home/art/art/workspace/src/lidar_potential_field_oa/lidar_potential_field_oa/tracking_nn/nn_models_lib/single_speed_MPC_IL_NN.keras')
        self.file = open("/home/art/art/workspace/src/lidar_potential_field_oa/lidar_potential_field_oa/paths/path1.csv")
        self.get_logger().info("opened file: %s" % self.file)
        self.ref_traj = np.loadtxt(self.file,delimiter=",")
        self.lookahead = 1.0
        self.intensity = []
        self.values = []
        self.reduced_lidar_data = []
        self.detect_obs = False
        # publishers and subscribers
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        self.sub_state = self.create_subscription(Body, '/chrono_ros_node/output/vehicle/state', self.state_callback, qos_profile)
        self.pub_vehicle_cmd = self.create_publisher(VehicleInput, '/chrono_ros_node/input/driver_inputs', 10)
        self.sub_PCdata = self.create_subscription(LaserScan,'/chrono_ros_node/output/lidar_2d/data/laser_scan',self.lidar_callback,qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)

    # function to process data this class subscribes to
    def state_callback(self, msg):
        # self.get_logger().info("Received '%s'" % msg)
        self.go = True
        self.state = msg
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        #convert quaternion to euler angles
        e0 = msg.pose.orientation.x
        e1 = msg.pose.orientation.y
        e2 = msg.pose.orientation.z
        e3 = msg.pose.orientation.w
        self.theta = np.arctan2(2*(e0*e3+e1*e2),e0**2+e1**2-e2**2-e3**2)
        self.v = np.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
        #self.get_logger().info("(x, y, theta, v): (%s,%s,%s,%s)" % (self.x, self.y ,self.theta,self.v))
        
    def lidar_callback(self,msg):
        
        #self.get_logger().info("received lidar data")
        self.lidar_data = msg
        self.raw_lidar_data = msg.ranges
        self.reduced_lidar_data = self.reduce_lidar()
    
    def reduce_lidar(self):
        reduced_lidar_data = [30.0 if x == 0.0 else x for x in self.raw_lidar_data]
        reduce_chunk = 10
        reduced_lidar_data = [min(reduced_lidar_data[i:i+reduce_chunk]) for i in range(0,len(reduced_lidar_data),reduce_chunk)]
        return reduced_lidar_data
    
    def distance_based_intensity(self):
        intensity = [0.0]*len(self.reduced_lidar_data)
        dis_threshold = 4.0
        for i in range(len(self.reduced_lidar_data)):
            if self.reduced_lidar_data[i] < dis_threshold:
                intensity[i] = 1.0 / abs(self.reduced_lidar_data[i])
            else:
                intensity[i] = 0.0
        return intensity
    
    def values_function(self):
        alpha = 0.4
        length = len(self.reduced_lidar_data)
        values = [0.0]*length
        pi_array = np.linspace(0,np.pi,length)
        midpoint = length // 2

        # Compute values for the first half using function 1
        for i in range(midpoint):
            values[i] = - alpha * (pi_array[i])**2
        # Compute values for the second half using function 2
        for i in range(midpoint, length):
            values[i] = -alpha * (np.pi - pi_array[i])**2

        return values
    
    def steering_function(self):
        length_array = np.array(self.reduced_lidar_data)
        min_index = np.argmin(length_array)
        steering = self.intensity[min_index] * self.values[min_index]
        if abs(steering) > 1e-5:
            self.detect_obs = True
        else:
            self.detect_obs = False
        return steering
    
        
        
    def error_state(self):
        x_current = self.x
        y_current = self.y
        theta_current = self.theta
        v_current = self.v
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(self.ref_traj[:,1])))
        for i in range(len(self.ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*self.lookahead-self.ref_traj[i][0])**2+(y_current+np.sin(theta_current)*self.lookahead-self.ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(self.ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, ref_state_current[3]-v_current]

        return error_state

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        if(not self.go):
            return
        ## get error state
        #e_flw = self.follow_error()
        e = self.error_state()
        error_state = np.array(e)
        ctrl_input = self.tracking_model.predict(np.array([error_state]))
        self.throttle = ctrl_input[0][0]
        steering_flw = ctrl_input[0][1]
        # do a steering correction based on lidar data
        self.intensity = self.distance_based_intensity()
        self.values = self.values_function()
        steering_avoid = self.steering_function()
        
        ## choose the steering method
        if self.detect_obs:
            steering = steering_avoid
            self.get_logger().info("Avoiding obstacle")
            self.get_logger().info("intentisy: %s" % self.intensity)
            self.get_logger().info("values: %s" % self.values)
            self.get_logger().info("steering_avoid: %s" % steering_avoid)
        else:
            self.get_logger().info("Tracking reference trajectory")
            steering = steering_flw
        ## smooth steering
        delta_steering = steering - self.steering
        if abs(delta_steering) > 0.1:
            self.steering = self.steering + 0.1 * delta_steering / abs(
                delta_steering
            )
        else:
            self.steering = steering
        ### for vehicle one
        msg = VehicleInput()
        msg.steering = np.clip(self.steering, -1.0, 1.0)
        msg.throttle = np.clip(self.throttle, 0, 1)
        msg.braking = np.clip(self.braking, 0, 1)
        # self.get_logger().info("sending vehicle inputs: %s" % msg)
        self.pub_vehicle_cmd.publish(msg)


        # self.get_logger().info('Inputs %s' % self.recorded_inputs[0,:])

        # self.get_logger().info('Inputs from file: (t=%s, (%s,%s,%s)),' % (t,self.throttle,self.braking,self.steering))

def main(args=None):
    rclpy.init(args=args)
    control = ControlNode()
    rclpy.spin(control)

    control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()