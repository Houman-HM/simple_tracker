#!/usr/bin/env python3
import rospy
import message_filters
from nav_msgs.msg import Odometry
import rospkg
import sys
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rospkg
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import threading

import queue
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import bernstein_coeff_order10_arbitinterval
import tracker_module as mpc_module
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import rospkg


robot_cmd_publisher = None

is_received = False
robot_cmd_publisher = None
robot_traj_publisher = None
robot_traj_marker = None
line_to_track_marker = None
line_to_track_publisher = None
global_path_publisher = None
global_path_marker = None

obstacle_points = jnp.zeros((200,2))

x_obs_pointcloud = np.ones((200,1)) * 100
y_obs_pointcloud = np.ones((200,1)) * 100

odom_mutex = threading.Lock()

x_drone_traj_global =  None
y_drone_traj_global = None
x_tracking_traj_global  = None
y_tracking_traj_global = None

x_tracking_traj_main_global  = None
y_tracking_traj_global_global = None


trajectory_updated = False

publish_traj_mutex = threading.Lock()
def createLineMarker(id = 1, frame_id = 'base', scale = [0.1,0,0], color = [1,0,0,1]):

    marker = Marker() 
    marker.id = id
    marker.header.frame_id = frame_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.pose.orientation.w = 1.0
    marker.color.a = color[0]
    marker.color.r = color[1]
    marker.color.g = color[2]
    marker.color.b = color[3]
    
    return marker

def odomCallback(robot_odom):

    # print("In the call back")
    global is_received, robot_pose_vel, target_pose_vel, obs_1_pose_vel, obs_2_pose_vel, obs_3_pose_vel, odom_mutex
    odom_mutex.acquire()
    robot_orientation_q = robot_odom.pose.pose.orientation
    robot_orientation_list = [robot_orientation_q.x, robot_orientation_q.y, robot_orientation_q.z, robot_orientation_q.w]

    (robot_roll, robot_pitch, robot_yaw) = euler_from_quaternion (robot_orientation_list)

    robot_pose_vel = [robot_odom.pose.pose.position.x, robot_odom.pose.pose.position.y, robot_yaw, 
                    robot_odom.twist.twist.linear.x, robot_odom.twist.twist.linear.y, robot_odom.twist.twist.angular.z]
    odom_mutex.release()


def publishTrajectories():

    global robot_traj_maker, robot_traj_publisher, line_to_track_marker, x_drone_traj_global, y_drone_traj_global,\
                x_tracking_traj_global, y_tracking_traj_global, trajectory_updated, x_tracking_traj_main_global, y_tracking_traj_main_global, publish_traj_mutex,\
                global_path_publisher, global_path_marker
    while (True):
        if (trajectory_updated):
            publish_traj_mutex.acquire()
            x_drone_traj, y_drone_traj, x_tracking_traj, y_tracking_traj = x_drone_traj_global, y_drone_traj_global,\
                x_tracking_traj_global, y_tracking_traj_global
            publish_traj_mutex.release()
            for i in range (x_drone_traj.shape[0]):
                drone_point = Point()
                tracking_point = Point()
                global_path_point = Point()

                drone_point.x = x_drone_traj[i]
                drone_point.y = y_drone_traj[i]
                tracking_point.x = x_tracking_traj[0,i]
                tracking_point.y = y_tracking_traj[0,i]
                global_path_point.x = x_tracking_traj_main_global[i * 100]
                global_path_point.y = y_tracking_traj_main_global[i* 100]


                line_to_track_marker.points.append(tracking_point)
                robot_traj_marker.points.append(drone_point)
                global_path_marker.points.append(global_path_point)


            
            robot_traj_publisher.publish(robot_traj_marker)
            line_to_track_publisher.publish(line_to_track_marker)
            global_path_publisher.publish(global_path_marker)
            rospy.sleep(0.001)
            
            robot_traj_marker.points = []
            line_to_track_marker.points = []
            global_path_marker.points = []

        rospy.sleep(0.001)

def generate_sinusoidal_trajectory(total_distance=10, num_points=10000):
  # Create an array of time points
  t = jnp.linspace(0, 2*np.pi, num_points)
  
  # Calculate the corresponding positions
  x = total_distance * jnp.sin(t)
  y = total_distance * jnp.cos(t)
  
  return t, x, y



def mpc():

    global is_received, robot_pose_vel, robot_cmd_publisher, x_drone_traj_global, y_drone_traj_global,\
                x_tracking_traj_global, y_tracking_traj_global, x_tracking_traj_main_global, y_tracking_traj_main_global, trajectory_updated,publish_traj_mutex
    rospy.loginfo("MPC thread started sucessfully!")

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("cem_vis_planner")

    v_max = 1.5 
    a_max = 1.5
    num_batch_projection = 500
    num_batch_cem = 100
    num_target = 1
    num_workspace = 1
    ellite_num_shift = 400

    rho_ineq = 1.0 
    rho_projection =  1.0
    rho_target = 1.0
    rho_workspace = 1.0
    maxiter_projection = 10
    maxiter_cem = 2
    maxiter_mpc = 5000
    weight_smoothness = 0.01
    weight_smoothness_psi = 0.1

    d_min_target = 0.0
    d_max_target = 0.05
    d_avg_target = (d_min_target+d_max_target)/2.0

    ############# parameters

    t_fin = 5.0
    num = 100
    tot_time = np.linspace(0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)
            
    P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    nvar = np.shape(P)[1]

    tot_time_jax = jnp.asarray(tot_time)

    ###################################
    t_update = 0.05
    num_up = 200
    dt_up = 0.01#t_fin/num_up
    tot_time_up = np.linspace(0, t_fin, num_up)
    tot_time_copy_up = tot_time_up.reshape(num_up, 1)

    P_up, Pdot_up, Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

    P_up_jax = jnp.asarray(P_up)
    Pdot_up_jax = jnp.asarray(Pdot_up)
    Pddot_up_jax = jnp.asarray(Pddot_up)
    


    ########################################

    x_init =  robot_pose_vel[0]
    vx_init = 0.0
    ax_init = 0.0

    y_init =  robot_pose_vel[1]
    vy_init = 0.0
    ay_init = 0.0


    x_target_init = 0
    y_target_init = 0

    vx_target = 1
    vy_target = 0.0


    x_target = x_target_init+vx_target*tot_time
    y_target = y_target_init+vy_target*tot_time

    t, x_target_trajectory, y_target_trajectory = generate_sinusoidal_trajectory()

    x_tracking_traj_main_global = np.asarray(x_target_trajectory)
    y_tracking_traj_main_global = np.asarray(y_target_trajectory)


    x_target = x_target.reshape(1, num)
    y_target = y_target.reshape(1, num)


    x_target_fin = x_target[0, -1]
    y_target_fin = y_target[0, -1]

    
    ###############################################################3
    A = np.diff(np.diff(np.identity(num), axis = 0), axis = 0)

    temp_1 = np.zeros(num)
    temp_2 = np.zeros(num)
    temp_3 = np.zeros(num)
    temp_4 = np.zeros(num)

    temp_1[0] = 1.0
    temp_2[0] = -2
    temp_2[1] = 1
    temp_3[-1] = -2
    temp_3[-2] = 1

    temp_4[-1] = 1.0


    x_workspace = 0*jnp.ones((num_workspace, num))
    y_workspace = 0*jnp.ones((num_workspace, num ))

    a_workspace = 1300.0
    b_workspace = 1300.0

    A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
    
    R = np.dot(A_mat.T, A_mat)
    mu = np.zeros(num)
    cov = np.linalg.pinv(R)

    ################# Gaussian Trajectory Sampling
    eps_k = np.random.multivariate_normal(mu, 0.001*cov, (num_batch_projection, ))
    
    goal_rot = -np.arctan2(y_target_fin-y_target_init, x_target_fin-x_target_init)
    
    x_init_temp = x_target_init*np.cos(goal_rot)-y_target_init*np.sin(goal_rot)
    y_init_temp = x_target_init*np.sin(goal_rot)+y_target_init*np.cos(goal_rot)


    x_fin_temp = x_target_fin*np.cos(goal_rot)-y_target_fin*np.sin(goal_rot)
    y_fin_temp = x_target_fin*np.sin(goal_rot)+y_target_fin*np.cos(goal_rot)


    x_interp = jnp.linspace(x_init_temp, x_fin_temp, num)
    y_interp = jnp.linspace(y_init_temp, y_fin_temp, num)

    x_guess_temp = jnp.asarray(x_interp+0.0*eps_k) 
    y_guess_temp = jnp.asarray(y_interp+eps_k)

    x_samples_init = x_guess_temp*jnp.cos(goal_rot)+y_guess_temp*jnp.sin(goal_rot)
    y_samples_init = -x_guess_temp*jnp.sin(goal_rot)+y_guess_temp*jnp.cos(goal_rot)

    x_samples_shift = x_samples_init[0:ellite_num_shift, :]
    y_samples_shift = y_samples_init[0:ellite_num_shift, :]
    
    ##############################################################

    occlusion_weight = 10000

    prob = mpc_module.batch_tracking(P, Pdot, Pddot, v_max, a_max, t_fin, num, num_batch_projection, 
                                                        num_batch_cem, tot_time, rho_ineq, maxiter_projection, rho_projection, rho_target, num_target, 
                                                        a_workspace, b_workspace, num_workspace, rho_workspace, maxiter_cem, d_min_target, d_max_target, 
                                                        P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight)

    lamda_x = jnp.zeros((num_batch_projection, nvar))
    lamda_y = jnp.zeros((num_batch_projection, nvar))

    d_a = a_max*jnp.ones((num_batch_projection, num))
    alpha_a = jnp.zeros((num_batch_projection, num))		

    d_v = v_max*jnp.ones((num_batch_projection, num))
    alpha_v = jnp.zeros((num_batch_projection, num))		

    alpha_workspace = jnp.zeros(( num_batch_projection, num_workspace*num))
    d_workspace = jnp.ones(( num_batch_projection, num_workspace*num))

    key = random.PRNGKey(0)

    alpha_init = np.arctan2(y_target_init - y_init, x_target_init - x_init)

    rospy.loginfo("Waiting for initial JAX compilation!")
    

    for i in range(0, maxiter_mpc):
        start_time = time.time()
        odom_mutex.acquire()
        jax_obstacle_points = jnp.asarray(obstacle_points)

        vx_target =1
        vy_target = 0
        alpha_init = 0#robot_pose_vel[2]
        odom_mutex.release()


        projected_points_on_tracking_line = (x_target_trajectory  - x_init)**2 + (y_target_trajectory - y_init)**2
        min_distance_index= jnp.argmin(projected_points_on_tracking_line)
        x_init_target = x_target_trajectory[min_distance_index]
        y_init_target = y_target_trajectory[min_distance_index]

        x_target = x_target_trajectory[i:i+prob.num*10:10].reshape(1,100)
        y_target = y_target_trajectory[i:i+prob.num*10:10].reshape(1,100)

        

        x_samples_init, y_samples_init = prob.compute_initial_samples(jnp.asarray(eps_k), x_target_init, y_target_init, x_target_fin, y_target_fin, x_samples_shift, y_samples_shift, ellite_num_shift, x_init, y_init)

        c_x_samples_init, c_y_samples_init, x_samples_init, y_samples_init = prob.compute_inital_guess( x_samples_init, y_samples_init)


        c_x_best, c_y_best, cost_track, x_best, y_best, alpha_v, d_v, alpha_a, d_a, \
             alpha_target, d_target, lamda_x, lamda_y, alpha_workspace, d_workspace, key, x_samples_shift, y_samples_shift = prob.compute_cem(key, x_init, vx_init, ax_init, y_init, vy_init, ay_init, alpha_a, d_a, alpha_v, d_v,
					                                x_target, y_target, lamda_x, lamda_y, x_samples_init, y_samples_init, x_workspace, y_workspace,
					                                alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init, vx_target, vy_target,
					                                d_avg_target, ellite_num_shift, jax_obstacle_points)

        vx_control_local, vy_control_local, ax_control, \
        ay_control, vangular_control, robot_traj_x, robot_traj_y, vx_control, vy_control= prob.compute_controls(c_x_best, c_y_best, dt_up, vx_target, vy_target, 
							                                                    t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                                                                                 x_target_init, y_target_init)


        if (i!=0):
            cmd = Twist()
            cmd.linear.x= vx_control
            cmd.linear.y= vy_control
            # cmd.angular.z = vangular_control
            robot_cmd_publisher.publish(cmd)
        time_taken = time.time() - start_time
        rospy.loginfo ("Time taken: %s", str(time_taken))

        odom_mutex.acquire()
        
        x_init = robot_pose_vel[0]
        y_init = robot_pose_vel[1]
        publish_traj_mutex.acquire()
        x_drone_traj_global, y_drone_traj_global, x_tracking_traj_global, y_tracking_traj_global, = np.asarray(x_best), np.asarray(y_best),\
            np.array(x_target), np.asarray(y_target)
        
        publish_traj_mutex.release()
        
        trajectory_updated = True

        vx_init = vx_control
        vy_init = vy_control

        ax_init = ax_control
        ay_init = ay_control

        x_target_fin = x_target[0,-1]
        y_target_fin = y_target[0,-1]
        odom_mutex.release()
    

if __name__ == "__main__":

	
    rospy.init_node('nn_mpc_node')
    rospack = rospkg.RosPack()
    robot_traj_marker = createLineMarker(color=[0.5,0,0,1])
    line_to_track_marker = createLineMarker(color=[0.5,0,0,0]) 
    global_path_marker = createLineMarker(color=[0.2,1,0,0])
    robot_cmd_publisher = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=10)

    robot_traj_publisher = rospy.Publisher('/robot_traj', Marker, queue_size=10)
    line_to_track_publisher = rospy.Publisher('/line_to_track', Marker, queue_size=10)
    global_path_publisher = rospy.Publisher('/global_path', Marker, queue_size=10)
    rospy.Subscriber("bebop/odom", Odometry, odomCallback)
    mpc_thread = threading.Thread(target=mpc)
    trajectory_publisher_thread = threading.Thread(target=publishTrajectories)
    mpc_thread.start()
    trajectory_publisher_thread.start()
    rospy.spin()


