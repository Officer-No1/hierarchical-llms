#! /usr/bin/env python
"""
Script for running multi-agent tracking simulation and visualization
Also used for defining ROS setup (to be added)

TrackerServer 类是一个综合性控制器，负责：
    1. 管理跟踪任务的执行
    2. 处理传感器数据
    3. 发布控制指令
    4. 可视化系统状态
    5. 根据不同实验模式切换工作流程

该服务器支持三种运行模式：
    1. 纯仿真模式：完全在软件环境中运行
    2. ROS仿真模式：使用ROS框架进行仿真
    3. ROS真实硬件模式：控制实际的机器人硬件
"""
import os
import rospy
import rospkg
import numpy as np
import functools

#### ros packages ####
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist 
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseStamped, Point
from llm_adaptive_server import LLMAdaptiveServer
from visualization_msgs.msg import Marker
#### local packages ####
from model.tracker_manger import TrackerManager
from model.config_loader import ConfigLoader
from utils.visualizer import Visualizer
from utils.visualizer_ros import VisualizerROS
import time
import yaml
import tf
import tf.transformations as tft
from nav_msgs.msg import Path


class TrackerServer:

    def __init__(self, exp_name):
        """
        初始化过程加载实验配置文件，创建配置加载器和跟踪管理器
        exp_name: .yaml config file to read
        """
        ######### general parameters #########
        try :
            path = rospkg.RosPack().get_path('tracker')
        except:
            #use relative path
            path = os.path.dirname(os.path.abspath(__file__)) + "/.."
        
        print("path is: ", path)
        #path = "/home/grasp/ma_ws/src/target_tracking/src/tracker"
        config_path = path + "/config/" + exp_name + ".yaml"
        if not os.path.exists(config_path):
            print("[tracker server]: Config file does not exist, the path is: ", config_path)
            return

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        print("========= loading config file from: ", config_path)

        self.config_loader = ConfigLoader(config)
        self.tracker_manager = TrackerManager(self.config_loader)
        self.exp = self.config_loader.exp
        self.frame_id = self.config_loader.frame_id
        self.dim = self.config_loader.dim

        self.target_ids = self.config_loader.targetID
        self.robot_ids = self.config_loader.robotID


        ######### save path #########
        self.save_path = path + "/results/" + str(self.tracker_manager.testID) + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ######### ROS parameters #########
        self.drone_cmd_pubs = []
        self.drone_odom_subs = []
        self.drone_vis_goal_pubs = []

        self.flags_pub = rospy.Publisher("/flags", Int8MultiArray, queue_size=10)
        ####### only for sim
        self.drone_vis_odom_pubs = []
        self.target_vis_odom_pubs = []

        ######### Experiment settings #########
        self.drone_ids  = self.config_loader.drone_ids
        self.car_ids = self.config_loader.car_ids
        self.global_origin  = np.array([0, 0, 0])
        self.drone_global_pos = np.zeros((len(self.drone_ids), 3))
        self.car_global_pos = np.zeros((len(self.car_ids), 3))

        self.drone_tfs = []
        self.detect_tfs = []
        tf = {"translation": [], "rotation": []}

        for i in range(len(self.drone_ids)):
            self.drone_tfs.append(tf)
            self.detect_tfs.append(tf)
        
        self.drones_waypoints_pubs = []
        self.apriltag_subs = []
        self.is_tf_ready = np.zeros(len(self.drone_ids))


        ############ visualization ############
        self.vis = Visualizer(self.save_path)
        self.vis_ros = None
        self.his_target_pos = []
        self.his_target_vel = []
        self.his_drone_pos = []
        self.his_drone_vel = []
        self.comm_dzones = []
        self.sens_dzones = []
        self.his_drone_cmd = []
        self.his_exitflag = []
        self.his_known_typeI_flags = []
        self.his_known_typeII_flags = []
        self.his_attacked_typeI_flags = []
        self.his_attacked_typeII_flags = []

        # run simulation

        if self.exp == "simulation":
            self.adapter = LLMAdaptiveServer(self.config_loader, path)
            self.tracker_manager.trac_prob.init_sim()
            self.simulation()
        elif self.exp == "ros simulation":
            self.tracker_manager.trac_prob.init_sim()
            self.ros_simulation()
        elif self.exp == "ros real":
            self.ros_real()
        else:
            print("Invalid experiment type")
            return
        
    def transform_point(self, point, trans, rot):
        rotation_matrix = tft.quaternion_matrix(rot)[:3, :3]
        point_transformed = np.dot(rotation_matrix, point) + np.array(trans)
        
        return point_transformed


    ####Odometry Callbacks####
    def drone_odom_callback(self, msg, index):
        """
        处理无人机的里程计数据，更新内部状态表示并发布可视化消息
        data: Crazyswarm GenericLogData
        id: int, id of the drone
        """
        time_stamp = msg.header.stamp
        ### need revision
        #index      = msg.frame_id 
        # extract the position
        odom       = msg.pose.pose.position
        x, y, z    = odom.x, odom.y, odom.z

        #print("index is {}, drone_pos is {}, time is {}".format(index, np.array([x, y, z]), time_stamp))
        
        # if self.drone_tfs[index]["translation"] != []:
        # drone_pos = self.transform_point(np.array([x, y, z]), 
        #                                 self.drone_tfs[index]["translation"], 
        #                                 self.drone_tfs[index]["rotation"])
        
        drone_pos = np.array([x, y, self.config_loader.robotHeights[index]])
        
        self.drone_global_pos[index] = drone_pos
        # update to the tracker manager
        #print("index is {}, drone_pos is {}, time is {}".format(index, drone_pos, time_stamp))
        self.tracker_manager.trac_prob.update_real_robot(drone_pos[:self.dim], index)
    
        self.vis_ros.publish_odom(self.drone_vis_odom_pubs[index], drone_pos, True)
        return
    
    
    ####Pose Callbacks####
    def apriltag_callback(self, msg, robot_index):
        '''
        回调函数处理AprilTag检测结果，用于在真实硬件实验中识别和跟踪目标
        '''
        time_stamp = msg.header.stamp
        #index      = msg.frame_id

        apriltags = msg.apriltags[0]
        tag_id = apriltags.id

        # if tag_id not in self.target_ids:
        #     return
        
        #find tag id index in the target_ids
        target_idx = -1
        print("the robot {} detected the tag {}".format(robot_index, tag_id))
        for i, id in enumerate(self.target_ids):
            if id == tag_id:
                target_idx = i
                break
        if target_idx == -1:
            return


        pose = msg.posearray.poses[0].position
        x, y, z = pose.x, pose.y, pose.z
        # if self.detect_tfs[robot_index]["translation"] != []:
        #     pose = msg.posearray.poses[0].position
        #     x, y, z = pose.x, pose.y, pose.z

        #     target_pos = self.transform_point(np.array([x, y, z]), 
        #                                     self.detect_tfs[robot_index]["translation"], 
        #                                     self.detect_tfs[robot_index]["rotation"])
            
        target_pos = np.array([x, y, 0.0])
        #print("index is {}, target_pos is {}, time is {}".format(robot_index, target_pos, time_stamp))

        self.tracker_manager.trac_prob.init_detect_target_odom(target_pos[:self.dim],
                                                                target_idx, robot_index)
        self.tracker_manager.trac_prob.update_detect_target_odom(target_pos[:self.dim], 
                                                                target_idx, robot_index)
        return
    

    def pub_waypoints(self, pos, index):
        '''
        将计算出的目标位置转换为ROS路径消息，发布给无人机控制系统
        '''
        
        # 创建带有高度的目标点
        goal = np.array([pos[0], pos[1], self.config_loader.robotHeights[index]])
        
        # 创建路径消息
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = rospy.Time.now()

        # 添加路径点
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = goal[0]
        pose.pose.position.y = goal[1]
        pose.pose.position.z = goal[2]
        pose.pose.orientation.w = 1.0

        path.poses.append(pose)

        # print(pos)
        # print(len(self.drones_waypoints_pubs))
        # print(index)

        # 发布路径点和可视化目标
        self.drones_waypoints_pubs[index].publish(path)
        self.vis_ros.publish_goal(goal, self.drone_vis_goal_pubs[index])
        return


    def weights_callback(self, msg):

        weights = np.array(msg.data).reshape(self.config_loader.N, self.config_loader.N)
        self.tracker_manager.trac_prob.weights = weights.copy() 
        return
    

    def assignment_matrix_callback(self, msg):

        assignment_matrix = np.array(msg.data).reshape(self.config_loader.N, self.config_loader.N)
        self.tracker_manager.trac_prob.assignment_matrix = assignment_matrix.copy()
        return



    def update_results(self, results):

        self.his_drone_pos.append(results["robot_pos"])
        self.his_drone_vel.append(results["robot_vel"])
        self.his_target_pos.append(results["target_pos"])
        self.his_drone_cmd.append(results["robot_cmd"])
        self.his_exitflag.append(results["exitflag"])
        self.his_known_typeI_flags.append(results["known_typeI_flags"])
        self.his_known_typeII_flags.append(results["known_typeII_flags"])
        self.his_attacked_typeI_flags.append(results["attacked_typeI_flags"])
        self.his_attacked_typeII_flags.append(results["attacked_typeII_flags"])

    def vis_sim(self):
        '''
        生成全面的可视化结果，包括地图、轨迹、攻击点和性能指标
        '''
        
        # 可视化地图和危险区域
        # load map
        print(" ========= loading map =========")
        self.vis.visualize_map(self.config_loader.x_bounds)
        self.vis.visualize_zones(self.config_loader.typeI_zones,
                                 self.config_loader.typeII_zones,
                                 self.his_known_typeI_flags[-1],
                                 self.his_known_typeII_flags[-1])

        # visualize results
        print(" ========= visualising results =========")
        # 可视化目标和机器人轨迹
        self.vis.visualize_target(self.his_target_pos, self.config_loader.targetID)
        self.vis.visualize_robot(self.his_drone_pos, self.config_loader.robotID)
        # self.vis.plot_dyn(self.his_drone_cmd, 
        #                   self.his_drone_vel)
        self.vis.plot_cmd(self.his_drone_cmd)

        # 打印性能指标
        print("typeI attacked_rate is ", \
              len(self.tracker_manager.typeI_attacked_pos) / self.tracker_manager.steps)
        print("typeII attacked_rate is ", \
              len(self.tracker_manager.typeII_attacked_pos) / self.tracker_manager.steps)
        

        print("average trace is ", sum(self.tracker_manager.trace_list)/len(self.tracker_manager.trace_list))
        print("total trace values are ", sum(self.tracker_manager.trace_list))
        print("total attacked typeI is ", len(self.tracker_manager.typeI_attacked_pos))
        print("total attacked typeII is ", len(self.tracker_manager.typeII_attacked_pos))
        
        # 计算轨迹长度
        accumulated_distance = 0
        for i in range(len(self.robot_ids)):
            for step in range(1, len(self.his_drone_pos)):
                accumulated_distance += np.linalg.norm(np.array(self.his_drone_pos[step][i]) - np.array(self.his_drone_pos[step-1][i]))

        print("total trajectory is ", accumulated_distance / (len(self.robot_ids))) 
        print("average trajectory is ", accumulated_distance / (len(self.robot_ids) * len(self.his_drone_pos)))
              


        # 可视化攻击点、跟踪误差等
        self.vis.plot_pts(self.tracker_manager.typeI_attacked_pos, "red")
        self.vis.plot_pts(self.tracker_manager.typeII_attacked_pos, "cyan")
        self.vis.plot_trace(self.tracker_manager.trace_list)
        self.vis.plot_trace_single(self.tracker_manager.trace_list_single)
        self.vis.plot_known_flags(self.his_known_typeI_flags, self.his_known_typeII_flags)
        self.vis.plot_exitflag(self.his_exitflag)
        self.vis.plot_attacked_flags(self.his_attacked_typeI_flags, self.his_attacked_typeII_flags)
        self.vis.show()
    
    def simulation(self):
        """
        执行纯仿真实验，定期调用LLM适配器更新系统参数
        steps: int, number of steps to run the simulation
        """
        
        # 初始化计数器
        # solve the problem
        print(" ========= solving problem =========")
        outer_idx = 0
        inner_idx = 0
        outer_update_cnt = int(self.config_loader.llm_outer_dur / self.config_loader.dt)
        inner_update_cnt = int(self.config_loader.llm_inner_dur / self.config_loader.dt)

        human_idx = [int(0.2 * self.config_loader.steps), 
                     int(0.5 * self.config_loader.steps),
                     int(0.8 * self.config_loader.steps)]

        # 主仿真循环
        for i in range(self.config_loader.steps):
            # 计算一步跟踪计算结果
            results = self.tracker_manager.solve_one_step()
            self.update_results(results)

            #print("results are ", results)
            #print("interpolate_results are ", self.adapter.interpolate_results(results))
            ###### parameters adapter ###### 
            # 更新LLM adapter结果
            self.adapter.update_results(results)

            # 按照设定的时间间隔调用LLM更新
            if self.config_loader.llm_on: 
                outer_idx += 1
                # if i in human_idx:

                #     inputs = "Focus more on tracking the targets, the trace is not good."
                #     prompt = "The human supervisor has inputs: " + inputs
                #     print("human" )
                #     self.tracker_manager.trac_prob.assignment_matrix =\
                #         self.adapter.task_adapter(prompt).copy()
                #     self.tracker_manager.trac_prob.weights = \
                #         self.adapter.weight_adapter(prompt).copy()
                #     continue


                if outer_idx%outer_update_cnt == 0:
                    outer_idx = 0
                    t1 = time.time()
                    print(" ========= updating LLM =========")
                    self.tracker_manager.trac_prob.assignment_matrix =\
                          self.adapter.task_adapter().copy()
                    t2 = time.time()
                    print(f"task adapter time: {t2-t1}")
            

                inner_idx += 1
                if inner_idx%inner_update_cnt == 0:
                    inner_idx = 0
                    t1 = time.time()
                    print(" ========= updating inner =========")
                    self.tracker_manager.trac_prob.weights = \
                        self.adapter.weight_adapter().copy()
                    t2 = time.time()
                    print(f"adapter time: {t2-t1}")

        self.adapter.save_data()
        self.vis_sim()

    

    def init_ros(self):
        '''
        初始化与ROS系统的通信接口，包括命令发布者、里程计发布者和目标可视化
        '''

        # 为每个机器人创建publisher
        for i in range(len(self.robot_ids)):

            name = "/drone" +  str(self.robot_ids[i]) 
            self.drone_cmd_pubs.append(rospy.Publisher(name + "/cmd_vel", 
                                                       Twist, queue_size=10))
            
            
            sim_frame_id = name + "/odom"

            self.drone_vis_odom_pubs.append(rospy.Publisher(sim_frame_id,
                                                       Odometry, queue_size=10))
            
            topic = name + "/goal"
            self.drone_vis_goal_pubs.append(rospy.Publisher(topic,
                                                            Marker, queue_size=10))

        # 为每个目标创建publisher
        for i in range(len(self.target_ids)):
            self.target_vis_odom_pubs.append(rospy.Publisher("/target" + str(self.target_ids[i]) + "/odom",
                                                            Odometry, queue_size=10))

        # 创建ROS可视化器
        self.vis_ros = VisualizerROS(self.frame_id,
                                     self.drone_cmd_pubs, 
                                     self.drone_vis_odom_pubs,
                                     self.target_vis_odom_pubs)
        
    
    def publish_results_msg(self, results, pub):
        '''
        将内部计算结果封装成ROS消息，提供给其他节点使用，特别是LLM adaptive server
        '''
        # 创建结果消息
        from tracker.msg import Results
        msg = Results()
        
        # 添加机器人位置、速度和控制命令
        for i in range(len(self.robot_ids)):
            pos = Point()
            pos.x = results["robot_pos"][i][0]
            pos.y = results["robot_pos"][i][1]
            pos.z = 1.0

            if results["robot_vel"].shape[1] > 0:
                print("robot_vel is ", results["robot_vel"])
                print("len is ", len(results["robot_vel"]))
                vel = Point()
                vel.x = results["robot_vel"][i][0]
                vel.y = results["robot_vel"][i][1]
                vel.z = 0.0
                msg.robot_vel.append(vel)

            cmd = Point()
            cmd.x = results["robot_cmd"][i][0]
            cmd.y = results["robot_cmd"][i][1]
            cmd.z = 0.0

            msg.robot_pos.append(pos)
            msg.robot_cmd.append(cmd)

        # 添加目标位置
        for i in range(len(self.target_ids)):
            pos = Point()
            pos.x = results["target_pos"][i][0]
            pos.y = results["target_pos"][i][1]
            pos.z = 0.0
            msg.target_pos.append(pos)

        # 添加其他状态信息（攻击标志、已知区域等）
        msg.exitflag = results["exitflag"]
        for i in range(len(self.robot_ids)):

            msg.attacked_typeI_flags.append(int(results["attacked_typeI_flags"][i]))
            msg.attacked_typeII_flags.append(int(results["attacked_typeII_flags"][i]))
            for j in range(self.config_loader.nTypeI):
                msg.known_typeI_flags.append(int(results["known_typeI_flags"][i][j]))
            for j in range(self.config_loader.nTypeII):
                msg.known_typeII_flags.append(int(results["known_typeII_flags"][i][j]))

        ##
        msg.trace = results["trace"]
        msg.weights = results["weights"]

        for i in range(len(self.robot_ids)):
            for j in range(len(self.target_ids)):
                msg.assignment_matrix.append(results["assignment_matrix"][i][j]) 
        #print("assignment_matrix is ", results["assignment_matrix"])
        #print("publishing results")
        pub.publish(msg)
        return
    





    

    def ros_simulation(self):
        """
        在ROS环境中执行仿真，与ROS节点交互并发布结果
        setup ROS parameters
        """
        
        # 初始化ros节点
        rospy.init_node('tracker_server', anonymous=True)
        rate = rospy.Rate(1/self.tracker_manager.dt)
        self.init_ros()
        
        # 设置航点publisher
        for i in range(len(self.robot_ids)):
            #### 2. publish to the waypoints topic
            name = "/dragonfly" + str(self.robot_ids[i])
            cmd_topic = name +  "/waypoints"

            print("cmd_topic is ", cmd_topic)
            self.drones_waypoints_pubs.append(rospy.Publisher(cmd_topic, 
                                                              Path, 
                                                              queue_size=10))

        # 结果publisher
        from tracker.msg import Results
        self.result_pub = rospy.Publisher("/server/results", Results, queue_size=10)
        
        # 主循环
        #print(" ========= setting up ROS subscribers =========")
        while not rospy.is_shutdown() and self.tracker_manager.cur_step < self.tracker_manager.steps:

            #print(" ========= solving problem =========")
            # 计算一步跟踪结果
            results = self.tracker_manager.solve_one_step()
            self.publish_results_msg(results, self.result_pub)
            
            # 发布航点和可视化信息
            for i in range(len(self.drone_ids)):
                self.pub_waypoints(results["robot_pos"][i], i)

            self.vis_ros.publish_all(results["target_pos"], 
                                     results["robot_pos"], 
                                     results["robot_cmd"],
                                     self.config_loader.robotHeights,
                                     self.config_loader.typeI_zones,
                                     self.config_loader.typeII_zones,
                                     results["known_typeI_flags"],
                                     results["known_typeII_flags"])
            rate.sleep()
        return

        

    def ros_real(self):
        '''
        控制实际的硬件系统，等待初始化完成后开始跟踪任务
        '''
        
        # 初始化ROS节点
        rospy.init_node('tracker_server', anonymous=True)
        rate = rospy.Rate(1/self.tracker_manager.dt)
        self.init_ros()

        # 设置AprilTag消息类型和结果发布者
        from apriltag_msgs.msg import ApriltagPoseStamped
        from tracker.msg import Results
        self.result_pub = rospy.Publisher("/server/results", Results, queue_size=10)
        

        # for each drone, we have a tf listener
        # /dragonfly21/quadrotor_ukf/control_odom
        # /dragonfly21/odom_tag
        # import tf
        # tf_listener = tf.TransformListener()
        # 为每个无人机设置订阅者和发布者
        print(" ========= setting up ROS subscribers =========")
        for i, id in enumerate(self.drone_ids):
            print(f"cf{id} has index: {i}")
            name = "/dragonfly" + str(id)

            #### 1. subscribe to the odom topic 订阅里程计
            topic = name + "/world_odom"
            print("topic is ", topic)
            self.drone_odom_subs.append(rospy.Subscriber(topic,
                                                         Odometry, 
                                                         functools.partial(self.drone_odom_callback, index=i)))
            
            
            #### 2. publish to the waypoints topic 发布航点
            cmd_topic = name +  "/global_waypoints"
            print("cmd_topic is ", cmd_topic)
            self.drones_waypoints_pubs.append(rospy.Publisher(cmd_topic, Path, queue_size=10))

            
            # #### 3. publish for visualization 发布可视化信息
            drone_frame_id = name + "/vis_odom"
            self.drone_vis_odom_pubs.append(rospy.Publisher(drone_frame_id,
                                                            Odometry, queue_size=10))
            

            #### 4. subscribe to the apriltag topic 订阅AprilTag检测
            apriltag_topic = name + "/world_apriltag_poses"
            print("apriltag_topic is ", apriltag_topic)
            self.apriltag_subs.append(rospy.Subscriber(apriltag_topic,
                                                        ApriltagPoseStamped,
                                                        functools.partial(self.apriltag_callback, robot_index=i)))
            


        #### for each car, we publish the visualized odom
        # for i, id in enumerate(self.car_ids):
        #     name = "/target" + str(id)
        #     topic = name + "/odom"
        #     self.target_vis_odom_pubs.append(rospy.Publisher(topic, Odometry, queue_size=10))
        #### test publish
        idx = 0
        while not rospy.is_shutdown():
            rate.sleep()
            if self.tracker_manager.trac_prob.is_real_init() == False:
                if idx % 500 == 0:
                    print("real init is not ready, idx is ", idx)
                idx += 1
                continue


            results = self.tracker_manager.solve_one_step()
            self.publish_results_msg(results, self.result_pub)
            for i in range(len(self.drone_ids)):
                print(" publish the result for robot")
                self.pub_waypoints(results["robot_pos"][i], i)


            # print("robot_pos is ", results["robot_pos"])
            # print("robot_cmd is ", results["robot_cmd"])
            # print("target_pos is ", results["target_pos"])
            # print("known_typeI_flags is ", results["known_typeI_flags"])


            self.vis_ros.publish_all_others(results["target_pos"], 
                                            self.config_loader.typeI_zones,
                                            self.config_loader.typeII_zones,
                                            results["known_typeI_flags"],
                                            results["known_typeII_flags"])
            

        # get the position of global origin from the tf
        #origin = tf_listener.lookupTransform("dragonfly21/odom", "global_origin", rospy.Time(0))


        

if __name__ == '__main__':

    import sys

    #try
    try:
        exp_name = sys.argv[1]
        print("exp is ", exp_name)
    except:

        exp_name = "exp3"

    tracker_server = TrackerServer(exp_name)