#! /usr/bin/env python
"""
Script for running multi-agent tracking simulation and visualization
Also used for defining ROS setup (to be added)

用于分析和可视化大语言模型(LLM)在多机器人目标跟踪任务中的性能数据，
主要功能是从实验结果文件中提取性能指标，并生成热力图(heatmap)来可视化不同机器人数量和目标数量组合下的LLM性能

它主要关注两个方面：
    1. 任务分配LLM的性能
    2. 权重调整LLM的性能
对每个方面，脚本分析两个关键指标：
    1. 成功率（LLM生成可行解决方案的比例）
    2. 平均令牌数（衡量计算资源消耗）
    
脚本生成四张热力图：
    1. 任务LLM成功率 - 不同机器人/目标组合下任务分配的成功率
    2. 行动LLM成功率 - 不同机器人/目标组合下权重调整的成功率
    3. 任务LLM平均令牌数 - 任务分配所需的计算资源
    4. 行动LLM平均令牌数 - 权重调整所需的计算资源
    
每张热力图都保存为PNG文件，并以{name1} {name2}.png命名，例如Task LLM Success Rate.png
"""
import os
import rospy
import rospkg
import numpy as np
import functools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results():
    '''
    这是脚本的主要函数，负责：
        1. 从指定路径读取所有实验结果文件
        2. 解析每个文件提取关键性能指标
        3. 构建数据矩阵用于可视化
        4. 调用plot_grid()生成热力图
    '''
    #results
    num_robot = []
    num_target = []
    average_trace = []
    robot_ability = []
    num_attacks   = []
    
    task_success_rate = []
    weight_success_rate = []

    llm_results = []
    # llm_result1 = [2, 2, 0.1912805419789908, 2, 28, 1.0, 0.92]
    # llm_result2 = [4, 4, 0.1911573390816325, 2, 26, 1.0, 0.9]
    # llm_result3 = [4, 6, 0.3445296702351883, 3, 26, 1.0, 0.94]
    

    # total ability * number of robots = number of targets * 2
    # total ability = number of targets * 2 / number of robots
    model = "3_model"
    path = "/home/wyw/Code/ZhouLab/llm_target_tracking/src/tracker/results/data/" + model + "/"

    for file in os.listdir(path):
        if file.endswith(".yaml"):
            new_llm_result = []
            num_robot = int(file.split("-")[0])
            num_target = int(file.split("-")[1])

            new_llm_result.append(num_robot)
            new_llm_result.append(num_target)


            with open(path + file, 'r') as f:
                data = f.readlines()
                for line in data:
                    if "task_correct_rate" in line:
                        task_success_rate = float(line.split(":")[1])
                        new_llm_result.append(task_success_rate)

                    if "weights_correct_rate" in line:
                        weights_correct_rate = float(line.split(":")[1])
                        new_llm_result.append(weights_correct_rate)

                    if "task_avg_token" in line:
                        task_avg_token = float(line.split(":")[1])
                        new_llm_result.append(task_avg_token)

                    if "weights_avg_token" in line:

                        weights_avg_token = float(line.split(":")[1])
                        new_llm_result.append(weights_avg_token)
    
            llm_results.append(new_llm_result)
    

    llm_results = np.array(llm_results)


    # plt.bar(x, y, color='blue', label='Task Success Rate')
    # plt.xlabel('Number of Robots')
    # plt.ylabel('Task Success Rate')

    # # print value on it
    # for a, b in zip(x, y):
    #     plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    # plt.show()

    print(llm_results)

    # 创建一个4x4矩阵，将不同机器人数量和目标数量组合下的任务成功率映射到矩阵位置
    # 行索引: 4 - target_num // 2 (目标数量从8到2递减)
    # 列索引: robot_num // 2 - 1 (机器人数量从2到8递增)
    task_success_matrix = np.zeros((4, 4))
    for i in range(len(llm_results)):
        robot_num = int(llm_results[i][0])
        target_num = int(llm_results[i][1])
        task_success_matrix[4 - target_num // 2, robot_num // 2 - 1] = llm_results[i][2]

    plot_gird(task_success_matrix, "Task LLM", "Success Rate")

    weight_success_matrix = np.zeros((4, 4))
    for i in range(len(llm_results)):
        robot_num = int(llm_results[i][0])
        target_num = int(llm_results[i][1])
        weight_success_matrix[4 -  target_num // 2, robot_num // 2 - 1] = llm_results[i][4]
    
    plot_gird(weight_success_matrix, "Action LLM", "Success Rate")

    

    task_token_matrix = np.zeros((4, 4))
    for i in range(len(llm_results)):
        robot_num = int(llm_results[i][0])
        target_num = int(llm_results[i][1])
        task_token_matrix[4 - target_num // 2, robot_num // 2 - 1] = llm_results[i][3]

    plot_gird(task_token_matrix, "Task LLM", "Average Token Number", range=(60, 310))

    weight_token_matrix = np.zeros((4, 4))
    for i in range(len(llm_results)):
        robot_num = int(llm_results[i][0])
        target_num = int(llm_results[i][1])
        weight_token_matrix[4 - target_num // 2, robot_num // 2 - 1] = llm_results[i][5]
    
    plot_gird(weight_token_matrix, "Action LLM", "Average Token Number", range=[60, 320])
    




    plt.show()

    


     
def plot_gird(matrix, name1, name2, color_map="vlag" ,range=[0.6, 1]):
    '''
    这个函数负责将数据矩阵可视化为热力图：
    
    参数：
        matrix: 4x4的数据矩阵，行表示目标数量，列表示机器人数量
        name1: 第一个标签名(例如"Task LLM")
        name2: 第二个标签名(例如"Success Rate")
        color_map: 热力图的颜色映射
        range: 颜色映射的值范围
    '''

    if matrix.shape != (4, 4):
        raise ValueError("Matrix shape must be (4, 4)")
    
    print(matrix)

    text_size = 15


    # Create a heatmap using seaborn
    plt.figure(figsize=(4, 3.2))
    #Greys RdBu vlag
    # Use seaborn heatmap to create the plot
    # cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    ax = sns.heatmap(matrix, annot=True, fmt=".2f",
                    vmin=range[0], vmax=range[1],
                    cmap=sns.cubehelix_palette(as_cmap=True), center=(range[0] + range[1]) / 2,
                    cbar_kws={'label': name2}, 
                    annot_kws={"size": 11},
                    linewidths=0.01, square=True)

   
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(text_size)  # Adjust font size here
    cbar.ax.tick_params(labelsize=text_size)  # Adjust font size for colorbar ticks

    # Optional: Adjusting the tick labels font size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=text_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=text_size)
    # Set axis labels and title
    plt.xlabel('Number of Robots', fontsize=text_size)
    plt.ylabel('Number of Targets', fontsize=text_size)

    # Adjust axis ticks (Replace with actual tick values if needed)
    plt.xticks(ticks=np.arange(4) + 0.5, labels=np.arange(2, 10, 2), fontsize=text_size)
    plt.yticks(ticks=np.arange(4) + 0.5, labels=np.arange(8,  0, -2), fontsize=text_size)
    
    #plt.title(name1, fontsize=text_size)
    plt.tight_layout()

    # Show the plot
    plt.savefig(name1 + " " + name2 + ".png")



#have args for exp_name
if __name__ == "__main__": 

    import sys


    tracker_server = plot_results()
    #rospy.spin()
