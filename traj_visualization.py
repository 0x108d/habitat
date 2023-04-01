import argparse
import json
import numpy as np
import os

import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
from habitat_sim.utils.common import quat_to_magnum
import cv2
from habitat_sim.nav import ShortestPath


def visualize_trajectory(json_file, scene_directory):
    # 加载JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    for episode in data["episodes"]:
        # 根据scene_id获取glb文件路径
        scene_file = os.path.join(scene_directory, episode["scene_id"])

        # 创建模拟器
        sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 0.5,
            "width": 1280,
            "height": 720,
        }
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = sim_settings["scene"]
        backend_cfg.enable_physics = False

        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.resolution = [sim_settings["height"], sim_settings["width"]]
        sensor_cfg.position = [0, sim_settings["sensor_height"], 0]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.uuid = "color_sensor"

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # 设置相机视角
        agent_state = habitat_sim.AgentState()
        agent = sim.initialize_agent(sim_settings["default_agent"], agent_state)

        # 获取路径
        path = episode["reference_path"]

        adjusted_path = []
        # 沿路径移动相机并捕捉图像
        shortest_path = ShortestPath()
        for i in range(len(path) - 1):
            position = np.array(path[i])
            next_position = np.array(path[i + 1])

            # 创建一个ShortestPath对象，设置起始点和目标点
            shortest_path.requested_start = position
            shortest_path.requested_end = next_position

            # 计算两个相邻点之间的最短路径
            sim.pathfinder.find_path(shortest_path)

            # 将找到的路径点添加到调整后的路径中
            for point in shortest_path.points:
                adjusted_path.append(point)

        for point in adjusted_path:
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([point[0], point[1], point[2]])
            agent_state.rotation = quat_from_coeffs([-0.0, 0.9659258262890683, -0.0, -0.25881904510252063])
            agent.set_state(agent_state, reset_sensors=True)

            # 捕捉并显示图像
            observations = sim.get_sensor_observations()
            rgb_observation = observations["color_sensor"]
            cv2.imshow("RGB", rgb_observation)
            cv2.waitKey(1)

        cv2.destroyAllWindows()


        # 关闭模拟器和窗口
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--scene-dir', type=str, required=True, help='Path to the directory containing glb files')
    args = parser.parse_args()

    visualize_trajectory(args.json, args.scene_dir)
