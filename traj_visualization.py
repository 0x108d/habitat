import argparse
import json
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
import cv2



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
            "sensor_width": 640,
            "width": 640,
            "height": 320,
        }
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = sim_settings["scene"]
        backend_cfg.enable_physics = False

        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.resolution = [sim_settings["height"], sim_settings["width"]]
        sensor_cfg.position = [0, sim_settings["sensor_height"], 0]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.uuid = "color_sensor"

        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.hfov = np.pi / 3
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        color_sensor_spec.channels_first = True
        color_sensor_spec.encoding = "png"

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # 设置相机视角
        agent_state = habitat_sim.AgentState()
        agent = sim.initialize_agent(sim_settings["default_agent"], agent_state)

        def catmull_rom_spline(points, num_interpolated_points):
            points = np.array(points)
            n = len(points)
            t = np.linspace(0, n - 1, num_interpolated_points)
            tck_x = CubicSpline(np.arange(n), points[:, 0], bc_type='clamped')
            tck_y = CubicSpline(np.arange(n), points[:, 1], bc_type='clamped')
            tck_z = CubicSpline(np.arange(n), points[:, 2], bc_type='clamped')
            return np.column_stack((tck_x(t), tck_y(t), tck_z(t)))

        # 获取路径
        path = episode["reference_path"]
        num_interpolated_points = 200
        interpolated_path = catmull_rom_spline(np.array(path), num_interpolated_points)

        agent_state = habitat_sim.AgentState()
        agent.set_state(agent_state, reset_sensors=True)
        # 沿路径移动相机并捕捉图像
        for point in interpolated_path:
            agent_state = habitat_sim.AgentState()
            agent_state.position = point
            agent_state.rotation = quat_from_coeffs([-0.0, 0.9659258262890683, -0.0, -0.25881904510252063])
            agent.set_state(agent_state, reset_sensors=True)

            # 捕捉并显示图像
            observations = sim.get_sensor_observations()
            rgb_observation = observations["color_sensor"]
            cv2.imshow("RGB", rgb_observation)
            cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--scene-dir', type=str, required=True, help='Path to the directory containing glb files')
    args = parser.parse_args()

    visualize_trajectory(args.json, args.scene_dir)
