import argparse
import json
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
import cv2
from scipy.spatial.transform import Rotation


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
        if "reference_path" not in episode:
            path = []
        else:
            path = episode["reference_path"]

        if not path:
            continue
        num_interpolated_points = 200
        interpolated_path = catmull_rom_spline(np.array(path), num_interpolated_points)

        agent_state = habitat_sim.AgentState()
        agent.set_state(agent_state, reset_sensors=True)
        # 沿路径移动相机并捕捉图像
        for i, point in enumerate(interpolated_path[:-1]):
            next_point = interpolated_path[i + 1]
            direction_vector = next_point - point
            direction_vector /= np.linalg.norm(direction_vector)

            # 计算相机朝向（四元数）
            up_vector = np.array([0, 1, 0])
            right_vector = np.cross(direction_vector, up_vector)
            right_vector /= np.linalg.norm(right_vector)
            up_vector = np.cross(right_vector, direction_vector)

            rotation_matrix = np.stack([right_vector, up_vector, -direction_vector], axis=1)
            rotation = Rotation.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()

            agent_state = habitat_sim.AgentState()
            agent_state.position = point
            agent_state.rotation = quat_from_coeffs(quaternion)
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
