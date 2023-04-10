import argparse
import json
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
import cv2
from scipy.spatial.transform import Rotation

def catmull_rom_spline(points, num_interpolated_points):
    points = np.array(points)
    n = len(points)
    t = np.linspace(0, n - 1, num_interpolated_points)
    tck_x = CubicSpline(np.arange(n), points[:, 0], bc_type='clamped')
    tck_y = CubicSpline(np.arange(n), points[:, 1], bc_type='clamped')
    tck_z = CubicSpline(np.arange(n), points[:, 2], bc_type='clamped')
    return np.column_stack((tck_x(t), tck_y(t), tck_z(t)))


def world_to_map_coordinates(world_coord, map_size, nav_bounds_min, nav_bounds_max):
    map_scale = (map_size[1] / (nav_bounds_max[2] - nav_bounds_min[2]), map_size[0] / (nav_bounds_max[0] - nav_bounds_min[0]))
    map_coord = ((world_coord[0] - nav_bounds_min[0]) * map_scale[1], (world_coord[2] - nav_bounds_min[2]) * map_scale[0])
    map_coord = (int(round(map_coord[0])), int(round(map_coord[1])))
    return map_coord


def visualize_trajectory(pred_json_file, val_seen_json_file, scene_directory):
    # 加载JSON文件
    with open(pred_json_file, 'r') as f:
        pred_data = json.load(f)

    with open(val_seen_json_file, 'r') as f:
        val_seen_data = json.load(f)

    for pred_episode_id, pred_episode_steps in pred_data.items():
        val_episode = None
        for episode in val_seen_data["episodes"]:
            if str(episode["episode_id"]) == pred_episode_id:
                val_episode = episode
                break

        if val_episode is None:
            print(f"Episode {pred_episode_id} not found in val_seen.json")
            continue
        scene_file = os.path.join(scene_directory, val_episode["scene_id"])
        filename_without_ext = os.path.splitext(os.path.basename(val_episode["scene_id"]))[0]
        navmesh_file = os.path.join(scene_directory, filename_without_ext + ".navmesh")

        print(f"Navmesh file path: {navmesh_file}")

        # 创建模拟器
        sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 0.7,
            "sensor_width": 320,
            "width": 320,
            "height": 240,
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

        sim.pathfinder.load_nav_mesh(navmesh_file)
        # 获取导航网格的边界
        nav_bounds_min, nav_bounds_max = sim.pathfinder.get_bounds()

        map_height = 320
        map_width = int(
            map_height * ((nav_bounds_max[0] - nav_bounds_min[0]) / (nav_bounds_max[2] - nav_bounds_min[2])))
        map_size = (map_width, map_height)

        blank_map = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)

        # 获取路径
        path = [step["position"] for step in pred_episode_steps if not step["stop"]]

        for i in range(0, map_size[0], 1):
            for j in range(0, map_size[1], 1):
                world_coord = np.array([nav_bounds_min[0] + (i / map_size[0]) * (nav_bounds_max[0] - nav_bounds_min[0]),
                                        0,
                                        nav_bounds_min[2] + (j / map_size[1]) * (
                                                nav_bounds_max[2] - nav_bounds_min[2])])
                x, y = world_to_map_coordinates(world_coord, map_size, nav_bounds_min, nav_bounds_max)
                if sim.pathfinder.is_navigable(world_coord):
                    blank_map[y, x] = [0, 0, 0]
                else:
                    blank_map[y, x] = 255

        for step in path:
            x, y = world_to_map_coordinates(step, map_size, nav_bounds_min, nav_bounds_max)
            cv2.circle(blank_map, (x, y), 2, (0, 0, 255), -1)

        if not path:
            continue
        num_interpolated_points = 600
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

            x, y = world_to_map_coordinates(point, map_size, nav_bounds_min, nav_bounds_max)

            cv2.circle(blank_map, (x, y), 1, (0, 255, 0), -1)

            # 捕捉并显示图像
            observations = sim.get_sensor_observations()
            rgb_observation = observations["color_sensor"]
            cv2.imshow("RGB", rgb_observation)
            # cv2.imshow("2D Map", blank_map)
            resized_map = cv2.resize(blank_map, (map_size[0] * 2, map_size[1] * 2), interpolation=cv2.INTER_AREA)
            cv2.imshow("2D Map", resized_map)
            cv2.waitKey(75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', type=str, required=True, help='Path to the pred_val_seen JSON file')
    parser.add_argument('--val-json', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--scene-dir', type=str, required=True, help='Path to the directory containing glb files')
    args = parser.parse_args()
    visualize_trajectory(args.pred_json, args.val_json, args.scene_dir)
