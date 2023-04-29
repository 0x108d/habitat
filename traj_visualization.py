import argparse
import json
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors
import cv2
from scipy.spatial.transform import Rotation, Slerp
import scipy.interpolate


def world_to_map_coordinates(world_coord, map_size, nav_bounds_min, nav_bounds_max):
    map_scale = (
        map_size[1] / (nav_bounds_max[2] - nav_bounds_min[2]), map_size[0] / (nav_bounds_max[0] - nav_bounds_min[0]))
    map_coord = (
        (world_coord[0] - nav_bounds_min[0]) * map_scale[1], (world_coord[2] - nav_bounds_min[2]) * map_scale[0])
    map_coord = (int(round(map_coord[0])), int(round(map_coord[1])))
    return map_coord


def catmull_rom_spline(points, num_interpolated_points):
    points = np.array(points)
    n = len(points)
    t = np.linspace(0, n - 1, num_interpolated_points)
    tck_x = CubicSpline(np.arange(n), points[:, 0], bc_type='clamped')
    tck_y = CubicSpline(np.arange(n), points[:, 1], bc_type='clamped')
    tck_z = CubicSpline(np.arange(n), points[:, 2], bc_type='clamped')
    return np.column_stack((tck_x(t), tck_y(t), tck_z(t)))


def visualize_trajectory(pred_json_file, val_seen_json_file, scene_directory):
    # loading JSON file
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

        # create simulator
        sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 0.7,
            "sensor_width": 1024,
            "width": 2048,
            "height": 1024,
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

        # 添加了topdown的传感器
        topdown_height = 30
        topdown_camera_x = 15
        topdown_camera_z = 0
        topdown_sensor_spec = habitat_sim.CameraSensorSpec()
        topdown_sensor_spec.uuid = "topdown_sensor"
        topdown_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        topdown_sensor_spec.resolution = [1024, 2048]
        topdown_sensor_spec.position = [topdown_camera_x, topdown_height, topdown_camera_z]
        topdown_sensor_spec.orientation = [-np.pi / 2, 0, 0]
        topdown_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg, topdown_sensor_spec]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        nav_bounds_min, nav_bounds_max = sim.pathfinder.get_bounds()
        # Calculate map size
        map_height = 1024
        map_width = int(
            map_height * ((nav_bounds_max[0] - nav_bounds_min[0]) / (nav_bounds_max[2] - nav_bounds_min[2])))
        map_size = (map_width, map_height)

        observations = sim.get_sensor_observations()
        topdown_observation = observations["topdown_sensor"]
        cv2.namedWindow("Top Down View", cv2.WINDOW_NORMAL)
        cv2.imshow("Top Down View", topdown_observation)
        # cv2.resizeWindow("Top Down View", map_size[0], map_size[1])

        # 3D Path
        # setting camera angle
        agent_state = habitat_sim.AgentState()
        agent = sim.initialize_agent(sim_settings["default_agent"], agent_state)

        # getpath
        path = [step["position"] for step in pred_episode_steps if not step["stop"]]

        if not path:
            continue
        num_interpolated_points = 600
        interpolated_path = catmull_rom_spline(np.array(path), num_interpolated_points)

        agent_state = habitat_sim.AgentState()
        agent.set_state(agent_state, reset_sensors=True)
        # Move the camera along the path and capture the image
        rotations = []
        skipped_indices = []
        for i, point in enumerate(path[:-1]):
            point = np.array(point)
            next_point = np.array(path[i + 1])
            direction_vector = next_point - point
            direction_norm = np.linalg.norm(direction_vector)

            # If the norm of the direction vector is too small, skip this point
            if direction_norm < 1e-6:
                skipped_indices.append(i)
                continue

            direction_vector /= direction_norm
            # Calculate camera orientation (quaternion)
            up_vector = np.array([0, 1, 0])
            right_vector = np.cross(direction_vector, up_vector)
            right_vector /= np.linalg.norm(right_vector)
            up_vector = np.cross(right_vector, direction_vector)

            rotation_matrix = np.stack([right_vector, up_vector, -direction_vector], axis=1)
            rotation = Rotation.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            rotations.append(quaternion)

        # Add the last point's rotation
        rotations.append(rotations[-1])
        for index in reversed(skipped_indices):
            del path[index]
        interpolator = scipy.interpolate.interp1d(np.arange(len(path)), path, axis=0)
        interpolated_path = interpolator(np.linspace(0, len(path) - 1, num_interpolated_points))

        # Interpolate rotations
        slerp = Slerp(np.linspace(0, len(path) - 1, len(path)), Rotation.from_quat(rotations))
        interpolated_rotations = slerp(np.linspace(0, len(path) - 1, num_interpolated_points)).as_quat()

        for i, point in enumerate(interpolated_path[:-1]):
            agent_state = habitat_sim.AgentState()
            agent_state.position = point
            agent_state.rotation = quat_from_coeffs(interpolated_rotations[i])
            agent.set_state(agent_state, reset_sensors=True)

            # Get sensor observations
            observations = sim.get_sensor_observations()

            # topdown_observation = observations["topdown_sensor"]
            # cv2.namedWindow("Top Down View", cv2.WINDOW_NORMAL)
            # cv2.imshow("Top Down View", topdown_observation)
            # cv2.resizeWindow("Top Down View", map_size[0], map_size[1])

            # Capture and display the image
            rgb_observation = observations["color_sensor"]

            cv2.imshow("RGB", rgb_observation)

            cv2.waitKey(75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', type=str, required=True, help='Path to the pred_val_seen JSON file')
    parser.add_argument('--val-json', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--scene-dir', type=str, required=True, help='Path to the directory containing glb files')
    args = parser.parse_args()
    visualize_trajectory(args.pred_json, args.val_json, args.scene_dir)
