import argparse
import json
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
import cv2
from scipy.spatial.transform import Rotation, Slerp
import scipy.interpolate


def parse_house_file(file_path):
    floor_heights = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("L"):
                data = line.strip().split()
                floor_number = int(data[1])
                min_height = float(data[5])
                max_height = float(data[11])
                floor_heights[floor_number] = (min_height, max_height)
    return floor_heights

def determine_floor(height, floor_heights):
    for floor, (min_h, max_h) in floor_heights.items():
        if min_h <= height < max_h:
            return floor
    return None


def catmull_rom_spline(points, num_interpolated_points):
    points = np.array(points)
    n = len(points)
    t = np.linspace(0, n - 1, num_interpolated_points)
    tck_x = CubicSpline(np.arange(n), points[:, 0], bc_type='clamped')
    tck_y = CubicSpline(np.arange(n), points[:, 1], bc_type='clamped')
    tck_z = CubicSpline(np.arange(n), points[:, 2], bc_type='clamped')
    return np.column_stack((tck_x(t), tck_y(t), tck_z(t)))


def world_to_map_coordinates(world_coord, map_size, nav_bounds_min, nav_bounds_max):
    map_scale = (
        map_size[1] / (nav_bounds_max[2] - nav_bounds_min[2]), map_size[0] / (nav_bounds_max[0] - nav_bounds_min[0]))
    map_coord = (
        (world_coord[0] - nav_bounds_min[0]) * map_scale[1], (world_coord[2] - nav_bounds_min[2]) * map_scale[0])
    map_coord = (int(round(map_coord[0])), int(round(map_coord[1])))
    return map_coord


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

                print(f"Episode ID: {pred_episode_id}")

                break

        if val_episode is None:
            print(f"Episode {pred_episode_id} not found in val_seen.json")
            continue
        reference_path = val_episode["reference_path"]
        print(f"Reference Path: {reference_path}")
        scene_file = os.path.join(scene_directory, val_episode["scene_id"])
        filename_without_ext = os.path.splitext(val_episode["scene_id"])[0]
        navmesh_file = os.path.join(scene_directory, filename_without_ext + ".navmesh")
        house_file = os.path.join(scene_directory, filename_without_ext + ".house")
        floor_heights = parse_house_file(house_file)
        print(f"Navmesh file path: {navmesh_file}")
        print(f"House file path: {house_file}")

        # getpath
        path = [step["position"] for step in pred_episode_steps if not step["stop"]]

        # create simulator
        sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 0.7,
            "sensor_width": 1024,
            "width": 1280,
            "height": 640,
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
        color_sensor_spec.hfov = np.pi / 3.5
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        color_sensor_spec.channels_first = True
        color_sensor_spec.encoding = "png"

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg, agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        sim.pathfinder.load_nav_mesh(navmesh_file)
        # Gets the boundaries of the navigation grid
        nav_bounds_min, nav_bounds_max = sim.pathfinder.get_bounds()

        map_height = 640
        map_width = int(
            map_height * ((nav_bounds_max[0] - nav_bounds_min[0]) / (nav_bounds_max[2] - nav_bounds_min[2])))
        map_size = (map_width, map_height)

        # create a blank 2d map
        blank_map = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)

        floor_colors = {
            0: (255, 0, 0),  # Red
            1: (0, 255, 0),  # Green
            2: (0, 0, 255)  # Blue
        }

        height_samples = 10  # Number of height samples to take within the range

        for i in range(0, map_size[0], 1):
            for j in range(0, map_size[1], 1):
                world_coord_base = np.array(
                    [nav_bounds_min[0] + (i / map_size[0]) * (nav_bounds_max[0] - nav_bounds_min[0]),
                     0,
                     nav_bounds_min[2] + (j / map_size[1]) * (nav_bounds_max[2] - nav_bounds_min[2])])
                x, y = world_to_map_coordinates(world_coord_base, map_size, nav_bounds_min, nav_bounds_max)

                navigable = False

                floor = determine_floor(world_coord_base[1], floor_heights)
                if floor is not None:
                    min_height, max_height = floor_heights[floor]
                    for h in np.linspace(min_height, max_height, height_samples):
                        world_coord = world_coord_base + np.array([0, h, 0])
                        if sim.pathfinder.is_navigable(world_coord):
                            navigable = True
                            break

                if navigable:
                    color = floor_colors.get(floor, (0, 0, 0))
                    blank_map[y, x] = color
                else:
                    blank_map[y, x] = 255
        # for step in path:
        for step in reference_path:
            print('Step:', step)
            x, _, z = step
            x, y = world_to_map_coordinates(step, map_size, nav_bounds_min, nav_bounds_max)
            cv2.circle(blank_map, (x, y), 5, (0, 0, 255), -1)

        # set up 3d top down view
        center_x = 0
        center_y = 2.5
        center_z = 0
        topdown_sensor_spec = habitat_sim.CameraSensorSpec()
        topdown_sensor_spec.uuid = "topdown_sensor"
        topdown_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        topdown_sensor_spec.resolution = [1280, 1080]
        topdown_sensor_spec.position = [center_x, center_y, center_z]

        topdown_sensor_spec.orientation = [-np.pi / 2.5, 0, 0]
        topdown_sensor_spec.vfov = np.pi * 45
        topdown_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        agent_cfg.sensor_specifications = [sensor_cfg, topdown_sensor_spec]

        sim.close()
        sim = habitat_sim.Simulator(cfg)

        # Set up the color sensor for the first agent
        sim.config.agents[0].sensor_specifications = [color_sensor_spec]

        # Set up the topdown sensor for the second agent
        sim.config.agents[1].sensor_specifications = [topdown_sensor_spec]

        agent_state = habitat_sim.AgentState()
        agent = sim.initialize_agent(sim_settings["default_agent"], agent_state)
        agent_state = habitat_sim.AgentState()
        agent.set_state(agent_state, reset_sensors=True)
        if not path:
            continue
        num_interpolated_points = 600
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

        rotation_increment = np.pi / 18  # Rotate by 10 degrees at a time
        agent_state = habitat_sim.AgentState()
        agent_state.position = interpolated_path[0]
        agent_state.rotation = quat_from_coeffs(interpolated_rotations[0])

        paused = False
        i = 0
        while i < len(interpolated_path) - 1:
            # Check for keyboard input
            key = cv2.waitKey(40) & 0xFF
            if key == ord(' '):  # Space bar to pause
                paused = not paused
            elif key == ord('r'):
                paused = False
            elif paused:  # Only allow movement and rotation while paused
                if key == ord('w'):
                    i = min(i + 1, len(interpolated_path) - 1)
                    agent_state.position = interpolated_path[i]
                elif key == ord('s'):
                    i = max(i - 1, 0)
                    agent_state.position = interpolated_path[i]
                elif key == ord('a'):
                    y_rotation = habitat_sim.utils.common.quat_from_angle_axis(rotation_increment, np.array([0, 1, 0]))
                    agent_state.rotation = y_rotation * agent_state.rotation
                elif key == ord('d'):
                    y_rotation = habitat_sim.utils.common.quat_from_angle_axis(-rotation_increment, np.array([0, 1, 0]))
                    agent_state.rotation = y_rotation * agent_state.rotation
            else:
                i += 1
                agent_state.position = interpolated_path[i]
                agent_state.rotation = quat_from_coeffs(interpolated_rotations[i])

            sim.agents[0].set_state(agent_state, reset_sensors=True)

            # Update the second agent (top-down camera) state
            agent_state_top_down = habitat_sim.AgentState()
            agent_state_top_down.position = agent_state.position
            agent_state_top_down.rotation = habitat_sim.utils.common.quat_from_angle_axis(-np.pi / 2,
                                                                                          np.array([1, 0, 0]))
            sim.agents[1].set_state(agent_state_top_down, reset_sensors=True)

            # Capture and display the image
            observations = sim.get_sensor_observations()
            topdown_observation = observations["topdown_sensor"]
            topdown_observation = cv2.cvtColor(topdown_observation, cv2.COLOR_RGB2BGR)
            rgb_observation = observations["color_sensor"]
            rgb_observation = cv2.cvtColor(rgb_observation, cv2.COLOR_RGB2BGR)

            x, y = world_to_map_coordinates(agent_state.position, map_size, nav_bounds_min, nav_bounds_max)
            cv2.circle(blank_map, (x, y), 3, (0, 255, 0), -1)

            resized_map = cv2.resize(blank_map, (map_size[0] * 2, map_size[1] * 2), interpolation=cv2.INTER_AREA)

            width = max(rgb_observation.shape[1], resized_map.shape[1], topdown_observation.shape[1])

            rgb_observation = cv2.resize(rgb_observation,
                                         (width, int(rgb_observation.shape[0] * width / rgb_observation.shape[1])))
            resized_map = cv2.resize(resized_map, (width, int(resized_map.shape[0] * width / resized_map.shape[1])))
            # topdown_observation = cv2.resize(topdown_observation, (
            #     width, int(topdown_observation.shape[0] * width / topdown_observation.shape[1])))

            # Then, vertically stack the images
            combined_image = np.vstack((rgb_observation, resized_map))

            # Now you can display the combined image
            cv2.imshow("Combined Image", combined_image)
            cv2.imshow("Top-down view", topdown_observation)

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-json', type=str, required=True, help='Path to the pred_val_seen JSON file')
    parser.add_argument('--val-json', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--scene-dir', type=str, required=True, help='Path to the directory containing glb files')
    args = parser.parse_args()
    visualize_trajectory(args.pred_json, args.val_json, args.scene_dir)
