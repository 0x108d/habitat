from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import json
from tkinter import filedialog, messagebox
import sys
from scipy.interpolate import CubicSpline
import numpy as np
import os
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
from scipy.spatial.transform import Rotation, Slerp
import scipy.interpolate
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar, QPushButton
import time


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


def adjust_floor_heights(floor_heights):
    adjusted_floor_heights = {}

    # Determine the min and max height of the building
    min_height = min(height[0] for height in floor_heights.values())
    max_height = max(height[1] for height in floor_heights.values())

    # Calculate the total height of the building
    total_height = max_height - min_height

    for floor in reversed(sorted(floor_heights.keys())):
        if floor == 0:
            # The bottom floor's min height is the minimum height of the building
            adjusted_floor_heights[floor] = (min_height, min_height + total_height * (
                    (floor_heights[floor][1] - floor_heights[floor][0]) / total_height))
        else:
            floor_height = total_height * ((floor_heights[floor][1] - floor_heights[floor][0]) / total_height)
            adjusted_floor_heights[floor] = (max_height - floor_height, max_height)
            max_height -= floor_height

    return adjusted_floor_heights


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


def load_data(pred_json_file, val_seen_json_file, selected_episode_id):
    # loading JSON file
    with open(pred_json_file, 'r') as f:
        pred_data = json.load(f)

    with open(val_seen_json_file, 'r') as f:
        val_seen_data = json.load(f)

    if selected_episode_id not in pred_data:
        print(f"Episode {selected_episode_id} not found in pred.json")
        return None, None, None

    pred_episode_steps = pred_data[selected_episode_id]
    val_episode = None
    for episode in val_seen_data["episodes"]:
        if str(episode["episode_id"]) == selected_episode_id:
            val_episode = episode
            break

    if val_episode is None:
        print(f"Episode {selected_episode_id} not found in val_seen.json")
        return None, None, None

    return pred_episode_steps, val_episode, val_seen_data


def create_simulator(scene_file, sensor_height, sensor_width, width, height, top_down_width, top_down_height, hfov,
                     vfov, topdown_vfov):
    # create simulator
    sim_settings = {
        "scene": scene_file,
        "default_agent": 0,
        "sensor_height": sensor_height,
        "sensor_width": sensor_width,
        "width": width,
        "height": height,
        "top-down width": top_down_width,
        "top-down height": top_down_height,
        "hfov": hfov,
        "vfov": vfov,
        "top_down_vfov": topdown_vfov

    }
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = sim_settings["scene"]
    backend_cfg.enable_physics = False

    # Set up the color sensor
    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [sim_settings["height"], sim_settings["width"]]
    sensor_cfg.position = [0, sim_settings["sensor_height"], 0]
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.uuid = "color_sensor"

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.hfov = np.pi / hfov  # 3.5
    color_sensor_spec.vfov = np.pi * vfov
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.channels_first = True
    color_sensor_spec.encoding = "png"

    # Set up 3D top-down view
    # center_x, center_y, center_z = sim_settings["center_coordinates"]
    topdown_sensor_spec = habitat_sim.CameraSensorSpec()
    topdown_sensor_spec.uuid = "topdown_sensor"
    topdown_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    topdown_sensor_spec.resolution = [sim_settings["top-down height"], sim_settings["top-down width"]]
    # topdown_sensor_spec.position = [center_x, center_y, center_z]
    topdown_sensor_spec.orientation = [-np.pi / 2.5, 0, 0]
    topdown_sensor_spec.vfov = np.pi * topdown_vfov
    topdown_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg, topdown_sensor_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg, agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Set up the color sensor for the first agent
    sim.config.agents[0].sensor_specifications = [color_sensor_spec]

    # Set up the topdown sensor for the second agent
    sim.config.agents[1].sensor_specifications = [topdown_sensor_spec]

    agent_state = habitat_sim.AgentState()
    agent = sim.initialize_agent(sim_settings["default_agent"], agent_state)
    agent_state = habitat_sim.AgentState()
    agent.set_state(agent_state, reset_sensors=True)

    return sim


def create_navmesh_map(sim, navmesh_file, map_height, floor_heights, height_samples=10):
    sim.pathfinder.load_nav_mesh(navmesh_file)
    nav_bounds_min, nav_bounds_max = sim.pathfinder.get_bounds()

    map_width = int(map_height * ((nav_bounds_max[0] - nav_bounds_min[0]) / (nav_bounds_max[2] - nav_bounds_min[2])))
    map_size = (map_width, map_height)

    blank_map = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)

    # Get the "viridis" colormap
    cmap = plt.get_cmap('viridis')

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
                # Map the floor value to a value between 0 and 1
                color = cmap(floor % cmap.N)[:3]  # Assuming floor is an integer index
                # Get the RGB color from the colormap
                color = tuple(int(val * 255) for val in color)
                blank_map[y, x] = color
            else:
                blank_map[y, x] = 255

    return blank_map, map_size, nav_bounds_min, nav_bounds_max


def draw_path_on_map(blank_map, path, map_size, nav_bounds_min, nav_bounds_max):
    # Get the "hot" colormap
    cmap = plt.get_cmap('hot')

    for i, step in enumerate(path):
        print('Step:', step)
        x, _, z = step
        x, y = world_to_map_coordinates(step, map_size, nav_bounds_min, nav_bounds_max)
        normalized_step_index = i / len(path)
        color = cmap(normalized_step_index)[:3]
        color = tuple(int(val * 255) for val in color)
        cv2.circle(blank_map, (x, y), 5, color, -1)


def calculate_rotations(path):
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
    return rotations, skipped_indices


def remove_skipped_indices(path, skipped_indices):
    for index in reversed(skipped_indices):
        del path[index]
    return path


def interpolate_path(path, num_interpolated_points):
    interpolator = scipy.interpolate.interp1d(np.arange(len(path)), path, axis=0)
    interpolated_path = interpolator(np.linspace(0, len(path) - 1, num_interpolated_points))
    return interpolated_path


def interpolate_rotations(rotations, num_interpolated_points):
    slerp = Slerp(np.linspace(0, len(rotations) - 1, len(rotations)), Rotation.from_quat(rotations))
    interpolated_rotations = slerp(np.linspace(0, len(rotations) - 1, num_interpolated_points)).as_quat()
    return interpolated_rotations


def visualize_trajectory(pred_json_file, val_seen_json_file, scene_directory, selected_episode_id, hfov,
                         vfov, topdown_vfov):
    # loading JSON file
    pred_episode_steps, val_episode, val_seen_data = load_data(pred_json_file, val_seen_json_file, selected_episode_id)
    if pred_episode_steps is None or val_episode is None or val_seen_data is None:
        return
    else:
        reference_path = val_episode["reference_path"]
        print(f"Reference Path: {reference_path}")
        instruction_text = val_episode["instruction"]
        print(f"Instruction Text: {instruction_text}")
        filename_without_ext = os.path.splitext(val_episode["scene_id"])[0]
        navmesh_file = os.path.join(scene_directory, filename_without_ext + ".navmesh")
        house_file = os.path.join(scene_directory, filename_without_ext + ".house")
        floor_heights = parse_house_file(house_file)
        floor_heights = adjust_floor_heights(floor_heights)
        print(f"Navmesh file path: {navmesh_file}")
        print(f"House file path: {house_file}")

        # getpath
        path = [step["position"] for step in pred_episode_steps if not step["stop"]]
        # set up 3d and top down view
        scene_file = os.path.join(scene_directory, val_episode["scene_id"])
        sensor_height = 0.7
        sensor_width = 1280
        width = 1280
        height = 640
        top_down_width = 720
        top_down_height = 1280
        # hfov = 3.5
        # vfov = 50
        # topdown_vfov = 45
        # center_coordinates = [0, 2.5, 0]

        sim = create_simulator(scene_file, sensor_height, sensor_width, width, height, top_down_width, top_down_height,
                               hfov,
                               vfov, topdown_vfov)

        # set up 2d map
        map_height = height  # specify desired map height
        blank_map, map_size, nav_bounds_min, nav_bounds_max = create_navmesh_map(sim, navmesh_file, map_height,
                                                                                 floor_heights)

        # Draw the path on the map
        draw_path_on_map(blank_map, reference_path, map_size, nav_bounds_min, nav_bounds_max)

        # if not path:
        #     continue
        num_interpolated_points = 600
        # Calculate rotations and skipped indices
        rotations, skipped_indices = calculate_rotations(path)

        # Remove skipped indices from the path
        path = remove_skipped_indices(path, skipped_indices)

        # Interpolate the path and rotations
        interpolated_path = interpolate_path(path, num_interpolated_points)

        interpolated_rotations = interpolate_rotations(rotations, num_interpolated_points)

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

            # Get the "cool" colormap
            cmap = plt.get_cmap('cool')
            color = cmap(0.0)[:3]
            agent_color = tuple(int(val * 255) for val in color)
            x, y = world_to_map_coordinates(agent_state.position, map_size, nav_bounds_min, nav_bounds_max)
            cv2.circle(blank_map, (x, y), 3, agent_color, -1)

            resized_map = cv2.resize(blank_map, (map_size[0], map_size[1]), interpolation=cv2.INTER_AREA)

            # Display the RGB observation
            cv2.imshow("RGB Observation", rgb_observation)

            # Display the map
            cv2.imshow("Map", resized_map)

            # Display the top-down view
            cv2.imshow("Top-down View", topdown_observation)

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()

        self.pred_json_file = None
        self.val_seen_json_file = None
        self.scene_directory = None

        self.initUI()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout()

        self.button_pred_json_file = QPushButton("Select Pred_val_seen", self)
        self.button_pred_json_file.clicked.connect(self.load_pred_json_file)

        self.button_val_seen_json_file = QPushButton("Select val_seen", self)
        self.button_val_seen_json_file.clicked.connect(self.load_val_seen_json_file)

        self.button_scene_directory = QPushButton("Select Scene Directory", self)
        self.button_scene_directory.clicked.connect(self.load_scene_directory)

        self.episode_id_label = QLabel("Enter Episode ID:", self)
        self.combo_episode_id = QComboBox(self)
        self.episode_range_label = QLabel(self)
        self.combo_episode_id.setEditable(True)

        self.hfov_label = QLabel("Enter hfov(Range 1.0-5):", self)
        self.hfov_entry = QLineEdit(self)

        self.vfov_label = QLabel("Enter vfov(Range 10-50):", self)
        self.vfov_entry = QLineEdit(self)

        self.topdown_vfov_label = QLabel("Enter topdown_vfov(Range 10-50):", self)
        self.topdown_vfov_entry = QLineEdit(self)

        self.instruction_label = QLabel("Instruction:", self)
        self.instruction_text_widget = QTextEdit(self)
        self.instruction_text_widget.setReadOnly(True)

        self.button_start = QPushButton("Start", self)
        self.button_start.clicked.connect(self.start)


        layout.addWidget(self.button_pred_json_file)
        layout.addWidget(self.button_val_seen_json_file)
        layout.addWidget(self.button_scene_directory)
        layout.addWidget(self.episode_id_label)
        layout.addWidget(self.combo_episode_id)
        layout.addWidget(self.episode_range_label)
        layout.addWidget(self.hfov_label)
        layout.addWidget(self.hfov_entry)
        layout.addWidget(self.vfov_label)
        layout.addWidget(self.vfov_entry)
        layout.addWidget(self.topdown_vfov_label)
        layout.addWidget(self.topdown_vfov_entry)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.instruction_text_widget)
        layout.addWidget(self.button_start)

        centralWidget.setLayout(layout)

    def load_pred_json_file(self):
        self.pred_json_file = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        # Update episode id choices based on the selected pred_json_file
        with open(self.pred_json_file, 'r') as f:
            pred_data = json.load(f)
        episode_ids = sorted(list(map(int, pred_data.keys())))
        self.combo_episode_id.addItems([str(e) for e in episode_ids])
        self.episode_range_label.setText(f"Episode range: {min(episode_ids)} - {max(episode_ids)}")

    def load_val_seen_json_file(self):
        self.val_seen_json_file = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]

    def load_scene_directory(self):
        self.scene_directory = QFileDialog.getExistingDirectory(self, "Select Directory")

    def start(self):
        selected_episode_id = self.combo_episode_id.currentText()
        if selected_episode_id not in [self.combo_episode_id.itemText(i) for i in range(self.combo_episode_id.count())]:
            QMessageBox.critical(self, "Invalid input", "The episode ID you entered does not exist.")
            return

        # Get the episode from the val_seen data
        with open(self.val_seen_json_file, 'r') as f:
            val_seen_data = json.load(f)
        # Find the corresponding episode in the val_seen data
        episode = next((ep for ep in val_seen_data['episodes'] if str(ep['episode_id']) == selected_episode_id),
                       None)
        if episode is None:  # No matching episode was found
            print(f"Episode {selected_episode_id} not found in val_seen.json")
            return

        # Display instruction
        self.instruction_text_widget.setPlainText("Instruction:\n" + str(episode['instruction']['instruction_text']))

        hfov = float(self.hfov_entry.text())
        vfov = int(self.vfov_entry.text())
        topdown_vfov = int(self.topdown_vfov_entry.text())
        visualize_trajectory(self.pred_json_file, self.val_seen_json_file, self.scene_directory, selected_episode_id,
                              hfov, vfov, topdown_vfov)


def main():
    app = QApplication(sys.argv)
    ex = GUI()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
