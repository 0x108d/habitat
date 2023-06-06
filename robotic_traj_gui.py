import tkinter as tk
from tkinter import filedialog, ttk
import json
import cv2
from traj_visualization import visualize_trajectory


class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.pred_json_file = None
        self.val_seen_json_file = None
        self.scene_directory = None
        self.episode_id = tk.StringVar()

        self.button_pred_json_file = tk.Button(self, text="Select pred_json_file", command=self.load_pred_json_file)
        self.button_pred_json_file.pack()

        self.button_val_seen_json_file = tk.Button(self, text="Select val_seen_json_file",
                                                   command=self.load_val_seen_json_file)
        self.button_val_seen_json_file.pack()

        self.button_scene_directory = tk.Button(self, text="Select Scene Directory", command=self.load_scene_directory)
        self.button_scene_directory.pack()

        self.combo_episode_id = ttk.Combobox(self, textvariable=self.episode_id)
        self.combo_episode_id.pack()

        self.button_start = tk.Button(self, text="Start", command=self.start)
        self.button_start.pack()

    def load_pred_json_file(self):
        self.pred_json_file = filedialog.askopenfilename()
        # Update episode id choices based on the selected pred_json_file
        with open(self.pred_json_file, 'r') as f:
            pred_data = json.load(f)
        self.combo_episode_id['values'] = list(pred_data.keys())

    def load_val_seen_json_file(self):
        self.val_seen_json_file = filedialog.askopenfilename()

    def load_scene_directory(self):
        self.scene_directory = filedialog.askdirectory()

    def start(self):
        selected_episode_id = self.episode_id.get()
        visualize_trajectory(self.pred_json_file, self.val_seen_json_file, self.scene_directory, selected_episode_id)


gui = GUI()
gui.mainloop()
