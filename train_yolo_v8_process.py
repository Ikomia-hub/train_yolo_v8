# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from train_yolo_v8.utils.ikutils import prepare_dataset
import os
import yaml
from ultralytics import YOLO
from datetime import datetime
import torch
from train_yolo_v8.utils import custom_callbacks
from ultralytics import download, settings

# Update a setting
settings.update({'mlflow': False})

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class TrainYoloV8Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)

        dataset_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "dataset")
        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "yolov8m"
        self.cfg["epochs"] = 100
        self.cfg["batch_size"] = 8
        self.cfg["input_size"] = 640
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["workers"] = 0
        self.cfg["optimizer"] = "auto"
        self.cfg["weight_decay"] = 0.0005
        self.cfg["momentum"] = 0.937
        self.cfg["lr0"] = 0.01
        self.cfg["lrf"] = 0.01
        self.cfg["config_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(
            os.path.realpath(__file__)) + "/runs/"

    def set_values(self, param_map):
        self.cfg["dataset_folder"] = str(param_map["dataset_folder"])
        self.cfg["model_name"] = str(param_map["model_name"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["workers"] = int(param_map["workers"])
        self.cfg["optimizer"] = str(param_map["optimizer"])
        self.cfg["weight_decay"] = float(param_map["weight_decay"])
        self.cfg["momentum"] = float(param_map["momentum"])
        self.cfg["lr0"] = float(param_map["lr0"])
        self.cfg["lrf"] = float(param_map["lrf"])
        self.cfg["config_file"] = param_map["config_file"]
        self.cfg["dataset_split_ratio"] = float(
            param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = str(param_map["output_folder"])


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainYoloV8(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Create parameters class
        if param is None:
            self.set_param_object(TrainYoloV8Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.model_weights = None
        self.enable_tensorboard(True)
        self.model = None
        self.stop_training = False
        self.repo = 'ultralytics/assets'
        self.version = 'v0.0.0'

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        param = self.get_param_object()
        dataset_input = self.get_input(0)

        # Conversion from Ikomia dataset to YoloV8 dataset
        print("Preparing dataset...")
        dataset_yaml = prepare_dataset(dataset_input, param.cfg["dataset_folder"],
                                       param.cfg["dataset_split_ratio"])

        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Create a YOLO model instance
        self.device = 0 if torch.cuda.is_available() else torch.device("cpu")
        if param.cfg["config_file"] != "":
            # Load the YAML config file
            with open(param.cfg["config_file"], 'r') as file:
                config_file = yaml.safe_load(file)
            self.model_weights = config_file["model"]
        else:
            # Set path
            model_folder = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "weights")
            self.model_weights = os.path.join(
                str(model_folder), f'{param.cfg["model_name"]}.pt')
            # Download model if not exist
            if not os.path.isfile(self.model_weights):
                url = f'https://github.com/{self.repo}/releases/download/{self.version}/{param.cfg["model_name"]}.pt'
                download(url=url, dir=model_folder, unzip=True)
        self.model = YOLO(self.model_weights)

        # Add custom MLflow callback to the model
        self.model.add_callback(
            'on_fit_epoch_end', custom_callbacks.on_fit_epoch_end)

        # Create output folder
        experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(param.cfg["output_folder"], exist_ok=True)
        output_folder = os.path.join(
            param.cfg["output_folder"], experiment_name)
        os.makedirs(output_folder, exist_ok=True)

        # Train the model
        if param.cfg["config_file"]:
            # Extract the custom argument-value pairs
            custom_args = {k: v for k, v in config_file.items()}
            self.model.train(**custom_args)

        else:
            self.model.train(
                data=dataset_yaml,
                epochs=param.cfg["epochs"],
                imgsz=param.cfg["input_size"],
                batch=param.cfg["batch_size"],
                workers=param.cfg["workers"],
                optimizer=param.cfg["optimizer"],
                momentum=param.cfg["momentum"],
                weight_decay=param.cfg["weight_decay"],
                lr0=param.cfg["lr0"],
                lrf=param.cfg["lrf"],
                pretrained=True,
                device=self.device,
                project=output_folder,
            )

        # Reset settings to default values
        settings.reset()

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

    def stop(self):
        super().stop()
        self.model.add_callback("on_train_epoch_end",
                                custom_callbacks.on_train_epoch_end)
        print("Stopping requested, training will stop after the current epoch.")
        self.stop_training = True

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------


class TrainYoloV8Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_yolo_v8"
        self.info.short_description = "Train YOLOv8 object detection models."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "2.1.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "AGPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://docs.ultralytics.com/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/train_yolo_v8"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        # Keywords used for search
        self.info.keywords = "YOLO, object, detection, ultralytics, real-time"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create process object
        return TrainYoloV8(self.info.name, param)
