import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*experimental class.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*deprecated.*")

import json
import os
import sys
import datetime
import requests
import traceback
import pkg_resources

import qdarktheme
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QTabWidget, QFileDialog, QVBoxLayout, QWidget, QPushButton, 
                             QLineEdit, QGroupBox, QLabel, QSpinBox, QComboBox, QCheckBox, 
                             QScrollArea, QDialog, QMessageBox, QApplication, QHBoxLayout,
                             QFormLayout, QDoubleSpinBox)

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import InteractiveBrowserCredential

try:
    # Import the SfM script from the local directory
    from src.SfM import SfMWorkflow
    icon_src = "src"
except:
    # Import the SfM script from the local directory
    from SfM import SfMWorkflow
    icon_src = ""


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_icon_path(icon_name):
    """

    :param icon_name:
    :return:
    """
    return pkg_resources.resource_filename(icon_src, f'icons/{icon_name}')


def get_now():
    """
    Returns a timestamp; used for file and folder names
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SfMWorkflowApp(QDialog):
    def __init__(self, parent=None):
        super(SfMWorkflowApp, self).__init__(parent)
        
        self.setWindowTitle("SfM Workflow")
        main_window_icon_path = get_icon_path("duck.png")
        self.setWindowIcon(QIcon(main_window_icon_path))
        self.resize(1500, 600)
        
        # Add window flags to allow minimize and maximize buttons and show in fullscreen
        self.setWindowFlags(self.windowFlags() | 
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint)
        
        # Config path
        self.config_path = os.path.expanduser("~/.azureml/config.json")

        # Initialize member variables
        self.subscription_id_input = None
        self.resource_group_input = None
        self.workspace_name_input = None
        self.device_input = None
        
        # Replace target_percentage_input with four new inputs
        self.reprojection_error_input = None
        self.reconstruction_uncertainty_input = None
        self.projection_accuracy_input = None
        
        # Add keypoint and tiepoint limit inputs
        self.keypoint_limit_input = None
        self.tiepoint_limit_input = None
        
        self.detect_markers_input = None
        self.computes_input = None
        self.computes_list = []
        self.input_path_input = None
        self.output_path_input = None
        self.output_name_input = None

        # Create dictionaries for building and export functions
        self.building_functions = {}
        self.export_functions = {}

        # Initialize UI first
        self.initUI()
        
        # Then load config
        self.load_config()

    def create_azure_panel(self):
        # Create Azure Parameters Group Box
        azure_group = QGroupBox("Azure Parameters")
        azure_layout = QVBoxLayout()

        # Create Cloud Credentials Group Box
        cloud_groupbox = QGroupBox("Cloud Credentials")
        cloud_layout = QVBoxLayout()

        # Cloud Credentials Section
        credentials_description = QLabel("\nEnter your Azure credentials below")
        credentials_description.setWordWrap(True)
        cloud_layout.addWidget(credentials_description)
        
        # Credentials Form
        credentials_form = QFormLayout()
        self.subscription_id_input = QLineEdit(self)
        self.resource_group_input = QLineEdit(self)
        self.workspace_name_input = QLineEdit(self)
        
        credentials_form.addRow('Subscription ID:', self.subscription_id_input)
        credentials_form.addRow('Resource Group:', self.resource_group_input)
        credentials_form.addRow('Workspace Name:', self.workspace_name_input)
        cloud_layout.addLayout(credentials_form)
        
        # Authentication Buttons - Now stacked vertically
        button_layout = QVBoxLayout()
        save_credentials_button = QPushButton('Save Credentials')
        save_credentials_button.clicked.connect(self.save_credentials)
        authenticate_button = QPushButton('Authenticate')
        authenticate_button.clicked.connect(self.authenticate)
        button_layout.addWidget(save_credentials_button)
        button_layout.addWidget(authenticate_button)
        cloud_layout.addLayout(button_layout)

        cloud_groupbox.setLayout(cloud_layout)
        azure_layout.addWidget(cloud_groupbox)
        
        # Create Compute Parameters Group Box
        compute_groupbox = QGroupBox("Compute Parameters")
        compute_layout = QVBoxLayout()
        
        compute_description = QLabel("\nSelect a compute instance to run the workflow after authenticating")
        compute_description.setWordWrap(True)
        compute_layout.addWidget(compute_description)
        
        compute_form = QFormLayout()
        self.computes_input = QComboBox(self)
        compute_form.addRow('Computes Running:', self.computes_input)
        compute_layout.addLayout(compute_form)
        
        compute_groupbox.setLayout(compute_layout)
        azure_layout.addWidget(compute_groupbox)

        azure_group.setLayout(azure_layout)
        return azure_group

    def create_io_panel(self):
        """Create a panel for input/output parameters."""
        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()

        io_description = QLabel("\nEnter the input and output paths below; URI (Azure) or local path")
        io_description.setWordWrap(True)
        io_layout.addWidget(io_description)
        
        io_form = QFormLayout()
        self.input_path_input = QLineEdit()
        self.output_path_input = QLineEdit()
        self.output_name_input = QLineEdit()
        self.output_name_input.setText(f"project_{get_now()}")
        
        io_form.addRow("Input Path:", self.input_path_input)
        io_form.addRow("Output Path:", self.output_path_input)
        io_form.addRow("Output Name:", self.output_name_input)
        io_layout.addLayout(io_form)

        io_group.setLayout(io_layout)
        return io_group

    def create_points_panel(self):
        """Create a panel for keypoint and tiepoint parameters."""
        points_group = QGroupBox("Point Limit Parameters")
        points_layout = QFormLayout()
        
        # Keypoint and Tiepoint Parameters
        self.keypoint_limit_input = QSpinBox(self)
        self.keypoint_limit_input.setRange(0, 1000000)
        self.keypoint_limit_input.setValue(40000)
        points_layout.addRow("Keypoint Limit:", self.keypoint_limit_input)
        
        self.tiepoint_limit_input = QSpinBox(self)
        self.tiepoint_limit_input.setRange(0, 1000000)
        self.tiepoint_limit_input.setValue(10000)
        points_layout.addRow("Tiepoint Limit:", self.tiepoint_limit_input)
        
        points_group.setLayout(points_layout)
        return points_group

    def create_error_reduction_panel(self):
        """Create a panel for error reduction parameters."""
        error_group = QGroupBox("Error Reduction Parameters")
        error_layout = QFormLayout()
        
        # Error Reduction Parameters
        self.reprojection_error_input = QSpinBox(self)
        self.reprojection_error_input.setRange(0, 100)
        self.reprojection_error_input.setValue(10)
        error_layout.addRow("Reprojection Error:", self.reprojection_error_input)
        
        self.reconstruction_uncertainty_input = QSpinBox(self)
        self.reconstruction_uncertainty_input.setRange(0, 100)
        self.reconstruction_uncertainty_input.setValue(50)
        error_layout.addRow("Reconstruction Uncertainty:", self.reconstruction_uncertainty_input)
        
        self.projection_accuracy_input = QSpinBox(self)
        self.projection_accuracy_input.setRange(0, 100)
        self.projection_accuracy_input.setValue(50)
        error_layout.addRow("Projection Accuracy:", self.projection_accuracy_input)
        
        error_group.setLayout(error_layout)
        return error_group

    def create_sfm_panel(self):
        # Create SfM Functions Group
        sfm_group = QGroupBox("SfM Parameters")
        sfm_layout = QVBoxLayout()
        
        sfm_description = QLabel("\nSelect the functions to run below")
        sfm_description.setWordWrap(True)
        sfm_layout.addWidget(sfm_description)
        
        # Processing Parameters
        params_form = QFormLayout()
        
        self.device_input = QSpinBox(self)
        self.device_input.setValue(0)
        params_form.addRow("Device:", self.device_input)
        
        self.detect_markers_input = QComboBox(self)
        self.detect_markers_input.addItems(['True', 'False'])
        self.detect_markers_input.setCurrentText('False')
        params_form.addRow("Detect Markers:", self.detect_markers_input)
        
        sfm_layout.addLayout(params_form)
        
        # Initialize checkboxes and combo boxes for building functions
        self.building_functions = {}
        self.building_combo_boxes = {}
        
        # Define options for each building function
        build_options = {
            "add_photos": [],
            "align_cameras": ["lowest", "low", "medium", "high", "highest"],
            "optimize_cameras": [],
            "build_depth_maps": ["lowest", "low", "medium", "high", "highest"],
            "build_point_cloud": [],
            "build_mesh": ["low", "medium", "high"],
            "build_texture": [],
            "build_dem": [],
            "build_ortho": []
        }
        
        # Create checkboxes and combo boxes
        for function_name, options in build_options.items():
            self.building_functions[function_name] = QCheckBox(function_name.replace('_', ' ').title(), self)
            combo = QComboBox(self)
            combo.addItems(options)
            self.building_combo_boxes[function_name] = combo
            
            if function_name == "align_cameras":
                combo.setCurrentText("medium")
                
            if function_name == "build_depth_maps":
                combo.setCurrentText("medium")
                
            if function_name == "build_mesh":
                combo.setCurrentText("medium")
        
        # Initialize export function checkboxes
        self.export_functions = {
            "export_viscore": QCheckBox("Export Viscore", self),
            "export_meta": QCheckBox("Export Meta", self),
            "export_cameras": QCheckBox("Export Cameras", self),
            "export_point_cloud": QCheckBox("Export Point Cloud", self),
            "export_potree": QCheckBox("Export Potree", self),
            "export_mesh": QCheckBox("Export Mesh", self),
            "export_texture": QCheckBox("Export Texture", self),
            "export_dem": QCheckBox("Export DEM", self),
            "export_ortho": QCheckBox("Export Orthomosaic", self),
            "export_report": QCheckBox("Export Report", self)
        }
        
        # Create horizontal layout for building functions and their options
        functions_layout = QHBoxLayout()
        
        # Options GroupBox - NEW
        options_group = QGroupBox("Building Options")
        options_layout = QFormLayout()
        
        # Add combo boxes to the options form layout
        for function_name, combo_box in self.building_combo_boxes.items():
            display_name = function_name.replace('_', ' ').title()
            options_layout.addRow(f"{display_name}:", combo_box)
        
        options_group.setLayout(options_layout)
        functions_layout.addWidget(options_group)
        
        # Building GroupBox
        building_group = QGroupBox("Building Functions")
        building_layout = QVBoxLayout()
        
        # Add just the checkboxes to the building layout
        for function_name, checkbox in self.building_functions.items():
            building_layout.addWidget(checkbox)
        
        building_layout.addStretch()
        building_group.setLayout(building_layout)
        functions_layout.addWidget(building_group)
        
        # Export GroupBox
        export_group = QGroupBox("Export Functions")
        export_layout = QVBoxLayout()
        for checkbox in self.export_functions.values():
            export_layout.addWidget(checkbox)
        export_layout.addStretch()
        export_group.setLayout(export_layout)
        functions_layout.addWidget(export_group)
        
        sfm_layout.addLayout(functions_layout)
        
        # Add points panel (moved from above)
        sfm_layout.addWidget(self.create_points_panel())
        # Add error reduction panel at the bottom (moved from above)
        sfm_layout.addWidget(self.create_error_reduction_panel())
        
        sfm_group.setLayout(sfm_layout)
        return sfm_group

    def initUI(self):
        # Create main layout
        main_layout = QHBoxLayout()
        
        # Left Panel
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.create_azure_panel())
        left_panel.addWidget(self.create_io_panel())  # Add IO panel to the left side
        left_panel.addStretch()
        
        # Right Panel
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.create_sfm_panel())
        
        # Run Button Group
        run_button_layout = QHBoxLayout()
        
        # Run Local Button
        run_local_button = QPushButton('Run Locally')
        run_local_button.clicked.connect(self.run_workflow_locally)
        run_local_button.setStyleSheet("background-color: rgba(255, 0, 0, 0.7); color: white;")
        run_button_layout.addWidget(run_local_button)
        
        # Run Azure Button
        run_azure_button = QPushButton('Run on Azure')
        run_azure_button.clicked.connect(self.run_workflow_azure)
        run_azure_button.setStyleSheet("background-color: rgba(0, 255, 0, 0.7); color: white;")
        run_button_layout.addWidget(run_azure_button)
        
        # Add run buttons to right panel
        right_panel.addLayout(run_button_layout)
        
        # Add panels to main layout
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        self.setLayout(main_layout)

    def download_sfm_script(self):
        """Method to download the SfM script from the GitHub repository."""
        try:
            # Specify the output directory and file path
            output_dir = os.path.expanduser("~/.azureml")
            output_path = os.path.join(output_dir, "SfM.py")
            if os.path.exists(output_path):
                os.remove(output_path)

            # Download the SfM script from the GitHub repository
            url = "https://raw.githubusercontent.com/Jordan-Pierce/Metashape-Azure/refs/heads/main/src/SfM.py"
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return output_path
            else:
                raise Exception("Failed to download the SfM script.")
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return ""

    def choose_sfm_script(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Choose SfM Script",
                                                   "",
                                                   "Python Files (*.py)",
                                                   options=options)
        if file_name:
            self.sfm_script_path_label.setText(file_name)  # Update the label with the selected file path

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as config_file:
                config = json.load(config_file)
                self.subscription_id_input.setText(config.get("subscription_id", ""))
                self.resource_group_input.setText(config.get("resource_group", ""))
                self.workspace_name_input.setText(config.get("workspace_name", ""))

    def save_credentials(self):
        """Method to handle saving the credentials when the 'Save Credentials' button is pressed."""
        credentials = {
            "subscription_id": self.subscription_id_input.text(),
            "resource_group": self.resource_group_input.text(),
            "workspace_name": self.workspace_name_input.text(),
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as config_file:
            json.dump(credentials, config_file, indent=4)
        QMessageBox.information(self, 'Success', 'Credentials saved successfully!')

        # Authenticate with the new credentials
        self.authenticate()

    def authenticate(self):
        """Method to handle authentication when the 'Authenticate' button is pressed."""

        # Make cursor busy
        self.setCursor(Qt.WaitCursor)

        # Get the Azure credentials
        self.get_azure_credentials()
        # Fill the computes dropdown with a list of compute options
        self.get_azure_compute_names()

        # Make cursor normal
        self.setCursor(Qt.ArrowCursor)

    def get_azure_credentials(self):
        """Method to handle authentication when the 'Authenticate' button is pressed."""
        try:
            self.creds = InteractiveBrowserCredential()
            self.ml_client = MLClient.from_config(credential=self.creds, path=self.config_path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return

    def get_azure_compute_names(self):
        """Method to fill the computes dropdown with a list of compute options."""
        self.computes_input.clear()
        try:
            compute_list = list(self.ml_client.compute.list())
            for compute in compute_list:
                if compute._type == 'amlcompute':
                    self.computes_input.addItem(f"{compute.name}")
                elif compute._type == 'computeinstance':
                    if 'Running' in compute._state:
                        self.computes_input.addItem(f"{compute.name}")
                else:
                    pass
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def get_azure_compute(self):
        """Method to get the selected compute option."""
        compute_name = self.computes_input.currentText()
        try:
            compute = self.ml_client.compute.get(compute_name)
            return compute
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return None

    def prepare_workflow(self):
        """Method to run the SfM workflow."""
        try:
            self.subscription_id = self.subscription_id_input.text()
            self.resource_group = self.resource_group_input.text()
            self.workspace_name = self.workspace_name_input.text()
            self.compute_name = self.computes_input.currentText()

            # Method calls to get input / output strings
            self.input_dir = self.input_path_input.text()
            self.output_dir = self.output_path_input.text()
            self.output_name = self.output_name_input.text()
            
            # Validate input / output strings
            if not self.input_dir:
                QMessageBox.critical(self, 'Error', "Please enter an input path!")
                return
            
            if not self.output_dir:
                QMessageBox.critical(self, 'Error', "Please enter an output path!")
                return
            
            if not self.output_name:
                self.output_name = get_now()
                
            # Convert drive letter paths to double-slashed format
            if ':\\' in self.input_dir:
                drive = self.input_dir.split(':\\')[0]
                path = self.input_dir.split(':\\')[1]
                self.input_dir = f'{drive}://{path}'
                    
            if ':\\' in self.output_dir:
                drive = self.output_dir.split(':\\')[0]
                path = self.output_dir.split(':\\')[1]
                self.output_dir = f'{drive}://{path}'
                    
            # Normalize path separators, preserving leading double slashes
            if '://' in self.input_dir:
                protocol = self.input_dir.split('://')[0] + '://'
                path = self.input_dir.split('://')[1]
                self.input_dir = protocol + os.path.normpath(path).replace('\\', '/')
            else:
                self.input_dir = os.path.normpath(self.input_dir).replace('\\', '/')
                
            if '://' in self.output_dir:
                protocol = self.output_dir.split('://')[0] + '://'
                path = self.output_dir.split('://')[1]
                self.output_dir = protocol + os.path.normpath(path).replace('\\', '/')
            else:
                self.output_dir = os.path.normpath(self.output_dir).replace('\\', '/')
            
            # Strip leading and trailing whitespace
            self.output_dir = self.output_dir.strip()
            self.output_name = self.output_name.strip()
            
            output_path = f"{self.output_dir}/{self.output_name}"
            print("Input Path:", self.input_dir)
            print("Output Path:", output_path)
            
            if os.path.exists(output_path):
                QMessageBox.critical(self, 'Error', "Output directory already exists!")
                return
                
            self.device = int(self.device_input.value())
            self.detect_markers = self.detect_markers_input.currentText() == 'True'

            self.reprojection_error = self.reprojection_error_input.value()
            self.reconstruction_uncertainty = self.reconstruction_uncertainty_input.value()
            self.projection_accuracy = self.projection_accuracy_input.value()
            self.keypoint_limit = self.keypoint_limit_input.value()
            self.tiepoint_limit = self.tiepoint_limit_input.value()

        except Exception as e:
            print(f"ERROR: {e}")
            QMessageBox.critical(self, 'Error', "Failed to complete all steps the workflow!")

    def run_workflow_locally(self):
        """Method to run the SfM workflow locally."""
        try:
            self.prepare_workflow()
        except Exception as e:
            print(f"ERROR: {e}")
            QMessageBox.critical(self, 'Error', "Failed to prepare the workflow!")
            return 

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Create a dictionary of building function parameters
            building_params = {}
            for function_name, checkbox in self.building_functions.items():
                if checkbox.isChecked():
                    # Get the selected option from the corresponding combo box
                    combo = self.building_combo_boxes[function_name]
                    if combo.count() > 0:
                        option = combo.currentText()
                    
                        # If the box isn't selected, it won't matter what the option is
                        building_params[function_name] = option
            
            # Pass main parameters to workflow
            SfMWorkflow(device=self.device,
                        input_dir=self.input_dir,
                        project_file="",
                        output_name=self.output_name,
                        output_dir=self.output_dir,
                        reprojection_error=self.reprojection_error,
                        reconstruction_uncertainty=self.reconstruction_uncertainty,
                        projection_accuracy=self.projection_accuracy,
                        keypoint_limit=self.keypoint_limit,
                        tiepoint_limit=self.tiepoint_limit,
                        detect_markers=self.detect_markers,
                        add_photos=self.building_functions['add_photos'].isChecked(),
                        align_cameras=self.building_functions['align_cameras'].isChecked(),
                        optimize_cameras=self.building_functions['optimize_cameras'].isChecked(),
                        build_depth_maps=self.building_functions['build_depth_maps'].isChecked(),
                        build_point_cloud=self.building_functions['build_point_cloud'].isChecked(),
                        build_mesh=self.building_functions['build_mesh'].isChecked(),
                        build_texture=self.building_functions['build_texture'].isChecked(),
                        build_dem=self.building_functions['build_dem'].isChecked(),
                        build_ortho=self.building_functions['build_ortho'].isChecked(),
                        export_viscore=self.export_functions['export_viscore'].isChecked(),
                        export_meta=self.export_functions['export_meta'].isChecked(),
                        export_cameras=self.export_functions['export_cameras'].isChecked(),
                        export_point_cloud=self.export_functions['export_point_cloud'].isChecked(),
                        export_potree=self.export_functions['export_potree'].isChecked(),
                        export_mesh=self.export_functions['export_mesh'].isChecked(),
                        export_texture=self.export_functions['export_texture'].isChecked(),
                        export_dem=self.export_functions['export_dem'].isChecked(),
                        export_ortho=self.export_functions['export_ortho'].isChecked(),
                        export_report=self.export_functions['export_report'].isChecked(),
                        building_params=building_params)  # Pass the building parameters
            
            # Success message
            QMessageBox.information(self, 'Success', 'Workflow completed successfully!')

        except Exception as e:
            print(f"ERROR: {e}")
            print(f"Failed to run all processes in the workflow locally!")
            traceback.print_exc()
            
        finally:
            # Make cursor normal
            QApplication.restoreOverrideCursor()

    def run_workflow_azure(self):
        """Method to run the SfM workflow on Azure."""
        try:
            self.prepare_workflow()
        except Exception as e:
            print(f"ERROR: {e}")
            QMessageBox.critical(self, 'Error', "Failed to prepare the workflow!")
            return
        
        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            if not self.compute_name:
                raise Exception("No compute selected. Please authenticate and select a compute.")

            # Get the SfM script path
            sfm_script_path = self.download_sfm_script()
            if not os.path.exists(sfm_script_path):
                raise Exception("SfM script file not found.")

            # Submit the job to the compute
            print(f"Submitting job to compute: {self.compute_name}")

            # load in the input data information from the function def
            input_dir = self.input_dir
            input_mode = InputOutputModes.DOWNLOAD

            # load in the output data info from the function def
            output_dir = f"{self.output_dir}/{self.output_name}"
            output_mode = InputOutputModes.UPLOAD

            # get a local instance of the compute info
            compute = self.ml_client.compute.get(self.compute_name)

            # create input and output dictionaries to use in the command calls later
            input = {
                "input_data": Input(type=AssetTypes.URI_FOLDER,
                                    path=input_dir,
                                    mode=input_mode)
            }
            output = {
                "output_data": Output(type=AssetTypes.URI_FOLDER,
                                      path=output_dir,
                                      mode=output_mode)
            }

            # create linux command line commands to be sent to the compute target
            command_args = [
                f'python SfM.py',
                f'--input_dir ${{inputs.input_data}}',
                f'--output_dir ${{outputs.output_data}}',
                f'--output_name {self.output_name}',
                f'--device {self.device_input.value()}',
                f'--reprojection_error {self.reprojection_error_input.value()}',
                f'--reconstruction_uncertainty {self.reconstruction_uncertainty_input.value()}',
                f'--projection_accuracy {self.projection_accuracy_input.value()}',
                f'--keypoint_limit {self.keypoint_limit_input.value()}',
                f'--tiepoint_limit {self.tiepoint_limit_input.value()}',
            ]

            if self.detect_markers_input.currentText() == 'True':
                command_args.append('--detect_markers')

            # Add building functions with their options
            for function_name, checkbox in self.building_functions.items():
                if checkbox.isChecked():
                    combo = self.building_combo_boxes[function_name]
                    if combo.count() > 0 and combo.currentText():
                        option = combo.currentText()
                        # For functions with options, pass them as arguments
                        command_args.append(f'--{function_name} {option}')
                    else:
                        # For functions without options, just add the flag
                        command_args.append(f'--{function_name}')

            # Add export functions
            for function_name, checkbox in self.export_functions.items():
                if checkbox.isChecked():
                    command_args.append(f'--{function_name}')

            command_str = ' '.join(command_args)

            transfer_data = command(
                code=sfm_script_path,
                command=command_str,
                inputs=input,
                outputs=output,
                environment="metashape-env@latest",
                compute=compute.name
            )

            # submit the job to the compute through the client
            returned_job = self.ml_client.jobs.create_or_update(transfer_data)
            print(f"STATUS: {returned_job.studio_url}")
            
            # Success message
            QMessageBox.information(self, 'Success', 'Workflow submitted successfully!')

        except Exception as e:
            QMessageBox.critical(self, 
                                 'Error', 
                                 'Failed to submit the job to the compute. Please check the parameters and try again.')
            print(f"ERROR: {e}")
            traceback.print_exc()
        
        finally:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            
            
# ----------------------------------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------------------------------


def metashape_app():
    app = QApplication.instance()
    parent = app.activeWindow()
    dlg = SfMWorkflowApp(parent)
    dlg.exec_()


def main_function():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    dlg = SfMWorkflowApp()
    dlg.exec_()


if __name__ == "__main__":
    try:
        import Metashape
        
        # If second argument is 'main', run main_function instead
        if len(sys.argv) > 1 and sys.argv[1] == 'app':
            main_function()
        else:
            label = "Scripts/Metashape-Azure"
            Metashape.app.addMenuItem(label, metashape_app)
            print("To execute this script press {}".format(label))
        
    except Exception as e:
        print("Failed to load Metashape module.")
        print(f"ERROR: {e}")

