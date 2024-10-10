import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*experimental class.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*deprecated.*")

import json
import os
import sys
import datetime
import pkg_resources

import requests
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QTabWidget, QFileDialog, QVBoxLayout, QWidget, QPushButton, QLineEdit, QGroupBox, QLabel,
                               QSpinBox, QComboBox, QCheckBox, QScrollArea, QDialog, QMessageBox, QApplication)
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
        self.resize(600, 600)

        self.config_path = os.path.expanduser("~/.azureml/config.json")

        self.subscription_id_input = QLineEdit(self)
        self.resource_group_input = QLineEdit(self)
        self.workspace_name_input = QLineEdit(self)

        self.device_input = QSpinBox(self)
        self.quality_input = QComboBox(self)
        self.target_percentage_input = QSpinBox(self)

        self.computes_input = QComboBox(self)  # Dropdown for computes
        self.computes_list = []  # Empty list to store compute options

        self.building_functions = {
            "add_photos": QCheckBox("Add Photos", self),
            "align_cameras": QCheckBox("Align Cameras", self),
            "optimize_cameras": QCheckBox("Optimize Cameras", self),
            "build_depth_maps": QCheckBox("Build Depth Maps", self),
            "build_point_cloud": QCheckBox("Build Point Cloud", self),
            "build_dem": QCheckBox("Build DEM", self),
            "build_ortho": QCheckBox("Build Orthomosaic", self)
        }

        self.export_functions = {
            "export_cameras": QCheckBox("Export Cameras", self),
            "export_point_cloud": QCheckBox("Export Point Cloud", self),
            "export_dem": QCheckBox("Export DEM", self),
            "export_ortho": QCheckBox("Export Orthomosaic", self),
            "export_report": QCheckBox("Export Report", self)
        }

        self.input_path_button = QPushButton("Choose Input Directory", self)
        self.input_path_button.clicked.connect(self.choose_input_directory)
        self.input_path_display = QLineEdit(self)
        self.input_path_display.setReadOnly(True)

        self.output_path_button = QPushButton("Choose Output Directory", self)
        self.output_path_button.clicked.connect(self.choose_output_directory)
        self.output_path_display = QLineEdit(self)
        self.output_path_display.setReadOnly(True)

        self.main_tab_widget = QTabWidget(self)
        self.main_tab_widget.currentChanged.connect(self.adjust_tab_height)

        self.initUI()
        self.load_config()

    def adjust_tab_height(self):
        current_tab_index = self.main_tab_widget.currentIndex()
        current_tab = self.main_tab_widget.widget(current_tab_index)
        current_tab_height = current_tab.sizeHint().height()
        self.main_tab_widget.setFixedHeight(current_tab_height)

    def initUI(self):
        layout = QVBoxLayout(self)

        ###
        # Cloud Credentials Panel
        cloud_credentials_group = QGroupBox("Cloud Credentials", self)
        cloud_credentials_layout = QVBoxLayout(cloud_credentials_group)

        description = QLabel("Enter your Azure credentials below")
        description.setWordWrap(True)
        cloud_credentials_layout.addWidget(description)

        subscription_id_label = QLabel('Subscription ID:', self)
        subscription_id_label.setToolTip("Enter your Azure subscription ID.")
        cloud_credentials_layout.addWidget(subscription_id_label)
        cloud_credentials_layout.addWidget(self.subscription_id_input)

        resource_group_label = QLabel('Resource Group:', self)
        resource_group_label.setToolTip("Enter your Azure resource group name.")
        cloud_credentials_layout.addWidget(resource_group_label)
        cloud_credentials_layout.addWidget(self.resource_group_input)

        workspace_name_label = QLabel('Workspace Name:', self)
        workspace_name_label.setToolTip("Enter your Azure workspace name.")
        cloud_credentials_layout.addWidget(workspace_name_label)
        cloud_credentials_layout.addWidget(self.workspace_name_input)

        # Save Credentials Button
        save_credentials_button = QPushButton('Save Credentials', self)
        save_credentials_button.clicked.connect(self.save_credentials)
        cloud_credentials_layout.addWidget(save_credentials_button)

        # Authenticate Button
        authenticate_button = QPushButton('Authenticate', self)
        authenticate_button.clicked.connect(self.authenticate)
        cloud_credentials_layout.addWidget(authenticate_button)

        # Computes Dropdown
        computes_label = QLabel('Computes:', self)
        computes_label.setToolTip("Select a compute option.")
        cloud_credentials_layout.addWidget(computes_label)
        cloud_credentials_layout.addWidget(self.computes_input)

        ###
        # Input & Output Directory

        # Local Tab
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)

        local_io_group = QGroupBox("Input / Output")
        local_io_layout = QVBoxLayout()

        description = QLabel("Select the input and output directories below")
        description.setWordWrap(True)
        local_io_layout.addWidget(description)

        input_path_label = QLabel("Input Directory:")
        local_io_layout.addWidget(input_path_label)
        local_io_layout.addWidget(self.input_path_button)
        local_io_layout.addWidget(self.input_path_display)

        output_path_label = QLabel("Output Directory:")
        local_io_layout.addWidget(output_path_label)
        local_io_layout.addWidget(self.output_path_button)
        local_io_layout.addWidget(self.output_path_display)

        local_io_group.setLayout(local_io_layout)
        local_layout.addWidget(local_io_group)

        self.main_tab_widget.addTab(local_tab, "Local")

        # Azure Tab
        azure_tab = QWidget()
        azure_layout = QVBoxLayout(azure_tab)

        azure_io_group = QGroupBox("Input / Output")
        azure_io_layout = QVBoxLayout()

        description = QLabel("Enter the input and output paths below")
        description.setWordWrap(True)
        azure_io_layout.addWidget(description)

        self.azure_io_tab_widget = QTabWidget()

        # URI Tab
        io_uri_widget = QWidget()
        io_uri_layout = QVBoxLayout()

        input_uri_label = QLabel("Input URI:")
        self.input_uri_input = QLineEdit()
        output_uri_label = QLabel("Output URI:")
        self.output_uri_input = QLineEdit()
        io_uri_layout.addWidget(input_uri_label)
        io_uri_layout.addWidget(self.input_uri_input)
        io_uri_layout.addWidget(output_uri_label)
        io_uri_layout.addWidget(self.output_uri_input)

        io_uri_widget.setLayout(io_uri_layout)

        # URL Tab
        io_url_widget = QWidget()
        io_url_layout = QVBoxLayout()

        input_url_label = QLabel("Input URL:")
        self.input_url_input = QLineEdit()
        output_url_label = QLabel("Output URL:")
        self.output_url_input = QLineEdit()
        io_url_layout.addWidget(input_url_label)
        io_url_layout.addWidget(self.input_url_input)
        io_url_layout.addWidget(output_url_label)
        io_url_layout.addWidget(self.output_url_input)

        io_url_widget.setLayout(io_url_layout)

        self.azure_io_tab_widget.addTab(io_uri_widget, "URI")
        self.azure_io_tab_widget.addTab(io_url_widget, "URL")

        azure_io_layout.addWidget(self.azure_io_tab_widget)
        azure_io_group.setLayout(azure_io_layout)

        azure_layout.addWidget(cloud_credentials_group)
        azure_layout.addWidget(azure_io_group)

        self.main_tab_widget.addTab(azure_tab, "Azure")

        # Add the main tab widget to the layout
        layout.addWidget(self.main_tab_widget)

        ###
        # SfM Group Panel
        sfm_functions_group = QGroupBox("SfM Functions", self)
        sfm_functions_layout = QVBoxLayout(sfm_functions_group)

        description = QLabel("Select the functions to run below")
        description.setWordWrap(True)
        sfm_functions_layout.addWidget(description)

        # Device Input
        device_label = QLabel('Device:')
        device_label.setToolTip("Enter the device index for processing.")
        sfm_functions_layout.addWidget(device_label)
        self.device_input.setValue(0)
        sfm_functions_layout.addWidget(self.device_input)

        # Quality
        quality_label = QLabel('Quality:')
        quality_label.setToolTip("Select the quality level for the processing.")
        sfm_functions_layout.addWidget(quality_label)
        self.quality_input.addItems(['lowest', 'low', 'medium', 'high', 'highest'])
        self.quality_input.setCurrentText('medium')
        sfm_functions_layout.addWidget(self.quality_input)

        # Target Percentage
        target_percentage_label = QLabel('Target Percentage:')
        target_percentage_label.setToolTip("Set the target percentage for filtering points.")
        sfm_functions_layout.addWidget(target_percentage_label)
        self.target_percentage_input.setRange(0, 100)
        self.target_percentage_input.setValue(75)
        sfm_functions_layout.addWidget(self.target_percentage_input)

        # Create a QTabWidget for Building and Export functions
        functions_tab_widget = QTabWidget(self)

        # Building Functions Tab
        building_tab = QWidget()
        building_layout = QVBoxLayout(building_tab)
        for function in self.building_functions.values():
            building_layout.addWidget(function)
        functions_tab_widget.addTab(building_tab, "Building")

        # Export Functions Tab
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        for function in self.export_functions.values():
            export_layout.addWidget(function)
        functions_tab_widget.addTab(export_tab, "Export")

        # Add the QTabWidget to the sfm_functions_layout
        sfm_functions_layout.addWidget(functions_tab_widget)

        layout.addWidget(sfm_functions_group)

        ###
        # Run Button
        run_button = QPushButton('Run Workflow')
        run_button.clicked.connect(self.run_workflow)
        layout.addWidget(run_button)

        # Create a scroll area and set the layout as its widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
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

    def choose_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_path_display.setText(directory)

    def choose_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path_display.setText(directory)

    def extract_input_value(self):
        current_tab = self.main_tab_widget.currentIndex()
        if current_tab == 0:  # Local
            return self.input_path_display.text()
        elif current_tab == 1:  # Azure
            current_azure_tab = self.azure_io_tab_widget.currentIndex()
            if current_azure_tab == 0:  # URI
                return self.input_uri_input.text()
            elif current_azure_tab == 1:  # URL
                return self.input_url_input.text()

    def extract_output_value(self):
        current_tab = self.main_tab_widget.currentIndex()
        if current_tab == 0:  # Local
            return self.output_path_display.text()
        elif current_tab == 1:  # Azure
            current_azure_tab = self.azure_io_tab_widget.currentIndex()
            if current_azure_tab == 0:  # URI
                return self.output_uri_input.text()
            elif current_azure_tab == 1:  # URL
                return self.output_url_input.text()

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
                self.computes_input.addItem(f"{compute.name}")
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def get_compute_status(self, compute):
        """Get the status of a compute instance."""
        try:
            # Check the provisioning state
            provisioning_state = getattr(compute, 'provisioning_state', None)
            print(compute.status)
            if provisioning_state:
                if provisioning_state.lower() == 'succeeded':
                    # Check if the compute is running
                    if hasattr(compute, 'status') and compute.status.lower() == 'running':
                        return "Running"
                    else:
                        return "Stopped"
                else:
                    return provisioning_state

            # If no provisioning state found, return unknown status
            return "Unknown status"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_azure_compute(self):
        """Method to get the selected compute option."""
        compute_name = self.computes_input.currentText()
        try:
            compute = self.ml_client.compute.get(compute_name)
            return compute
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return None

    def run_workflow(self):
        """Method to run the SfM workflow."""
        try:
            self.subscription_id = self.subscription_id_input.text()
            self.resource_group = self.resource_group_input.text()
            self.workspace_name = self.workspace_name_input.text()
            self.compute_name = self.computes_input.currentText()

            self.device = int(self.device_input.value())

            # Method calls to get input / output strings
            self.input_dir = self.extract_input_value()
            self.output_dir = os.path.join(self.extract_output_value(), get_now())

            self.quality = self.quality_input.currentText()
            self.target_percentage = self.target_percentage_input.value()

            # Make the cursor busy
            self.setCursor(Qt.WaitCursor)

            if os.path.exists(self.input_dir):
                # Run locally
                self.run_workflow_locally()
                QMessageBox.information(self, 'Success', 'Workflow completed!')
            else:
                # Run on Azure
                self.run_workflow_azure()
                QMessageBox.information(self, 'Success', 'Workflow submitted to Azure!')

        except Exception as e:
            print(f"ERROR: {e}")
            QMessageBox.critical(self, 'Error', "Failed to run the workflow!")

        # Make the cursor normal
        self.setCursor(Qt.ArrowCursor)

    def run_workflow_locally(self):
        """Method to run the SfM workflow locally."""
        try:
            workflow = SfMWorkflow(device=self.device,
                                   input_dir=self.input_dir,
                                   project_file="",
                                   output_dir=self.output_dir,
                                   quality=self.quality,
                                   target_percentage=self.target_percentage,
                                   add_photos=self.building_functions['add_photos'].isChecked(),
                                   align_cameras=self.building_functions['align_cameras'].isChecked(),
                                   optimize_cameras=self.building_functions['optimize_cameras'].isChecked(),
                                   build_depth_maps=self.building_functions['build_depth_maps'].isChecked(),
                                   build_point_cloud=self.building_functions['build_point_cloud'].isChecked(),
                                   build_dem=self.building_functions['build_dem'].isChecked(),
                                   build_ortho=self.building_functions['build_ortho'].isChecked(),
                                   export_cameras=self.export_functions['export_cameras'].isChecked(),
                                   export_point_cloud=self.export_functions['export_point_cloud'].isChecked(),
                                   export_dem=self.export_functions['export_dem'].isChecked(),
                                   export_ortho=self.export_functions['export_ortho'].isChecked(),
                                   export_report=self.export_functions['export_report'].isChecked())

            for function_name, checkbox in self.building_functions.items():
                if checkbox.isChecked():
                    getattr(workflow, function_name)()

            for function_name, checkbox in self.export_functions.items():
                if checkbox.isChecked():
                    getattr(workflow, function_name)()

        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception(f"Failed to run the workflow locally!")

    def run_workflow_azure(self):
        """Method to run the SfM workflow on Azure."""
        try:
            if not self.compute_name:
                raise Exception("No compute selected. Please authenticate and select a compute.")

            # Get the SfM script path
            sfm_script_path = self.download_sfm_script()
            if not os.path.exists(sfm_script_path):
                raise Exception("SfM script file not found.")

            # Submit the job to the compute
            print(f"Submitting job to compute: {self.compute_name}")

            # load in the input data information from the function def
            input_path = self.input_dir
            input_mode = InputOutputModes.DOWNLOAD  # RO_MOUNT

            # load in the output data info from the function def
            output_path = self.output_dir
            output_mode = InputOutputModes.UPLOAD

            # get a local instance of the compute info
            compute = self.ml_client.compute.get(self.compute_name)

            # create input and output dictionaries to use in the command calls later

            input = {
                "input_data": Input(type=AssetTypes.URI_FOLDER,  # URI_FILE
                                    path=input_path,
                                    mode=input_mode)
            }
            output = {
                "output_data": Output(type=AssetTypes.URI_FOLDER,
                                      path=output_path,
                                      mode=output_mode)
            }

            # create linux command line commands to be sent to the compute target
            command_args = [
                f'python SfM.py ${{inputs.input_data}} ${{outputs.output_data}}',
                f'--device {self.device_input.value()}',
                f'--quality {self.quality_input.currentText()}',
                f'--target_percentage {self.target_percentage_input.value()}'
            ]

            for function_name, checkbox in self.building_functions.items():
                if checkbox.isChecked():
                    command_args.append(f'--{function_name}')

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

        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception(f"Failed to run the workflow on Azure!")


def metashape_app():
    app = QApplication.instance()
    parent = app.activeWindow()
    dlg = SfMWorkflowApp(parent)
    dlg.exec_()

def main_function():
    app = QApplication(sys.argv)
    dlg = SfMWorkflowApp()
    dlg.exec_()


if __name__ == "__main__":
    try:
        import Metashape

        label = "Scripts/Metashape-Azure"
        Metashape.app.addMenuItem(label, metashape_app)
        print("To execute this script press {}".format(label))
    except Exception as e:
        main_function()