from SfM import Metashape
from SfM import SfMWorkflow

from PySide2 import QtWidgets


class SfMWorkflowApp(QtWidgets.QDialog):
    def __init__(self, parent):
        super(SfMWorkflowApp, self).__init__(parent)
        self.setWindowTitle("SfM Workflow Interface")

        self.subscription_id_input = QtWidgets.QLineEdit()
        self.resource_group_input = QtWidgets.QLineEdit()
        self.workspace_name_input = QtWidgets.QLineEdit()

        self.device_input = QtWidgets.QLineEdit()
        self.input_dir_input = QtWidgets.QLineEdit()
        self.project_file_input = QtWidgets.QLineEdit()
        self.output_dir_input = QtWidgets.QLineEdit()
        self.quality_input = QtWidgets.QComboBox()
        self.target_percentage_input = QtWidgets.QSpinBox()

        self.building_functions = {
            "add_photos": QtWidgets.QCheckBox("Add Photos"),
            "align_cameras": QtWidgets.QCheckBox("Align Cameras"),
            "optimize_cameras": QtWidgets.QCheckBox("Optimize Cameras"),
            "build_depth_maps": QtWidgets.QCheckBox("Build Depth Maps"),
            "build_point_cloud": QtWidgets.QCheckBox("Build Point Cloud"),
            "build_dem": QtWidgets.QCheckBox("Build DEM"),
            "build_ortho": QtWidgets.QCheckBox("Build Orthomosaic")
        }

        self.export_functions = {
            "export_cameras": QtWidgets.QCheckBox("Export Cameras"),
            "export_point_cloud": QtWidgets.QCheckBox("Export Point Cloud"),
            "export_dem": QtWidgets.QCheckBox("Export DEM"),
            "export_ortho": QtWidgets.QCheckBox("Export Orthomosaic"),
            "export_report": QtWidgets.QCheckBox("Export Report")
        }

        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        # Cloud Credentials Panel
        cloud_credentials_group = QtWidgets.QGroupBox("Cloud Credentials")
        cloud_credentials_layout = QtWidgets.QVBoxLayout()

        subscription_id_label = QtWidgets.QLabel('Subscription ID:')
        subscription_id_label.setToolTip("Enter your Azure subscription ID.")
        cloud_credentials_layout.addWidget(subscription_id_label)
        cloud_credentials_layout.addWidget(self.subscription_id_input)

        resource_group_label = QtWidgets.QLabel('Resource Group:')
        resource_group_label.setToolTip("Enter your Azure resource group name.")
        cloud_credentials_layout.addWidget(resource_group_label)
        cloud_credentials_layout.addWidget(self.resource_group_input)

        workspace_name_label = QtWidgets.QLabel('Workspace Name:')
        workspace_name_label.setToolTip("Enter your Azure workspace name.")
        cloud_credentials_layout.addWidget(workspace_name_label)
        cloud_credentials_layout.addWidget(self.workspace_name_input)

        cloud_credentials_group.setLayout(cloud_credentials_layout)
        layout.addWidget(cloud_credentials_group)

        # Overall description
        description = QtWidgets.QLabel("This interface allows you to configure and run the Structure from Motion (SfM) workflow. "
                                       "Select the desired functions, set the parameters, and click 'Run Workflow' to start the process.")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Device Input
        device_label = QtWidgets.QLabel('Device:')
        device_label.setToolTip("Enter the device index for processing.")
        layout.addWidget(device_label)
        layout.addWidget(self.device_input)

        # Input Directory
        input_dir_label = QtWidgets.QLabel('Input Directory:')
        input_dir_label.setToolTip("Select the directory containing the input images.")
        layout.addWidget(input_dir_label)
        layout.addWidget(self.input_dir_input)
        input_dir_button = QtWidgets.QPushButton('Browse')
        input_dir_button.clicked.connect(self.browse_input_dir)
        layout.addWidget(input_dir_button)

        # Project File
        project_file_label = QtWidgets.QLabel('Project File:')
        project_file_label.setToolTip("Select the project file to load or create a new one.")
        layout.addWidget(project_file_label)
        layout.addWidget(self.project_file_input)
        project_file_button = QtWidgets.QPushButton('Browse')
        project_file_button.clicked.connect(self.browse_project_file)
        layout.addWidget(project_file_button)

        # Output Directory
        output_dir_label = QtWidgets.QLabel('Output Directory:')
        output_dir_label.setToolTip("Select the directory where the output files will be saved.")
        layout.addWidget(output_dir_label)
        layout.addWidget(self.output_dir_input)
        output_dir_button = QtWidgets.QPushButton('Browse')
        output_dir_button.clicked.connect(self.browse_output_dir)
        layout.addWidget(output_dir_button)

        # Quality
        quality_label = QtWidgets.QLabel('Quality:')
        quality_label.setToolTip("Select the quality level for the processing.")
        layout.addWidget(quality_label)
        self.quality_input.addItems(['lowest', 'low', 'medium', 'high', 'highest'])
        self.quality_input.setCurrentText('medium')
        layout.addWidget(self.quality_input)

        # Target Percentage
        target_percentage_label = QtWidgets.QLabel('Target Percentage:')
        target_percentage_label.setToolTip("Set the target percentage for filtering points.")
        layout.addWidget(target_percentage_label)
        self.target_percentage_input.setRange(0, 100)
        self.target_percentage_input.setValue(75)
        layout.addWidget(self.target_percentage_input)

        # Building Functions
        building_group = QtWidgets.QGroupBox("Building Functions")
        building_group.setFixedWidth(400)
        building_layout = QtWidgets.QVBoxLayout()
        for function in self.building_functions.values():
            building_layout.addWidget(function)
        building_group.setLayout(building_layout)
        layout.addWidget(building_group)

        # Export Functions
        export_group = QtWidgets.QGroupBox("Export Functions")
        export_group.setFixedWidth(400)
        export_layout = QtWidgets.QVBoxLayout()
        for function in self.export_functions.values():
            export_layout.addWidget(function)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Run Button
        run_button = QtWidgets.QPushButton('Run Workflow')
        run_button.clicked.connect(self.run_workflow)
        layout.addWidget(run_button)

        # Create a scroll area and set the layout as its widget
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def browse_input_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Input Directory')
        if dir_path:
            self.input_dir_input.setText(dir_path)

    def browse_project_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Project File')
        if file_path:
            self.project_file_input.setText(file_path)

    def browse_output_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def run_workflow(self):
        try:
            subscription_id = self.subscription_id_input.text()
            resource_group = self.resource_group_input.text()
            workspace_name = self.workspace_name_input.text()

            device = int(self.device_input.text())
            input_dir = self.input_dir_input.text()
            project_file = self.project_file_input.text()
            output_dir = self.output_dir_input.text()
            quality = self.quality_input.currentText()
            target_percentage = self.target_percentage_input.value()

            workflow = SfMWorkflow(device=device,
                                   input_dir=input_dir,
                                   project_file=project_file,
                                   output_dir=output_dir,
                                   quality=quality,
                                   target_percentage=target_percentage)

            for function_name, checkbox in self.building_functions.items():
                if checkbox.isChecked():
                    getattr(workflow, function_name)()

            for function_name, checkbox in self.export_functions.items():
                if checkbox.isChecked():
                    getattr(workflow, function_name)()

            QtWidgets.QMessageBox.information(self, 'Success', 'Workflow completed successfully!')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))

def launch_app():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = SfMWorkflowApp(parent)
    dlg.exec_()

label = "Scripts/Metashape-Azure"
Metashape.app.addMenuItem(label, launch_app)
print("To execute this script press {}".format(label))