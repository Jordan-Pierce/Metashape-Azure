import sys

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
                             QSpinBox, QFileDialog, QMessageBox, QRadioButton, QGroupBox, QVBoxLayout)


from SfM import Metashape
from SfM import SfMWorkflow


# -----------------------------------------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------------------------------------

class SfMWorkflowApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SfM Workflow Interface')
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()

        # Device Input
        device_layout = QHBoxLayout()
        device_label = QLabel('Device:')
        self.device_input = QLineEdit()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_input)
        layout.addLayout(device_layout)

        # Input Directory
        input_dir_layout = QHBoxLayout()
        input_dir_label = QLabel('Input Directory:')
        self.input_dir_input = QLineEdit()
        input_dir_button = QPushButton('Browse')
        input_dir_button.clicked.connect(self.browse_input_dir)
        input_dir_layout.addWidget(input_dir_label)
        input_dir_layout.addWidget(self.input_dir_input)
        input_dir_layout.addWidget(input_dir_button)
        layout.addLayout(input_dir_layout)

        # Project File
        project_file_layout = QHBoxLayout()
        project_file_label = QLabel('Project File:')
        self.project_file_input = QLineEdit()
        project_file_button = QPushButton('Browse')
        project_file_button.clicked.connect(self.browse_project_file)
        project_file_layout.addWidget(project_file_label)
        project_file_layout.addWidget(self.project_file_input)
        project_file_layout.addWidget(project_file_button)
        layout.addLayout(project_file_layout)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel('Output Directory:')
        self.output_dir_input = QLineEdit()
        output_dir_button = QPushButton('Browse')
        output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(output_dir_button)
        layout.addLayout(output_dir_layout)

        # Quality
        quality_layout = QHBoxLayout()
        quality_label = QLabel('Quality:')
        self.quality_input = QComboBox()
        self.quality_input.addItems(['lowest', 'low', 'medium', 'high', 'highest'])
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_input)
        layout.addLayout(quality_layout)

        # Target Percentage
        target_percentage_layout = QHBoxLayout()
        target_percentage_label = QLabel('Target Percentage:')
        self.target_percentage_input = QSpinBox()
        self.target_percentage_input.setRange(0, 100)
        target_percentage_layout.addWidget(target_percentage_label)
        target_percentage_layout.addWidget(self.target_percentage_input)
        layout.addLayout(target_percentage_layout)

        # Function Selection
        self.function_group = QGroupBox("Select Functions to Run")
        function_layout = QVBoxLayout()
        self.functions = {
            "add_photos": QRadioButton("Add Photos"),
            "align_cameras": QRadioButton("Align Cameras"),
            "optimize_cameras": QRadioButton("Optimize Cameras"),
            "build_depth_maps": QRadioButton("Build Depth Maps"),
            "build_point_cloud": QRadioButton("Build Point Cloud"),
            "build_dem": QRadioButton("Build DEM"),
            "build_ortho": QRadioButton("Build Orthomosaic"),
            "export_cameras": QRadioButton("Export Cameras"),
            "export_point_cloud": QRadioButton("Export Point Cloud"),
            "export_dem": QRadioButton("Export DEM"),
            "export_ortho": QRadioButton("Export Orthomosaic"),
            "export_report": QRadioButton("Export Report")
        }
        for function in self.functions.values():
            function_layout.addWidget(function)
        self.function_group.setLayout(function_layout)
        layout.addWidget(self.function_group)

        # Run Button
        run_button = QPushButton('Run Workflow')
        run_button.clicked.connect(self.run_workflow)
        layout.addWidget(run_button)

        self.setLayout(layout)

    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Input Directory')
        if dir_path:
            self.input_dir_input.setText(dir_path)

    def browse_project_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Project File')
        if file_path:
            self.project_file_input.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def run_workflow(self):
        try:
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

            for function_name, radio_button in self.functions.items():
                if radio_button.isChecked():
                    getattr(workflow, function_name)()

            QMessageBox.information(self, 'Success', 'Workflow completed successfully!')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def launch_app():
    app = QApplication(sys.argv)
    ex = SfMWorkflowApp()
    ex.show()
    sys.exit(app.exec_())

# -----------------------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------------------

label = "Scripts/Metashape-Azure"
Metashape.app.addMenuItem(label, launch_app)
print("To execute this script press {}".format(label))