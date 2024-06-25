import os
import sys
import time
import datetime
import traceback

import numpy as np


# -----------------------------------------------------------------------------------------------------------
# Version Checks
# -----------------------------------------------------------------------------------------------------------

try:
    import Metashape

except Exception as e:
    raise Exception(f'ERROR: {e}')

# Check that the Metashape version is compatible with this script
compatible_version = "2.0.2"
found_version = Metashape.app.version

if found_version != compatible_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_version, compatible_version))


# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------

def announce(announcement: str):
    """
    Gabriel's message to the world.
    """
    print("\n###############################################")
    print(announcement)
    print("###############################################\n")


def get_gpu_mask(device: int):
    """
    Calculates a GPU mask for Metashape. Instead of specifying the
    device index, Metashape expects a mask; for example, a device
    with 5 GPUs, the first 4 on, would be: '11110'. A device with
    3 GPUs, the first and last on would be: '101'.

    This function takes in the device index and calculates the mask.
    """
    # GPU binary string
    gpuBinary = ""

    # Get the total number of GPU devices on machine
    gpus = Metashape.app.enumGPUDevices()

    # Loop through all devices, and only turn on
    # the one specified by device; leave others off.
    for index, gpu in enumerate(gpus):
        if index == device:
            gpuBinary += "1"
        else:
            gpuBinary += "0"

    # Convert binary string to int
    gpuMask = int(gpuBinary, 2)

    return gpuMask


def get_now():
    """
    Returns a timestamp; used for file and folder names
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


def print_progress(p: int):
    """
    Prints progress to user
    """
    print('Current task progress: {:.2f}%'.format(p))


def find_files(folder: str, types: list):
    """
    Takes in a folder and a list of file types, returns a list of file paths
    that end with any of the specified extensions. Searches only one level deep.
    """
    matching_files = []

    # Search in the specified folder
    for entry in os.scandir(folder):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in types:
            matching_files.append(entry.path)

    # Search in the immediate subdirectories
    for entry in os.scandir(folder):
        if entry.is_dir():
            for subentry in os.scandir(entry.path):
                if subentry.is_file() and os.path.splitext(subentry.name)[1].lower() in types:
                    matching_files.append(subentry.path)

    return matching_files


# -----------------------------------------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------------------------------------

class SfMWorkflow:
    def __init__(self, device, input_dir, project_file, output_dir, quality='high', target_percentage=75):

        # Ensure the license is activated
        self.validate_license()

        # Set the device index
        self.device = int(device)

        # Check that input directory exists
        if os.path.exists(input_dir):
            self.input_dir = input_dir
        else:
            raise Exception("ERROR: Input directory provided doesn't exist; please check input")

        # If user passes a previous project file, use it
        if os.path.exists(project_file):
            self.project_file = project_file
        else:
            # Create a new one inside the input directory
            self.project_file = f"{self.input_dir}/project.psx"

        # Create the output directory
        self.output_dir = f"{output_dir}/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create filenames for data outputs
        self.output_dem = f"{self.output_dir}/DEM.tif"
        self.output_dense = f"{self.output_dir}/Dense_Cloud.ply"
        self.output_ortho = f"{self.output_dir}/Orthomosaic.tif"
        self.output_cameras = f"{self.output_dir}/Cameras.xml"
        self.output_report = f"{self.output_dir}/Report.pdf"

        # Validate and set the quality
        self.quality = quality
        self.validate_quality()

        # Validate and set the target percentage
        self.target_percentage = int(target_percentage)
        self.validate_target_percentage()

        Metashape.app.gpu_mask = get_gpu_mask(device=self.device)
        self.doc = Metashape.Document()

        if not os.path.exists(self.project_file):
            print(f"NOTE: Creating new project file")
            self.doc.save(self.project_file)
        else:
            print(f"NOTE: Opening existing project file")
            self.doc.open(self.project_file,
                          read_only=False,
                          ignore_lock=True,
                          archive=True)

        if self.doc.chunk is None:
            self.doc.addChunk()
            self.doc.save()

    def validate_license(self):
        """

        """
        if not Metashape.License().valid:
            raise Exception("ERROR: Metashape License not valid on this machine")

    def validate_quality(self):
        """

        """
        if self.quality.lower() not in ["lowest", "low", "medium", "high", "highest"]:
            raise Exception(f"ERROR: Quality must be low, medium, or high")

    def validate_target_percentage(self):
        """

        """
        if not type(self.target_percentage) == int and 0 < self.target_percentage < 100:
            raise Exception(f"ERROR: Target percentage must be int between 0 and 100")

    def add_photos(self):
        """

        """
        chunk = self.doc.chunk

        if not chunk.cameras:
            announce("Finding Photos")
            # Find the available files
            photos = find_files(self.input_dir, [".jpg", ".jpeg", ".tiff", ".tif", ".png"])

            if not photos:
                raise Exception(f"ERROR: Image directory provided does not contain any usable images")

            announce("Adding photos")
            chunk.addPhotos(photos, progress=print_progress)
            print(str(len(chunk.cameras)) + " images loaded")

            print("")
            print("Process Successful!")
            self.doc.save()

    def align_cameras(self):
        """

        """
        chunk = self.doc.chunk

        if not chunk.tie_points:

            announce("Matching photos")
            chunk.detectMarkers(target_type=Metashape.CircularTarget12bit)

            downscale = {"lowest": 8,
                         "low": 4,
                         "medium": 2,
                         "high": 1,
                         "highest": 0}[self.quality.lower()]

            chunk.matchPhotos(
                keypoint_limit=40000,
                tiepoint_limit=10000,
                generic_preselection=True,
                reference_preselection=True,
                downscale=downscale,
                progress=print_progress
            )

            chunk.alignCameras()

            print("")
            print("Process Successful!")
            self.doc.save()

    def optimize_cameras(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.tie_points:
            announce("Performing camera optimization")

            points = chunk.tie_points.points
            selections = [Metashape.TiePoints.Filter.ReprojectionError,
                          Metashape.TiePoints.Filter.ReconstructionUncertainty,
                          Metashape.TiePoints.Filter.ProjectionAccuracy,
                          Metashape.TiePoints.Filter.ImageCount]

            for s_idx, selection in enumerate(selections):

                try:
                    f = Metashape.TiePoints.Filter()

                    if s_idx == 3:
                        f.init(chunk, selection)
                        f.removePoints(1)
                    else:
                        list_values = f.values
                        list_values_valid = [list_values[i] for i in range(len(list_values)) if points[i].valid]
                        list_values_valid.sort()
                        target = int(len(list_values_valid) * self.target_percentage / 100)
                        threshold = list_values_valid[target]
                        f.selectPoints(threshold)
                        f.removePoints(threshold)

                    chunk.optimizeCameras(
                        fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                        fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True,
                        fit_p1=True, fit_p2=True, fit_p3=True, fit_p4=True,
                        adaptive_fitting=False, tiepoint_covariance=False
                    )

                except Exception as e:
                    print(f"WARNING: Could not filter points based on selection method {s_idx}")

            print("")
            print("Process Successful!")
            self.doc.save()

    def build_depth_maps(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.tie_points and not chunk.depth_maps:
            announce("Building depth maps")
            downscale = {"lowest": 16,
                         "low": 8,
                         "medium": 4,
                         "high": 2,
                         "highest": 1}[self.quality.lower()]

            chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering,
                                 downscale=downscale,
                                 progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def build_point_cloud(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.depth_maps and not chunk.point_cloud:
            announce("Building dense point cloud")
            chunk.buildPointCloud(source_data=Metashape.DepthMapsData,
                                  progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def build_dem(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.point_cloud and not chunk.elevation:
            announce("Building DEM")
            chunk.buildDem(source_data=Metashape.PointCloudData,
                           interpolation=Metashape.Interpolation.DisabledInterpolation,
                           progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def build_ortho(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.point_cloud and not chunk.orthomosaic:
            announce("Building orthomosaic")
            chunk.buildOrthomosaic(surface_data=Metashape.ElevationData,
                                   blending_mode=Metashape.BlendingMode.MosaicBlending,
                                   fill_holes=False,
                                   progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_cameras(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.tie_points:
            announce("Exporting Camera Positions")
            chunk.exportCameras(path=self.output_cameras,
                                progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_point_cloud(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.point_cloud and not os.path.exists(self.output_dense):
            announce("Exporting dense point cloud")
            chunk.exportPointCloud(path=self.output_dense,
                                   save_point_color=True,
                                   save_point_classification=True,
                                   save_point_normal=True,
                                   save_point_confidence=True,
                                   crs=chunk.crs,
                                   progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_dem(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.elevation and not os.path.exists(self.output_dem):
            announce("Exporting DEM")
            chunk.exportRaster(path=self.output_dem,
                               source_data=Metashape.ElevationData,
                               progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_ortho(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.orthomosaic and not os.path.exists(self.output_ortho):
            announce("Exporting orthomosaic")
            compression = Metashape.ImageCompression()
            compression.tiff_big = True
            chunk.exportRaster(path=self.output_ortho,
                               source_data=Metashape.OrthomosaicData,
                               image_compression=compression,
                               progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_report(self):
        """

        """
        chunk = self.doc.chunk
        announce("Exporting Report")
        chunk.exportReport(path=self.output_report)
        print("")
        print("Process Successful!")
        self.doc.save()

    def run(self):
        """

        """
        announce("Structure from Motion")
        t0 = time.time()

        self.add_photos()
        self.align_cameras()
        self.optimize_cameras()
        self.build_depth_maps()
        self.build_point_cloud()
        self.build_dem()
        self.build_ortho()

        self.export_cameras()
        self.export_point_cloud()
        self.export_dem()
        self.export_ortho()
        self.export_report()

        announce("Workflow Completed")
        print(f"NOTE: Processing finished, results saved to {self.output_dir}")
        print(f"NOTE: Completed in {np.around(((time.time() - t0) / 60), 2)} minutes")
        self.doc.save()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main(device, input_path, project_file, output_path, quality, target_percentage):
    """

    """

    try:
        workflow = SfMWorkflow(device=device,
                               input_dir=input_path,
                               project_file=project_file,
                               output_dir=output_path,
                               quality=quality,
                               target_percentage=target_percentage)
        workflow.run()

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())


if __name__ == '__main__':

    main(sys.argv[1],  # Device
         sys.argv[2],  # Input Path
         sys.argv[3],  # Project File
         sys.argv[4],  # Output Path
         sys.argv[5],  # Quality
         sys.argv[6])  # Target Percentage


