import os
import time
import datetime
import json
import argparse
import traceback

import numpy as np
from packaging import version


# -----------------------------------------------------------------------------------------------------------
# Version Checks
# -----------------------------------------------------------------------------------------------------------


try:
    import Metashape

except Exception as e:
    raise Exception(f'ERROR: {e}')

# Check that the Metashape version is compatible with this script
compatible_version = version.parse("2.1.2")
found_version = version.parse(str(Metashape.app.version))

if found_version < compatible_version:
    raise Exception(f"Found version {found_version}, but expecting at least {compatible_version}")


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
    
    
def get_now():
    """
    Returns a timestamp; used for file and folder names
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


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
    def __init__(self,
                 device,
                 input_dir,
                 project_file,
                 output_name,
                 output_dir,
                 quality='high',
                 target_percentage=10,
                 detect_markers=True,
                 add_photos=True,
                 align_cameras=True,
                 optimize_cameras=True,
                 build_depth_maps=True,
                 build_point_cloud=True,
                 build_mesh=True,
                 build_texture=True,
                 build_dem=True,
                 build_ortho=True,
                 export_viscore=True,
                 export_meta=True,
                 export_cameras=True,
                 export_point_cloud=True,
                 export_potree=True,
                 export_mesh=True,
                 export_texture=True,
                 export_dem=True,
                 export_ortho=True,
                 export_report=True):

        # Ensure the license is activated
        self.borrow_license()
        self.validate_license()

        # Set the device index
        self.device = int(device)

        # Check that input directory exists
        if os.path.exists(input_dir):
            self.input_dir = input_dir
        else:
            raise Exception("ERROR: Input directory provided doesn't exist; please check input")

        # Create the output directory
        if output_name:
            self.output_name = output_name
        else:
            self.output_name = get_now()
            
        self.output_dir = f"{output_dir}/{self.output_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create filenames for data outputs
        self.output_dem = f"{self.output_dir}/DEM.tif"
        self.output_dense = f"{self.output_dir}/Dense_Cloud.ply"
        self.output_potree = f"{self.output_dir}/Potree.zip"
        self.output_mesh = f"{self.output_dir}/Mesh.ply"
        self.output_texture = f"{self.output_dir}/Texture.jpg"
        self.output_ortho = f"{self.output_dir}/Orthomosaic.tif"
        self.output_cameras = f"{self.output_dir}/Cameras.xml"
        self.output_meta = f"{self.output_dir}/Meta.json"
        self.output_report = f"{self.output_dir}/Report.pdf"
        
        # Set the export viscore flag
        self.export_viscore_flag = export_viscore
        
        if self.export_viscore_flag:
            self.output_dense = f"{self.output_dir}/{self.output_name}.ply"
            self.output_cameras = f"{self.output_dir}/{self.output_name}.cams.xml"
            self.output_meta = f"{self.output_dir}/{self.output_name}.meta.json"

        # Validate and set the quality
        self.quality = quality
        self.validate_quality()

        # Validate and set the target percentage
        self.target_percentage = int(target_percentage)
        self.validate_target_percentage()

        # Detect markers
        self.detect_markers_flag = detect_markers

        Metashape.app.gpu_mask = get_gpu_mask(device=self.device)
        self.doc = Metashape.Document()

        self.project_file = f"{self.output_name}.psx"
        if not os.path.exists(self.project_file):
            print(f"NOTE: Creating new project file")
            self.project_file = f"{self.output_dir}/{self.output_name}.psx"
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

        # Store the boolean parameters
        self.add_photos_flag = add_photos
        self.align_cameras_flag = align_cameras
        self.optimize_cameras_flag = optimize_cameras
        self.build_depth_maps_flag = build_depth_maps
        self.build_point_cloud_flag = build_point_cloud
        self.build_mesh_flag = build_mesh
        self.build_texture_flag = build_texture
        self.build_dem_flag = build_dem
        self.build_ortho_flag = build_ortho

        self.export_viscore_flag = export_viscore
        self.export_meta_flag = export_meta
        self.export_cameras_flag = export_cameras
        self.export_point_cloud_flag = export_point_cloud
        self.export_potree_flag = export_potree
        self.export_mesh_flag = export_mesh
        self.export_texture_flag = export_texture
        self.export_dem_flag = export_dem
        self.export_ortho_flag = export_ortho
        self.export_report_flag = export_report

        # Run the workflow
        self.run_workflow()

    def borrow_license(self):
        """

        """
        try:
            Metashape.License().borrowLicense(3600)
            print("NOTE: License borrowed successfully")
        except Exception as e:
            Exception(f"ERROR: Could not borrow license: {e}")

    def return_license(self):
        """

        """
        try:
            Metashape.License().returnLicense()
            print("NOTE: License returned successfully")
        except Exception as e:
            print(f"WARNING: Could not return license: {e}")

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

            if self.detect_markers_flag:
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
                                  point_confidence=True,
                                  progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def build_mesh(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.point_cloud and not chunk.model:
            announce("Building mesh")
            chunk.buildModel(surface_type=Metashape.Arbitrary,
                             interpolation=Metashape.EnabledInterpolation,
                             face_count=Metashape.HighFaceCount,
                             source_data=Metashape.PointCloudData,
                             progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def build_texture(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.model:
            announce("Building texture")
            chunk.buildUV()
            chunk.buildTexture(blending_mode=Metashape.BlendingMode.MosaicBlending,
                               texture_size=4096,
                               texture_type=Metashape.Model.TextureType.DiffuseMap,
                               source_model=chunk.model,
                               transfer_texture=False)
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
            
    def export_meta(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.cameras:
            announce("Exporting Meta Data")
            
            camera_data = {}
            # Loop through all cameras
            for camera in chunk.cameras:
                # Get camera center coordinates
                center = None
                if camera.center is not None:
                    geo = chunk.transform.matrix.mulp(camera.center)
                    center = list(chunk.crs.project(geo)) if chunk.crs else list(camera.center)

                # Get camera transform matrix 
                transform = None
                if camera.transform:
                    transform = [list(camera.transform.row(n)) for n in range(camera.transform.size[1])]

                # Store camera metadata
                camera_data[camera.key] = {
                    'path': camera.photo.path,
                    'center': center,
                    'transform': transform
                }

            # Write metadata to JSON file
            with open(self.output_meta, 'w') as f:
                json.dump({'cameras': camera_data}, f, indent=4)

            print("Successfully exported camera metadata")
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
                                   save_point_classification=False,
                                   save_point_normal=True,
                                   save_point_confidence=True,
                                   crs=chunk.crs,
                                   progress=print_progress)

            print("")
            print("Process Successful!")
            self.doc.save()

    def export_potree(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.point_cloud and not os.path.exists(self.output_potree):
            announce("Exporting dense point cloud")
            chunk.exportPointCloud(path=self.output_potree,
                                   format=Metashape.PointCloudFormatPotree,
                                   save_point_color=True,
                                   save_point_classification=True,
                                   save_point_normal=True,
                                   save_point_confidence=True,
                                   crs=chunk.crs,
                                   progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_mesh(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.model and not os.path.exists(self.output_mesh):
            announce("Exporting mesh")
            chunk.exportModel(path=self.output_mesh,
                              progress=print_progress)
            print("")
            print("Process Successful!")
            self.doc.save()

    def export_texture(self):
        """

        """
        chunk = self.doc.chunk

        if chunk.model and not os.path.exists(self.output_texture):
            announce("Exporting texture")
            chunk.model.saveTexture(self.output_texture)
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

    def run_workflow(self):
        announce("Structure from Motion")
        t0 = time.time()

        if self.add_photos_flag:
            try:
                self.add_photos()
            except Exception as e:
                print(f"ERROR in add_photos: {e}")

        if self.align_cameras_flag:
            try:
                self.align_cameras()
            except Exception as e:
                print(f"ERROR in align_cameras: {e}")

        if self.optimize_cameras_flag:
            try:
                self.optimize_cameras()
            except Exception as e:
                print(f"ERROR in optimize_cameras: {e}")

        if self.build_depth_maps_flag:
            try:
                self.build_depth_maps()
            except Exception as e:
                print(f"ERROR in build_depth_maps: {e}")

        if self.build_point_cloud_flag:
            try:
                self.build_point_cloud()
            except Exception as e:
                print(f"ERROR in build_point_cloud: {e}")

        if self.build_mesh_flag:
            try:
                self.build_mesh()
            except Exception as e:
                print(f"ERROR in build_mesh: {e}")

        if self.build_texture_flag:
            try:
                self.build_texture()
            except Exception as e:
                print(f"ERROR in build_texture: {e}")

        if self.build_dem_flag:
            try:
                self.build_dem()
            except Exception as e:
                print(f"ERROR in build_dem: {e}")

        if self.build_ortho_flag:
            try:
                self.build_ortho()
            except Exception as e:
                print(f"ERROR in build_ortho: {e}")
                
        if self.export_meta_flag:
            try:
                self.export_meta()
            except Exception as e:
                print(f"ERROR in export_meta: {e}")

        if self.export_cameras_flag:
            try:
                self.export_cameras()
            except Exception as e:
                print(f"ERROR in export_cameras: {e}")

        if self.export_point_cloud_flag:
            try:
                self.export_point_cloud()
            except Exception as e:
                print(f"ERROR in export_point_cloud: {e}")

        if self.export_potree_flag:
            try:
                self.export_potree()
            except Exception as e:
                print(f"ERROR in export_potree: {e}")

        if self.export_mesh_flag:
            try:
                self.export_mesh()
            except Exception as e:
                print(f"ERROR in export_mesh: {e}")

        if self.export_texture_flag:
            try:
                self.export_texture()
            except Exception as e:
                print(f"ERROR in export_texture: {e}")

        if self.export_dem_flag:
            try:
                self.export_dem()
            except Exception as e:
                print(f"ERROR in export_dem: {e}")

        if self.export_ortho_flag:
            try:
                self.export_ortho()
            except Exception as e:
                print(f"ERROR in export_ortho: {e}")

        if self.export_report_flag:
            try:
                self.export_report()
            except Exception as e:
                print(f"ERROR in export_report: {e}")

        announce("Workflow Completed")
        print(f"NOTE: Processing finished, results saved to {self.output_dir}")
        print(f"NOTE: Completed in {np.around(((time.time() - t0) / 60), 2)} minutes")
        self.doc.save()

        # Return the license
        self.return_license()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run the Structure from Motion workflow.')
    parser.add_argument('input_path', type=str,
                        help='Path to the input directory')
    
    parser.add_argument('project_file', type=str, default="",
                        help='Path to the project file')
    
    parser.add_argument('output_name', type=str, default="",
                        help='Name of the output project')

    parser.add_argument('output_path', type=str,
                        help='Path to the output directory')

    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (default: 0)')

    parser.add_argument('--quality', type=str, default='medium',
                        choices=['lowest', 'low', 'medium', 'high', 'highest'],
                        help='Quality of the workflow (default: medium)')

    parser.add_argument('--target_percentage', type=int, default=10,
                        help='Target percentage for optimization (default: 10)')

    parser.add_argument('--detect_markers', action='store_true',
                        help='Detect markers in photos')

    parser.add_argument('--add_photos', action='store_true',
                        help='Add photos to the project')

    parser.add_argument('--align_cameras', action='store_true',
                        help='Align cameras')

    parser.add_argument('--optimize_cameras', action='store_true',
                        help='Optimize cameras')

    parser.add_argument('--build_depth_maps', action='store_true',
                        help='Build depth maps')

    parser.add_argument('--build_point_cloud', action='store_true',
                        help='Build point cloud')

    parser.add_argument('--build_mesh', action='store_true',
                        help='Build mesh')

    parser.add_argument('--build_texture', action='store_true',
                        help='Build texture')

    parser.add_argument('--build_dem', action='store_true',
                        help='Build DEM')

    parser.add_argument('--build_ortho', action='store_true',
                        help='Build orthomosaic')
    
    parser.add_argument('--export_viscore', action='store_true',
                        help='Export Viscore formatted metadata')
    
    parser.add_argument('--export_meta', action='store_true',
                        help='Export meta formatted metadata')

    parser.add_argument('--export_cameras', action='store_true',
                        help='Export cameras')

    parser.add_argument('--export_point_cloud', action='store_true',
                        help='Export point cloud')

    parser.add_argument('--export_potree', action='store_true',
                        help='Export Potree')

    parser.add_argument('--export_mesh', action='store_true',
                        help='Export mesh')

    parser.add_argument('--export_texture', action='store_true',
                        help='Export texture')

    parser.add_argument('--export_dem', action='store_true',
                        help='Export DEM')

    parser.add_argument('--export_ortho', action='store_true',
                        help='Export orthomosaic')

    parser.add_argument('--export_report', action='store_true',
                        help='Export report')

    args = parser.parse_args()

    try:
        workflow = SfMWorkflow(device=args.device,
                               input_dir=args.input_path,
                               project_file=args.project_file,
                               output_name=args.output_name,
                               output_dir=args.output_path,
                               quality=args.quality,
                               target_percentage=args.target_percentage,
                               detect_markers=args.detect_markers,
                               add_photos=args.add_photos,
                               align_cameras=args.align_cameras,
                               optimize_cameras=args.optimize_cameras,
                               build_depth_maps=args.build_depth_maps,
                               build_point_cloud=args.build_point_cloud,
                               build_mesh=args.build_mesh,
                               build_texture=args.build_texture,
                               build_dem=args.build_dem,
                               build_ortho=args.build_ortho,
                               export_viscore=args.export_viscore,
                               export_meta=args.export_meta,
                               export_cameras=args.export_cameras,
                               export_point_cloud=args.export_point_cloud,
                               export_potree=args.export_potree,
                               export_mesh=args.export_mesh,
                               export_texture=args.export_texture,
                               export_dem=args.export_dem,
                               export_ortho=args.export_ortho,
                               export_report=args.export_report)
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())

if __name__ == '__main__':
    main()