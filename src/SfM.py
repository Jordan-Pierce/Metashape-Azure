import os
import sys
import time
import argparse
import datetime
import traceback
import configparser

import numpy as np

try:
    import Metashape

except Exception as e:
    raise Exception(f'ERROR: Metashape environment not correctly launched or loaded; '
                    f'see message below\n{e}')

finally:
    pass

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0.2"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version,
                                                                      compatible_major_version))


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


def print_progress(p):
    """
    Prints progress to user
    """
    print('Current task progress: {:.2f}%'.format(p))


def find_files(folder, types):
    """
    Takes in a folder and a list of file types, returns a list of file paths
    that end with any of the specified extensions.
    """
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]


def sfm_workflow(args):
    """
    Takes in an input folder, runs SfM Workflow on all images in it,
    outputs the results in the output folder.
    """
    announce("Structure from Motion")

    # Start the timer
    t0 = time.time()

    # If user passes a previous project file, use it;
    # Assume that the project directory is where the
    # project file is located.
    if args.project_file:
        if os.path.exists(args.project_file):
            project_file = args.project_file
            project_dir = f"{os.path.dirname(project_file)}/"
        else:
            raise Exception(f"ERROR: Could not open project file {args.project_file}")

    elif os.path.exists(args.output_dir):
        output_dir = f"{args.output_dir}/"
        project_dir = f"{output_dir}{get_now()}/"
        os.makedirs(project_dir, exist_ok=True)
        project_file = f"{project_dir}project.psx"

    else:
        raise Exception(f"ERROR: Must provide either existing project file or output directory")

    # Create filenames for data outputs
    output_dem = project_dir + "DEM.tif"
    output_dense = project_dir + "Dense_Cloud.ply"
    output_ortho = project_dir + "Orthomosaic.tif"
    output_cameras = project_dir + "Cameras.xml"
    output_report = project_dir + "Report.pdf"

    # Quality checking
    if args.quality.lower() not in ["lowest", "low", "medium", "high", "highest"]:
        raise Exception(f"ERROR: Quality must be low, medium, or high")

    # ------------------------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------------------------
    # Set the GPU to use; future feature would to include multiple
    Metashape.app.gpu_mask = get_gpu_mask(device=args.device)

    # Create a metashape doc object
    doc = Metashape.Document()

    if not os.path.exists(project_file):
        print(f"NOTE: Creating new project file")
        # Create a new Metashape document and save it as a project file in the output folder.
        doc.save(project_file)
    else:
        print(f"NOTE: Opening existing project file")
        # Open existing project file.
        doc.open(project_file,
                 read_only=False,
                 ignore_lock=True,
                 archive=True)

    # Create a new chunk (3D model) in the Metashape document.
    if doc.chunk is None:
        doc.addChunk()
        doc.save()

    # Assign the chunk
    chunk = doc.chunk

    # Add the photos to the chunk
    if not chunk.cameras:

        # Check that input folder exists
        if os.path.exists(args.input_dir):
            input_dir = args.input_dir
        else:
            raise Exception("ERROR: Input directory provided doesn't exist; please check input")

        # Call the "find_files" function to get a list of photo file paths
        # with specified extensions from the image folder.
        photos = find_files(input_dir, [".jpg", ".jpeg", ".tiff", ".tif", ".png"])

        if not photos:
            raise Exception(f"ERROR: Image directory provided does not contain any usable images; please check input")

        announce("Adding photos")
        chunk.addPhotos(photos, progress=print_progress)
        print(str(len(chunk.cameras)) + " images loaded")

        print("")
        print("Process Successful!")
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:
        announce("Matching photos")
        # Detect markers (if they're there)
        chunk.detectMarkers(target_type=Metashape.CircularTarget12bit)

        # Quality
        downscale = {"lowest": 8,
                     "low": 4,
                     "medium": 2,
                     "high": 1,
                     "highest": 0}[args.quality.lower()]

        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=downscale,
                          progress=print_progress)

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()

        print("")
        print("Process Successful!")
        doc.save()

    # Perform gradual selection to remove messy points
    if chunk.tie_points:

        announce("Performing gradual selection and camera optimization")
        # Target percentage for gradual selection
        if 0 <= args.target_percentage <= 99:
            target_percentage = args.target_percentage
        else:
            raise Exception(f"ERROR: Target Percentage provided not in range [0, 99]; check input provided")

        # Obtain the tie points from the chunk
        points = chunk.tie_points.points

        # Filter selection methods
        selections = [Metashape.TiePoints.Filter.ReprojectionError,
                      Metashape.TiePoints.Filter.ReconstructionUncertainty,
                      Metashape.TiePoints.Filter.ProjectionAccuracy,
                      Metashape.TiePoints.Filter.ImageCount]

        # Loop through each of the selections, identify target percentage, remove, optimize
        for s_idx, selection in enumerate(selections):

            try:
                # Tie point filter
                f = Metashape.TiePoints.Filter()

                if s_idx == 3:
                    # ImageCount selection method
                    f.init(chunk, selection)
                    f.removePoints(1)
                else:
                    # Other selection methods
                    list_values = f.values
                    list_values_valid = list()
                    for i in range(len(list_values)):
                        if points[i].valid:
                            list_values_valid.append(list_values[i])
                    list_values_valid.sort()
                    # Find point values based on threshold
                    target = int(len(list_values_valid) * target_percentage / 100)
                    threshold = list_values_valid[target]
                    # Select and remove
                    f.selectPoints(threshold)
                    f.removePoints(threshold)

                # Optimize cameras
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,
                                      fit_b1=True, fit_b2=True, fit_k1=True,
                                      fit_k2=True, fit_k3=True, fit_k4=True,
                                      fit_p1=True, fit_p2=True, fit_p3=True,
                                      fit_p4=True, adaptive_fitting=False, tiepoint_covariance=False)

            except Exception as e:
                print(f"WARNING: Could not filter points based on selection method {s_idx}")

        print("")
        print("Process Successful!")
        doc.save()

    # Export Camera positions
    if chunk.tie_points:
        announce("Exporting Camera Positions")
        chunk.exportCameras(path=output_cameras,
                            progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:
        announce("Building depth maps")
        # Quality
        downscale = {"lowest": 16,
                     "low": 8,
                     "medium": 4,
                     "high": 2,
                     "highest": 1}[args.quality.lower()]

        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering,
                             downscale=downscale,
                             progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a dense point cloud using the depth maps.
    if chunk.depth_maps and not chunk.point_cloud:
        announce("Building dense point cloud")
        chunk.buildPointCloud(source_data=Metashape.DepthMapsData,
                              progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a DEM from the point cloud.
    if chunk.point_cloud and not chunk.elevation:
        announce("Building DEM")
        chunk.buildDem(source_data=Metashape.PointCloudData,
                       interpolation=Metashape.Interpolation.DisabledInterpolation,
                       progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build an orthomosaic from the point cloud.
    if chunk.point_cloud and not chunk.orthomosaic:
        announce("Building orthomosaic")
        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.PointCloudData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending,
                               fill_holes=False,
                               progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the dense point cloud if it exists in the chunk.
    if chunk.point_cloud and not os.path.exists(output_dense):
        announce("Exporting dense point cloud")
        chunk.exportPointCloud(path=output_dense,
                               save_point_color=True,
                               save_point_classification=True,
                               save_point_normal=True,
                               save_point_confidence=True,
                               crs=chunk.crs,
                               progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the DEM if it exists in the chunk.
    if chunk.elevation and not os.path.exists(output_dem):
        announce("Exporting DEM")
        chunk.exportRaster(path=output_dem,
                           source_data=Metashape.ElevationData,
                           progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Export the orthomosaic as a TIFF file if it exists in the chunk.
    if chunk.orthomosaic and not os.path.exists(output_ortho):
        announce("Exporting orthomosaic")
        # Set compression parameters (otherwise bigtiff error)
        compression = Metashape.ImageCompression()
        compression.tiff_big = True

        chunk.exportRaster(path=output_ortho,
                           source_data=Metashape.OrthomosaicData,
                           image_compression=compression,
                           progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Finally, export the report
    announce("Exporting Report")
    chunk.exportReport(path=output_report)

    print("")
    print("Process Successful!")
    doc.save()

    # Print a message indicating that the processing has finished and the results have been saved.
    print(f"\nNOTE: Processing finished, results saved to {project_dir}")
    print(f"NOTE: Completed in {np.around(((time.time() - t0) / 60), 2)} minutes")


def sfm(args):
    """

    """
    try:
        # First, check that the license is activated
        # on current machine; if not exit early.
        if not Metashape.License().valid:
            raise Exception("License not activated on current machine!")

    except Exception as e:
        raise Exception(f"ERROR: {e}")

    try:
        # If the license is valid, run the workflow
        print("NOTE: Running Workflow...")
        sfm_workflow(args)

    except Exception as e:
        print(f"ERROR: Could not finish workflow!\n{e}")
        print(f"ERROR: {traceback.print_exc()}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main(device: str = None, project_file: str = None, input_path: str = None, output_path: str = None):
    """

    """
    try:
        device = int(device)
    except Exception as e:
        raise Exception(f"ERROR: Invalid value for 'device'")

    # Get the configuration file
    config = configparser.ConfigParser()
    config.read('/home/metashape/config.ini')

    # Accessing values in SfM section
    input_dir = input_path
    output_dir = output_path
    project_file = project_file

    quality = config.get('SfM', 'quality')
    target_percentage = config.get('SfM', 'target_percentage')

    # Redundant, remove later
    parser = argparse.ArgumentParser(description='SfM Workflow')
    args = parser.parse_args([])

    # Manually fill in argparse values
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.project_file = project_file
    args.device = device
    args.quality = quality
    args.target_percentage = int(target_percentage)

    # Double check
    print("Input Directory:", args.input_dir)
    print("Num. Files: ", len(os.listdir(input_dir)))
    print("Output Directory:", args.output_dir)
    print("Project File:", args.project_file)
    print("Quality:", args.quality)
    print("Target Percentage:", args.target_percentage)

    # Create the output directory (which is mounted)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Run the workflow
        sfm(args)
        print("Completed SfM Workflow!.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())


if __name__ == '__main__':
    # Parse all but the first argument
    _, device, project_file, input_path, output_path = sys.argv
    # Call the main function to kick off the workflow
    main(device=device, project_file=project_file, input_path=input_path, output_path=output_path)

