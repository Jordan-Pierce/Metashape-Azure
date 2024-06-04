import os
import sys
import time
import argparse
import datetime
import traceback
import configparser
from pathlib import Path
import shutil
import numpy as np


try:
    import Metashape
except:
    print('error, metashape environment not correctly launched or loaded')
    sys.exit(1)
    
finally:
    pass

# from Common import print_progress
# Check that the Metashape version is compatible with this script

compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------
def announce(announcement: str):
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

    # If user passes a previous project dir use it
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
    output_mesh = project_dir + "Mesh.ply"
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
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, fit_k1=True,
                                      fit_k2=True, fit_k3=True, fit_k4=True, fit_p1=True, fit_p2=True, fit_p3=True,
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

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:

        announce("Building mesh")
        # Quality
        facecount = {"lowest": Metashape.FaceCount.LowFaceCount,
                     "low": Metashape.FaceCount.LowFaceCount,
                     "medium": Metashape.FaceCount.MediumFaceCount,
                     "high": Metashape.FaceCount.HighFaceCount,
                     "highest": Metashape.FaceCount.HighFaceCount}[args.quality.lower()]

        chunk.buildModel(source_data=Metashape.DepthMapsData,
                         interpolation=Metashape.Interpolation.DisabledInterpolation,
                         face_count=facecount,
                         progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build a DEM from the 3D model.
    if chunk.model and not chunk.elevation:

        announce("Building DEM")
        chunk.buildDem(source_data=Metashape.ModelData,
                       interpolation=Metashape.Interpolation.DisabledInterpolation,
                       progress=print_progress)

        print("")
        print("Process Successful!")
        doc.save()

    # Build an orthomosaic from the 3D model.
    if chunk.model and not chunk.orthomosaic:

        announce("Building orthomosaic")
        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
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

    # Export the mesh if it exists in the chunk.
    if chunk.model and not os.path.exists(output_mesh):

        announce("Exporting mesh")
        chunk.exportModel(path=output_mesh, progress=print_progress)

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

    metashape_license = args.metashape_license

    try:
        # First, just try to activate Metashape; if the license isn't provided
        # exit the script early.
        if metashape_license in ["", None]:
            raise Exception("ERROR: You must pass in a Metashape License.")

        # Get the Metashape License stored in the environmental variable
        print("NOTE: Activating license...")
        Metashape.License().activate(metashape_license)

    except Exception as e:
        print(f"ERROR: {e}")
        raise(e)

    try:
        # If the license is valid, run the workflow
        print("NOTE: Running Workflow...")
        sfm_workflow(args)

    except Exception as e:
        print(f"ERROR: Could not finish workflow!\n{e}")
        print(f"ERROR: {traceback.print_exc()}")

    finally:
        # Always deactivate after script regardless
        try:
            print("NOTE: Deactivating License...")
            Metashape.License().deactivate()
        except:
            pass

        if not Metashape.License().valid:
            print("NOTE: License deactivated or was not active to begin with.")
        else:
            print("ERROR: License was not deactivated; do not delete compute without Deactivating!")


def test_license(args):
    metashape_license = args.metashape_license

    try:
        # First, just try to activate Metashape; if the license isn't provided
        # exit the script early.
        if metashape_license in ["", None]:
            raise Exception("ERROR: You must pass in a Metashape License.")

        # Get the Metashape License stored in the environmental variable
        print("NOTE: Activating license...")
        Metashape.License().activate(metashape_license)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    finally:
        # Always deactivate after script regardless
        try:
            print("NOTE: Deactivating License...")
            Metashape.License().deactivate()
        except:
            pass

        if not Metashape.License().valid:
            print("NOTE: License deactivated or was not active to begin with.")
        else:
            print("ERROR: License was not deactivated; do not delete compute without Deactivating!")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main(license:str = None, input_path:str = None, output_path:str = None):
    """

    """
    
    config = configparser.ConfigParser()
    config.read('/home/metashape/config.ini')

    # Get the license stored as command line argument
    metashape_license = license

    # Accessing values in SfM
    
    input_dir = input_path
    output_dir = output_path
    project_file = config.get('SfM', 'project_file')
    quality = config.get('SfM', 'quality')
    target_percentage = config.get('SfM', 'target_percentage')
    
        

    # Redundant, remove later
    parser = argparse.ArgumentParser(description='SfM Workflow')
    args = parser.parse_args([])

    # Manually fill in argparse values
    args.metashape_license = metashape_license
    
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.project_file = project_file
    args.quality = quality
    args.target_percentage = int(target_percentage)
    

    # Double check
    print("Metashape License:", args.metashape_license)
    print("Input Directory:", args.input_dir)
    print("Num. Files: ", len(os.listdir(input_dir)))
    print("Output Directory:", args.output_dir)
    print("Project File:", args.project_file)
    print("Quality:", args.quality)
    print("Target Percentage:", args.target_percentage)

    # Testing output and volume mount
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except:
        announce("Output Directory Already Exists")
    test_output = args.output_dir + "output.txt"

    with open(test_output, 'w') as file:
        # Write content to the file
        file.write('This was output from the container!')

    print(f"TEST: {test_output} {os.path.exists(test_output)}")

    try:
        announce("Activating License")
        announce(args.metashape_license)
        #test_license(args)
        announce("End of License Test")
        
        # Run the workflow
        
        sfm(args)
        print("Completed SfM Workflow!.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())


if __name__ == '__main__':
   
    #assumes passing of license as a command line argument    
    args = sys.argv[1:]
        
    main(license = args[0], input_path = args[1], output_path = args[2])

    
