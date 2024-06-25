#!/usr/bin/env python
# coding: utf-8

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, ComputeTarget
from azureml.core.runconfig import DockerConfiguration
import os


# A callable function that packages input and output data to be submitted for metashape processing
# takes in relevant information about the command and loads it into ML Studio Python SDK.
def submit_metashape_job(input_asset: str = None,
                         input_version: str = None,
                         output_asset: str = None,
                         output_version: str = None,
                         device: int = None,
                         quality: str = 'high',
                         target_percent: int = 75,
                         compute_target: str = None):
    # get default creds
    creds = DefaultAzureCredential()
    # use creds to make a client
    local_client = MLClient.from_config(credential=creds)

    # load in the input data information from the function def
    input_asset_name = input_asset
    input_asset_version = input_version
    input_asset = local_client.data.get(name=input_asset_name, version=input_asset_version)
    input_path = input_asset.path
    input_mode = InputOutputModes.DOWNLOAD

    # load in the output data info from the function def
    output_asset_name = output_asset
    output_asset_version = output_version
    output_asset = local_client.data.get(name=output_asset_name, version=output_asset_version)
    output_path = output_asset.path
    output_mode = InputOutputModes.UPLOAD

    # get a local instance of the compute info
    compute = local_client.compute.get(compute_target)

    # get the device number where it needs to be
    DEVICE = device
    QUALITY = quality
    TAR_PERCENT = target_percent

    # TODO Pass the project file (PSX) file to SfM.py
    # PROJECT_FILE = ...
    # If passed an empty string, class methods will deal with it

    # make sure the data types are the same
    data_type = local_client.data.get(name=input_asset_name, version=input_asset_version).type

    # create an instance of the output data store in case it doesn't exist, definitely redundant, needs to be removed
    output_asset = Data(
        path=output_path,
        type=data_type,
        description="output path to save outputs from a job submission",
        name=output_asset_name,
        version=output_asset_version
    )

    # try to make the store if it doesn't exist (it will)
    try:
        local_client.data.create_or_update(output_asset)
    except:
        # don't do anything if the store exists
        print(f'NOTE: "output_asset" already exists.')

    # create input and output dictionaries to use in the command calls later
    input = {
        "input_data": Input(type=data_type,
                            path=input_path,
                            mode=input_mode)
    }
    output = {
        "output_data": Output(type=data_type,
                              path=output_path,
                              mode=output_mode,

                              )
    }

    cmd = (f'python '
           f'SfM.py '
           f'{DEVICE} '
           f'${{inputs.input_data}} '
           # TODO ??? {PROJECT_FILE} ??? 
           f'${{outputs.output_data}} '
           f'{QUALITY} '
           f'{TAR_PERCENT}')

    # create linux command line commands to be sent to the compute target
    transfer_data = command(
        code='./SfM.py',
        command=cmd,
        inputs=input,
        outputs=output,
        environment='metashape-env@latest',
        compute=compute.name
    )

    # submit the job to the compute through the client
    returned_job = local_client.jobs.create_or_update(transfer_data)
    returned_job.studio_url
