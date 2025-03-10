# Metashape Azure

```bash
# cmd

conda create --name sfm python=3.8 -y
conda activate sfm

pip install "git+https://github.com/Jordan-Pierce/Metashape-Azure.git"
```

If you are also running Metashape `locally`, you will need to run the `install.py` script as well:

```bash
# cmd

python install.py
```

This will add dependencies to the `Metashape` python environment. Finally, to run the application, use the following 
command:

```bash
# cmd

metashape-azure-mls
```

<p align="center">
  <img src="figures/GUI.PNG" alt="Metashape-Azure-MLS">
</p>

### Running the Application

1.) Go to [Azure Machine Learning Studio](https://ml.azure.com/) on `Microsoft Edge`; login
2.) Navigate to your workspace
2.) Copy the following account parameter credentials (top-right) into the application:
  - Subscription ID
  - Resource Group
  - Workspace Name
3.) Save your credentials for the future
4.) Authenticate
5.) Choose the compute (cluster)
6.) Provide the input path (folder containing images) as an Azure URI
7.) Provide the output path (folder to contain the project) as an Azure URI
8.) Provide the project name
  - The following cannot already exist {output path}/{project name}
9.) Choose the SfM parameters
10.) Click "Run on Azure"

### Notes

- Azure Machine Learning Studio and Authentication must be done in `Microsoft Edge`
- You must be connected to the network (i.e., `VPN`) to access the `Azure` services.

