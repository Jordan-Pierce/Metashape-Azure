# Metashape Azure

```bash
# cmd

conda create --name sfm python=3.8 -y
conda activate sfm

pip install "git+https://github.com/Jordan-Pierce/Metashape-Azure.git"
```

If you are also running Metashape locally, you will need to run the install script as well:

```bash
# cmd

conda activate sfm

python install.py
```

Finally, to run the application, use the following command:

```bash
# cmd

metashape-azure-mls
```

### Notes

- If authentication opens any browser other than `Edge`, please copy the URL and paste it into `Edge` to authenticate, as required from the `Azure` SDK.
- You must be connected to the network (i.e., `VPN`) to access the `Azure` services.

<p align="center">
  <img src="figures/GUI.PNG" alt="Metashape-Azure-MLS">
</p>