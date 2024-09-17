import os.path
import subprocess

metashape_exe = rf"C:\\Program Files\\Agisoft\\Metashape Pro\\python\\python.exe"
assert os.path.exists(metashape_exe), "Metashape executable not found"

requirements = ["numpy==1.21.4",
                "azure-identity",
                "azure-ai-ml",
                "azure-cli",
                "pyqt5",
                "./packages/Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl"]

# Define the command to install the requirements
commands = [metashape_exe,
            '-m',
            'pip',
            'install',
            *requirements]

# Use subprocess to run the command in the terminal
result = subprocess.run(commands, shell=True, capture_output=True)

# Check if the command was successful and output the result
if result.returncode == 0:
    print("Requirements installed successfully!")
else:
    print("Failed to install requirements. Error message:")
    print(result.stdout.decode() if result.stdout else result.stderr.decode())