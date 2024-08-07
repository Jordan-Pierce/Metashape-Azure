# Metashape Azure

### Setup
```python
# Pull the repo
git clone https://github.com/Jordan-Pierce/Metashape-Azure.git
```

```python
import os
os.environ['AGISOFT_FLS'] = "host:port"
import Metashape [2.1.2]
Metashape.License().borrow(N)

# SfM.py

if __name__ == '__main__':

    main(sys.argv[1],  # Device                 0
         sys.argv[2],  # Input Path             ./input
         sys.argv[3],  # Project File           "" OR ./input/project.psx
         sys.argv[4],  # Output Path            ./output
         sys.argv[5],  # Quality                'high'
         sys.argv[6])  # Target Percentage      75

    Metashape.License().returnLicense()
```

### Notes
- `project_file` can be left as empty string, or a path
- `project_file` needs it's sidecar `project.files`, and the folder containing images
