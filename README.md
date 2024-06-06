# Metashape Azure

### Setup
```python
# Have Docker installed...

# Pull the repo
git clone https://github.com/Jordan-Pierce/Metashape-Azure.git

### I/O
`./input` - (Copied over) Where all images of a scene should be  
`./output` - (Mounted volume) Where all outputs (`.psx` and data products) will be placed

# From root
docker-compose build
docker-compose up
```