# Plasma Spray iCoating

Thermal spray coating is a manufacturing process where a solid substrate is coated with a material by a plasma stream. A carrier gas is converted to plasma by a high-power electrode and has particles of a coating material introduced into the flow. The plasma carries the material to the substrate where it hopefully adheres to the surface. Possible coatings include paint, metal and polymers. The process needs to be monitored to ensure that the particles adhere, a smooth coating is applied and defects aren't introduced into the coating. 

Possible conclusions include 
  - The coating particles not being hot enough to adhere
  - Particles bouncing off the surface due to an obtuse collision angle
  - Particles being too slow to adhere

## External requirements

- [Exiftool](https://exiftool.org/)
    + Required for reading metadata FLIR files
    + Needed for [PyExifTool](https://pypi.org/project/PyExifTool/)
    + Needs to be downloaded and placed on PATH (for Windows)
    + The path to EXE can also be provided in Python wrapper
 
## Local Installation

It can be installed locally by just directing pip towards the folder

```bash
python -m pip install .
```

## Build + Install
The following commands build the scripts into a WHL, installs it and quickly tests that it can be imported.

```bash
python -m pip install setuptools build
python -m build
python -m pip install dist\Trenchcoat-1.0.0-py3-none-any.whl
python -c "import trenchcoat"
```

## Recommendations

- Use [VSCode](https://code.visualstudio.com/) or similar to make it easier to interact with Jupyter notebooks
- Setup your environment using [venv](https://docs.python.org/3/library/venv.html) or similar to make it easier to test and manage.
  
## Link to Data
The CSQ and NPZ files from the paper can be found here

[https://doi.org/10.15131/shef.data.c.7375201](https://doi.org/10.15131/shef.data.c.7375201)

## Link to Papers

## Project Structure

- [scripts](src) : Program files for connecting to sensors, processing the data and generating results
  + [trenchcoat](src/trenchcoat) :Python API for processing the thermal data produced by a FLIR T540 and the Acoustic Emission Sensors
      * [seq](src/trenchcoat/seq) : Class for opening SEQ files and parsing the data
      * [dataparser](src/trenchcoat/dataparser) : For parsing TDMS files, image CSV files and converting to ther data types.
      * [improcessing](src/trenchcoat/improcessing) : Process and calculate statistics about the thermal imaging data
      * [parse_ae](src/trenchcoat/parse_ae) : For parsing AE TDMS or parquet files
      * [plotting](src/trenchcoat/plotting) : Utilities for plotting information about data files
      * [repackcsv](src/trenchcoat/repackcsv) : Repacks exported FLIR CSV files into NPZ files
      * [repackcsvp](src/trenchcoat/repackcsvp) : Repacks exported FLIR CSV files into NPZ files with multiprocessing
  + [notebooks](notebooks) : Notebooks explaining some of the main features in the API and how to replicate some of the plots in the papers

## Examples
### SEQCustomFile
The SEQCustomFile in [seq](src/trenchcoat/seq) is a class for parsing and decoding the data from a SEQ file.

**Note: The class only supports SEQ files packed using JPEG-LS. Use exiftool to check image type**

```python
from trenchcoat.seq import SEQCustomFile
import cv2

# open and map file
seq = SEQCustomFile(r"path\to\file")
# get number of frames
print("number of frames", seq.numframes())
# frame rate
print("framerate", seq.frame_rate())
# number of frames and frame shape (HxWxN)
print("frame shape", seq.shape())

# iterate over raw frames
for frame in seq:
    cv2.imshow("raw", frame)
    cv2.waitKey(1)

# iterate over frame in degrees C
for frame in seq.tempiter(E=0.96, units="C"):
    ## do something with the frame ##

# iterate over frame in terms of radiance
for frame in seq.temprad(E=0.96):
    ## do something with the frame ##
```
