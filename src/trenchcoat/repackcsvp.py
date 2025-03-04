"""Repacks CSV to NPZ (MT version).

This program is for converting a zip-compressed FLIR CSV file to NPZ or TDMS.

Make sure you have a decent amount of memory and space to support loading the files.

When temperature is exported from FLIR Thermal Studio, it exports the data as a CSV file with header data at the top followed by the temperature data.
Each frame is just stacked one after the other with no delimiter. These files are ridiculously big. To make it easier to store and a bit more Python friendly, 
the file is unpacked and converted to a NPZ file. NPZ files are compressed and a bit easier to deal with in numpy.

The program assumes that the CSV files are zip compressed since that's how I was storing them.

This program does the following:
    - Unpacks the CSV file
    - Loads each frame into a list
    - Convert to a numpy array
    - Write to NPZ

It is generally recommended to skip TDMS since they can be a bit messier to deal with. TDMS support was added on request.

Minimal Example:
    python -m repackcsvp.py --i csv_in --o npz_out --skiptdms
"""
from __future__ import annotations

import argparse
import os
import subprocess
from glob import glob
from multiprocessing.dummy import Pool
from pathlib import Path

import dataparser as dp
import numpy as np


def imageCSVSave(csv:str|Path,path_output:str|Path) -> Path:
    """Read images from exported FLIR CSV and save them to a NPZ compressed file.

    The data is exported with header data at the top followed by the images one after the other.
    The width of the images is the number of columns. No separator between frames.

    The data is loaded, reshaped into a 3D array and then saved as a NPZ file under the default key (arr_0)

    Inputs:
        csv: Path to CSV file exported by FLIR Thermal Studio
    """
    # convert to compressed CSV
    data = dp.pandasReadImageCSV(csv)
    # reshape to correct size
    data = data.values.reshape((-1, 348, 464)) if len(data.values.shape) == 2 else data.values
    # save CSV as compressed NPZ under default key arr_0
    np.savez_compressed(path_output,data)
    return Path(path_output)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Repack compressed CSVs as NPZ and TDMS files. The compressed files are extracted to an intermediary file and deleted once used.")
    # add an option for specifying 7z path
    parser.add_argument("--zip",nargs="?",const="7z",default="7z",type=str,help="Path to 7z executable. If not specified, then it is assumed that 7z is on path")
    # add an option for specifying intermediary path
    parser.add_argument("--ext",nargs="?",const="",default="",type=str,help="Path to intermediary folder where the CSVs are extracted. If not specified then it extracts to local folder")
    # input path
    parser.add_argument("--i","-input",help="Wildcard path to compressed CSV files",required=True)
    # output path
    parser.add_argument("--o","-output",help="Folder path to place the output",required=True)
    parser.add_argument("--skiptdms",const=True,action="store_const",help="Skip making TDMS")
    # parse arguments
    args = parser.parse_args()

    ## check arguments
    # if the input path is blank
    # default to local
    path_input = args.i
    # check if 7z exists
    zip_exe = args.zip
    # get the rest
    ext_path = "ext_csv"

    path_output = args.o

    # iterate over the files in the current folder
    # change the path to wherever the 7z files are stored
    for path in glob(path_input):
        # call 7z to extract the file to the specified folder
        # this can take a while
        if os.path.splitext(path)[1] == ".7z":
            subprocess.call([zip_exe,"e",path,"-o"+ext_path])  # noqa: S603
        # find the file
        csv = None
        for csv in glob(os.path.join(ext_path,"*.csv")):
            fname = os.path.splitext(os.path.basename(csv))[0]

            # convert the file to TDMS
            with Pool(3) as pool:
                if not args.skiptdms:
                    res_csv = pool.starmap(dp.convertCSVToTDMS,[(csv,os.path.join(path_output,fname+".tdms"))])
                res_npz = pool.starmap(imageCSVSave,[(csv,os.path.join(path_output,fname+".npz"))])
        # delete extracted CSV
        # this can be commented out, but given the size of the CSV files
        # can become difficult to manage
        if csv is not None:
            Path(csv).unlink(missing_ok=True)
