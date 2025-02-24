import subprocess
from glob import glob
import dataparser as dp
import os
import numpy as np
import argparse

"""
    Repack CSV to NPZ (Single threaded)

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
        - Write array to NPZ and TDMS file

    It is generally recommended to skip TDMS since they can be a bit messier to deal with. TDMS support was added on request.

    Minimal Example:
        python -m repackcsv.py --i csv_in --o npz_out --skiptdms
"""

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Repack compressed CSVs as NPZ and TDMS files. The compressed files are extracted to an intermediary file and deleted once used.")
    # add an option for specifying 7z path
    parser.add_argument("--zip",nargs='?',const="7z",default="7z",type=str,help="Path to 7z executable. If not specified, then it is assumed that 7z is on path")
    # add an option for specifying intermediary path
    parser.add_argument("--ext",nargs='?',const="",default="",type=str,help="Path to intermediary folder where the CSVs are extracted. If not specified then it extracts to local folder")
    # flag for being verbose
    parser.add_argument("--verbose",const=True,action="store_const",help="Print details about the current status of the conversion")
    parser.add_argument("--skiptdms",const=True,action="store_const",help="Skip making TDMS")
    # input path
    parser.add_argument("--i","-input",help="Wildcard path to compressed CSV files",required=True)
    # output path
    parser.add_argument("--o","-output",help="Folder path to place the output",required=True)
    # parse arguments
    args = parser.parse_args()

    verbose = args.verbose
    def printv(*args,**kwargs):
        if verbose:
            print(*args)

    ## check arguments
    # if the input path is blank
    # default to local
    path_input = args.i
    print("TEST",path_input, os.path.splitext(path_input)[1])
    if (os.path.splitext(path_input)[1] != '.7z') and (os.path.splitext(path_input)[1] != '.zip'):
        raise ValueError("Wildcard input path must point towards 7z or zip files!")
    # check if 7z exists
    zip_exe = args.zip
    # get the rest
    ext_path = "ext_csv"
    
    path_output = args.o
    print(path_input)

    # iterate over the files in the current folder
    # change the path to wherever the 7z files are stored
    for path in glob(path_input):
        # call 7z to extract the file to the specified folder
        # this can take a while
        printv(f"Extracting {path_input} to {ext_path}")
        subprocess.call([zip_exe,"e",path,"-o"+ext_path])
        # find the file
        csv = None
        for csv in glob(os.path.join(ext_path,"*.csv")):
            fname = os.path.splitext(os.path.basename(csv))[0]
            # convert the file to TDMS
            if not args.skiptdms:
                printv(f"Converting {csv} to TDMS file at {os.path.join(path_output,fname+'.tdms')}")
                dp.convertCSVToTDMS(csv,opath=os.path.join(path_output,fname+'.tdms'))
            printv(f"Reading in {csv}")
            # convert to compressed CSV
            try:
                data = dp.pandasReadImageCSV(csv)
                print(f"Parsed data. Got dataframe of shape {data.shape}\nHas cols {data.columns}\ndtypes {set(list(data.dtypes))}")
                # reshape to correct size
                if len(data.values.shape)==2:
                    printv("Reshaping the 2D data to 3D")
                    data = data.values.reshape((-1,348,464))
                else:
                    data = data.values
            except Exception as e:
                print(f"Pandas failed! {e}")
                print("Trying backup method")
                data = dp.altImageCSVload(csv)
            printv(f"Converting {csv} to {os.path.join(path_output,fname+'.npz')}")
            # save CSV as compressed NPZ under default key arr_0
            np.savez_compressed(os.path.join(path_output,fname+'.npz'),data)
        # delete extracted CSV
        # this can be commented out, but given the size of the CSV files
        # can become difficult to manage
        if csv is not None:
            os.remove(csv)
