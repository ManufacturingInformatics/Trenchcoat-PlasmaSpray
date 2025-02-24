import numpy as np
import pandas as pd
from nptdms import TdmsFile, TdmsWriter, RootObject, ChannelObject
import os
import warnings
import multiprocessing as mp


def getParamsHeader(path: str) -> dict:
    """
        Extract the header from the exported FLIR CSV file

        Input:
            path : Path to target CSV

        Returns dictionary of values
    """
    params = {}
    with open(path,'r') as source:
        # first line is the source file path
        line = source.readline().strip()
        # source path is on other side of the comma
        params["file"] = line.split(',')[1]
        # next line is empty
        source.readline()
        # next line as the emissivity and a parameters header
        line = source.readline().replace('Parameters:,','')
        for i in range(7):
            pts = line.split(',')
            pts[0] = pts[0].split(':')[0]
            params[pts[0]] = pts[1].strip()
            line = source.readline()[1:].strip()
    return params


def printTDMSInfo(path: str,to_log:bool=True) -> str | None:
    """
        Print the structure of the TDMS file using the tdmsinfo command line tool installed with nptdms

        Inputs:
            path : File path to target file
            to_log : Write the returned string to a text file. If True, then the file is based on the input filename. If a string, then that is used as the output filename.

        Returns None if to_log if a string or True else returns the string created by the tdmsinfo command
    """
    # get filename
    fname = os.path.splitext(os.path.basename(path))[0]
    import subprocess
    td_struct = subprocess.check_output(["tdmsinfo","-p",path])
    td_str = td_struct.decode("utf-8")
    if not to_log:
        return td_str
    else:
        # if the user specified an output file
        if isinstance(to_log,str):
            open(to_log,'w').write(td_str)
        elif isinstance(to_log,bool):
            open(f"{fname}.txt",'w').write(td_str)


def parseImageCSV(path: str,params:bool=True,img_shape:tuple=(464,348),**kwargs) -> np.ndarray:
    """
        Processes the exported CSV file from FLIR Report Studio and returns a numpy array.

        Inputs:
            path : Path to single exported temperature CSV file.
            params : Flag or int indicating if the CSV file contains parameters in the file header. If True, then it skips the first 10 rows.
                    If False, then it only skipts the first 2 which is the filename. If an integer is given, then that number of rows is skipped in the file.
            img_shape : Two element tuple defining the shape of the image data. Default (464,348).
            jump_to : Jump to a specific number of seconds or specific frame to create the CSV file from. If the user gives a float, then it's taken as number of seconds. If an integer,
                        it's taken as the specific frame to start from..
            framerate : Framerate of the data. Default 30 Hz.

        Returns a numpy array of type float64

    """
    # if params is boolean and False
    if isinstance(params,bool) and (not params):
        # set number of rows to skip to 2
        sh = 2
    # if there are parameters in the data file
    # skip the first 10 rows
    elif isinstance(params,bool) and params:
        sh = 10
    # if the user has specified a custom number
    # store that value
    elif isinstance(params,int):
        sh = params
    # if the user wants to jump to a specific frame
    if 'jump_to' in kwargs:
        # get jump period
        jump = kwargs['jump_to']
        # get frame rate
        # default to 30Hz
        fr = int(kwargs.get('framerate',30))
        # if the user specified a float
        # take as number of seconds
        if isinstance(jump,float):
            # convert to frame count using framerate
            # times by frame height to get number of frames to skip
            sh += int(fr * jump * img_shape[1])
        # if the user specified an integer
        # take as frame to jump to
        elif isinstance(jump,int):
            sh += int((jump-1) * img_shape[1])
    # process data according to set parameters
    data = np.genfromtxt(path,dtype=float,delimiter=',',skip_header=sh)
    # skip first column
    data = data[:,1:]
    return np.dstack(np.vsplit(data,data.shape[0]//img_shape[0]))


def altImageCSVload(path: str) -> np.ndarray:
    frame_list = []
    current_frame = []
    with open(path) as data:
        # skip filename and new line
        data.readline()
        data.readline()
        # get filesize
        filesize = os.fstat(data.fileno()).st_size
        # loops until EOF
        # using this instead of empty strings as occasionally empty strings occur in the file
        while data.tell() != filesize:
            # frames have a Frame X followed by the comma separated values which are newline delimited
            line = data.readline().rstrip()
            # ignore only new lines
            if len(line) == 0:
                #print("skipping new line")
                continue
            pts = line.split(',')
            # if it's the start of a new frame
            if "Frame" in pts[0]:
                # append frame to current list
                if len(current_frame) > 0:
                    frame_list.append(np.vstack(current_frame))
                current_frame.clear()
                # remove the "Frame X" at start of string
                pts.pop(0)
            # remove zero length strings
            pts = list(filter(bool,pts))
            # convert values to a numpy row
            values = np.fromiter(map(float, pts), np.float64).reshape((1,464))
            current_frame.append(values)
    arr = np.dstack(frame_list)
    return arr


def pandasReadImageCSV(path: str,has_head: bool=True,img_shape: tuple=(464,348),add_fidx: bool=False,use_chunks: bool=True,**kwargs) -> pd.DataFrame:
    """
        Process the exported CSV file from FLIR Report Studio and returns a pandas dataframe

        Designed for use with larger CSVs are Pandas processes them faster

        Inputs:
            path : Input file path
            has_head : Flag for stating of the CSV has the parameters header and it should be skipped
            img_shape : Shape of each frame according to camera datasheet
            add_fidx : Add a column for frame index
            jump_to : Jump to a specific number of seconds or specific frame to create the CSV file from. If the user gives a float, then it's taken as number of seconds. If an integer,
                        it's taken as the specific frame to start from..
            framerate : Framerate of the data. Default 30 Hz.

        Returns pandas dataframe where each column is a row of the image
    """
    # check number of values defining image shape
    if len(img_shape)!=2:
        raise ValueError(f"Image shape must be 2 elements len(img_shape)=={len(img_shape)}")
    # set chunk size to None
    # if None, then the data ia read straight to a DataFrame
    csize = None
    # if the user simply stated true
    if (isinstance(use_chunks,bool) and use_chunks):
        # set chunk size to per-frame
        csize = int(img_shape[1])
    # if user stated an int size
    elif (isinstance(use_chunks,int) and use_chunks):
        if use_chunks < 0:
            raise ValueError("Chunk size cannot be negative!")
        csize = use_chunks
    # set number of rows to skip to 10
    # skip header
    sr = 10
    # if the user wants to jump to a specific frame
    if 'jump_to' in kwargs:
        # get jump period
        jump = kwargs['jump_to']
        # get frame rate
        # default to 30Hz
        fr = int(kwargs.get('framerate',30))
        # if the user specified a float
        # take as number of seconds
        if isinstance(jump,float):
            # convert to frame count using framerate
            # times by frame height to get number of frames to skip
            sr += int(fr * jump * img_shape[1])
        # if the user specified an integer
        # take as frame to jump to
        elif isinstance(jump,int):
            sr += int((jump-1) * img_shape[1])
    # read using pandas
    # if a chunksize is given then 
    data = pd.read_csv(path,delimiter=',',skiprows=sr if has_head else 2, # if the the file has the parameter header, then skip it
                       header=None, # this is to stop it using the first row of values as the header names
                       usecols=[i for i in range(1,img_shape[0]+1)], # skip first column
                       dtype="float64", # datatype
                       chunksize=csize) # chunksize for v. large datasets set to 1 frame
    # if a chunk size is given then pandas returns a TextFileReader class
    # need to concat the chunks together to form a DataFrame
    if csize is not None:
        data = pd.concat(data,ignore_index=True)
    # if the user doesn't want the frame index column
    if not add_fidx:
        return data
    else:
        # insert column at start representing frame index
        data.insert(0,"NFRAME",np.arange(0,data.shape[0]) // img_shape[1],True)
        return data


def convertCSVToTDMS(ipath: str,opath:str|None = None,**kwargs):
    """
        Convert the target CSV file to a TDMS file

        Opens the CSV file as a chunked data set, reads the chunks in and writes them to a TDMS file. This approach is used as the target
        CSV files can be several GBs in size so reading them into memory is impractical. The chunk size is set to the number of rows meaning
        each chunk is 1 frame of data.

        If the has_head flag is True, then the metadata stored at the top of the CSV is added to the TDMS. Regardless, the image shape given by
        the user is stored as metadata so the vector of values can be reshaped back to its original size

        Input:
            ipath : Path to target file
            opath : Output filepath. If None, it is based off the input filename
            img_shape : Shape of the image data (ww,hh)
            has_head : Flag to indicate if the CSV file has a parameter header. If True, then the metadata stored is added to the TDMS file in the root
    """
    # get image shape
    img_shape = kwargs.get('img_shape',(464,348))
    # get flag indicating if there are parameters
    params = kwargs.get('has_head',True)
    # create empty dict
    metadata = {}
    # get metadata from top of the CSV file
    if params:
        if os.path.splitext(ipath)[1] in ['.7z','.zip']:
            warnings.warn("Skipping extracting headers as it can't be retrieved from a compressed file!")
        else:
            metadata = getParamsHeader(ipath)
    # add image shape
    metadata["img_shape"] = str(img_shape)
    # load data into pandas array
    data = pd.read_csv(ipath,delimiter=',',skiprows=10 if params else 2, # if the the file has the parameter header, then skip it
                       header=None, # this is to stop it using the first row of values as the header names
                       usecols=[i for i in range(1,img_shape[0]+1)], # skip first column
                       dtype="float64", # datatype
                       chunksize=int(img_shape[1])) # chunksize for v. large datasets set to 1 frame
    # create tdms file
    # set output path to either the given one or base it off the input filename
    # file is set in append mode
    with TdmsWriter(opath if opath is not None else os.path.splitext(os.path.basename(ipath))[0]+'.tdms') as tdms_writer:
        # create root object
        root = RootObject(properties=metadata)
        # iterate over the chunks
        for chunk in data:
            tdms_writer.write_segment([root,ChannelObject("flir","temp",chunk.values.flatten())])


def convertCSVToHDF5(ipath: str,opath: str|None = None,**kwargs):
    """
        Convert the target CSV file to a compressed HDF5 file

        The advantages of this over TDMS is it retains the 3D shape and compressed the data, but has a performance penalty.

        Data is saved as a 3D dataset called temp. If the CSV file has a header, then it is added as Attributes to the dataset.
        See Attributes under h5py docs.

        Inputs:
            path : Path to CSV file
            opath : Output path. If None, then the output file is based on the input filepath
            has_head : Flag to indicate that there's heaer information in the CSV. If True, then the header info is added as attributes.
            img_shape : 2-element list or tuple representing the width and height of the image
            cmethod : Compression method. See h5py documentation for supported methods. Default gzip
            clevel : Level the data is compressed to. Higher means more compressed. Max level and effect depends on compression method. See h5py docs. Default 9
    """
    import h5py
    # get image shape
    img_shape = kwargs.get('img_shape',(464,348))
    # get flag indicating if there are parameters
    params = kwargs.get('has_head',True)
    # create empty dict
    metadata = {}
    # get metadata from top of the CSV file
    if params:
        if os.path.splitext(ipath)[1] in ['.7z','.zip']:
            warnings.warn("Skipping extracting headers as it can't be retrieved from a compressed file!")
        else:
            metadata = getParamsHeader(ipath)
    # add image shape
    metadata["img_shape"] = str(img_shape)
    csize = int(img_shape[1])
    data = pd.read_csv(ipath,delimiter=',',skiprows=10 if params else 2, # if the the file has the parameter header, then skip it
                       header=None, # this is to stop it using the first row of values as the header names
                       usecols=[i for i in range(1,img_shape[0]+1)], # skip first column
                       dtype="float64", # datatype
                       chunksize=csize) # chunksize for v. large datasets set to 1 frame

    with h5py.File(opath if opath is not None else os.path.splitext(os.path.basename(ipath))[0]+'.hdf5','w') as dest:
        chunk = data.get_chunk(csize).values.reshape((img_shape[0],img_shape[1],1))
        # create dataset write first chunk
        data_set = dest.create_dataset('temp',data=chunk,compression=kwargs.get("cmethod","gzip"),compression_opts=kwargs.get("clevel",9),shape=(img_shape[1],img_shape[0],1),maxshape=(img_shape[1],img_shape[0],None))
        for k,v in metadata.items():
            data_set.attrs.create(k,v)
        # iterate over other chunks
        for chunk in data:
            data_set.resize((dest['temp'].shape[-1] + 1),axis=2)
            data_set[:,:,-1] = chunk.values


def getCalibrationParams(params_dict: dict) -> dict:
    """ Filter the CSQ metadta to just the calibration parameters """
    return dict(filter(lambda x : x[0] in ["FLIR:PlanckR2","FLIR:PlanckR1","FLIR:PlanckO","FLIR:PlanckB","FLIR:PlanckF","FLIR:ReflectedApparentTemperature"],params_dict.items()))


# function to convert temperature of object to raw value object
def Tobj2Robj(T:np.ndarray|float,R2:float,R1:float,O:float,B:float,F:float) -> np.ndarray | float:   # noqa: E741
    """ Convert object temperature to raw object value """
    # reverse calculation
    # from https://exiftool.org/forum/index.php?topic=4898.msg23972#msg23972
    exp_BT = np.exp(B/T)
    # numerator
    raw_num = R1 + R2*O*(F-exp_BT)
    # denominator
    raw_den = R2*(exp_BT-F)
    return raw_num/raw_den

# formula adapted from
# Thermimage R package https://rdrr.io/cran/Thermimage/man/raw2temp.html
def Robj2Tobj(Robj,R2,R1,B,F,O):  # noqa: E741
    """ Convert raw object values to object temperature Kelvin """
    return B/np.log(R1/(R2*(Robj+O))+F)

# function to convert reflected temperature to raw value of reflectance
def Tref2Rref(Tref,R2,R1,B,F,O):  # noqa: E741
    """ Convert reflected temperature in Kelvin to a raw value """
    return R1/(R2*(np.exp(B/Tref)-F))-O

# function to convert raw reflected value to reflected temperature
def Rref2Tref(Rref,R2,R1,B,F,O):  # noqa: E741
    """ Convert raw reflected value to reflected temperature in Kelvin"""
    return B/(np.log((R1+R2*((O*F)+(F*Rref)-O))/(Rref*R2)))

# function to convert raw reflectance to raw object
def Raw2Robj(S,E,Rref):
    """ Convert Raw FLIR value to Raw value for object """
    return (S-(1-E)*Rref)/E

def Robj2Raw(Robj,E,Rref):
    """ Convert raw value for object to raw FLIR value """
    return (Robj*E)+(1-E)*Rref

def changeE(temp,Eold,Enew,R2,R1,O,B,F,Tref):  # noqa: E741
    """
        Re-calculate the temperature frame under a new emissivity value

        Large temperature recordings tend to eat up a lot of memory and take a while.
        Only use 2D temperature arrays for this function.

        For 3D temperature arrays, see changeEmissivity.

        Inputs:
            temp : 2D temperature array
    """
    # convert reflected temperature to raw reflected value
    Rref = Tref2Rref(Tref,R2,R2,B,F,O)
    # convert temperature to raw object value
    Robj = Tobj2Robj(temp,R2,R1,O,B,F)
    # convert raw object to raw values
    raw = Robj2Raw(Robj,Eold,Rref)
    # forward calculate to get new raw object under new emissivity
    Robj = Raw2Robj(raw,Enew,Rref)
    # convert new to temperature
    return Robj2Tobj(Robj,R2,R1,B,F,O)

def changeEmissivity(temp,Eold,Enew,frame_first=False,**params_dict) -> np.ndarray:
    """
        Batch convert the emissivity of the given temperature array.

        The temperature values go through multiple stages of conversion to get the raw values and then
        do the forward calculation under the new emissivity value.

        If the given temperature is 2D then the changeE function is called and returned.

        If the given temperature is 3D, then each frame is passed to a multiprocessing.Pool to speed up processing.

        Currently the number of processes is set to 10.

        Inputs:
            temp: 2D or 3D array of temperature in Kelvin.
            Eold : Emissivity of the recording
            Enew : New emissivity value to convert to
            frame_first : Flag to indicate if the temperature array is organised frames first or not. If True, then it's organised (frames, width, height). Default False.
            **params_dict : Dictionary of calibration parameters. Can be the loaded CSQ metadata or organised by parameter names.
    """
    # extract calibration parameters from the dictionary
    try:
        R2 = params_dict["FLIR:PlanckR2"]
    except KeyError:
        R2 = params_dict["R2"]

    try:
        R1 = params_dict["FLIR:PlanckR1"]
    except KeyError:
        R1 = params_dict["R1"]

    try:
        O = params_dict["FLIR:PlanckO"]  # noqa: E741
    except KeyError:
        O = params_dict["O"]  # noqa: E741

    try:
        B = params_dict["FLIR:PlanckB"]
    except KeyError:
        B = params_dict["B"]

    try:
        F = params_dict["FLIR:PlanckF"]
    except KeyError:
        F = params_dict["F"]

    try:
        Tref = params_dict["FLIR:ReflectedApparentTemperature"]
    except KeyError:
        Tref = params_dict["Tref"]

    # if the temperature array is 2D i.e. a single frame
    # call changeE
    if len(temp.shape)==2:
        return changeE(temp,Eold,Enew,R2,R1,O,B,F,Tref)
    # if it's an array of frames
    elif len(temp.shape)==3:
        # create array to hole results
        temp_new = np.empty(temp.shape)
        # create process pool
        with mp.Pool(processes=10) as pool:
            # if the frame index is first
            if frame_first:
                # pass each frame to convert and then add the results to raw
                res = pool.map(changeE,((temp[z,:,:],Eold,Enew,R2,R1,O,B,F,Tref) for z in range(temp.shape[0])))
                for z in range(temp.shape[0]):
                    temp_new[z,:,:]=res[z]
            # if the frame index is last
            else:
                # pass each frame to convert and then add the results to raw
                res = pool.map(changeE,((temp[:,:,z],Eold,Enew,R2,R1,O,B,F,Tref) for z in range(temp.shape[-1])))
                for z in range(temp.shape[-1]):
                    temp_new[:,:,z]=res[z]
        return temp_new

def batchConvert(raw,fn,frame_first=False,**params_dict):
    """
        Batch convert data using multiprocessing pool

        Currently the number of processes is set to 10.

        Inputs:
            raw: 3D array of values.
            fn : Function to apply to each frame of the dataset
            Eold : Emissivity of the recording
            Enew : New emissivity value to convert to
            frame_first : Flag to indicate if the temperature array is organised frames first or not. If True, then it's organised (frames, width, height). Default False.
            **params_dict : Dictionary of calibration parameters. Can be the loaded CSQ metadata or organised by parameter names.
    """
    # extract calibration parameters from the dictionary
    try:
        R2 = params_dict["FLIR:PlanckR2"]
    except KeyError:
        R2 = params_dict.get("R2",0.0)

    try:
        R1 = params_dict["FLIR:PlanckR1"]
    except KeyError:
        R1 = params_dict.get("R1",0.0)

    try:
        O = params_dict["FLIR:PlanckO"]  # noqa: E741
    except KeyError:
        O = params_dict.get("O",0.0)  # noqa: E741

    try:
        B = params_dict["FLIR:PlanckB"]
    except KeyError:
        B = params_dict.get("B",0.0)

    try:
        F = params_dict["FLIR:PlanckF"]
    except KeyError:
        F = params_dict.get("F",0.0)

    try:
        Tref = params_dict["FLIR:ReflectedApparentTemperature"]
    except KeyError:
        Tref = params_dict.get("Tref",0.0)

    try:
        Eold = params_dict["FLIR:Emissivity"]
    except KeyError:
        Eold = params_dict.get("Eold",0.0)

    Enew = params_dict["Enew"]

    # create array to hole results
    raw_new = np.empty(raw.shape)
    # create process pool
    with mp.Pool(processes=10) as pool:
        # if the frame index is first
        if frame_first:
            # pass each frame to convert and then add the results to raw
            res = pool.map(fn,((raw[z,:,:],Eold,Enew,R2,R1,O,B,F,Tref) for z in range(raw.shape[0])))
            for z in range(raw.shape[0]):
                raw_new[z,:,:]=res[z]
        # if the frame index is last
        else:
            # pass each frame to convert and then add the results to raw
            res = pool.map(fn,((raw[:,:,z],Eold,Enew,R2,R1,O,B,F,Tref) for z in range(raw.shape[-1])))
            for z in range(raw.shape[-1]):
                raw_new[:,:,z]=res[z]
    return raw
        
def temp2raw(temp,frame_first=True,**params_dict):
    """
        Convert FLIR temperature from to raw data values

        Requires the calibration parameters from the file. They can be extracted using the flirmagic.getMetadata function
        or by passing a dictionary of the parameters.

        The required parameters must be accessible under the FLIR Tags e.g. FLIR:PlanckR2 or by their parameter name R2

        Even though the function can work with the full 3D temperature array, for large datasets it might be worth feeding it in frame by frame

        Inputs:
            temp : N-D temperature array.
            **params_dict : Dictionary of camera parameters.
    """
    # extract calibration parameters from the dictionary
    try:
        R2 = params_dict["FLIR:PlanckR2"]
    except KeyError:
        R2 = params_dict["R2"]

    try:
        R1 = params_dict["FLIR:PlanckR1"]
    except KeyError:
        R1 = params_dict["R1"]

    try:
        O = params_dict["FLIR:PlanckO"]  # noqa: E741
    except KeyError:
        O = params_dict["O"]  # noqa: E741

    try:
        B = params_dict["FLIR:PlanckB"]
    except KeyError:
        B = params_dict["B"]

    try:
        F = params_dict["FLIR:PlanckF"]
    except KeyError:
        F = params_dict["F"]

    # if the input data is 2D
    # no real need to create process pool
    if len(temp.shape)==2:
        return temp2raw(temp,R2,R1,O,B,F)
    # if input is 3D
    elif len(temp.shape)==3:
        # create array to hole results
        raw = np.empty(temp.shape)
        # create process pool
        with mp.Pool(processes=10) as pool:
            # if the frame index is first
            if frame_first:
                # pass each frame to convert and then add the results to raw
                res = pool.map(temp2raw,((temp[z,:,:],R2,R1,O,B,F) for z in range(temp.shape[0])))
                for z in range(temp.shape[0]):
                    raw[z,:,:]=res[z]
            # if the frame index is last
            else:
                # pass each frame to convert and then add the results to raw
                res = pool.map(temp2raw,((temp[:,:,z],R2,R1,O,B,F) for z in range(temp.shape[-1])))
                for z in range(temp.shape[-1]):
                    raw[:,:,z]=res[z]
        return raw

def raw2temp(raw,E,frame_first=True,**params_dict):
    """
        Convert the raw values to temperature values in degrees C

        The required parameters must be accessible under the FLIR Tags e.g. FLIR:PlanckR2 or by their parameter name R2

        Returns a numpy array of camera 

        Inputs:
            raw : Numpy array of raw values
            E : Target emissivity value
            **params_dict : Dictionary of camera paramters
    """
    # extract calibration parameters from the dictionary
    try:
        R2 = params_dict["FLIR:PlanckR2"]
    except KeyError:
        R2 = params_dict["R2"]

    try:
        R1 = params_dict["FLIR:PlanckR1"]
    except KeyError:
        R1 = params_dict["R1"]

    try:
        O = params_dict["FLIR:PlanckO"]  # noqa: E741
    except KeyError:
        O = params_dict["O"]  # noqa: E741

    try:
        B = params_dict["FLIR:PlanckB"]
    except KeyError:
        B = params_dict["B"]

    try:
        F = params_dict["FLIR:PlanckF"]
    except KeyError:
        F = params_dict["F"]

    if len(raw.shape)==2:
        return raw2temp(raw,E,R2,R1,O,B,F)
    # if the input is 3d
    elif len(raw.shape)==3:
        # create array to hole results
        temp = np.empty(raw.shape)
        # create process pool
        with mp.Pool(processes=10) as pool:
            # if the frame index is first
            if frame_first:
                # pass each frame to convert and then add the results to raw
                res = pool.map(raw2temp,((raw[z,:,:],E,R2,R1,O,B,F) for z in range(raw.shape[0])))
                for z in range(raw.shape[0]):
                    temp[z,:,:]=res[z]
            # if the frame index is last
            else:
                # pass each frame to convert and then add the results to raw
                res = pool.map(raw2temp,((raw[:,:,z],E,R2,R1,O,B,F) for z in range(raw.shape[-1])))
                for z in range(raw.shape[-1]):
                    temp[:,:,z]=res[z]
        return temp

def loadTDMSData(path,inc_time=True):
    """
        Load the data from non-empty channels in a TDMS file

        Iterates over groups and channels in the target TDMS file and adds the non-empty ones to
        a dictionary organised by channel names.

        Each array is a numpy array of data. If inc_time is True, then the time vector is generates
        and added to the array changing it into a 2 column array.

        Inputs:
            path : Path to target TDMS file
            inc_time : Flag to include

        Returns a dictionary of found channel data
    """
    data = {}
    with TdmsFile(path) as of:
        for gg in of.groups():
            for cc in gg.channels():
                if cc.data.shape[0]>0:
                    if inc_time:
                        data[cc.name] = np.hstack((cc.time_track().reshape(-1,1),cc.data.reshape(-1,1)))
                    else:
                        data[cc.name] = cc.data
    return data

class R_pca:
    def __init__(self, D, mu=None, lmbda=None):
        """
            Constructor for RPCA

            Inputs:
                D : Matrix to perform RPCA on.
                mu,lmbda : Used in fitting
        """
        self.D = D
        #self.n, self.d = self.D.shape
        # initialie matrix for fitting
        #self.S = np.zeros(self.D.shape)
        #self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
        self._fitted = False

    def is_fitted(self):
        return self._fitted

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=None):
        """
            Fit RPCA to the data given earlier

            Inputs:
                tol : Tolerance for fitting. If None, set to result of frobenius_norm on D
                max_iter : Max number of iterations to try. Default 1000
                iter_print : At what multiple of iterations a print statement is made. If None,
                            then nothing is printed. Default None.
        """
        # current iteration
        it = 0
        # error to fitting
        err = np.Inf
        # matrices
        Sk = np.zeros(self.D.shape)
        Yk = np.zeros(self.D.shape)
        Lk = np.zeros(self.D.shape)
        # tolerance for error
        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * np.linalg.norm(self.D)
        # reset fitted flag
        self._fitted = False
        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and it < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk += self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            it += 1
        self._fitted = True
        return Lk, Sk

def parseSeqCSV(fn: np.array) -> np.ndarray:
    """ 
        Parse exported FLIR CSV files and stack them into a 2D numpy array

        Input:
            fn : Input file path

        Returns numpy array of stacked rows
    """
    # open file
    with open(fn,'r') as of:
        # read until frame counter
        line = of.readline()
        while 'Frame' not in line:
            continue
        # collecting remaining lines into parts
        pts = [line.rstrip().split(','),] + [line.rstrip().split(',')[1:] for line in of]
    pts.pop(-1)
    print("finished!")
    return np.row_stack(pts)


class StackSeqCSV:
    """ 
        OOP approach for reading several exported FLIR CSV files and stacking them into a single file

        Can prob write it as a single function. Just wanted to stage it as methods

        Input:
            path : Folder where the CSV files are located

        Examples:
            stack_csv = StackSeqCSV(f"csv_files/*.csv")
            stack_csv.stack("stacked_csv.csv")
    """
    def __init__(self,path: str):
        # get and sort seq frames into a list
        self._paths = sorted(glob(path),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1].split(' ')[0]))
        self._nf = len(self._paths)
        self._skip = 0
        self._shape = ()
    
    # est final size of the stacked CSV
    def estSize(self):
        self._skip=1
        with open(self._paths[0],'r') as of:
            line = of.readline()
            # read until Frame is found
            # set set as number of rows to skip for every other fils
            while 'Frame' not in line:
                line = of.readline()
                self._skip+=1
            # split line into parts to get number of columns
            cols = len(line.rstrip().split(',')[1:])
            # init col list to pass to pandas
            self._cols = [i for i in range(1,cols+1)]
            i = 0
            for line in of:
                i+=1
        self._shape = (self._nf,i,len(self._cols))
        return self._shape

    def stack(self,fout:str,convert:bool=True,progress:bool=True):
        # if user wants to convert the stack immediately afterwards
        # set the temp out file to temp.csv and assume fout is the output filename
        fout_csv = 'temp.csv' if convert else fout
        # initialize progress variables
        ci = 0
        if progress:
            th = 1000 if isinstance(progress,bool) and progress else progress
            pi = 100*(th/len(self._paths))
            p = 0
        with open(fout_csv,'w') as of:
            for fi,fn in enumerate(self._paths):
                if progress:
                    if (ci == th):
                        p += pi
                        print(f"{p:.2f}%...")
                        ci = 0
                with open(fn,'r') as inf:
                    # skip header
                    line = inf.readline()
                    while 'Frame' not in line:
                        line = inf.readline()
                    # write remaining lines to file
                    for line in inf:
                        if line != '\n':
                            of.write(line)
                ci += 1
        if convert:
            data = pd.read_csv(fout_csv,sep=',',dtype='float16',header=None,usecols=np.arange(1,465))
            print("pandas shape ",data.shape)
            np.savez_compressed(fout,data.values.reshape((-1,347,464))[::-1,...])
            # remove temp CSV to hold stack
            os.remove(fout_csv)

    def checkFiles(self,progress=True):
        # initialize progress variables
        ci = 0
        if progress:
            th = 1000 if isinstance(progress,bool) and progress else progress
            pi = 100*(th/len(self._paths))
            p = 0
        skip = set()
        rows = set()
        cols = set()
        for fi,fn in enumerate(self._paths):
            if progress:
                if (ci == th):
                    p += pi
                    print(f"{p:.2f}%...")
                    ci = 0
            with open(fn,'r') as inf:
                # skip header
                line = inf.readline()
                i = 0
                while 'Frame' not in line:
                    line = inf.readline()
                    i += 1
                skip.add(i)
                sk = i
                # write remaining lines to file
                for line in inf:
                    if line != '\n':
                        i += 1
                        cols.add(len(line.rstrip().split(',')))
                rows.add(i-sk)
            ci += 1
        return skip,rows,cols

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    from glob import glob
    import matplotlib.pyplot as plt
    #from improcessing import findOrderedStripes, plotMaxStripeTemperature, frame2gray

    #masks_list = findOrderedStripes("lsbu-doe-stripes-masks/masks/cleanuplsbu_doe_powder_stripes_0003_mask_stripes.jpg",as_ct=False)
    #pathA = r"D:\FLIR Studio Output\20221117 194938\*.npz"
    #pathB = r"D:\FLIR Studio Output\20221123 120606\lsbu_doe_powder_stripes_0003.npz"

    #fn = r"D:\FLIR Studio Output\20221123 135003\lsbu_doe_powder_stripes_0003_0001.csv"
    #D = parseSeqCSV(fn)
    #stackSeqCSVs(r"D:\FLIR Studio Output\20221123 135003\*.csv")
    #for i in range(1,6):
    #    StackSeqCSV(rf"D:\FLIR Studio Output\20221202 172418\em_0{i}\*.csv").stack(f'lsbu_doe_powder_stripes_0003_em0{i}.npz')
    
##    f,ax = plt.subplots(constrained_layout=True)
##    ax.set(xlabel='Frame Index',ylabel='Temperature ($^\circ$C)',title='Striped Plate Coating, Different Emissivity')
##    for fn in glob('lsbu_doe_powder_stripes_0003_em0*.npz'):
##        em = os.path.splitext(fn)[0].split('_')[-1].split('em')[1]
##        em = em[0]+'.'+em[1]
##        data = np.load(fn)["arr_0"]
##        ax.plot(data.max((1,2)),label=em)
##    ax.legend()
##    plt.show()

    # load temperature data
    # clipping to avoid deadspace and to save memory
    data = np.load('coating_plate_1/lsbu_plasma_coating_plate_1.npz')["arr_0"]
    
    data = data[:,::4,::4]
    nf,r,c = data.shape
    data = data.reshape((r,c,nf))
    #data.shape = (r,c,nf)
    # re-arrange to be column
    L,S = R_pca(data).fit()
    print(L.shape,S.shape)
    
