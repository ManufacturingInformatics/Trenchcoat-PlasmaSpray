"""Plotting module."""

from __future__ import annotations

import json
import os
import warnings
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataparser import loadTDMSData
from matplotlib import animation
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from nptdms import TdmsFile
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import splev, splrep
from scipy.signal import find_peaks, stft

if TYPE_CHECKING:
    from matplotlib.container import BarContainer


def plotMaxTemperaturePowder(path:str|Path,tclip:float=1000.0,c:int=10,**kwargs) -> plt.Figure:
    r"""Plots the max temperature of the powder in the file.

    The data is clipped by c columns until the global max temperature is below tclip degrees.

    If the data is reduced to 0 columns, then the function returns None and a warnings is printed
    If the data was succesfully clipped, then the max temperature for each frame is plotted and the
    figure object is returned.

    Inputs:
        path : Path to NPZ temperature file
        tclip : Temperature threshold in degrees C. Default 1000.0.
        c : Number of columns to clip each time until the max temp is below tclip. Default 10.
        xlabel : X-axis text label. Default Frame Number.
        ylabel : Y-axis text label. Default Max Frame Temperature ($^\circ$C).
        title : Axis text title. Max Temperature tclip={tclip}($^\circ$C)

    Returns matplotlib Figure object if the data could be successfully clipped else None
    """  # noqa: D401
    # check inputs to see if they're valid
    c = int(c)
    if c <=0:
        raise ValueError("Column Resolution cannot be less than or equal to 0")
    # load file
    data = np.load(path)["arr_0"]
    # attempt to clip data whilst global max is greater than threshold
    while data.max()>=tclip:
        data = data[:,:,:-c]
        # if the resulting dataset has zero columns exit as attempting to get a max will raise an error
        if data.shape[-1]==0:
            warnings.warn(f"Failed to clip {path} such that max temperature is less then {tclip}!",stacklevel=2)
            return None
    # if successful plot data
    f,ax = plt.subplots()
    ax.plot(data.max((1,2)))
    ax.set(xlabel=kwargs.get("xlabel","Frame Number"),ylabel=kwargs.get("ylabel","Max Frame Temperature ($^\circ$C)"),title=kwargs.get("title",f"Max Temperature tclip={tclip}($^\circ$C)"))  # noqa: W605
    return f

def plotMaxTempPowderSB(path:str|Path,tclip:float=950.0,**kwargs) -> plt.Figure:
    # load array
    data = np.load(path)["arr_0"]
    nf = data.shape[0]
    # mask array
    data[data>1000] = 0.0
    max_temp_df = pd.DataFrame.from_dict({"Frame Index" : np.arange(nf),"Max Temp" : data.max((1,2))})
    ax = sns.lineplot(max_temp_df,x="Frame Index",y="Max Temp")
    ax.set(xlabel="Frame Index",ylabel=kwargs.get("ylabel","Max Frame Temperature ($^\circ$C)"),title=kwargs.get("title",f"Max Temperature tclip={tclip}($^\circ$C)"))  # noqa: W605
    return ax.figure

def batchMaxTemperaturePowder(path:str|Path,tclip:float=1200.0,c:int=10,**kwargs):
    r"""Plots the max temperature of the powder in each of the found files.

    The data is clipped by c columns until the global max temperature is below tclip degrees.

    If the data is reduced to 0 columns, then the file is skipped and a warning is printed
    If the data was succesfully clipped, then the max temperature for each frame is plotted

    Inputs:
        path : Wildcard path to NPZ temperature file
        tclip : Temperature threshold in degrees C. Default 1000.0.
        c : Number of columns to clip each time until the max temp is below tclip. Default 10.
        xlabel : X-axis text label. Default Frame Number.
        ylabel : Y-axis text label. Default Max Frame Temperature ($^\circ$C).
        title : Axis text title. Max Temperature tclip={tclip}($^\circ$C)

    Returns matplotlib Figure object if the data could be successfully clipped else None
    """  # noqa: D401
    sns.set_theme("paper")
    # check inputs to see if they're valid
    c = int(c)
    if c<=0:
        raise ValueError("Column Resolution cannot be less than or equal to 0")
    # convert path to list of filenames
    # else assume something iterable
    if isinstance(path,[str,Path]):
        path = glob(path)
    labels = kwargs.get("labels")
    if labels is None:
        labels = len(path)*[None,]
    if kwargs.get("return_data",False):
        data_list = []
    # create axis
    f,ax = plt.subplots(constrained_layout=True)
    # iterate over files
    for fn,label in zip(path,labels):
        # get filename
        fname = os.path.splitext(os.path.basename(fn))[0]
        # if label not set use filename
        if label is None:
            label = fname  # noqa: PLW2901
        # load data
        data = np.load(fn)["arr_0"]
        # attempt to clip data
        while data.max()>=tclip:
            data = data[:,:,:-c]
            # if the resulting dataset has zero columns exit as attempting to get a max will raise an error
            if data.shape[-1]==0:
                print(f"Failed to clip {fn} such that max temperature is less then {tclip}!")
                break
        # skip file is number of columns is 0
        if data.shape[-1]==0:
            continue
        # plot per frame max temperature using filename as legend label
        if kwargs.get("return_data",False):
            data_list.append(data.max((1,2)))
            ax.plot(data_list[-1],label=label)
        else:
            ax.plot(data.max((1,2)),label=label)
    # set axis labels and title
    ax.set(xlabel=kwargs.get("xlabel","Frame Number"),ylabel=kwargs.get("ylabel","Max Frame Temperature ($^\circ$C)"),title=kwargs.get("title",f"Max Temperature tclip={tclip} ($^\circ$C)"))  # noqa: W605
    ax.legend()
    if kwargs.get("return_data",False):
        return f,data_list
    return f

def showTempAcoustic(temp:str|Path,ac:str|Path,flir_fps:float=30.0,clip_time:bool=False):
    """Creates a slider to compare a specific acoustic timestamp against accompanying temperature values.

    Creates two plots on two rows. The top plot is the colormapped temperature data. The bottom plot
    is the acoustic voltage from a single channel. The slider goes through the different acoustic samples
    and finds the closest matching in time temperature frame and displays it.

    Inputs:
        temp : Path to NPZ temperature file or numpy array of temperature data.
        ac : Path to TDMS AC file or numpy array of AC data
        flir_fps : FPS of temperature data. Used to calculate time vector. Default 30.0.
        clip_time : Flag to shift
    """  # noqa: D401
    # load data
    if isinstance(temp,str):
        temp = np.load(temp)["arr_0"]
    if isinstance(ac,str):
        from dataparser import loadTDMSData
        ac = loadTDMSData(ac,True)["Input 0"]  # noqa: FBT003
    nf = temp.shape[0]
    # construct time for temperature
    flir_time = np.arange(0.0,nf/flir_fps,1/flir_fps)
    if clip_time:
        # find difference between acoustic and temp
        diff = flir_time.max() - ac[:,0].max()
        # clip that many frames
        if diff > 0:
            temp = temp[int(flir_fps*diff):,:,:]
        else:
            ac = ac[int(1e6*diff):,:]
    # recalculate for clipped array
    flir_time = np.arange(0.0,nf/flir_fps,1/flir_fps)
    # make an axis
    f,ax = plt.subplots(nrows=2)
    # create a dot to move across the surface of the acoustic plot
    marker = ax[1].scatter(ac[0,0],ac[0,1],100,'r',zorder=2)
    # plot the signal data
    ax[1].plot(ac[:,0],ac[:,1],'b',zorder=1)
    ax[1].set(xlabel="Time (s)",ylabel="Acoustic Sensor Voltage (V)",title="Sensor Voltage (V)")
    ax[0].imshow(temp[0,:,:],cmap=plt.cm.hot,interpolation="nearest")
    ax[0].set_title("Thermal Image")

    axidx = plt.axes([0.25, 0.05, 0.65, 0.03])
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=ac.shape[0]-1,
        valinit=0,
        valstep=1,
        valfmt="%d",
    )

    def update(_) -> None:
        # get index from slider for AC time
        cid = int(idx_slider.val)
        marker.set_offsets([ac[cid,0],ac[cid,1]])
        # find closest matching frame
        ii = np.argmin(np.abs(flir_time-ac[cid,0]))
        ax[0].imshow(temp[ii,:,:],cmap=plt.cm.hot,interpolation="nearest")

    idx_slider.on_changed(update)

    plt.show()

def plotTemperatureEntropy(path:str|Path,**kwargs) -> plt.Figure:
    """Plots the entropy of the Temperature frames over the lifetime of the tool.

    Entropy is calculated from unique value passed to scipy.stats.entropy

    Inputs:
        path : Path to NPZ file
        title : Axes title. Default Signal Energy

    Returns generated figure object
    """  # noqa: D401
    from scipy.stats import entropy
    data = np.load(path)["arr_0"]
    ent = [entropy(np.unique(data[ii,:,:],return_counts=True)[1],base=None) for ii in range(data.shape[0])]
    f,ax = plt.subplots()
    ax.plot(ent,"b")
    ax.set(xlabel="Frame Number",ylabel="Entropy",title=kwargs.get("title","Temperature Frame Entropy"))
    return f

def grayscaleFrame(frame:np.ndarray) -> np.ndarray:
    """Converts temperature frame to a grayscale image."""  # noqa: D401
    # norm to 0-1 scale
    frame -= frame.min()
    frame /= frame.max()
    # multiply my 255
    frame *= 255
    return frame.astype("uint8")

def showFramesTDMS(path:str|Path,**kwargs) -> None:
    """Iterates over temperature data stored in a TDMS file and display.

    Inputs:
        path : Path to TDMS file
        img_shape : Shape of the temperature frames
    """  # noqa: D401
    # get image shape
    img_shape = kwargs.get("img_shape",(464,348))
    # open file
    with TdmsFile(path) as tdms_file:
        ## search for nonempty channel
        cfound = None
        gfound = None
        for gg in tdms_file.groups():
            for cc in gg.channels():
                if len(cc):
                    cfound = cc.name
                    break
            if cfound:
                gfound = gg.name
        # get number of chunks in the file
        nchunks = len(tdms_file[gfound][cfound])//(img_shape[0]*img_shape[1])
        # get first chunk
        chunk = tdms_file[gfound][cfound][:(img_shape[0]*img_shape[1])].reshape(img_shape)
    # create mesh grids for plotting
    x = np.arange(img_shape[1])
    y = np.arange(img_shape[0])
    X,Y = np.meshgrid(x,y)  # noqa: N806
    # create axes
    f,ax = plt.subplots()
    # plot first contourf
    ax.contourf(X,Y,chunk)
    # create axes for chunk index
    axidx = plt.axes([0.25, 0.1, 0.65, 0.03])
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=nchunks-1,
        valinit=0,
        valstep=1,
        valfmt="%d",
    )
    # create axes for number of contours
    axlevel = plt.axes([0.25, 0.15, 0.65, 0.03])
    level_slider = Slider(
        ax=axlevel,
        label="Number of Contour Levels",
        valmin=2,
        valmax=10,
        valinit=2,
        valstep=1,
        valfmt = "%d",
    )

    def update(_) -> None:
        # remove contours
        for coll in ax.collections:
            ax.collections.remove(coll)
        # get chunk id
        cid = int(idx_slider.val)
        # load next chunk
        with TdmsFile(path) as tdms_file:
            chunk = tdms_file[gfound][cfound][img_shape[0]*(cid*img_shape[1]):(img_shape[0]*((cid+1)*img_shape[1]))]
        # reshape into frame
        if min(chunk.shape)==0:
            return
        chunk = chunk.reshape(X.shape)
        # show
        ax.contourf(X,Y,chunk,levels=int(level_slider.val))

    idx_slider.on_changed(update)
    level_slider.on_changed(update)
    plt.show()

def showFramesCSV(path:str|Path,**kwargs) -> None:
    # get image shape
    img_shape = kwargs.get("img_shape",(464,348))
    # has header metadata
    has_head = kwargs.get("has_head",True)
    # get number of rows
    nrows = sum(1 for _ in open(path)) - 10 if has_head else 2  # noqa: SIM115
    # set chunk size
    csize = img_shape[1]
    # number of chunks
    nchunks = nrows // csize
    # read first chunk
    chunk = pd.read_csv(path,delimiter=",",skiprows=10 if has_head else 2, # if the the file has the parameter header, then skip it
                       header=None, # this is to stop it using the first row of values as the header names
                       usecols=list(range(1,img_shape[0]+1)), # skip first column
                       dtype="float64", # datatype
                       nrows=csize) # number of rows to read
    # create mesh grids for plotting
    x = np.arange(img_shape[0])
    y = np.arange(img_shape[1])
    X,Y = np.meshgrid(x,y)
    # create axes
    f,ax = plt.subplots()
    # plot first chunk
    ax.contourf(X,Y,chunk.values)
    # create axes for chunk index
    axidx = plt.axes([0.25, 0.1, 0.65, 0.03])
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=nchunks-1,
        valinit=0,
        valstep=1,
        valfmt="%d",
    )
    # create axes for number of contours
    axlevel = plt.axes([0.25, 0.15, 0.65, 0.03])
    level_slider = Slider(
        ax=axlevel,
        label="Number of Contour Levels",
        valmin=2,
        valmax=10,
        valinit=2,
        valstep=1,
        valfmt = "%d",
    )

    def update(_) -> None:
        for coll in ax.collections:
            ax.collections.remove(coll)
        # get target chunk
        cid = int(idx_slider.val)
        # get target chunk
        chunk = pd.read_csv(path,delimiter=",",skiprows=10+(cid*img_shape[1]) if has_head else 2+(cid*img_shape[1]), # if the the file has the parameter header, then skip it
                       header=None, # this is to stop it using the first row of values as the header names
                       usecols=list(range(1,img_shape[0]+1)), # skip first column
                       dtype="float64", # datatype
                       nrows=csize) # number of rows to read
        # plot first chunk
        ax.contourf(X,Y,chunk.values)

    idx_slider.on_changed(update)
    level_slider.on_changed(update)
    plt.show()

def getMaxTemp(path:str|Path,**kwargs) -> list|None:
    """Retrieves the max temperature of a TDMS or CSV temperature file.

    Reads the temperature data stored in either a TDMS or CSV file.
    Each call returns a list of max temperature value for each frame

    Inputs:
        path : Filepath to input file
        img_shape : Shape of each frame. Default (464,348)
        has_head : Flag indicating of the CSV has header information.

    Returns a list of temperature frames

    """  # noqa: D401
    # get file extension
    ext = os.path.splitext(path)[1]
    if ext in [".csv",".tdms"]:
        msg = f"Unsupported filetype! This function only supports ['.csv','.tdms']. Received {ext}"
        raise ValueError(msg)
    # get image shape
    img_shape = kwargs.get("img_shape",(464,348))
    # has header metadata
    has_head = kwargs.get("has_head",True)
    # if a CSV file
    if ext == ".csv":
        nrows = sum(1 for _ in open(path))  # noqa: SIM115
        nchunks = nrows // img_shape[1]
        nrows = nrows - 10 if has_head else 2
        def read_chunk() -> float:
            for _ in range(nchunks):
                chunk = pd.read_csv(path,delimiter=",",skiprows=10 if has_head else 2, # if the the file has the parameter header, then skip it
                           header=None, # this is to stop it using the first row of values as the header namesFVide
                           usecols=list(range(1,img_shape[0]+1)), # skip first column
                           dtype="float64", # datatype
                           nrows=img_shape[1]) # number of rows to read
                yield chunk.to_numpy().max(axis=[0,1])
        return list(read_chunk())
    # if a TDMS file
    if ext == ".tdms":
        # open the target file
        with TdmsFile(path) as tdms_file:
            # find non empty channel
            cfound = None
            gfound = None
            for gg in tdms_file.groups():
                for cc in gg.channels():
                    if len(cc):
                        cfound = cc.name
                        break
                if cfound:
                    gfound = gg.name
            nchunks = tdms_file[gg][cc].shape[0] // (img_shape[0]*img_shape[1])
            def read_chunk() -> float:
                for i in range(nchunks):
                    # read chunk
                    chunk = tdms_file[gfound][cfound][i*(img_shape[0]*img_shape[1]):(i+1)*(img_shape[0]*img_shape[1])]
                    # find max
                    yield chunk.max()
            return list(read_chunk())
    return None

def plotMaxTempNPZ(path:str|Path,**kwargs) -> plt.Figure:
    r"""Plots the per-frame max temperature.

    Inputs:
        path : Input file path to NPZ temperature file
        title : Axis title. Default {filename}\nPer-Frame Max Temperature

    Return generated figure object
    """  # noqa: D401
    data = np.load(path)["arr_0"]
    f,ax = plt.subplots()
    # calculate max temperature
    max_temp = data.max(axis=(1,2))
    ax.plot(max_temp)
    ax.set(xlabel="Frames",ylabel="Max Temperature (C)",title=kwargs.get("title",f"{os.path.splitext(os.path.basename(path))[0]}\nPer-Frame Max Temperature"))
    return f

def plotMaxTempAcousticSimple(path:str|Path,acpath:str|Path,**kwargs) -> plt.Figure:
    from dataparser import loadTDMSData
    # load data
    temp = np.load(path)["arr_0"]
    ac = loadTDMSData(acpath)["Input 0"]
    tclip = kwargs.get("tclip",-1)
    # clip to target time
    if tclip>0:
        ac = ac[ac[:,0]>tclip,:]
        temp = temp[int(30.0*tclip):,:,:]
    # create axes
    f,ax = plt.subplots(nrows=2,constrained_layout=True)
    # calculate max temperature
    max_temp = temp.max(axis=(1,2))
    ax[0].plot(max_temp)
    ax[0].set(xlabel="Frames",ylabel="Max Temperature (C)",title=kwargs.get("title",f"{os.path.splitext(os.path.basename(path))[0]} Per-Frame Max Temperature"))
    ax[1].plot(ac[:,0],ac[:,1])
    ax[1].set(xlabel="Time (s)",ylabel="Sensor Voltage (V)",title=kwargs.get("actitle",f"{os.path.splitext(os.path.basename(acpath))[0]} Sensor Voltage Input 0"))
    return f

def plotTempsMultipleE(path:str|Path="stripes_npz/*/*.npz",**kwargs) -> dict:
    """Plots the files spread across folders representing different emissivity values.

    This function iterates over a folder structure containing the same filenames.
    Each folder represents a different emissivity value (e.g. em001 -> em 0.01).
    A figure is produced for each unique filename and has lines for the per-frame
    max temperature for each directory. A dictionary is returned organised by
    unique filename and the generated figure.

    Inputs:
        path : Wildcard path to multiple directories representing different emissivities.
        title : Plot title. Defaults to filename.

    Returns a dictionary of unique filenames and corresponding figure
    """  # noqa: D401
    # create blank dictionary
    dirs = {}
    # iterate over all found files
    for temp in glob(path):
        # get filename
        dd = os.path.splitext(os.path.basename(temp))[0]
        # add paths to files with the same filename under the dictionary
        if dd not in dirs:
            dirs[dd] = [temp]
        else:
            dirs[dd].append(temp)
    figs = {}
    # iterate over dictionary
    for kk,vv in dirs.items():
        # for a given filename
        f,ax = plt.subplots()
        # plot max temperature and set label to the directory name
        for temp in vv:
            data = np.load(temp)["arr_0"]
            ax.plot(data.max(axis=(1,2)),label=os.path.dirname(temp).split("\\")[-1])
        ax.legend()
        ax.set(xlabel="Frames",ylabel="Temperature (C)",title=kwargs.get("title",os.path.splitext(os.path.basename(temp))[0]))
        figs[kk] = f
    return figs

def attemptLineUpTempAC(path:str|Path,acpath:str|Path,time:float=100.0,vclip:float=0.04,tclip:float=1000.0,temp_fps:float=30.0) -> np.ndarray:
    """Attempts to clip the acoustic and temperature files so they line up at the start.

    Files are clipped to after time seconds and location of the first peak is searched for.
    The min threshold of the peaks for temperature and AC is set using tclip and vclip respectively.

    The functions returns the files such that the first value is the first identified peaks in both files.
    Meant to line up with when the powder first hits the plate.
    Inputs:
        path : Input path for temperature NPZ file
        acpath : Path for accompanying AC file
        time : Time after which to search for peaks in both files
        vclip : Min height to search for peaks in AC
        tclip : Min height to search for peaks in temperature
        temp_fps : FPS of temperature file
    """  # noqa: D401
    # load files
    temp = np.load(path)["arr_0"]
    ac = loadTDMSData(acpath)["Input 0"]
    # clip to target time
    ac_clip = ac[ac[:,0]>time,:]
    temp_clip = temp[int(temp_fps*time):,:,:]
    # searcb for peaks in AC file
    # choose first peak
    pk_ac = find_peaks(ac_clip[:,1],height=vclip)[0]
    if len(pk_ac)==0:
        return None,None
    pk_ac = pk_ac[0]
    # search for peaks in temperature file
    # choose first peak
    pk_temp = find_peaks(temp_clip.max(axis=(1,2)),height=tclip)[0]
    if len(pk_temp)==0:
        return None,None
    # clip both files to when first peak occurs
    return ac[int(time*1e6)+pk_ac:,:],temp[int(time*temp_fps):,:]

def plotLineUpTempAC(path:str|Path,acpath:str|Path,time:float=100.0,vclip:float=0.04,tclip:float=1000.0,temp_fps:float=30.0,**kwargs) -> plt.Figure:
    # process the files so they line up
    ac,temp = attemptLineUpTempAC(path,acpath,time,vclip,tclip,temp_fps)
    if (ac is None) or (temp is None):
        msg = f"Failed to line up files {path} and {acpath}!"
        raise ValueError(msg)
    # create axes
    f,ax = plt.subplots(nrows=2,constrained_layout=True)
    # get max temperature
    temp_max = temp.max(axis=(1,2))
    # construct time vector
    temp_time = np.arange(0.0,temp_max.shape[0]*(1/temp_fps),1/temp_fps)
    # sometimes the time vector is +/- 1 the size of the temperature vector
    # this ensures they are the correct size
    temp_max = temp_max[:min(temp_max.shape[0],temp_time.shape[0])]
    # plot the temperature
    ax[0].plot(temp_time,temp_max)
    ax[0].set(xlabel="Time (s)",ylabel="Max Temperature (C)",title=kwargs.get("title",f"{os.path.splitext(os.path.basename(path))[0]} Per-Frame Max Temperature"))
    # plot the acoustic voltage
    ax[1].plot(ac[:,0],ac[:,1])
    ax[1].set(xlabel="Time (s)",ylabel="Sensor Voltage (V)",title=kwargs.get("actitle",f"{os.path.splitext(os.path.basename(acpath))[0]} Sensor Voltage Input 0"))
    return f

def plotPSD(data:np.ndarray,**kwargs) -> plt.Figure:
    """Plots the PSD of given data. Intended for acoustic data.

    The data can be clipped to a certain period using keyterms.
    tclip clips to after a certain timestamp. This can be used to skip
    the early part of the signal wehen the machine is being setup.
    peak is used to skip to when a peak occurs. By default it skips
    to when the max peak occurs. If first_peak flag is True, then it
    skips to the first peak.

    e.g. plotPSD(data,tclip=100,peak=0.02,first_peak=True)

    Inputs:
        data : 2D Numpy array of time and acoustic signals preferable from loadTDMSData.
        tclip : Timestamp to clip to.
        peak : Threshold above which to search for peaks for. If True, then all peaks are found.
                If a float, then it is passed to the height parameter in find_peaks.
        first_peak : Flag to use the first peak instead of the max. Default True.
        title : Axis title. Default PSD.

    Returns the generated figure
    """  # noqa: D401
    if isinstance(data,str):
        from dataparser import loadTDMSData
        data = loadTDMSData(data)["Input 0"]
    ## filter samples according to set parameters
    if "tclip" in kwargs:
        tclip = kwargs.get("tclip")
        if tclip <0:
            msg = f"Time threshold cannot be less than 0! Received {tclip}!"
            raise ValueError(msg)
        # clip according to time
        data = data[data[:,0]>tclip,:]
    # clip according to peak
    if "peak" in kwargs:
        peak = kwargs.get("peak")
        # search for peaks
        # set threshold if peak is a float value
        pp,_ = find_peaks(data[:,1],height=peak if isinstance(peak,float) else None)
        # if using the first peak
        if kwargs.get("first_peak",True):  # noqa: SIM108
            pp = pp[0]
        # elif using highest peak
        else:
            # find max peak
            pp = data[pp,1].argmax()
        # clip file to from peak onwards
        data = data[pp:,:]
    f,ax = plt.subplots(constrained_layout=True)
    ax.set_yscale(kwargs.get("scale","linear"))
    ec = ax.specgram(data[:,1],Fs=1e6)[-1]
    cbar = f.colorbar(ec)
    cbar.set_label("Magnitude (dB)")
    ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=kwargs.get("title","PSD"))
    return f

def plotAllFilesFFT(path:str|Path,**kwargs) -> plt.Figure:
    """Iterates over all found TDMS files and plot the FFT of the data.

    RFFT is used to minimize the amount of data required and impact on the machine

    Specify which key input channel is to be used

    Each row of axes is for each file

    Inputs:
        path : Wildcard input path
        key : Which key (Input 0 or Input 1) to use

    Returns figure object
    """  # noqa: D401
    from dataparser import loadTDMSData
    from scipy.fft import rfft, rfftfreq
    search = glob(path)
    f,ax = plt.subplots(nrows=len(search),ncols=2,sharex=True,constrained_layout=True)
    for fi,fn in enumerate(search):
        fname = os.path.splitext(os.path.basename(fn))[0]
        ac = loadTDMSData(fn)[kwargs.get("key","Input 0")]
        sh = ac.shape[0]
        yf = rfft(ac[:,1])
        xf = rfftfreq(sh)
        del ac
        ax[fi,0].plot(xf,np.abs(yf))
        ax[fi,0].set(xlabel="Freq (Hz)",ylabel="Magnitude",title=f"{fname} FFT Magnitude")
        ax[fi,1].plot(xf,np.angle(yf),"r")
        ax[fi,1].set(xlabel="Freq (Hz)",ylabel="Phase (radians)",title=f"{fname} FFT Phase")
    for aa in ax.flatten():
        aa.set_yscale("log")
    return f

def plotHistogramVideo(path:str|Path,of:str | Path | None=None,nbins:int=20,esource:str="global",**kwargs):
    """Creates video of temperature histogram and plot it as a video.

    Histogram edges are calculated globally for the file and then for each
    frame update the height of the bars.

    Creates a matplotlib animation plotting the histogram for each frame

    Inputs:
        path : Filepath to NPZ file or numpy array of temperature data
        of : Output file path. If data is specified by a filepath and of is None, then it's
            made from that file path. Default None.
        nbins : Number of bins to use in the histogram
        esource : Source of edges for the histogram. If local then it's calculated for each frame.
                If global then it's calculated from the entire file. Default global.
        edgecol : Edge color used in the histogram plot
        facecol : Face color used in the hisogram plot
        alpha : Transparency value used for the histogram
        rclip : Clip row range in image used to generate plot. Default -1.
        cclip : Clip column range in image used to generate plot. Default -1.
        tclip : Mask temperatures to above the specified temperature. Helps mask the background noise.
    """  # noqa: D401
    # check if edge source is valid
    if esource not in ["local","global"]:
        msg = f"Invalid histogram edge source! Received {esource}"
        raise ValueError(msg)
    # load data file
    data = np.load(path)["arr_0"] if isinstance(path,[str,Path]) else path
    nf,r,c = data.shape

    # if not clipping to a specific column range
    cclip = kwargs.get("cclip",-1)
    # check if the upper column is outside the range of the frame
    if (cclip>c):
        msg = f"Column clipping number is larger than the size of the frame! Received {cclip}"
        raise ValueError(msg)
    # if the column range is negative and not -1
    if (cclip != -1) and (cclip < -1):
        msg = f"Column clipping number cannot be a negative number other than -1! Received {cclip}"
        raise ValueError(msg)

    # if not clipping to a specific column range
    rclip = kwargs.get("rclip",-1)
    # check if the upper column is outside the range of the frame
    if (rclip>r):
        msg = f"Row clipping number is larger than the size of the frame! Received {rclip}"
        raise ValueError(msg)
    # if the column range is negative and not -1
    if (rclip != -1) and (cclip < -1):
        msg = f"Row clipping number cannot be a negative number other than -1! Received {rclip}"
        raise ValueError(msg)

    # check temperature limiting
    # min temperature to clip above
    tclip = kwargs.get('tclip',0)
    if (tclip < 0):
        msg = f"Clipping Temperature cannot be less than 0! Received {tclip}"
        raise ValueError(msg)

    # load data
    if isinstance(path,[str,Path]) and (of is None):
        of = os.path.splitext(os.path.basename(path))[0]+"_temp_hist"
    elif (of is None):
        raise ValueError("Output filename not specified!")
    # append column clip
    if cclip != -1:
        of += f"_cclip_{cclip}"
    # append row clip
    if rclip != -1:
        of += f"_rclip_{rclip}"
    # append temp clipping
    if tclip >0:
        of += f"_tclip_{str(tclip).replace('.','-')}"
    # append local
    if esource == "local":
        of += "esource-local"
    # append file extension
    of += ".avi"
    # init histogram bins to None
    # if esource is local it gets updated each loop
    HIST_BINS = None  # noqa: N806
    # if edge source is global
    if esource == "global":
        # clip to target area
        frame = data[:,:rclip,:cclip].flatten()

        # find min and max temperature
        tmin = frame[frame>tclip].min()
        tmax = frame[frame>tclip].max()
        # if the source of the edges is global
        HIST_BINS = np.linspace(tmin,tmax,nbins)  # noqa: N806
    # create axes
    fig,ax = plt.subplots()
    ax.set(xlabel="Frame Temperature (C)",ylabel="Population",title=kwargs.get("title","Temperature Histogram"))
    # set first frame
    _,_,bar_container = ax.hist(data[0,:,:].flatten(),HIST_BINS,lw=1,ec=kwargs.get("edgecol","yellow"),fc=kwargs.get("facecol","green"),alpha=kwargs.get("alpha",0.5))

    # function for generating each frame
    def prepare_animation(bar_container:BarContainer|list=bar_container) -> callable:
        def animate(frame_number:int,HIST_BINS:int=HIST_BINS,bar_container:BarContainer|list=bar_container) -> None:  # noqa: N803
            # get next frame
            # clip to target area and flatten
            frame = data[frame_number,:rclip,:cclip].flatten()
            # clip to temperature range
            frame = frame[frame>tclip]
            # if there are no pixels after filtering
            # set all heights to 0
            if frame.shape[0] == 0:
                for rect in bar_container.patches:
                    rect.set_height(0)
            # if the source of the edges is local
            if esource == "local":
                # find temperature limits for frame
                tmin = frame.min()
                tmax = frame.max()
                # calculate histogram bins
                HIST_BINS = np.linspace(tmin,tmax,nbins)  # noqa: N806
                n,_ = np.histogram(frame,HIST_BINS)
                # get width
                w = HIST_BINS[1]-HIST_BINS[0]
                # reposition and resize histogram patches
                for rect,bb,count in zip(bar_container.patches,HIST_BINS,n):
                    rect.set_xy((bb,0))
                    rect.set_height(count)
                    rect.set_width(w)
                # adjust axis limits
                ax.set_xlim(HIST_BINS[0],HIST_BINS[-1]+rect.get_width())
                ax.set_ylim(0,n.max()+10)
            # if the source of the edges is global
            if esource == "global":
                n,_ = np.histogram(frame,HIST_BINS)
                # update the height of each bar
                for count,rect in zip(n,bar_container.patches):
                    rect.set_height(count)
                # adjust y axis limits
                ylim = ax.get_ylim()[1]
                ax.set_ylim(0,max(n.max()+10,ylim))
                # return the patches
            return bar_container.patches
        return animate
    # make animation
    ani = animation.FuncAnimation(fig,prepare_animation(bar_container),nf,repeat=False,blit=True)
    # save animation
    writervideo = animation.FFMpegWriter(fps=15)
    ani.save(of,writer=writervideo)

def plotMaxTempAcoustic(temp:str|Path,ac:str|Path,**kwargs):
    """Plots the Max Temp vs Time vs Acoustic Voltage.

    The acoustic data is downsampled to the same sample rate and array shape

    Works best when the files approximately line up

    Inputs:
        temp : Filepath to NPZ temperature file or an already loaded numpy array.
        ac : Filepath to TDMS acoustic sensor file or an already loaded numpy array.
        mode : Interpolation mode supported by scipy.interpolate. Default linear.
        fps : Frame rate of the temperature data. Default 15
        title : Plot title. Default Max Temperature vs Acoustic Voltage

    Returns the generated figure object handle.
    """  # noqa: D401
    # get FPS
    fps = kwargs.get("fps",15)
    Tfps = 1/fps  # noqa: N806
    # load data
    if isinstance(temp,str):
        temp = np.load(temp)["arr_0"]
    if isinstance(ac,str):
        ac = loadTDMSData(ac,True)["Input 0"]  # noqa: FBT003
    # get shape of the temperature data
    tnf,tr,tc = temp.shape

    # get per frame max temperature
    max_temp = temp.max(axis=(1,2))

    # create time vector for temperature
    temp_time = np.arange(0.0,tnf*Tfps,Tfps)
    # clip to match same size at max temp
    temp_time = temp_time[:min((tnf,temp_time.shape[0]))]

    # get last time stamp
    ac_time = ac[-1,0]
    # acoustic recording was started after FLIR
    # moving starting point of temperature forwards so it better lines up
    td = abs(ac_time-temp_time.max())
    itd = int(td/fps)
    max_temp = max_temp[itd:]
    temp_time = temp_time[itd:]

    # create axes
    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    ac = ac[:,1]
    # downsample the acoustic data from 1MHz to fps
    ac = ac[::int(np.floor(1e6/fps))]

    # get the min size shared by all
    minsz = min([max_temp.shape[0],ac.shape[0]])
    ax.scatter(temp_time[:minsz],max_temp[:minsz],ac[:minsz])
    ax.set(xlabel="Time (s)",ylabel="Max Frame Temperature (C)",zlabel="Acoustic Sensor Voltage (V)")
    f.suptitle(kwargs.get("title","Max Temperature vs Acoustic Voltage"))
    return f

def plotMaxTempAcousticInterp(temp:str|Path,ac:str|Path,mode:str="linear",**kwargs) -> plt.Figure:
    """Plots the Max Temp vs Time vs Acoustic Voltage.

    An interpolation model is fitted to the max temperature data with generated timestamps.
    The model is then used to interpolate at the accompanying TDMS timestamps.

    This can take a while as the max temperature values tend to be 1000s of values whilst the TDMS
    files are 1E6s of values.

    Inputs:
        temp : Filepath to NPZ temperature file or an already loaded numpy array.
        ac : Filepath to TDMS acoustic sensor file or an already loaded numpy array.
        mode : Interpolation mode supported by scipy.interpolate. Default linear.
        fps : Frame rate of the temperature data. Default 15
        title : Plot title. Default Max Temperature vs Acoustic Voltage

    Returns the generated figure object handle.
    """  # noqa: D401
    from scipy.interpolate import interp1d
    # get FPS
    fps = kwargs.get("fps",15)
    Tfps = 1/fps  # noqa: N806
    # load data
    if isinstance(temp,str):
        temp = np.load(temp)["arr_0"]
    if isinstance(ac,str):
        ac = loadTDMSData(ac,True)["Input 0"]
    # get shape of the temperature data
    tnf,tr,tc = temp.shape

    # get max temperature
    temp_max = temp.max(axis=(1,2))
    # create time vector
    temp_time = np.arange(0.0,tnf*Tfps,Tfps)

    # fit model to the max temperature data
    interp = interp1d(temp_time,temp_max,kind=mode,fill_value="extrapolate")
    # interpolate max temperature using timestamps of AC data
    temp_max_interp = interp(ac[:,0].squeeze())
    # create axes
    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    ax.scatter(ac[:,0],temp_max_interp,ac[:,0])
    ax.set(xlabel="Time (s)",ylabel="Max Frame Temperature (C)",zlabel="Acoustic Sensor Voltage (V)")
    f.suptitle(kwargs.get("title","Max Temperature vs Acoustic Voltage"))
    return f

def plotMaxTempAcousticSplev(path:str|Path,acpath:str|Path,**kwargs) -> plt.Figure:
    """Plots the Max Temp vs Time vs Acoustic Voltage.

    An spline model is fitted to the max temperature data with generated timestamps.
    The model is then used to interpolate at the accompanying TDMS timestamps.

    This can take a while as the max temperature values tend to be 1000s of values whilst the TDMS
    files are 1E6s of values.

    Inputs:
        temp : Filepath to NPZ temperature file or an already loaded numpy array.
        ac : Filepath to TDMS acoustic sensor file or an already loaded numpy array.
        mode : Interpolation mode supported by scipy.interpolate. Default linear.
        fps : Frame rate of the temperature data. Default 15
        title : Plot title. Default Max Temperature vs Acoustic Voltage

    Returns the generated figure object handle.
    """  # noqa: D401
    # get FPS
    fps = kwargs.get("fps",15)
    Tfps = 1/fps  # noqa: N806
    # load data
    temp = np.load(path)["arr_0"] if isinstance(path,[str,Path]) else path
    # load AE data
    ac = loadTDMSData(acpath,True)["Input 0"] if isinstance(acpath,str) else acpath  # noqa: FBT003
    # get shape of the temperature data
    tnf,tr,tc = temp.shape
    # get max temperature
    temp_max = temp.max(axis=(1,2))
    # create time vector for max temperature
    temp_time = np.arange(0.0,tnf*Tfps,Tfps)
    # sometimes the time vector is 1 element larger than the temperature
    if (temp_max.shape[0] != temp_time.shape[0]):
        msz = min((temp_max.shape[0],temp_time.shape[0]))
        temp_max = temp_max[:msz]
        temp_time = temp_time[:msz]
    # correct for time
    clip_time = kwargs.get("clip_time",True)
    # if just True, then try and adjust based on difference in length
    # assumes both were stopped at roughly the same time
    if isinstance(clip_time,bool) and clip_time:
        # assuming temperature was started first
        # which it was for the stripe runs
        diff = temp_time[-1] - ac[-1,0]
        # convert time difference to frames in the same frame rate as temperature
        diff_frames = int(fps*diff)
        # cut that number of frames
        temp_max = temp_max[diff_frames:]
        temp_time = temp_time[diff_frames:]
        # reset time to 0
        temp_time -= temp_time.min()
    # clip both to a specific time
    elif isinstance(clip_time,float):
        # clip temperature
        tframe = int(fps*clip_time)
        temp_max = temp_max[tframe:]
        temp_time = temp_time[tframe:]
        temp_time -= temp_time.min()
        # clip acoustic sensor
        tframe = int(1e6*clip_time)
        ac = ac[:tframe,:]
        ac[:,0] -= ac[:,0].min()

    # fit model to the max temperature data
    rep = splrep(temp_time,temp_max,k=kwargs.get("k",3),full_output=True)
    # save the spline representation as a text file or other if specified
    # can then be reloaded in to save refitting each time
    if "save_rep" in kwargs:
        srep = kwargs.get("save_rep")
        ## convert rep to JSON
        srep_json = {}
        # iterate over the values returns by splrep and associating keys
        for k,v in zip(["knots","weighted_sum_residuals","success","msg"],rep):
            # if it's the knots then it's a tuple of two arrays and an integer
            if k == "knots":
                srep_json[k] = [[float(knot) for knot in v[0]],
                                [float(knot) for knot in v[1]],
                                v[2]]
            # remaining keys are a float, integer and a str, both don't need converting
            else:
                srep_json[k] = v
        # if save_rep is True then try and build path from filename
        if isinstance(srep,bool) and srep:
            if isinstance(path,[str,Path]):
                with open(f"{os.path.splitext(os.path.basename(path))[0]}-srep.json","w") as dump:
                    json.dump(srep_json,dump)
            else:
                warnings.warn("Cannot save spline representation to file! Temperature wasn't a file path so output path cannot be inferred!", stacklevel=2)
        elif isinstance(srep,str):
            with open(srep,"w") as dump:
                json.dump(srep_json,dump)
    # interpolate max temperature using timestamps of AC data
    temp_max_interp = splev(ac[:,0].squeeze(),rep[0])
    # create axes
    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    ax.scatter(ac[:,0],temp_max_interp,ac[:,0])
    ax.set(xlabel="Time (s)",ylabel="Max Frame Temperature (C)",zlabel="Acoustic Sensor Voltage (V)")
    f.suptitle(kwargs.get("title","Max Temperature vs Acoustic Voltage"))
    # pickle the file as it takes quite a while to generate so storing and inspecting afterwards
    # would be better in the long term
    if "pickle_fig" in kwargs:
        import pickle
        pfig = kwargs.get("pickle_fig")
        # if pickle_fig is True
        if isinstance(pfig,bool) and pfig:
            # if the temperature was given as a path
            # construct path from the input values
            if isinstance(path,[str,Path]):
                with open(f"{os.path.splitext(os.path.basename(path))[0]}-acoustic-temp-spline.pickle",'wb')  as dump:
                    pickle.dump(f,dump)
            else:
                warnings.warn("Cannot pickle figure as path could not be inferred from input!", stacklevel=2)
        # if pickle_fig is a string treat it as a path
        elif isinstance(pfig,str):
            with open(pfig,"wb") as dump:
                pickle.dump(f,dump)
    # return the figure object
    return f

def plotFFT(path:str|Path,cache_res:bool=True,no_plot:bool=False,**kwargs) -> plt.Figure|None:
    """Plots the FFT of the given acoustic emission data.

    Plots the fft using RFFT and RFFTFREQ functions. The user
    specifies which channel of dta they want plotting using key keyword.
    e.g. Input 0 or Input 1.

    Inputs:
        path : File path to TDMS file or numpy array
        key : Channel of data to process. Can either be Input 0 or Input 1. Default Input 0.

    Returns matplotlib figure object
    """  # noqa: D401
    key = kwargs.get("key","Input 0")
    if isinstance(path,[str,Path]):
        ac = loadTDMSData(path,inc_time=False)[key]
    print(ac.shape)
    yf = rfft(ac)
    print(yf.shape,yf.dtype)
    xf = rfftfreq(ac.shape[0],d=1./1e6)
    print(xf.shape,xf.dtype,xf.min(),xf.max())

    if cache_res:
        print("caching results")
        if isinstance(cache_res,bool):
            of = f"{os.path.splitext(os.path.basename(path))[0]}-fft-cache-{key}.bin"
        # save to specific filename
        elif isinstance(cache_res,str):
            of = cache_res
        yf.tofile(of)
    if no_plot:
        return None
    f,ax = plt.subplots(nrows=2,sharex=True)
    ax[0].plot(xf,np.abs(yf))
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Amplitude",title="Amplitude")
    ax[1].plot(xf,np.angle(yf))
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase (rads)")
    f.suptitle(kwargs.get("title",f"FFT of AC Signal, {key}"))
    return f

def plotAllFFT(path:str|Path,**kwargs) -> plt.Figure:
    """Plots the FFT for each acoustic file found on the same axis.

    Y-axis is set to log scale.

    Plots the fft using RFFT and RFFTFREQ functions. The user
    specifies which channel of dta they want plotting using key keyword.
    e.g. Input 0 or Input 1.

    Inputs:
        path : File path to TDMS file or numpy array
        key : Channel of data to process. Can either be Input 0 or Input 1. Default Input 0.

    Returns matplotlib figure object
    """  # noqa: D401
    if isinstance(path,[str,Path]):
        path = glob(path)
    period = 1./1e6
    # make the figure
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    # iterate over paths
    for fn in path:
        # load the paths
        if isinstance(fn,[str,Path]):
            path = loadTDMSData(fn)[kwargs.get("key","Input 0")]
        # get the rfft and plot it
        yf = rfft(path[:,1])
        xf = rfftfreq(path.shape[0],d=period)
        ax[0].plot(xf,np.abs(yf),label=os.path.splitext(os.path.basename(fn))[0])
        ax[1].plot(xf,np.angle(yf),label=os.path.splitext(os.path.basename(fn))[0])

    # set params and labels
    plt.rcParams["agg.path.chunksize"] = 150
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Amplitude",title="Amplitude")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase (rads)")
    f.suptitle(kwargs.get("title",f"FFT of AC Signal, {kwargs.get('key','Input 0')}"))
    ax[0].legend()
    ax[1].legend()
    return f

def plotMaxTemperatureMask(path:str|Path|np.ndarray,mask:str|Path|np.ndarray,**kwargs) -> plt.Figure:
    # load data
    data = np.load(path)["arr_0"] if isinstance(path,[str,Path]) else path
    # load mask
    mask_img = cv2.imread(mask,cv2.IMREAD_GRAYSCALE) if isinstance(mask,str) else mask
    # calculate max temperature within masked area for each frame
    max_temp = [data[ni,mask_img==255].max() for ni in range(data.shape[0])]
    # plot the masked max temperature
    f,ax = plt.subplots()
    ax.plot(max_temp)
    ax.set(xlabel="Frame",ylabel="Max Temperature (C)",title=kwargs.get("title","Max Temperature Mask"))
    return f

def videoMaskTemperature(path:np.ndarray|str|Path,mask:np.ndarray|str|Path,opath:str|Path,cmap:int=cv2.COLORMAP_HOT) -> None:
    data = np.load(path)["arr_0"] if isinstance(path,[str,Path]) else path
    nf,rows,width = data.shape
    # load mask
    if isinstance(mask,str):
        mask_img = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask,np.ndarray):
        mask_img = mask.copy()

    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),int(cmap != "gray"))
    if not out.isOpened():
        msg = f"Failed to open video file at {opath}!"
        raise ValueError(msg)

    for ni in range(nf):
        frame = data[ni,:,:]
        frame[mask != 255] = 0
        frame_norm = (255*((frame-frame.min())/abs(frame.max()-frame.min()))).astype("uint8")
        # apply mask
        frame_norm = cv2.bitwise_and(frame_norm, frame_norm, mask=mask_img)
        if cmap == "gray":
            out.write(frame_norm)
        else:
            out.write(cv2.applyColorMap(frame_norm,cmap))
    out.release()

def convertFFTCacheCSVToParquet(pathA:str,pathB:str,outpath:str|None=None) -> None:  # noqa: N803
    fft_cache_df = pd.DataFrame()

    with open(pathA) as file:
        print("reading path A")
        fft_cache_df["Magnitude 0"] = [float(f) for f in file.readline().split(",")]
        fft_cache_df["Phase 0"] = [float(f) for f in file.readline().split(",")]
        fft_cache_df["Freq"] = [float(f) for f in file.readline().split(",")]

    with open(pathB) as file:
        print("reading path B")
        fft_cache_df["Magnitude 1"] = [float(f) for f in file.readline().split(",")]
        fft_cache_df["Phase 1"] = [float(f) for f in file.readline().split(",")]

    if outpath is None:
        outpath = os.path.splitext(os.path.basename(pathA))[0].split("-Input")[0]
        fft_cache_df.to_parquet(outpath+".parquet")
    else:
        fft_cache_df.to_parquet(outpath)

def replotFFTCacheAll(*args:list,**kwargs) -> plt.Figure:
    import vaex
    mpl.use("Agg")
    # memory map the binary data
    f,ax = plt.subplots(nrows=2,sharex=True)
    labels = kwargs.get("label",[os.path.splitext(os.path.basename(fn))[0] for fn in args])
    for path,label in zip(args,labels):
        data = np.memmap(path,"complex128","r")
        # create frequency data
        freq = rfftfreq(2*(data.shape[0]-1),d=1./1e6)
        # convert to vaex data array containing the 
        arr = vaex.from_arrays(mag=np.abs(data),phase=np.angle(data),freq=freq)
        del freq
        del data
        # plot mag
        ax[0].plot(arr.freq,arr.mag,label=label)
        ax[1].plot(arr.freq,arr.phase,label=label)
    for aa in ax.flatten():
        aa.legend()
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="Magnitude")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (radians)",title="Phase (radians")
    f.suptitle(kwargs.get("title","FFT Compare"))
    return f

def replotFFTCacheSeparate(*args:list) -> None:
    import vaex
    mpl.use("Agg")
    # memory map the binary data
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="Magnitude")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (radians)",title="Phase (radians")

    line_mag = None
    line_phase = None
    max_shape = 0
    period = 1./1e6
    for p,path in enumerate(args):
        data = np.memmap(path,"complex128","r")
        # create frequency data
        freq = rfftfreq(2*(data.shape[0]-1),d=period)
        # convert to vaex data array containing the 
        arr = vaex.from_arrays(mag=np.abs(data),phase=np.angle(data),freq=freq)
        print(arr.shape)

        # create initial line objects
        if p==0 or (max_shape != arr.shape[0]):
            line_mag=ax[0].plot(arr.freq.to_numpy().flatten(),arr.mag.to_numpy().flatten())[0]
            line_phase=ax[1].plot(arr.freq.to_numpy().flatten(),arr.phase.to_numpy().flatten())[0]
        elif (arr.shape[0] != max_shape):
            line_mag.set_data(arr.freq.to_numpy().flatten(),arr.mag.to_numpy().flatten())
            line_phase.set_data(arr.phase.to_numpy().flatten())
        else:
            line_mag.set_ydata(arr.mag.to_numpy().flatten())
            line_phase.set_ydata(arr.phase.to_numpy().flatten())
        max_shape = max(max_shape,arr.shape[0])
        f.suptitle(os.path.splitext(os.path.basename(path))[0])
        f.savefig(os.path.splitext(os.path.basename(path))[0]+".png")

def findFFTPeaks(path:str|Path,freq_gap:float=1e5,return_mag:bool=False) -> tuple|None:
    if isinstance(path,[str,Path]):
        data = np.memmap(path,"complex128","r")
        # get abs data
        mag = np.abs(data)
        max_len = mag.shape[0]
    else:
        mag=path
        max_len = mag.shape[0]
    # create frequency data
    freq = rfftfreq(2*(max_len-1),d=1./1e6)
    # find how many samples should be between peaks
    gap = freq[freq<=freq_gap].shape[0]
    # find peaks with a gap of 1e5 Hz between them
    pks = find_peaks(mag,distance=gap,height=500)[0]
    return freq[pks],mag[pks] if return_mag else None

def maskToPeaks(path:str|Path,width:int=10000,**kwargs) -> None:
    """Finds & masks to the target peaks in the given FFT and reconstruct the signal to see the impact."""  # noqa: D401
    plt.rcParams["agg.path.chunksize"] = 10000
    # get filename
    fname = os.path.splitext(os.path.basename(path))[0]
    T = 1./1e6
    # open the file using a memory map
    data = np.memmap(path,"complex128","r")
    # create frequency data
    max_len = 2*(data.shape[0]-1)
    # re-create frequency vector
    freq = rfftfreq(max_len,d=T)
    freq_max = freq.max()
    # recreate time vector
    time = np.arange(0,max_len*T,T)
    # find the magnitude of the signal
    mag = np.abs(data)
    # find the maximum
    mag_max = mag.max()
    # find peaks
    freq_peaks,_ = findFFTPeaks(path,return_mag=False,**kwargs)
    # create a copy to update
    data_inv = np.zeros(data.shape[0],data.dtype)
    # icreate axes
    f,ax = plt.subplots(ncols=2,constrained_layout=True)
    # force y limits
    ax[1].set_ylim(0,mag_max)
    # plot the fft
    ax[1].plot(freq,mag,"b")
    # objects for holding polygon span
    freq_span = None
    # object for holding the reconstructed data line
    recon_data = None
    # two line representing the range of frequency
    for ff in freq_peaks:
        # find freq limits clipping where appropriate
        fmin = max(0,ff-(width/2))
        fmax = min(freq_max,ff+(width/2))
        # create mask for target frequency
        mask = (freq >= fmin)&(freq<= fmax)
        # on the fft plot fill an area
        if freq_span is None:
            freq_span = ax[1].axvspan(fmin,fmax,0,mag_max,color="red",alpha=0.2)
        else:
            # get current vertices of the span
            _ndarray = freq_span.get_xy()
            # update x coordinates
            _ndarray[:,0] = [fmin,fmin,fmax,fmax,fmin]
            freq_span.set_xy(_ndarray)
        # update data with the values we want
        data_inv[mask] = data[mask]
        # reconstruct data
        data_recon = irfft(data_inv)
        # plot reconstructed data on the right
        if recon_data is None:
            recon_data = ax[0].plot(time,data_recon)[0]
        else:
            recon_data.set_ydata(data_recon)
            ax[0].set_ylim(data_recon.min(),data_recon.max())
        # setup labels
        ax[0].set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"Reconstructed Data f=[{fmin:.2e},{fmax:.2e}]")
        ax[1].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="FFT")
        f.suptitle(fname)
        # if the user wants to save each one now rather than display them
        f.savefig(f"{fname}-fft-peak-mask-width-{int(width)}-{fmin:.2e}-{fmax:.2e}.png")
    plt.close(f)

def maskToPeaksSTFT(path:str|Path,bin_file:str|None=None,width:int=10000,**kwargs):
    """Finds & masks to the target peaks in the given FFT and reconstruct the signal to see the impact."""  # noqa: D401
    # load original data data
    ac = loadTDMSData(path,True)[kwargs.get("key","Input 0")]  # noqa: FBT003
    # get shape of the data
    # used to set length of vectors
    max_len = ac.shape[0]
    # recreate time vector
    time = ac[:,0]
    # increase plotting res
    plt.rcParams["agg.path.chunksize"] = 10000
    # get filename
    fname = os.path.splitext(os.path.basename(path))[0]
    # set time period
    period = 1./1e6
    # perform STFT
    fst,tst,Zxx = stft(ac[:,1],fs=1e6,nperseg=int(1e6))  # noqa: N806
    # get magnitude of STFT
    mag_st = np.abs(Zxx)
    # if the binary file isn't specified
    if bin_file is None:
        # make frequency vector
        freq = rfftfreq(max_len,d=period)
        # calculate RFFT
        data_bin = rfft(ac[:,1])
    elif isinstance(bin_file,str):
        # load binary data
        data_bin = np.memmap(bin_file,"complex128","r")
        # re-create frequency vector
        freq = rfftfreq(max_len,d=period)
    # get max frequency
    freq_max = freq.max()
    # find the magnitude of the signal
    mag = np.abs(data_bin)
    # find the maximum
    mag_max = mag.max()
    # find peaks
    freq_peaks,_ = findFFTPeaks(mag,return_mag=False,**kwargs)
    # create a copy to update
    data_inv = np.zeros(data_bin.shape[0],dtype="complex128")
    # create axes
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    # force y limits
    ax[1].set_ylim(0,mag_max)
    # plot the fft
    ax[1].plot(freq,mag,"b")
    # objects for holding polygon span
    freq_span = None
    # reconstructed data fike
    recon_data = None
    # update colormesh
    stmesh = None
    print(freq.shape,time.shape,mag_st.shape)
    # two line representing the range of frequency
    for ff in freq_peaks:
        # find freq limits clipping where appropriate
        fmin = max(0,ff-(width/2))
        fmax = min(freq_max,ff+(width/2))
        # create mask for target frequency
        mask = (freq >= fmin)&(freq<= fmax)
        # on the fft plot fill an area
        if freq_span is None:
            freq_span = ax[1].axvspan(fmin,fmax,0,mag_max,color="red",alpha=0.2)
        else:
            # get current vertices of the span
            _ndarray = freq_span.get_xy()
            # update x coordinates
            _ndarray[:,0] = [fmin,fmin,fmax,fmax,fmin]
            freq_span.set_xy(_ndarray)
        # update data with the values we want
        data_inv[mask] = data_bin[mask]
        # reconstruct data
        data_recon = irfft(data_inv)
        # plot reconstructed data on the right
        if recon_data is None:
            recon_data = ax[0].plot(time,data_recon)[0]
        else:
            recon_data.set_ydata(data_recon)
            ax[0].set_ylim(data_recon.min(),data_recon.max())
        # plot STFT for target region
        mask_st = (fst >= fmin)&(fst<= fmax)
        fst_mask = fst[mask_st]
        # remove mesh
        if stmesh is not None:
            stmesh.remove()
        stmesh = ax[2].pcolormesh(tst,fst_mask,mag_st[mask_st,:],cmap=kwargs.get("cmap","hot"),shading=kwargs.get("shading","auto"),norm=LogNorm())

        ax[2].set_ylim(fst_mask.min(),fst_mask.max())
        # setup labels
        ax[0].set(xlabel="Time (s)",ylabel="Voltage (V)",title="Reconstructed Data")
        ax[1].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="FFT")
        ax[2].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT")
        f.suptitle(f"{fname}, f=[{fmin:.2e}, {fmax:.2e}]")
        # if the user wants to save each one now rather than display them
        f.savefig(f"{fname}-fft-peak-mask-width-{int(width)}-{fmin:.2e}-{fmax:.2e}-inc-stft.png")
    plt.close(f)

def plotAcousticSTFT(path:str|Path,key:str="Input 0",inc_data:bool=False,cmap:str="hot") -> plt.Figure:
    """Plot. the SFTF of the given TDMS acoustic signal."""
    fname = os.path.splitext(os.path.basename(path))[0]
    # load the target data
    data = loadTDMSData(path,True)[key]  # noqa: FBT003
    # calculate the STFT of the signal
    f,t,Zxx = stft(data[:,1],fs=1e6,nperseg=1000)  # noqa: N806
    mag = np.abs(Zxx)

    # create plots
    fig,ax = plt.subplots(ncols=2 if inc_data else 1,constrained_layout=True)
    # if not including data then we're only plotting on one axis
    if not inc_data:
        surf = ax.pcolormesh(t,f,mag,shading="auto",norm=LogNorm(),cmap=cmap)
        ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT Magnitude")
        plt.colorbar(surf)
    else:
        ax[0].plot(data[:,0],data[:,1])
        ax[0].set(xlabel="Time (s)",ylabel="Voltage (V)",title="Data")
        ax[1].pcolormesh(t,f,mag,shading="auto",cmap=cmap,norm=LogNorm())
        ax[1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT Magnitude")
    fig.suptitle(fname)
    return fig

def compareAcousticSTFT(pathA:str,pathB:str,timeA:float | None=None,timeB:float | None=None,key:str="Input 0",inc_data:bool=True,cmap:str="hot",freq_mask:float|None=None,use_log:bool=False,detrend:str="linear"):  # noqa: N803
    # get filenames
    fnameA = Path(pathA).stem  # noqa: N806
    fnameB = Path(pathB).stem  # noqa: N806
    # create axes
    fig,ax = plt.subplots(nrows=2 if inc_data else 1,ncols=2,constrained_layout=True)
    # load file A
    data = loadTDMSData(pathA,True)[key]  # noqa: FBT003
    time = data[:,0]
    volt = data[:,1]
    del data
    # if a time period was specified
    def _plotSTFT(time:np.ndarray,data:np.ndarray,aa:plt.Axes,time_per:tuple|None=None) -> plt.Figure:
        if time_per is not None:
            mask = time<=time_per if isinstance(timeA,float) else (time>=time_per[0]) & (time<=time_per[1])
            time_mask = time[mask]
            data_mask = data[mask]
        # compute STFT
        f,t,Zxx = stft(data_mask,fs=1e6,nperseg=1000,detrend=detrend)  # noqa: N806
        # if the user specified a frequency mask
        if freq_mask is not None:
            mask = f>=freq_mask if isinstance(freq_mask,float) else (f>=freq_mask[0]) & (f<=freq_mask[1])
            f = f[mask]
            Zxx = Zxx[mask,:]  # noqa: N806
        # find magnitude
        mag = np.abs(Zxx)
        # plot the STFT + colorbar
        if len(aa)>1:
            aa[0].plot(time_mask,data_mask)
            pcm = aa[1].pcolormesh(t,f,mag,shading="auto",cmap=cmap,norm=LogNorm() if use_log else None)
            aa[1].figure.colorbar(pcm,ax=aa[1])
        else:
            pcm = aa.pcolormesh(t,f,mag,shading="auto",cmap=cmap,norm=LogNorm() if use_log else None)
            aa.figure.colorbar(pcm,ax=aa)
    # plot STFT
    _plotSTFT(time,volt,ax[0,:] if inc_data else ax[0],timeA)
    # load file B
    data = loadTDMSData(pathB,True)[key]  # noqa: FBT003
    time = data[:,0]
    volt = data[:,1]
    del data
    # plot STFT
    _plotSTFT(time,volt,ax[1,:] if inc_data else ax[1],timeB)

    # set labels
    fig.suptitle(f"{fnameA}, {fnameB}\n{key}")
    if inc_data:
        ax[0,0].set(xlabel="Time (s)",ylabel="Voltage (V)",title="Data")
        ax[1,0].set(xlabel="Time (s)",ylabel="Voltage (V)",title="Data")
        ax[0,1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT")
        ax[1,1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT")
    else:
        ax[0].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT")
        ax[1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",title="STFT")
    return fig

def plotAcousticSTFT3D(path:str|Path,key:str="Input 0") -> plt.Figure:
    """Plots the STFT of the given TDMS acoustic signal as a 3D surface plot."""  # noqa: D401
    # load the target data
    data = loadTDMSData(path,True)[key]  # noqa: FBT003
    # calculate the STFT of the signal
    f,t,Zxx = stft(data[:,1],fs=1e6,nperseg=1000)  # noqa: N806
    # create plots
    fig = plt.figure()
    ax = [None,None]
    ax[0] = fig.add_subplot(1,2,1)
    ax[1] = fig.add_subplot(1,2,2,projection="3d")
    ax[0].plot(data[:,0],data[:,1])

    time,freq = np.meshgrid(t,f)
    surf = ax[1].plot_surface(time,freq,np.abs(Zxx),cmap="hot",norm=LogNorm())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax[1].set_zlim(np.abs(Zxx).min(),np.abs(Zxx).max())

    ax[0].set(xlabel="Time (s)",ylabel="Voltage (V)",title="Data")
    ax[1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",zlabel="STFT Magnitude",title="STFT")
    fig.suptitle(os.path.splitext(os.path.basename(path))[0])

    return fig

def plotAllPSD(*args:list) -> plt.Figure:
    f,ax = plt.subplots(ncols=2)
    for fn in args:
        data = loadTDMSData(fn,False)  # noqa: FBT003
        ax[0].psd(data["Input 0"],int(1e6),label=os.path.splitext(os.path.basename(fn))[0])
        ax[1].psd(data["Input 1"],int(1e6),label=os.path.splitext(os.path.basename(fn))[0])
    plt.legend()
    ax[0].set_title("Input 0")
    ax[1].set_title("Input 1")
    return f

def maskFFT(path:str|Path,mask_freq:float,**kwargs) -> tuple[np.ndarray,np.ndarray]:
    """Apply a High pass filter to the target file.

    If the target file is a .bin file, then it's assumed to be a FFT cache file from running plotFFT

    If the target file is a .tdms file, then it's assumed to be the original signal data from which the FFT
    is calculated.

    The target is masked by setting all values from mask_freq and below to zero and performing IRFFT on the result.

    Inputs:
        path : Path to either BIN or TDMS file
        mask_freq : Frequency in Hz to mask at
        key : Target channel to use. Default Input 0.

    Returns time vector and masked data
    """
    ext = os.path.splitext(path)[-1]
    period=1./1e6
    # if the file is a binary file of the FFT data
    if ext == ".bin":
        data = np.memmap(path,"complex128",'r')

        freq = rfftfreq(2*(data.shape[0]-1),period)
        time = np.arange(0,2*(data.shape[0]-1)*period,period)
    # if it's the TDMS file then the FFT needs to be calculated
    elif ext == ".tdms":
        data = loadTDMSData(path,True)[kwargs.get("key","Input 0")]  # noqa: FBT003
        time = data[:,0]
        data = rfft(data[:,1])
        max_len = data.shape[0]
        freq = rfftfreq(max_len,period)
    # mask clear those that are below target
    data_mask = np.zeros(data.shape,data.dtype)
    data_mask = data.copy()
    data_mask[freq<=mask_freq]=0.0
    # reconstruct data
    return time,irfft(data_mask)

def findTimePeaks(path:str|Path,time_skip:int=100,limit:float=0.95,time_dist:float=0.5) -> plt.Figure:
    # load time data
    data = loadTDMSData(path,True)  # noqa: FBT003
    # create plot
    f,ax = plt.subplots(ncols=2,sharex=True)
    # iterate over data and corresponding axes
    for (kk,vv),aa in zip(data.items(),ax):
        aa.plot(vv[:,0],vv[:,1],"b-")
        mask = vv[:,0]>=time_skip
        peaks = find_peaks(vv[mask,1],height=limit*vv[mask,1].max(),distance=int((time_dist/(1./1e6))))[0]  # noqa: UP034
        # filter peaks to after
        aa.plot(vv[mask,0][peaks],vv[mask,1][peaks],"rx")
        aa.set(xlabel="Time (s)",ylabel="Voltage (V)",title=kk)
    f.suptitle(os.path.splitext(os.path.basename(path))[0])
    return f

def rollingMax(path:str|Path,win:float=1.5) -> plt.Figure:
    # load time data
    data = loadTDMSData(path,True)  # noqa: FBT003
    period = 1./1e6
    # create plot
    f,ax = plt.subplots(ncols=2,sharex=True)
    for (kk,vv),aa in zip(data.items(),ax):
        aa.plot(vv[:,0],vv[:,1],"b-")
        aa.set(xlabel="Time (s)",ylabel="Voltage (V)",title=kk)
        bb = aa.twinx()
        bb.plot(vv[:,0],pd.DataFrame(vv[:,1]).rolling(int(win/period),min_periods=1).max(),"r")
        bb.set_ylabel("Rolling Max Voltage (V)")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}, Rolling Window={win}")
    return f

def findTimePeaksRolling(path:str|Path,time_skip:int=100,limit:float=0.5,time_dist:float=0.1,win:float=1.5) -> plt.Figure:
    # load time data
    data = loadTDMSData(path,True)  # noqa: FBT003
    # create plot
    f,ax = plt.subplots(ncols=2,sharex=True)
    period = 1./1e6
    # iterate over data and corresponding axes
    for (kk,vv),aa in zip(data.items(),ax):
        # plot original data
        aa.plot(vv[:,0],vv[:,1],"b-")
        # skip the required amount of time
        mask = vv[:,0]>=time_skip
        # find rolling max
        roll_max = pd.DataFrame(vv[mask,1]).rolling(int(win/period),min_periods=1).max().to_numpy().flatten()
        # make a twin axis to plot the rolling max
        bb = aa.twinx()
        bb.plot(vv[mask,0],roll_max,"r-")
        # find peaks in rolling max
        peaks = find_peaks(roll_max,height=limit*roll_max.max(),distance=int(time_dist/period))[0]
        # filter peaks to after 
        aa.plot(vv[mask,0][peaks],roll_max[peaks],"gx")
        aa.set(xlabel="Time (s)",ylabel="Voltage (V)",title=kk)

    f.suptitle(os.path.splitext(os.path.basename(path))[0])
    return f

def makeTimePeakMasks(path:str|Path,time_skip:int=100,limit:float=0.5,time_dist:int=0.1,win:float=1.5) -> plt.Figure:
    # load time data
    data = loadTDMSData(path,True)  # noqa: FBT003
    f,ax = plt.subplots(ncols=2,sharex=True)
    offset = int(time_skip/(1./1e6))
    period = 1./1e6
    for vv,aa in zip(data.values(),ax):
        vv[:offset,1] = 0
        # find rolling max
        roll_max = pd.DataFrame(vv[:,1]).rolling(int(win/period),min_periods=1).max().to_numpy().flatten()
        peaks = find_peaks(roll_max,height=limit*roll_max.max(),distance=int(time_dist/period))[0]
        # calculate half the window size
        win_half = int(win/period)//2
        # make array to update
        data_mask = np.zeros(vv.shape[0],roll_max.dtype)
        # iterate over the peaks
        for pp in peaks:
            data_mask[pp-win_half:pp+win_half] = vv[pp-win_half:pp+win_half,1]
        # plot
        aa.plot(vv[:,0],vv[:,1],"b-")
        aa.plot(vv[:,0],data_mask,"r-")
        aa.plot(vv[peaks,0],roll_max[peaks],"gx")
    return f
