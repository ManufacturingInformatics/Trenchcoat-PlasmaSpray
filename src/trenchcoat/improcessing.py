from __future__ import annotations

import multiprocessing as mp
from functools import partial
from glob import glob
from pathlib import Path
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Slider
from scipy.spatial.distance import euclidean
from scipy.fft import rfft2
from scipy.signal import convolve2d
from scipy.stats import entropy, kurtosis, norm, skew
from scipy.optimize import curve_fit


MAX_8_BIT = 255

def estPlasmaSize(path: str,thresh:float=800.0,cmap: str="hot",frames: str="all",return_data: bool=False,**kwargs) -> list[plt.Figure, list]:
    """Estimate the size of the plasma in the data and track it over time.

    The size is estimated using image processing methods and is returned in terms of pix^2.

    The input Tlim is the temperature in degrees C used to mask the pixels to only those above or eq
    to the threshold. In the case of multiple contours, the largest one is selected according to
    cv2.contourArea.

    The edges of the contour are drawn on an image using a colour map to denote the time. The idea
    is to show how the area of the contour increases and decreases over time in the image space.
    The input cmap is a string to denote which matplotlib colourmap to use. It is up to the user
    to choose an appropriate colourmap for their data.

    If return_data is True, then the time and area lists are returned as a tuple alongside the drawing.
    i.e. return (time,area),stack

    If return_data is False, the data is plotted and the figure object is returned alongside the drawing.
    i.e. return f,stack

    Inputs:
        path : Path to NPZ file
        thresh : Temperature limit in degrees C. Default 800
        cmap : Matplotlib colour map used in drawing. Default hot.
        frames : Subset of images to use. If "all" then all frames are used
                else it is treated as a 2-element list of integers representing
                lower and upper limit. Default "all"
        return_data : Flag controlling whether the data or the plotted figure is returned.
        title : Title of the plotted figure when return_data is False.

    Returns (index,time,area),drawing when return_data is True or figure,drawing when return_data is False
    """
    # load dataset
    data = np.load(path)["arr_0"]
    # get target subset of frames
    if frames != "all":
        data = data[frames[0]:frames[1],...]
    nf,r,c = data.shape
    # frame to hold normalized data
    norm = np.zeros((r,c),np.uint8)
    # get colormap
    cmap = colormaps.get_cmap(cmap)
    # lists for area and timestamp
    area = []
    time = []
    index = []
    stack_imgs = []
    for i in range(nf):
        # get frame
        frame = data[i,...]
        # mask pixels
        frame[frame<thresh] = frame.min()
        # norm to 0-255
        norm = cv2.normalize(frame,norm,0,255,cv2.NORM_MINMAX).astype("uint8")
        # find contours
        cts,_ = cv2.findContours(norm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # convert index to timestamp
        time.append(i/30.0)
        # store index
        index.append(i)
        # if contours were found
        if len(cts)>0:
            # get largest one
            ct = max(cts,key=lambda x : cv2.contourArea(x))
            # calculate area
            area.append(cv2.contourArea(ct))
            # draw on the stack image
            stack = np.zeros((r,c,3),np.uint8)
            stack = cv2.drawContours(stack,[ct],-1,[int(c*255) for c in cmap(i/nf)[:-1]],1)
            stack_imgs.append(stack)
        else:
            area.append(0)
    # if the user wants the data and not the plot
    if return_data:
        return (index,time,area),stack_imgs
    # make the axes
    f,ax = plt.subplots()
    # plot time against area
    ax.plot(time,area,"x-")
    # set the labels
    ax.set(xlabel="Time (s)",ylabel="Contour Area $(pix^2)$",title=kwargs.get("title",f"{Path(path).stem} Est. Plasma Area vs Time"))
    return f,stack_imgs


def collectPlasmaStats(path: str,thresh: float = 800.0,**kwargs) -> pd.DataFrame:
    """Collect statistics about the area identified in the plasma.

    Based off estPlasmaSize, the temperature is masked to those above Tlim and the
    contour around the edge is drawn.
    The contour is used to create a masked area from which several statistics are 
    collected.

    Current list:
        - Contour area in pix^2
        - Timestamp in seconds est from the index.
        - Frame index of measurements
        - Temperature Variance in degs C.
        - Standard deviation in degs C
        - Mean Temperature in degs C
        - Min Temperature in degs C
        - Width of the contour in pix
        - Height of the contour in pix
        - Aspect Ratio of width/height
        - Temperature Threshold deg C

    These statistics are combined together into a dataframe with appropriate
    column names

    The temperature limit Tlim can either be a single floating point value or a list.
    If it"s a collection, then the threshold is applied multiple times to a dataset and
     the results appended to the combined
    dataframe.

    Inputs:
        path : File path to NPZ file
        thresh : Single floating number of list of Temperature limit to mask above

    Returns pandas dataframe
    """
    # load dataset
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    # statistics to collect
    area = []   # contour area in pix^2
    time = []   # timestamp est from index and FPS
    index = []  # frame index of measurement
    var = []    # temp var
    std_dev = []# temp std dev
    mean_temp = []  # mean temp
    min_temp = []   # min temp
    max_temp = []   # max temp
    width = []  # contour weidth
    height = [] # contour height
    asprat = [] # aspect ratio w/h
    temp_limit = []   # temperature limit used for masking
    # frame to hold normalized data
    norm = np.zeros((r,c),np.uint8)
    if isinstance(thresh,(float,int)):
        thresh = [thresh]
    for tl in thresh:
        for i in range(nf):
            # get frame
            frame = data[i,:,:]
            # mask pixels
            frame[frame<tl] = frame.min()
            # norm to 0-255
            norm = cv2.normalize(frame,norm,0,255,cv2.NORM_MINMAX).astype("uint8")
            # find contours
            cts,_ = cv2.findContours(norm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            time.append(i/30.0)
            index.append(i)
            temp_limit.append(tl)
            # if contours were found
            if len(cts)>0:
                # get largest one
                ct = max(cts,key=lambda x : cv2.contourArea(x))
                # draw filled contour area on a mask
                mask = np.zeros((r,c),np.uint8)
                mask = cv2.drawContours(mask,[ct],-1,(255,255,255),-1)
                # fit bounding rectangle around contour to get width and height
                _,_,w,h = cv2.boundingRect(ct)
                # create masked array
                mask_arr = np.ma.MaskedArray(data[i,...],mask=mask==0).compressed()
                # update stats
                area.append(cv2.contourArea(ct))
                var.append(mask_arr.var())
                std_dev.append(mask_arr.std())
                mean_temp.append(mask_arr.mean())
                min_temp.append(mask_arr.min())
                max_temp.append(mask_arr.max())
                height.append(h)
                width.append(w)
                asprat.append(w/h)
            else:
                area.append(0)
                var.append(0)
                std_dev.append(0)
                mean_temp.append(0)
                min_temp.append(0)
                max_temp.append(0)
                height.append(0)
                width.append(0)
                asprat.append(0)
    # form into dataframe
    return pd.DataFrame({"Frame Index":index,"Time (s)":time,"Contour Area (pix^2)":area,"Variance (deg C)":var,
                         "Std Dev (deg C)":std_dev,"Mean Temperature (deg C)":mean_temp,"Min. Temperature (deg C)":min_temp,
                         "Max. Temperature (deg C)":max_temp,"Width (pix)":width,"Height (pix)":height,"Aspect Ratio (Width/Height)":asprat,"Temperature Threshold (deg C)":temp_limit})


def combinePlasmaStats(path: str,thresh:float=800.0,**kwargs) -> pd.DataFrame:
    """Stacks result of collectPlasmaStats applied to each of the files found in the wildcard path.

    See collectPlasmaStats for more details on the structure and stats collected.

    Two new columns File Path and Label are added to each dataframe before stacking so the stats from each file can be distinguished.
    File Path is the complete file path to the original file so it can be traced back.
    Label is a unique label separate from the File Path indicating some sort of metadata that"s more useful e.g. Gas Flow Rate.
    These labels are be specified via the "labels" keyword or will be set to the filename without extension.

    The temperature limit Tlim can either be a single floating point value or a list. If it"s a collection,
    then the threshold is applied multiple times to a dataset and the results appended to the combined
    dataframe.

    Inputs:
        path : Wildcard path to files or list of specified files.
        Tlim : Single floating number of list of Temperature limit to mask above. Default 800.0 C
        labels : List of unique labels for each file. If not specified, it defaults to the filename without the extension

    Returns a single pandas DataFrame with all the stats combined into one
    """
    if isinstance(path,str):
        path = glob(path)
    # if no labels are given, default to filename without extension
    labels = kwargs.get("labels",[Path(fn).stem for fn in path])
    all_dfs = []
    for fn,label in zip(path,labels):
        # collect statistics using target settings
        stats = collectPlasmaStats(fn,thresh,**kwargs)
        # store the file path
        stats["File Path"] = fn
        # store the assigned label
        # can be used as a Hue parameter for plotting
        stats["Label"] = label
        # add to list
        all_dfs.append(stats)
    # stack together to form giant
    return pd.concat(all_dfs,ignore_index=True)


def findTrueArea(df: pd.DataFrame) -> pd.DataFrame:
    """Find the true area between the thresholds area values in the dataframe from combinePlasmaStats.

    Because the area was calculated using a single thresholds, the area at one threshold will have the combined area of the higher
    thresholds above it.

    e.g. if Tlim is 600 C, the area will contain the areas for 700, 800, 900 etc.

    The purpose of this function is to find the area between pairs of threshold limits

    So if Tlim was [500,600,700,800] then this finds the areas for the following

    500-600 = Area(500) - Area (600)
    600-700 = Area(600) - Area (700)

    ------------------------- T=600 C
    +++++++++++++++++++++++++
    ------------------------- T=700 C

    A new dict is formed where keys are the the unique labels and contain another dictionary with the time
    and area for specific temperature limits.

    As the area vectors can be different lengths, they needed to be padded to ensure they"re the same length.
    They are padded with pad_value.

    The dicts are then combined together into a single dataframe

    Inputs:
        df : Dataframe from combinePlasmaStats.

    Returns DataFrame
    """
    # get unique trhesholds
    ths = df["Temperature Threshold (deg C)"].unique()
    res_label = {}
    # iterate over labels
    for lb in df["Label"].unique():
        res = {}
        filt = df[df["Label"]==lb]
        # iterate over thresholds as pairs
        for ta,tb in zip(ths,ths[1:]):
            # get contour area for both thresholds
            ca = filt[filt["Temperature Threshold (deg C)"]==ta]["Contour Area (pix^2)"]
            cb = filt[filt["Temperature Threshold (deg C)"]==tb]["Contour Area (pix^2)"]
            ## piecewise interpolate to lineup the times
            # use the dataset with more points as the new timepoints
            # combine timestamps together
            ref_time = np.array(sorted(set(np.append(filt[filt["Temperature Threshold (deg C)"]==ta]["Time (s)"].values,filt[filt["Temperature Threshold (deg C)"]==tb]["Time (s)"].values))))
            ca_interp = np.interp(ref_time,filt[filt["Temperature Threshold (deg C)"]==ta]["Time (s)"],ca)
            cb_interp = np.interp(ref_time,filt[filt["Temperature Threshold (deg C)"]==tb]["Time (s)"],cb)
            # subtract the larger threshold contour area array from lower threshold
            # A(ta) - A(tb) -> tA >= T > tB
            res[f"Area {ta}C - {tb}C"] = ca_interp - cb_interp
            res[f"Frame Index {ta}C - {tb}C"] = ref_time/30.0
            res[f"Time (s) {ta}C - {tb}C"] = ref_time
        # add to dict under label
        res_label[lb] = res
    # convert each dict into a dataframe and combine
    all_dfs = []
    for k,v in res_label.items():
        area_df = pd.DataFrame(v)
        # assign label as a column
        area_df["Label"]=k
        all_dfs.append(area_df)
    # combine together
    return pd.concat(all_dfs)


def estAllPlasmaSize(path: str,thresh:float=800.0,**kwargs) -> plt.Figure | list[plt.Figure, dict, dict]:
    """Estimate the plasma size of several files and plot them on the same axis.

    The size is estimated using image processing methods and is returned in terms of pix^2.

    The input Tlim is the temperature in degrees C used to mask the pixels to only those above or eq
    to the threshold. In the case of multiple contours, the largest one is selected according to
    cv2.contourArea.

    If rel_area is True, then the area is converted to relative to the max area. This can help correct
    for different recording distances and see relative changes in area instead.

    By default the legend labels are set to the filenames of the inputs. This can be overriden to something
    more informative using the labels keyword.

    As there is usually an offset between the traces due to when the plasma is started, they won"t be aligned.
    The input align_area is a parameter that sets time to 0 where the contour area becomes equal to greater
    than this value. By default this is set to 300.0 pix^2 which works well. If align_area is set to 0,
    then no alignment is performed.

    When return_data is True, a dict containing all the data is returned. This is organised by full file path
    and contains a sub dict containing the assigned label from the labels input and the data. Under "data"
    is a 3 column numpy array containing the index, timestamp and contour area in that order.

    Inputs:
        path : Wildcard path to multiple files
        thresh : Temperature threshold in degrees C. Default 800.0
        labels: Legend labels for each file. Default is filenames without extension.
        align_area : Area value to align the signals. If zero, then it"s ignored. Default 0.0
        title: Figure title. Default Estimated Plasma Area vs Time

    Return figure object and data dictionary when return_data is True else just the figure object.
    """
    if isinstance(path,str):
        path = glob(path)
    # axis for plotting
    f,ax = plt.subplots()
    # lists for metrics
    area = [] # contour area
    time = [] # time stamp from index
    index = [] # frame index
    width = [] # width of the contour
    height = [] # height of the contour
    aspect_ratio = [] # aspect ratio 2/
    res_imgs = {}
    res = {}
    # set plotting labels
    # by default it goes to filenames
    if len(kwargs.get("labels",[]))>0:
        labels = kwargs.get("labels",[])
    else:
        labels = [Path(fn).stem for fn in path]
    # ensure it is a list
    if isinstance(thresh,(float,int)):
        thresh = [thresh]
    # iterate over each file
    for fn,label in zip(path,labels):
        # load file
        data = np.load(fn)["arr_0"]
        nf,r,c = data.shape
        res_imgs[fn] = {}
        res[fn] = {}
        for tl in thresh:
            # clear the metrics list
            area.clear()
            time.clear()
            index.clear()
            height.clear()
            width.clear()
            aspect_ratio.clear()
            # setup the images directory
            res_imgs[fn][str(tl)] = []
            # normalized frame
            norm = np.zeros((r,c),np.uint8)
            # iterate over frames
            for i in range(nf):
                frame = data[i,...]
                # mask
                frame[frame<tl] = frame.min()
                norm = cv2.normalize(frame,norm,0,255,cv2.NORM_MINMAX).astype("uint8")
                cts,_ = cv2.findContours(norm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if len(cts)>0:
                    # choose largest contour
                    ct = max(cts,key=lambda x : cv2.contourArea(x))
                    # find the width and height
                    _,_,w,h = cv2.boundingRect(ct)
                    width.append(w)
                    height.append(h)
                    aspect_ratio.append(w/h)
                    # draw the contour
                    img = np.zeros((r,c),np.uint8)
                    img = cv2.drawContours(img,[ct],-1,255,1)
                    res_imgs[fn][str(tl)].append(img)
                    # calculate area
                    area.append(cv2.contourArea(ct))
                    # timestamp
                    time.append(i/30.0)
                    index.append(i)
            # find timestamps to align the time
            ref = 0
            if kwargs.get("align_area",0.0) > 0:
                ref = time[np.where(np.array(area)>=kwargs.get("align_area",300.0))[0].min()]
            time = [t-ref for t in time]
            res[fn][str(tl)] = {"label":label,"data":np.column_stack([index,time,area,np.array(area)/max(area),width,height,aspect_ratio]),"cols":["Frame Index","Time (s)","Contour Area ($pix^2$)","Relative Contour Area","Width (pix)","Height (pix)","Aspect Ratio"]}
            # plot data
            ax.plot(time,area,label=f"{label},T>={tl}")
    # set the plotting labels
    # if a non-zero align_area value was given, then the x-axis will say Aligned Time (s) instead of Time (s)
    ax.set(xlabel=f"{"Aligned " if kwargs.get("align_area",0.0)>0 else ""}Time (s)",ylabel="Contour Area ($pix^2$)")
    f.suptitle(kwargs.get("title",fr"Estimated Plasma Area vs Time (Tlim={thresh} $^\circ$C)"))
    ax.legend()
    if kwargs.get("return_data",False):
        return f,res,res_imgs
    return f


def fitPolyToPlasmaArea(data: str,**kwargs) -> tuple[plt.Figure, pd.DataFrame]:
    r"""Fit a 1D line to a specific time period of each data Using the data from estAllPlasmaSize, .

    The goal is to estimate the gradient of specific portions of the data to compare the different
    curves. By default the period 0 to 1s is used as it"s the initial growth of the shape.

    NOTE: this works best when align_area is set in estAllPlasmaSize to align the signals so a single time
    period is used.

    In case the signal is not aligned, the user can specify between which limits the line is fitted over.
    Currently supported are time period ("t") and contour area ("area"). Either can be specified by a dictionary where the keys
    are the file paths from the original dict or a two-element tuple/list applied to all signals

    e.g. if t=[0,1] then the signal is fitted over the time period 0-1s.

    If no period is specified, a line is fitted over the entire signal.

    Three plots are made:
        1) Original data + fitted lines
        2) Only fitted lines
        3) Gradient of the fitted lines

    For the final plot, the x-axis tick labels are set to the labels in the data dict. This can also be overriden
    using the new_ticks keyword.

    Inputs:
        data : Dict from estAllPlasmaSize
        t : Two-element list or dict containing two-element lists specifying the time period to fit over.
        area : Two-element list or dict containing two-element lists specifying area period to fit over.
        xlabel : x-axis label used in the first plot. Default "Time (s)".
        ylabel : y-axis label used in the first plot. Default "Contour Area ($pix^2$)".
        title : Axis title used in the first plot. Default "Plasma Contour Area vs Time with Fitted Lines".
        fit_xlabel : x-axis label used in the second plot. Default "Time (s)".
        fit_ylabel : y-axis label used in the second plot. Default "Contour Area ($pix^2$)".
        fit_title : Axis title used in the first plot. Default "Fitted Lines to Plasma Contour Area".
        gradient_xlabel : x-axis label used in the second plot. Default "Plasma Gas Flow Rate (SL/min)".
        gradient_ylabel : y-axis label used in the second plot. Default "Fitted Line Gradient".
        gradient_title : Axis title used in the first plot. Default f"Gradient of Fitted Lines to Plasma Contour Area\nt=[{tA},{tB}]".
        new_ticks : New tick labels for third plot. Default is labels in input dictionary.

    Returns figure of original data + fitted lines, figure of only fitted lines and a figure of gradient of the fitted lines
    """
    use_time = False
    use_area = False
    use_rel_area = False
    suffix = ""
    if "t" in kwargs:
        if isinstance(kwargs["t"],(list,tuple)):
            tlow, thigh = kwargs["t"]
            use_area = True
            suffix=f"t=[{tlow},{thigh}]"
        else:
            suffix="masked by time"
    elif "area" in kwargs:
        if isinstance(kwargs["area"],(list,tuple)):
            areadown, areaup = kwargs["area"]
            if ((areadown>=0) and (areadown<=1)) and ((areaup>=0) and (areaup<=1)):
                use_rel_area = True
                suffix=f"rel. area=[{areadown},{areaup}]"
            else:
                use_area = True
                suffix=f"area=[{areadown},{areaup}]"
        else:
            suffix="masked by area"
    # reformat dictionary to be threshold first rather than filename first
    # it is arranged this way so a new figure is made for each threshold
    new_dict = {}
    for fn,v in data.items():
        for t,d in v.items():
            if t not in new_dict:
                new_dict[t] = {}
            new_dict[t][fn] = d

    # gradient value
    grad = []
    off = []
    # max area
    maxm = []
    # mean area
    meanm = []
    # label used for x ticks
    labels = []
    # dict of figures generated
    figures = {}
    # dataframes to combine together
    dfs = []
    # iterate over dictionary
    for t,v in new_dict.items():
        # create plot for the data and fitted
        f,ax = plt.subplots()
        # fitted only
        ff,axf = plt.subplots()
        # gradient plots
        fz,axz = plt.subplots(ncols=2,constrained_layout=True)
        # clear gradient and offset lists
        grad.clear()
        off.clear()
        maxm.clear()
        meanm.clear()
        labels.clear()
        # iterate over data for each file
        for fn,d in v.items():
            # extract the data points
            pts = d["data"]
            # extract the time period
            if use_time:
                if isinstance(kwargs["t"],dict):
                    tlow,thigh = kwargs["t"][fn]
                # make to time period
                mask = np.where((pts[:,1]>=tlow) & (pts[:,1]<=thigh))[0]
            # extract area period
            elif use_area:
                if isinstance(kwargs["area"],dict):
                    arealow,areahigh = kwargs["area"][fn]
                # else use regular area col
                else:
                    mask = np.where((pts[:,2]>=arealow) & (pts[:,2]<=areahigh))[0]
                # apply additional mask to only used masked data within a 1 sec period
                mask = mask[np.where(pts[mask,1] <= (pts[mask,1].min()+kwargs.get("tp",5.0)))[0]]
            elif use_rel_area:
                if isinstance(kwargs["area"],dict):
                    arealow,areahigh = kwargs["area"][fn]
                # if it"s between 0 and 1 use relative area column
                if ((arealow>=0) and (arealow<=1)) and ((areahigh>=0) and (areahigh<=1)):
                    mask = np.where((pts[:,3]>=arealow) & (pts[:,3]<=areahigh))[0]
                # else use regular area col
                else:
                    mask = np.where((pts[:,3]>=arealow) & (pts[:,3]<=areahigh))[0]
                mask = mask[np.where(pts[mask,1] <= (pts[mask,1].min()+kwargs.get("tp",5.0)))[0]]
            # if no mask
            else:
                mask = np.arange(0,pts.shape[0],1)
            pts_mask = pts[mask,:]
            # plot the original data
            offset=0
            if kwargs.get("use_reltime",True):
                offset = pts_mask[:,1].min()
            # fit a poly to the data returning coeffs
            if use_rel_area:
                z = np.polyfit(pts_mask[:,1],pts_mask[:,3],1)
                ax.plot(pts_mask[:,1]-offset,pts_mask[:,3],".-",label=d["label"])
            else:
                z = np.polyfit(pts_mask[:,1],pts_mask[:,2],1)
                ax.plot(pts_mask[:,1]-offset,pts_mask[:,2],".-",label=d["label"])
            # store gradient
            grad.append(z[0])
            off.append(z[1])
            maxm.append(d["data"][:,2].max())
            meanm.append(d["data"][:,2].mean())
            labels.append(d["label"])
            # create points to feed into function 
            xp = np.linspace(pts_mask[:,1].min(),pts_mask[:,1].max())
            yp = np.poly1d(z)(xp)
            offset=0
            if kwargs.get("use_reltime",True):
                offset = xp.min()
            # plot the fitted data
            ax.plot(xp-offset,yp,"--",label=f"{d["label"]} Fit")
            # plot the fitted data on the other axis
            axf.plot(xp-offset,yp,"--",label=f"{d["label"]} Fit")

        # store the data in a dataframe
        poly_data = pd.DataFrame({f"Gradient ({suffix})":grad,f"Offset ({suffix})":off,f"Max Area ({suffix}":maxm})
        poly_data["File"] = list(v.keys())
        poly_data["Label"] = [d["label"] for d in v.values()]
        poly_data["Threshold (C)"] = t
        dfs.append(poly_data)
        ax.legend()
        axf.legend()
        # set labels
        axf.set(xlabel=kwargs.get("fit_xlabel","Relative Time (s)" if kwargs.get("use_reltime",True) else "Time (s)"),ylabel="Contour Area ($pix^2$)" if use_time or use_area else "Relative Contour Area")
        ff.suptitle(kwargs.get("fit_title",rf"Fitted Lines to Plasma Contour Area\n{suffix} T>={t} $^\circ$C"))
        # set the label for og + fitted
        ax.set(xlabel=kwargs.get("xlabel","Relative Time (s)" if kwargs.get("use_reltime",True) else "Time (s)"),ylabel="Contour Area ($pix^2$)" if use_time or use_area else "Relative Contour Area")
        f.suptitle(kwargs.get("title",rf"Plasma Contour Area vs Time with Fitted Lines\n{suffix} T>={t} $^\circ$C"))
        # plot the gradient of each fitted line
        axz[0].plot(grad,"b-x")
        axz[1].plot(off,"r-o")
        # change each x-tick to the target label
        axz[0].set_xticks(range(len(grad)),labels)
        axz[1].set_xticks(range(len(off)),labels)
        # set the axis labels
        axz[0].set(xlabel=kwargs.get("gradient_xlabel","Plasma Gas Flow Rate (SL/min)"),ylabel="Fitted Line Gradient",title="Fitted Line Gradient")
        axz[1].set(xlabel=kwargs.get("gradient_xlabel","Plasma Gas Flow Rate (SL/min)"),ylabel="Fitted Line Offset",title="Fitted Line Offset")
        # set the title
        fz.suptitle(kwargs.get("gradient_title",fr"Plasma Contour Area Fitted Line Paramaters {suffix}\nT>={t} $^\circ$C"))

        # create a plot of the max area
        fm,axm = plt.subplots()
        axm.plot(maxm,"-x")
        axm.set_xticks(range(len(labels)),labels)
        axm.set(xlabel="Plasma Gas Flow Rate (SL/min)",ylabel="Max Plasma Area ($pix^2$)")
        fm.suptitle(fr"Max Plasma Area ($pix^2$) vs Plasma Gas Flow Rate (SL/min)\nT>={t} $^\circ$C")

        # create a plot of the mean area
        fmn,axmn = plt.subplots()
        axmn.plot(meanm,"-x")
        axmn.set_xticks(range(len(labels)),labels)
        axmn.set(xlabel="Plasma Gas Flow Rate (SL/min)",ylabel="Average Plasma Area ($pix^2$)")
        fmn.suptitle(fr"Average Plasma Area ($pix^2$) vs Plasma Gas Flow Rate (SL/min)\nT>={t} $^\circ$C")
        # update figure dict
        figures[t] = [f,fz,fm,fmn]

    return figures, pd.concat(dfs,ignore_index=True)


def makeColorBar(cmap: str,width: int,height: int) -> np.ndarray:
    """Construct an image of the target size for the target colourmap.

    Inputs:
        cmap: Colour map name
        width : Width of the colour map
        height : Height of the colour map
        transpose: Flag to transpose the result

    Returns constructed image
    """
    # make a vector of 0-255 values the required height
    col = np.linspace(0,255,height.shape[0],dtype=np.uint8)
    # stack to form a grayscale version we want to apply
    channel = np.column_stack(width*(col,))
    if cmap == "hsv":
        bar = np.dstack((channel,255*np.ones(channel.shape,np.uint8),255*np.ones(channel.shape,np.uint8)))
    elif cmap == "yellow":
        bar = np.dstack((channel,channel,np.zeros(channel.shape,np.uint8)))
    elif cmap == "gray":
        bar = np.dstack(3*(channel,))
    # if it"s an integer take it to be an opencv colormap
    elif isinstance(cmap,int):
        bar = cv2.applyColorMap(channel,cmap)
    # if not a matching string or an interger then try and call it as a function passing the grayscale channel
    else:
        bar = cmap(channel)
    return bar

def makeColorBarTrans(cmap: str,width: int,height: int) -> np.ndarray:
    """Construct a transposed image of the target size for the target colourmap.

    Inputs:
        cmap: Colour map name
        width : Width of the colour map
        height : Height of the colour map
        transpose: Flag to transpose the result

    Returns constructed image
    """
    # make a vector of 0-255 values the required height
    col = np.linspace(0,255,height.shape[0],dtype=np.uint8)
    # stack to form a grayscale version we want to apply
    channel = np.column_stack(width*(col,))
    if cmap == "hsv":
        bar = np.dstack((channel,255*np.ones(channel.shape,np.uint8),255*np.ones(channel.shape,np.uint8)))
    elif cmap == "yellow":
        bar = np.dstack((channel,channel,np.zeros(channel.shape,np.uint8)))
    elif cmap == "gray":
        bar = np.dstack(3*(channel,))
    # if it"s an integer take it to be an opencv colormap
    elif isinstance(cmap,int):
        bar = cv2.applyColorMap(channel,cmap)
    # if not a matching string or an interger then try and call it as a function passing the grayscale channel
    else:
        bar = cmap(channel)
    return bar.T

def convertTempToVideo(data: np.ndarray,opath: str | None = None,cmap: int = cv2.COLORMAP_HOT) -> None:
    """Convert the temperature dataset to video.

    A colormap is applied to each frame of the dataset and the results are written to the video.

    The video is written to the opath location or if not specifed then the default path is temp-video.avi

    Inputs:
        data : Temperature dataset organised frames first (i.e. (num frames x rows x cols))
        opath : Output path to write the video to. If None, opath is set to temp-video.avi
        cmap : OpenCV colormap or gray. Gray converts to grayscale
    """
    if isinstance(data,str):
        data = np.load(data)["arr_0"]
    # if opath is None, set to a default value
    if opath is None:
        opath = "temp-video.avi"
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    if cmap == "gray":
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),0)
    else:
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),1)
    # iterate over frames
    for ff in range(nf):
        # get frame
        frame = data[ff,:,:]
        # normalize
        frame -= frame.min()
        frame /= frame.max()
        frame *= 255.0
        frame = frame.astype("uint8")
        if cmap == "gray":
            out.write(frame)
        # else apply colormap to the data
        else:
            frame = cv2.applyColorMap(frame,cmap)
            # write data to video file
            out.write(frame)
    # release video writer
    out.release()


def recolormapCsplit(data: str | np.ndarray,csplit: int,opath: str,cmap: int = cv2.COLORMAP_HOT) -> None:
    """Colormap the target data normalizing the data in two parts specified by csplit.

    Each frame is load and split columnwise about the csplit index. Each part is
    normalized and combined afterwards

    The output video is written to opath or temp-video.avi if not specified

    Inputs:
        data : Path to NPZ file or loaded NP array
        csplit : Column index to split the image at
        opath : Output path to write the video to.
        cmap : OpenCV colormap index. Default cv2.COLORMAP_HOT
    """
    # if opath is None, set to a default value
    if opath is None:
        opath = "temp-video.avi"
    if isinstance(data,str):
        data = np.load(data)["arr_0"]
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    if cmap == "gray":
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),0)
    else:
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),1)
    # iterate over frames
    for ff in range(nf):
        # get frame
        frame = data[ff,:,:]
        # split into 2 parts about the specified column
        left = frame[:,:csplit]
        right = frame[:,csplit:]
        # normalize
        left -= left.min()
        left /= left.max()
        left *= 255.0
        left = left.astype("uint8")

        right -= right.min()
        right /= right.max()
        right *= 255.0
        right = right.astype("uint8")

        if cmap == "gray":
            frame = cv2.hconcat((left,right))
            out.write(frame)
        # else apply colormap to the data
        else:
            leftcol = cv2.applyColorMap(left,cmap)
            rightcol = cv2.applyColorMap(right,cmap)
            frame = cv2.hconcat([leftcol,rightcol])
            # write data to video file
            out.write(frame)
    # release video writer
    out.release()


def recolormapSections(data: str | np.ndarray,secs: int,opath: str,cmap: int = cv2.COLORMAP_HOT,axis: int = 1) -> None:
    """Colormap the target data normalizing the data in two parts specified by csplit.

    Each frame is loaded and split as specifed by specs. Specs and axis is supplied to the
    np.array_split function. See np.array_split docs.

    The output video is written to opath or temp-video.avi if not specified

    Inputs:
        data : Path to NPZ file or loaded NP array
        secs : Column index to split the image at
        opath : Output path to write the video to.
        cmap : OpenCV colormap index. Default cv2.COLORMAP_HOT
        axis : Axis for splitting. Default 1.
    """
    # if opath is None, set to a default value
    if opath is None:
        opath = "temp-video.avi"
    if axis ==0:
        def combine(secs:np.ndarray) -> np.ndarray:
            return cv2.vconcat(secs)
    elif axis==1:
        def combine(secs:np.ndarray) -> np.ndarray:
            return cv2.hconcat(secs)
    else:
        msg = f"Axis selection has to be either 0 or 1! Received {axis}!"
        raise ValueError(msg)
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    if cmap == "gray":
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),0)
    else:
        out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),1)
    # iterate over frames
    for ff in range(nf):
        # get frame
        frame = data[ff,:,:]
        sections = np.array_split(frame,secs,axis)
        for i in range(len(sections)):
            sections[i] -= sections[i].min()
            sections[i] /= sections[i].max()
            sections[i] *= 255.0
            sections[i] = sections[i].astype("uint8")

        if cmap == "gray":
            frame = combine(sections)
            out.write(frame)
        # else apply colormap to the data
        else:
            for i in range(len(sections)):
                sections[i] = cv2.applyColorMap(sections[i],cmap)
            frame = combine(sections)
            # write data to video file
            out.write(frame)
    # release video writer
    out.release()


def pltTempToVideo(data: str | np.ndarray,opath: str,cmap: str = "hot") -> None:
    """Convert NPZ data into an animation using matplotlib FuncAnimation.

    Uses a FuncAnimation with the following settings:
        - interval : int((1./30.0)
        - blit : True
    And all frames area written to the video

    The cmap parameter is a string colormap reference supported by Matplotlib

    NPZ needs to be frame first i,e, nf,r,c

    Inputs:
        data : Path to NPZ file or already loaded NPZ array
        opath : Output path for animation
        cmap : String to matplotlib colormap. Default hot
    """
    from matplotlib.animation import FuncAnimation
    # load the npz file
    if isinstance(data,str):
        data = np.load(data)["arr_0"]
    # setup the figure and axis
    f,ax = plt.subplots(constrained_layout=True)
    nf,r,c = data.shape
    # create mappable using first frame
    im = ax.imshow(data[0,:,:],cmap=cmap)
    # function for updating
    def update(i:int) -> list:
        im.set_array(data[i,:,:])
        return [im]
    # create animation
    ani = FuncAnimation(f,update,frames=nf,interval=int((1./30.0)*1000),blit=True)
    ani.save(opath)


def drawContoursVideo(data: np.ndarray,opath: str | None = None,cmap: int = cv2.COLORMAP_HOT,color: tuple[int, int, int] = (0,255,0)) -> None:
    """Create a video drawing contours around the found objects in the temperature data.

    Uses findObjectsInTemp to find the objects.

    Linewidth of contour currently set to 1 due to the image size being v. small

    Inputs:
        data : 3D data array of temperature values
        opath : Output path to save the data to. If None, it is set to temp-contour-video.avi
        cmap : OpenCV colormap or gray
        color : Line color for contour. Default green.
    """
    # if opath is None, set to a default value
    if opath is None:
        opath = "temp-contour-video.avi"
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows))
    # iterate over frames
    for ff in range(nf):
        # get frame
        frame = data[ff,:,:]
        # normalize
        img = frame - frame.min()
        img /= img.max()
        img *= 255.0
        img = frame.astype("uint8")
        # if user specified grayscale
        img = np.dstack(3*[img]) if cmap=="gray" else cv2.applyColorMap(img,cmap)

        cts = findObjectsInTemp(frame)
        if len(cts)>0:
            # draw contours
            cv2.drawContours(img,cts,-1,color,1)
        # write data to video file
        out.write(img)
    # release video writer
    out.release()


def binFrame(data: np.ndarray,levels: int = 5) -> np.ndarray:
    """Bin the data into the set number of levels and sets pixels to the index.

    Calculates the bin edges using np.histogram and creates a
    new matrix where the values within the edges are the index

    e.g. bins = [b0,b1,...bN]
    binned[(data>=b0)&(data<=b1)] = 0

    Returns the binned values
    """
    # flatten and perform histogram
    edges = np.histogram_bin_edges(data.flatten(),levels)
    # make an empty array
    # set to uint8 so it can be easily viewed and processed
    binned = np.zeros(data.shape,dtype="uint8")
    # iterate over bin edges and set the pixels that fall in the edges
    # to its index
    for bi,(left,right) in enumerate(zip(edges,edges[1:])):
        binned[(data>=left)&(data<=right)] = bi
    return binned


def binFrameValues(data: np.ndarray,levels: int = 5) -> np.ndarray:
    """Bin the data into the set number of levels and sets pixels to the index.

    Calculates the bin edges using np.histogram and creates a
    new matrix where the values within the edges are the set to
    the upper edge

    e.g. bins = [b0,b1,...bN]
    binned[(data>=b0)&(data<=b1)] = 0

    Returns the binned values
    """
    # flatten and perform histogram
    edges = np.histogram_bin_edges(data.flatten(),levels)
    # make an empty array
    # set to uint8 so it can be easily viewed and processed
    binned = np.zeros(data.shape,dtype=data.dtype)
    # iterate over bin edges and set the pixels that fall in the edges
    # to its index
    for left,right in zip(edges,edges[1:]):
        binned[(data>=left)&(data<=right)] = right
    return binned


def binFrameArray(data: np.ndarray,levels: int = 5) -> np.ndarray:
    """Bin the data into the set number of levels and sets pixels to the upper edge.

    Calculates the bin edges using np.histogram and creates a
    new matrix where the values within the edges are the set to
    the upper edge

    e.g. bins = [b0,b1,...bN]
    binned[(data>=b0)&(data<=b1)] = b1

    Returns the binned values
    """
    # if the data is a 2D array
    # just pass it to binFrame
    if len(data.shape)==2:
        return binFrame(data,levels)
    # if it"s a 3D array
    if len(data.shape)==3:
        # freeze the binFrame function to the desired parameters
        convert = partial(binFrame,levels)
        # create a process pool
        with mp.Pool(processes=4) as pool:
            # feed in each frame from the data
            res = pool.map(convert,(data[z,:,:] for z in range(data.shape[0])))
            # create empty array
            binned = np.zeros(data.shape,dtype="uint8")
            # iterate over results + update binned array
            for z in range(data.shape[0]):
                binned[z,:,:]=res[z]
        # return theed updated result
        return binned
    return None


def binFrameVideo(data: np.ndarray,opath: str | None = None,levels: int = 5,mode: str = "index") -> None:
    """Apply binFrameArray on each frame in the array, normalizes and writes the result to a video.

    The number of binning levels is set up levels.

    The mode parameter controls how the frames are normalized:
        - index : Calls binFrame then cv2.normalize
        - values : Calls binFramesValues then manually normalizes
    """
    if mode not in ["index","values"]:
        msg = f"Unsupported mode {mode}! Should be in ['index','values']"
        raise ValueError(msg)
    # if opath is None, set to a default value
    if opath is None:
        opath = f"temp-binned-levels-{levels}-{mode}-video.avi"
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    # set to grayscale
    out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),0)
    # iterate over frames
    for ff in range(nf):
        # get frame
        frame = data[ff,:,:]
        # if the user wants the index
        if mode == "index":
            binned = binFrame(frame,levels)
            # normalize to make it grayscale
            img = cv2.normalize(binned,0,255,cv2.NORM_MINMAX)
        elif mode == "values":
            binned = binFrameValues(frame,levels)
            # normalize 0-1
            binned = (binned - binned.min())/(binned.min()-binned.max())
            binned *= 255.0
            img = binned.astype("uint8")
        # write data to video file
        out.write(img)
    # release video writer
    out.release()


def createTempDiffToVideo(data: np.ndarray,opath: str | None = None,cmap: int = cv2.COLORMAP_HOT):
    """Create a video from the absolute temperature difference between frames.

    A colormap is applied to each difference frame and written to video

    Inputs:
        data : Temperature dataset organised frames first (i.e. (num frames x rows x cols))
        opath : Output path to write the video to. If None, then temp-diff-video.avi
        cmap : OpenCV colormap or gray. Gray converts to grayscale
    """
    # if opath is None, set to a default value
    if opath is None:
        opath = "temp-diff-video.avi"
    # get shape of data
    nf,rows,width = data.shape
    # create fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows))
    # create list of frames
    num_frames = list(range(nf))
    # iterate over frames
    for prevs,now in zip(num_frames,num_frames[1:]):
        # get frame
        diff = np.abs(data[prevs,:,:] - data[now,:,:]) 
        # normalize
        diff -= diff.min()
        diff /= diff.max()
        diff *= 255.0
        diff = diff.astype("uint8")
        # make 3 channel
        if cmap=="gray":
            diff = np.dstack(3*[diff])
        # else apply colormap to the data
        else:
            diff = cv2.applyColorMap(diff,cmap)
        # write data to video file
        out.write(diff)
    # release video writer
    out.release()


def findObjectsInTemp(data: np.ndarray) -> np.ndarray:
    """Find all objects in the given temperature frame.

    The temperature frame is normalized, converted to grayscale and then thresholded.
    It is thresholded to binary using Otsu. OpenCVs findContours method is then applied
    with the RETR_LIST so a list of contour points is returned.

    Inputs:
        data : 2D temperature frame

    Returns list of contour points for found objects
    """
    # convert temperature frame to grayscale
    data -= data.min()
    data /= data.max()
    data *= 255.0
    data = data.astype("uint8")
    # threshold
    _,thresh = cv2.threshold(data,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find contours around objects
    return cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]


class PlasmaGaussian:
    """Fits a Gaussian model to the shape of the plasma head.

    If lim is between 0-1 then it is calculated based on the max temperature of the frame
    rather than an absolute value. Anything else is treated as direct temperature limits

    # make the class
    clf = PlasmaGaussian()
    # load a temperature source
    data = np.load(path)["arr_0"]
    # fit the gaussian
    hist = clf.fit_guess(data[0,:,:])
    # draw the fitting
    draw = clf.draw(True,True,True)
    """

    def __init__(self,lim:float=0.9):  # noqa: ANN204
        """Find and fit a gaussian model to the shape of the plasma head.

        Inputs:
            lim : Limit of temperatures to mask to. Default 0.9
        """
        # if the limit is -ve raise error
        if lim <= 0:
            msg = f"Temperature limit cannot be 0! Received {lim}!"
            raise ValueError(msg)
        # store limit
        self._lim = lim
        # store flag indicating if it"s negative or not
        self._rel = (lim >=0.0) and (lim <= 1.0)
        # store last optimal values 
        self._opt = None
        # store covariance of fitted values
        self._cov = None
        # last frame
        self._source = None
        # draw
        self._draw = None
        # last frame
        self._frame = None
        # last mid point
        self._mid = None
        # last plasma peak
        self._plasmaPk = None
        # model mode
        self.__mode = "standard"
        # offset from right side
        self.__offset = None

    def getMode(self):
        return self.__mode

    # take the image and clip to hot values
    def _clip(self,frame: np.ndarray) -> np.ndarray:
        """Mask temperature to where the the largest hot object is and process coordinates so it"s relative to the right side of the image.
            
        The temperature limits where set when the class was created

        Inputs:
            D : Input temperature image

        Returns row and col coordinates
        """
        self._shape = frame.shape
        r,c = frame.shape
        # get the max temperature
        maxt = frame.max()
        # make blank frame
        self._thresh = np.zeros(frame.shape,dtype="uint8")
        # set pixels where the max temperature is above lim times max
        if self._rel:
            self._thresh[(self._lim*maxt) <= frame] = 255
        else:
            self._thresh[self._lim <= frame] = 255
        # if the threshold is all white then return early
        # this is usually when there is little going on in the frame so all the pixels are a similar temperature
        if (self._thresh==MAX_8_BIT).all():
            self._thresh[...] = 0
            return None,None
        # find contours around edge of the plasma
        ct = cv2.findContours(self._thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
        # if there are no contours
        if len(ct)==0:
            return None,None
        # choose the largest
        ct = max(ct,key=cv2.contourArea)
        self._fullct = ct
        # remove the points following the border of the image
        ii = np.where(ct[:,:,0]<(c-1))[0]
        self._ct = ct[ii,:,:]
        # return the coordinates as row (y) column (x)
        return self._ct[:,:,1].flatten(),self._ct[:,:,0].flatten()

    def draw(self,draw_ct:bool=False,draw_text:bool=False,draw_gauss:bool=False,draw_symmangle:bool=False,**kwargs) -> np.ndarray:  # noqa: ANN003, FBT001, FBT002
        """Draw the results on an image.

        The results stored from the previously run fit_guess are drawn on the
        thresholded temperature.

        The input flags control what is drawn.

        When draw_ct is True, then the filtered contour from the edge of the plasma is
        drawn. The colour of the line is set using ct_col.
        When draw_text is True, then a string of the fitted parameters is drawn.
        The colour of the text is set using text_col.
        When draw_gauss is True, then the filtered gaussian is drawn. The colour of the
        line is drawn using gauss_col.

        The default format of the draw text is:

            p= {height} {mean} and {std dev}

        A new formatting function can be passed via text_format keyword. Needs to accept
        the height, mean and std dev in that order.
        e.g txt = kwargs["text_format"](height,mean,std)

        The mean is the row location where the peak occurs
        The height is relative to the right side of the image

        Inputs:
            draw_ct : Flag to draw the edge contour around the gaussian. Default False.
            draw_text : Flag to write the gaussian prameters on the image. Default False.
            draw_gauss : Flag to draw the fitted Gaussian
            ct_col : BGR color used for drawing the found contour. Default (0,0,255).
            text_col : BGR color used for drawing the parameter text. Default (255,255,255).
            gauss_col : BGR color used for drawing the fitted gaussian. Default (255,0,0).
            buff : Buffer either side of samples used to plot gaussian curve. Default 10.

        Returns 3-channel image of the results drawn on the thresholded temperature
        """# create drawing canvas
        if kwargs.get("use_thresh",False):
            self._draw = np.dstack(3*(self._thresh,))
        else:
            self._draw = cv2.applyColorMap(self._frame,cv2.COLORMAP_HOT)
        # if no paramters then return just the threshold
        if self._opt is None:
            return self._draw
        # draw found contour
        if draw_ct:
            self._draw = cv2.polylines(self._draw,[self._ct],False,kwargs.get("ct_col",(0,0,255)),3)  # noqa: FBT003
        # draw text with the fitted parameters
        if draw_text and self._opt:
            if kwargs.get("text_format",None):
                txt = kwargs["text_format"](*self._opt)
            else:
                txt ="p="+" ".join([str(o) for o in self._opt])
            self._draw = cv2.putText(self._draw,txt,(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,kwargs.get("text_col",(255,255,255)),2,cv2.LINE_AA)
        # draw the gaussian
        if draw_gauss and self._opt:
            # create sampling vector to draw the distribution
            buff = kwargs.get("buff",10)
            x_axis = np.arange(self._ct[:,:,1].min()-buff,self._ct[:,:,1].max()+buff,1,dtype=np.int16)
            if self.__mode == "standard":
                g = norm.pdf(x_axis,loc=self._opt[1],scale=self._opt[2])
                # scale to 0- height as result is 0-1 scale
                # change to be relative to left side of image
                g = self._shape[1]-(self._opt[0]*((g-g.min())/(g.max()-g.min()))).astype("int32")
            elif self.__mode == "exp":
                g = PlasmaGaussian.fit_egauss(x_axis, *self._opt)
            elif self.__mode == "legacy" or kwargs.get("use_legacy", False):
                g = norm.pdf(x_axis,loc=self._opt[1],scale=self._opt[2])
                # scale to 0- height
                # change to be relative to left side of image
                g = self._shape[1]-(self.height*((g-g.min())/(g.max()-g.min()))).astype("int32")
            # combine into a set of coordinates
            gct = np.column_stack((g,x_axis)).reshape((-1,1,2))
            # draw as a blue line
            self._draw = cv2.polylines(self._draw,[gct],False,kwargs.get("gauss_col",(255,0,0)),2)  # noqa: FBT003
        # draw the line from peak to identified mid point
        if draw_symmangle and self._opt:
            ct = self._ct.squeeze()
            # find loc of peak of the contour
            # the peak is furthest from the right side or min x value
            pki = ct[:,0].flatten().argmin()
            # find the row where this occurs
            pk_row = ct[pki,1]
            pk_col = ct[pki,0]
            self.estSymmAngle()
            self._draw = cv2.line(self._draw,[pk_col,pk_row],[self._shape[1],int(self._mid)],kwargs.get("midpt_col",(255,0,255)),2)
        # return the the drawing
        return self._draw

    def estRMSE(self) -> float:
        """Calculate RMSE between the last fitted gaussian and contour.

        Calculated between height of contour relative to left side of image and drawn gaussian
        relative to left side of image

        If a gaussian has not been fitted, np.nan is returned

        Returns RMSE
        """
        # if there is no line fitted
        if not self._opt:
            return np.nan
        if self.__mode in ("standard", "model"):
            #g = norm.pdf(self._ct[:,:,1],loc=self._opt[1],scale=self._opt[2])
            g = PlasmaGaussian.gauss(self._ct[:,:,1],*self._opt)
        elif self.__mode == "exp":
            g = PlasmaGaussian.fit_egauss(self._ct[:,:,1], *self._opt)
        # scale to 0- height
        # change to be relative to left side of image
        #g = self._shape[1]-(self._opt[0]*((g-g.min())/(g.max()-g.min()))).astype("int32")
        # get "height" value of contour
        # this is already relative to the left side due to coordinate system
        gt = self._ct[:,:,0].flatten()

        # find RMSE between contour (gt) and (g)
        return np.sqrt(np.sum((g-gt)**2)/g.shape[0])

    def estMAE(self) -> float:
        """Calculate MAE between the last fitted gaussian and contour.

        Calculated between height of contour relative to left side of image and drawn gaussian
        relative to left side of image

        If a gaussian has not been fitted, np.nan is returned

        Returns MAE
        """
        if not self._opt:
            return np.nan
        if not self._opt:
            return np.nan
        if self.__mode in ("standard", "model"):
            #g = norm.pdf(self._ct[:,:,1],loc=self._opt[1],scale=self._opt[2])
            g = PlasmaGaussian.gauss(self._ct[:,:,1],)
        elif self.__mode == "exp":
            g = PlasmaGaussian.fit_egauss(self._ct[:,:,1], *self._opt)
            
        # this is already relative to the left side due to coordinate system
        gt = self._ct[:,:,0].flatten()
        return np.abs(g-gt).mean()

    def estCtArea(self):
        """Find the area of the full contour before trimming.

        Returns estimated contour area
        """
        if not self._opt:
            return np.nan
        return cv2.contourArea(self._fullct)


    def estSymmAngle(self,mid:int | None = None):
        """Estimate the angle between the peak of the contour and the mid point along row.

        An ideal symmetrical gaussian will have an angle of 0 as the the peak is inline
        with the mid point.

        The midpoint can be defined by the user (mid). If mid is None, The midpoint is the median
        of the row coordinates of the last contour. This can change a bit so the ability to specify
        a fixed point was added.

        The peak is defined as the contour point that has the smallest column (i.e. furthest away
        from the right side).

            tan(theta) = dy/dx -> theta = arctan(dy/dx)

        Inputs:
            mid : Row index used as reference for calculating the angle. If None, it is calculated from
                    the contour. Default None.

        Returns estimates angle in radians
        """
        if not self._opt:
            return np.nan
        # if there is only one point then ignore
        if self._ct.shape[0]==1:
            return np.nan
        ct = self._ct.squeeze()
        # find loc of peak of the contour
        # the peak is furthest from the right side or min x value
        pki = ct[:,0].flatten().argmin()
        # find the row where this occurs
        pk_row = ct[pki,1]
        pk_col = ct[pki,0]
        self._plasmaPk = [pk_row,pk_col]
        # find the median of the row values
        # the height is going to be the edge of the image
        # the sort is to ensure it is sorted
        if mid is None:
            mid = self._ct[:,:,1].min()+((self._ct[:,:,1].max()-self._ct[:,:,1].min())/2)
        self._mid = mid
        # find the angle as np.arctan(dy/dx) -> np.arctan(diff. in cols/diff. in rows)
        return np.arctan(abs(mid-pk_row)/(abs(pk_col-self._shape[1])))


    def estSkewKurtosis(self) -> list[float, float, float]:
        """Calculate the Skew, Kurtosis (Pearson) and Kurtosis (Fisher) and the last contour points as a gaussian distribution.

        Returns the skew, kurtosis (pearson) and kurtosis (fisher)
        """
        if not self._opt:
            return np.nan,np.nan,np.nan
        if self._ct.shape[0]==1:
            return np.nan,np.nan,np.nan
        return [skew(self._ct[:,:,0])[0],kurtosis(self._ct[:,:,0],fisher=True)[0],kurtosis(self._ct[:,:,0],fisher=False)[0]]


    def estAreaSplit(self,all_fill_value:float=5.0):
        """Split the contour area down the middle and find the ratio of area above and below the midline.

        Used as an metric of gaussian asymmetry.

        When the contour is a single data point or the area is 0, the value all_fill_value is returned.
        This is designed to be set to an absurd or distinct value that can be easily distinguished from normal
        ratios.

        Inputs:
            all_fill_value : Value to return when the contour is a single data point or the area is 1. Default 5.0

        Returns ratio of area above and below the midline
        """
        if not self._opt:
            return np.nan
        # if there is only one point then ignore
        if self._ct.shape[0]==1:
            return np.nan
        # find the median of the row values        # the sort is to ensure it is sorted
        mid = self._fullct[:,:,1].min()+((self._fullct[:,:,1].max()-self._fullct[:,:,1].min())/2)
        # split the contour to those ABOVE the mid point
        pt = self._fullct[self._fullct[:,:,1].flatten()>=mid,:,:]
        # find the area
        area = cv2.contourArea(pt)
        # if it"s a single or no data points then the area will be 0 raising an error
        if (area==0) or (pt.shape[0]<=1):
            return all_fill_value
        # split the contour to those BELOW the mid point
        pt = self._fullct[self._fullct[:,:,1].flatten()<=mid,:,:]
        bottom_area = cv2.contourArea(pt)
        if (bottom_area==0) or (pt.shape[0]<=1):
            return all_fill_value
        # return the ratio between area
        return area/bottom_area


    def fit_guess(self,frame: np.ndarray) -> list:
        """Assuming the plasma is a symmetrical gaussian distribution then the mean and std can be est from the coordinates without curve_fit.

        Inputs:
            D : Input temperature image
            show_debug : Draw an image showing some debugging features

        Return list containing height, mean and std of the distribution
        """
        # process frame to get coordintes representing sample of gaussian
        row,col = self._clip(frame)
        self._frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        # if clipping failed to find the plasma and returned None
        # set parameters to None and return
        if (row is None) or (col is None) or (row.shape[0] == 0) or (col.shape[0] == 0):
            self._opt = None
            self._ocov = None
            return self._opt
        # get shape of the image
        height,width = self._shape
        # peak of the gaussian is furthest from the right side of the image
        # so is the smallest x value
        xmii = np.argmin(col)
        # what row that occurs at is the mean
        self.mean = row[xmii]
        # the height of the curve relative to the right side of the image
        self.height = width-col[xmii]
        # shift height to be relative to the right side of the image
        col = width - col
        # calculate the standard deviation
        std = int(np.std(col))
        # set fitted values for mean and standard deivation
        #self._opt = [col.min(), height,mean,std]
        # get result without scaling factor
        pred = PlasmaGaussian.gauss(row, (width-col).min(), 1, self.mean, std)
        # calculate scaling factor by dividing this original by the result
        scale = col/pred
        self._opt = [col.min(), scale.mean() ,self.mean,std]
        return self._opt

    # function for gaussian model
    @staticmethod
    def gauss(x: np.ndarray, offset:float, amp:float, mean:float, sigma:float) -> np.ndarray:
        """Symmetrical Gaussian modelling function

        The adjustment needed to put it in the right location on the image is left up to the user.
        This is just a basic 

        Inputs:
            x : Input array
            amp : Scaling factor on output
            mean : X coordinate of distribution mean
            sigma : Std dev of the distribution

        Returns output of the model
        """
        return offset + amp/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    def fit_model(self, frame: np.ndarray, **kwargs) -> list:
        """Fit a symmetrical Gaussian to the data using curve_fit.

        Inputs:
            frame : Frame of temperature values to process
            sort_contour : Flag to identify the unqiue contour points and sort them in order of y-coordinate.
                            This was found to shorten fitting times and reduce error. Default True.
            method : Method used for fitting. See scipy.optimise.curve_fit. Default dogbox.
            maxfev : Maximum number of evaluations when fitting. See scipy.optimise.curve_fit. Default 1e3.

        Returns list of parameters found by curve fit
        """
        # process frame to get coordintes representing sample of gaussian (y,x)
        row,col = self._clip(frame)
        self._frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        # if clipping failed to find the plasma and returned None
        # set parameters to None and return
        if (row is None) or (col is None) or (row.shape[0] == 0) or (col.shape[0] == 0):
            self._opt = None
            self._ocov = None
            return self._opt
        # need at least 4 points to fit to 4 parameters
        if row.shape[0] < 4:
            self._opt = None
            self._ocov = None
            return self._opt

        # get shape of the image
        frame_height,frame_width = self._shape
        
        # only get unique coordinates
        # this is required to avoid it looping back on itself
        # data has to be positively increasing
        if kwargs.get("sort_contour", True):
            pts = []
            for yy,xx in zip(row,col):
                pts.append([xx,yy])
            pts_arr = np.array(pts, np.int32)
            pts_tup = [tuple(row) for row in pts_arr]
            coords_uq = np.unique(pts_tup,axis=0)
            coords_uq = np.array(coords_uq)
            # sort by y-coordinate
            coords_uq = coords_uq[coords_uq[:, 1].argsort()]
            # change the scale so it starts from 0
            coords_uq[:,0] = frame_width - coords_uq[:,0]
            #coords_uq[:,1] -= coords_uq[:,1].min()
            row = coords_uq[:,1]
            col = coords_uq[:,0]

        # find the row for the peak of the distribution
        xmii = np.argmin(col)
        # get the location of the mean
        mean = row[xmii]
        # the height of the curve
        self.height = frame_width - col[xmii]
        # calculate the standard deviation
        std = np.nanstd(frame_width-col)
        # offset, amp, mean, sigma
        # fit curve to the data
        # std dev has non zero lower limit to avoid div zero errors
        #print("p0", [0, 300, mean, std])
        #print("bounds", [(0,10,0,1),(3,500,frame_height,10)])
        popt,pcov=curve_fit(PlasmaGaussian.gauss, row, col,
                    #p0=[coords_uq[:,0].min(), 1e-6, mean, max(1e-6,std)], # initial guess based on data
                    p0 = [0, 300, mean, std],
                    bounds=[(0,10,0,1), # set bounds to stop it going too high
                            (3,500,frame_height,13)], 
                    method=kwargs.get("method","dogbox"), # method to use
                    maxfev=kwargs.get("maxfev",1e3)) # number of iterations
        self._opt = popt.tolist()
        return self._opt


    def fit_emodel(self, frame: np.ndarray, **kwargs) -> list:
        """Fit an exponential modulated Gaussian to the data using curve_fit.

        Inputs:
            frame : Frame of temperature values to process
            **kwargs : See scipy.curve_fit
        """
        # process frame to get coordintes representing sample of gaussian
        y,x = self._clip(frame)
        self._frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        # if clipping failed to find the plasma and returned None
        # set parameters to None and return
        if (y is None) or (x is None) or (y.shape[0] == 0) or (x.shape[0] == 0):
            self._opt = None
            self._ocov = None
            return self._opt
        # get shape of the image
        r,c = self._shape
        # fit curve to the data
        popt,pcov=(curve_fit(PlasmaGaussian.fit_egauss, x, y, **kwargs))
        self._opt = popt.tolist()
        return self._opt


    @staticmethod
    def fit_egauss(x: np.ndarray, scale: float, lbda: float, std_dev: float, mean: float) -> np.ndarray:
        """Apply exponentially modulated Gaussian using x-coordinates.

        Inputs:
            x : Array of x-coordinates
            scale : Scaling parameter
            lbda : Lambda parameter
            std_dev : Standard deviation
            mean : Mean location of distribution  
        """
        import scipy.special as sse
        # from https://stackoverflow.com/q/36900920
        # exponential gaussian
        return scale*(0.5*lbda*np.exp(0.5*lbda*(2*mean+lbda*std_dev*std_dev-2*x))*sse.erfc((mean+lbda*std_dev*std_dev-x)/(np.sqrt(2)*std_dev)))


# generic method to use with CoaxialPlane.frame
def calcFrameMetric(frame: np.ndarray,use_metric,segs:int) -> np.ndarray:
    r,c = frame.shape
    metric = np.zeros((r,c),dtype="float32")
    # iterate over rows in jumps
    for ri in range(r//segs):
        # iterate over cols in jumps
        for ci in range(c//segs):
            # fill area in matrix with entropy of temperature region
            metric[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)] = use_metric(frame[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)])
    return metric


class CoaxialPlane:
    """Estimate the coaxial plane gradient of the powder distribution by fitting a polynomial to the entropy of the temperature image.

    The polynomial is fitted by finding regions of the image where the entropy is noticeably high. The size of the chunks checked
    is controlled by nsegs parameter. The integer size must be a factor of the image width and height. 29 is the default as
    it fits into both dimensions.

    When the threshold flag is True, the entropy is converted to grayscale (0-255) and thresholded using Otsu. This helps separate
    the low and high end entropy regions. The locations of white pixels in the thresholded mask are used to fit the polynomial.
    When the threshold flag is False, the entropy is normalized (0-1) and the non-zero values are passed as weights to the polynomial fitting.
    Both tend to produce similar results.

    When the trim flag is True, the found points are reduced to just those along the edges of the segments where entropy is calculated.
    This is to speed up the operation by drastically reducing the number of points used.

    The line_lims parameter controls the range of values the fitted line is drawn over.
    Supported values:
        0 : Range is over the columns of thresholded sections
        1 : Range is over the entire width of the image
        2 : Range is over the entire width with the right most point being on the mid point
    """
    
    def __init__(self,k:int=1,nsegs:int=29,trim:bool=True,line_lims:int=0,threshold:bool=False,**kwargs) -> None:  # noqa: ANN003, FBT001, FBT002
        """Estimate the coaxial plane gradient of the powder distribution by fitting a polynomial to the entropy of the temperature image.

        Inputs:
        k : Order of fitting
        nsegs : Size of the segments to check along each axis. Default 29
        trim : Trim the number of points used to fit the lines to those on the boundaries of the segments.
        line_trims : Integer controlling the range of x values to draw the fitted line over.
        threshold : Flag to threshold the entropy regions via image processing. If False, the normalized
                    entropy is used as fitting weights.
        min_segs : Min number of segments after thresholding (threshold=True) for a fitting to occur
        """
        # order of fitting
        self._k = k
        # size of segments along each axis
        self._segs = nsegs
        self._trim = trim
        self.gray = None
        self._last = []
        self._exp = []
        self._rw = []
        self._cw = []
        self._shape = ()
        self._source = None
        self._thresh = threshold
        self._threshMask = None
        self._w = None
        # angle
        self._theta = 0.0
        if line_lims not in [0,1,2]:
            msg = f"Unsupported line limit mode {line_lims}!"
            raise ValueError(msg)
        self._llims = line_lims
        self._minsegs = kwargs.get("min_segs",0)

    # function for checking that the number of segments is a factor of the image size
    def _checkInput(self,frame: np.ndarray):
        """Check input against class settings.

        Conditions:
            - Matrix must not be empty (e.g. 0x4)
            - The set number of segments must fit inside the rows and columns

        If none of the conditions are satisfied, an exception is raised
        """
        r,c = frame.shape
        # check for empty matrix
        if (r==0) or (c==0):
            msg = f"Frame cannot be empty! Frame shape {r},{c}"
            raise ValueError(msg)
        # check if it fits in the height/rows
        if (r%self._segs)!=0:
            msg = f"Frame does not equally break up into {self._segs} segments along row axis!"
            raise ValueError(msg)
        # check if it fits in the width/columns
        if (c%self._segs)!=0:
            msg = f"Frame does not equally break up into {self._segs} segments along column axis!"
            raise ValueError(msg)
        # store image shape which is used later
        self._shape = [r,c]

    # break the frame into chunks and calculate entropy
    def frameEntropy(self,frame: np.ndarray) -> np.ndarray:
        """Calculate entropy of given temperature frame using class settings.

        The entropy was calculated from the counts of unique temperature values in each region of the frame

        # REG is the region of the image currently being analysed
        uq,cts = np.unique(reg,return_counts=True)
        reg_entr = scipy.stats.entropy(reg,base=None)

        The entropy is calculated using Shannon entropy is calculated as H = -sum(pk * log(pk))

        This creates an image of entropy where regions of high activity such as areas where the powder travels have high values
        and areas of low activity like the background or the plasma have low values

        Inputs:
            frame : Temperature frame

        Returns float32 array of entropy values
        """
        r,c = self._shape
        # create empty matrix to update
        ent = np.zeros(self._shape,dtype="float32")
        # iterate over rows in jumps
        for ri,ci in [(ri,ci) for ci in range(c//self._segs) for ri in range(r//self._segs)]:
            ent[self._segs*ri:self._segs*(ri+1),self._segs*ci:self._segs*(ci+1)] = entropy(np.unique(frame[self._segs*ri:self._segs*(ri+1),self._segs*ci:self._segs*(ci+1)],return_counts=True)[1],base=None)
        return ent


    def frameTempStd(self,frame: np.ndarray) -> np.ndarray:
        """Calculate standard deviation of given temperature frame using class settings.

        Inputs:
            frame : Temperature frame

        Returns float32 array of entropy values
        """
        r,c = frame.shape
        metric = np.zeros((r,c),dtype="float32")
        # iterate over rows in jumps
        for ri in range(r//self._segs):
            # iterate over cols in jumps
            for ci in range(c//self._segs):
                # fill area in matrix with entropy of temperature region
                metric[self._segs*ri:self._segs*(ri+1),self._segs*ci:self._segs*(ci+1)] = np.std(frame[self._segs*ri:self._segs*(ri+1),self._segs*ci:self._segs*(ci+1)])
        return metric


    def drawLastLine(self,show_marks:bool=False,on_source:bool=False,**kwargs) -> np.ndarray:  # noqa: ANN003, FBT001, FBT002
        """Draw last fitted line on an image.

        If on_source is False, then the last grayscale entropy matrix is drawn on.
        If on_source is True, then the last temperature frame is colormapped and drawn on.

        The show_marks flag, draws crosses on the points used to fit the line.

        Inputs:
            show_marks : Flag to draw crosses on the pixels used to fit the line
            on_source : Draw the line on the last input colormapped.
            line_col : Colour of the drawn line as a 3-element tuple. Default (255,0,0).
            line_width : Width of the drawn line. Default 3.

        Returns drawn image
        """#print(self._last)
        # make an image to draw the line on
        # either the previous source data colormapped
        if on_source:
            if self._source is None:
                msg = "No prior source! Need to call estPlane"
                raise ValueError(msg)
            gray = CoaxialPlane.frame2gray(self._source)
            draw = cv2.applyColorMap(gray.copy(),cv2.COLORMAP_HOT)
        # or the grayscaled entropy mask
        else:
            if self.gray is None:
                msg = "No previous entropy grayscale!"
                raise ValueError(msg)
            gray = CoaxialPlane.frame2gray(self.gray)
            draw = np.dstack(3*(gray,))
        # if a line hasn"t been fitted or failed to fit
        # return the the drawing unchanged
        if (self._last is None) or (len(self._last) ==0):
            return draw
        # if the user wants the coordinates to be used in the fitting marked
        if show_marks:
            # form into coordinate pairs
            coords = list(zip(self._cw,self._rw))
            # mark with green crosses
            for cc in coords:
                cv2.drawMarker(draw,cc,(0,255,0),cv2.MARKER_CROSS,5,3)
        # create a poly using fitted parameters
        p = np.poly1d(self._last)
        # create X coordinates covering the x coordinates of fitted points
        if self._llims == 0:
            px = np.arange(self._cw.min(),self._cw.max(),1,dtype="int16")
        # create X coordinates coverinng the width of the image
        elif self._llims == 1:
            px = np.arange(0,self._shape[1],1,dtype="int16")
        # create X coordinates covering the width of the image minus 1
        # the last point is fitted to the middle of the image/
        elif self._llims == 2:
            px = np.arange(0,self._shape[1]-1,1,dtype="int16")
        # evaluate values
        py = p(px)
        # form into coordinate pairs
        coords = [(int(cc),int(rr)) for cc,rr in zip(px,py)]
        # iterate over coordinates drawing a blue line between points
        for pleft,pright in zip(coords,coords[1:]):
            draw = cv2.line(draw,pleft,pright,kwargs.get("line_col",(255,0,0)),kwargs.get("line_width",3))
        # if line mode is 2 when draw the final point at the middle row of the image on the right hand side
        if self._llims == 2:
            draw = cv2.line(draw,pright,(self._shape[1]-1,(self._shape[0]//2)-1),(255,0,0),3)
        return draw


    def estTheta(self,offset:float=-np.pi/2) -> float:
        """Estimate the angle of the coaxial plane from the fitted line.

            The angle is estimated by taking the first and last points of the curve, treating
            them as vertices of a triangle and using arctan to find the angle between them.

            This uses the last fitted line.

            The purpose of offset is to change the reference point. By default a horizontal
            line reads as approx pi/2. By applying an offset of -pi/2, a horizontal line
            has an angle of 0.

            Inputs:
                offset : Angle offset added from the calculated angle in radians. Default 0

            Returns estimated coaxial plane angle in raidans
        """
        if len(self._last)==0:
            msg = "Unable to find theta! Line has not been fitted"
            raise ValueError(msg)
        # create a poly using fitted parameters
        p = np.poly1d(self._last)
        # create X coordinates covering the x coordinates of fitted points
        if self._llims == 0:
            px = np.arange(self._cw.min(),self._cw.max(),1,dtype="int16")
        # create X coordinates coverinng the width of the image
        elif self._llims == 1:
            px = np.arange(0,self._shape[1],1,dtype="int16")
        # create X coordinates covering the width of the image minus 1
        # the last point is fitted to the middle of the image/
        elif self._llims == 2:
            px = np.arange(0,self._shape[1]-1,1,dtype="int16")
        # evaluate values
        py = p(px)
        # form into coordinate pairs
        coords = [(int(cc),int(rr)) for cc,rr in zip(px,py)]
        # use tan to find angle
        dc = abs(coords[-1][0]-coords[0][0])
        dr = abs(coords[-1][1]-coords[0][1])
        # to capture divide by zero error
        if (dc==0) or (dr==0):
            self._theta = 0.0
            return self._theta
        self._theta = np.arctan(dc/dr)
        return self._theta+offset

    @staticmethod
    def frame2gray(frame: np.ndarray) -> np.ndarray:
        """Convert given matrix to grayscale."""
        return (255*((frame-frame.min())/abs(frame.max()-frame.min()))).astype("uint8")

    def estPlane(self,frame: np.ndarray) -> np.ndarray:
        """Fit a line across the high entropy regions of the image.

        The entropy is calculated across regions of the image according
        to settings in the constructor. The pixels of high entropy
        are treated as samples of a polynomial where the input is the
        x position and the output is the y position.

        The poly is fitted using numpy"s polyfit.

        The resulting coefficients are returned in decreasing orderes of power.

        Inputs:
            frame : Input data matrix to process

        Returns an array of polynomial coefficients in decreasing order of power.
        """
        # check size of input to see if it could be broken up into equal blocks
        self._checkInput(frame)
        self._source = frame.copy()
        if self.gray is None:
            self.gray = np.zeros(self._shape,frame.dtype)
        # reset parameters
        self._last = [None for _ in range(self._k)]
        # calculate entropy
        ent = self.frameEntropy(frame)
        # if entropy is all zero
        # return the None
        if not ent.any():
            self._last = None
            return self._last
        # if there is some activity
        # convert entropy to grayscale
        self.gray = CoaxialPlane.frame2gray(ent)
        # if thresholding the grayscale entropy
        if self._thresh:
            _,self._threshMask = cv2.threshold(self.gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # find the row and columns pixels are 255
            self._rw,self._cw = np.where(self._threshMask==MAX_8_BIT)
        # if using the entropy as weights
        else:
            # convert gray to 0-1
            norm = self.gray.astype("float32")/255.0
            r,c = self.gray.shape
            self._threshMask = np.zeros((r,c),dtype=np.uint8)
            self._threshMask[norm>0] = 255
            # find where norm is greater than 0
            self._rw,self._cw = np.where(norm>0)
            # get the corresponding weight values
            self._w = norm[self._rw,self._cw].flatten()
        # divide the number of identified points by the area of a segment (segment^2) to estimate the number of segments identified
        # kind of equivalent number of segments
        if (self._rw.shape[0]/(self._segs**2)) < self._minsegs:
            self._last = None
            return self._last
        # if there are masked regions
        if self._rw.shape[0]!=0:
            # if the user wants the coordinates trimmed to speed up time
            if self._trim:
                self._w = self._w[(self._cw%self._segs)==0]
                self._rw = self._rw[(self._rw%self._segs)==0]
                self._cw = self._cw[(self._cw%self._segs)==0]
            # taking the coordinates as points on a line
            # fit a line to the data
            self._last = np.polyfit(self._cw,self._rw,self._k,w=self._w)
        # return the fitted parameters
        return self._last

    def estPlaneUsing(self,frame: np.ndarray,use_metric):
        """Fit a line across the high entropy regions of the image.

        The entropy is calculated across regions of the image according
        to settings in the constructor. The pixels of high entropy
        are treated as samples of a polynomial where the input is the
        x position and the output is the y position.

        The poly is fitted using numpy"s polyfit.

        The resulting coefficients are returned in decreasing orderes of power.

        Inputs:
            D : Input data matrix to process
            use_metric : Process the frame using the specified function instead of
                        frame entropy

        Returns an array of polynomial coefficients in decreasing order of power.
        """# check size of input to see if it could be broken up into equal blocks
        self._checkInput(frame)
        self._source = frame
        if self.gray is None:
            self.gray = np.zeros(self._shape,frame.dtype)
        # reset parameters
        self._last = [None for _ in range(self._k)]
        # calculate metric
        ent = use_metric(frame,self._segs)
        # if entropy is all zero
        # return the None
        if not ent.any():
            self._last = None
            return self._last
        # if there is some activity
        # convert entropy to grayscale
        self.gray = CoaxialPlane.frame2gray(ent)
        # if thresholding the grayscale entropy
        if self._thresh:
            _,self._threshMask = cv2.threshold(self.gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # find which pixels are 255
            self._rw,self._cw = np.where(self._threshMask==MAX_8_BIT)
        # if using the entropy as weights
        else:
            # convert gray to 0-1
            norm = self.gray.astype("float32")/255
            r,c = self.gray.shape
            self._threshMask = np.zeros((r,c),dtype=np.uint8)
            self._threshMask[norm>0] = 255
            # find where norm is greater than 0
            self._rw,self._cw = np.where(norm>0)
            # get the corresponding weight values
            self._w = norm[self._rw,self._cw].flatten()
        # if the data is less than min length
        # set to None and return None
        if (self._rw.shape[0]/self._segs**2) < self._minsegs:
            self._last = None
            return self._last
        # if there are masked regions
        if self._rw.shape[0]!=0:
            # if the user wants the coordinates trimmed to speed up time
            if self._trim:
                self._w = self._w[(self._cw%self._segs)==0]
                self._rw = self._rw[(self._rw%self._segs)==0]
                self._cw = self._cw[(self._cw%self._segs)==0]
            # taking the coordinates as points on a line
            # fit a line to the data
            self._last = np.polyfit(self._cw,self._rw,self._k,w=self._w)

        # return the fitted parameters
        return self._last

    def getTempStats(self):
        """Calculate statistics about the masked area used to calculate coaxial plane.

        estPlane must have been called recently to set the appropriate variables

        Current statistics in order:
            - Min Temperature
            - Max Temperature
            - Mean Temperature
            - Std Dev Temperature
            - Variance Temperature
            - Entropy
            - Contour Width
            - Contour Height
            - Conour Aspect Ratio (W/H)

        Returns statistics as a list in the specified order
        """
        from scipy.stats import entropy
        if self._last is None:
            return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        if self._last is None:
            return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        # get the data from the mask
        thresh_data = self._source[self._threshMask==MAX_8_BIT]
        # find the outlining mask
        cts,_ = cv2.findContours(self._threshMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # find the largest contour
        ct = max(cts,key=lambda x : cv2.contourArea(x))
        # find the bounding rectangle around the contour to get the width and height
        _,_,w,h = cv2.boundingRect(ct)
        # find the statistics and return
        return [thresh_data.min(),thresh_data.max(),thresh_data.mean(),thresh_data.std(),thresh_data.var(),entropy(np.unique(thresh_data,return_counts=True)[1],base=None),w,h,w/h]


    @staticmethod
    def _scaledSine(x: np.ndarray,scale:float,freq:float,offset:float) -> np.ndarray:
        return (scale*x*np.sin(x*(freq/2*np.pi)))+offset


    def estPlaneExperimental(self,frame: np.ndarray):
        """Slightly different version of estPlane that attempts to fit a scaled sine wave to the entropy values using scipy.optimize.curve_fit function.

        Very different results compared to the original.

        Fitted parameters are stored under a different variable and do not affect the stable
        fitting.
        """
        from scipy.optimize import curve_fit
        # check size of input to see if it could be broken up into equal blocks
        self._checkInput(frame)
        self._source = frame.copy()
        if self.gray is None:
            self.gray = np.zeros(self._shape,frame.dtype)
        # reset parameters
        self._last = [None for _ in range(self._k)]
        # calculate entropy
        ent = self.frameEntropy(frame)
        # if entropy is all zero
        # return the None
        if not ent.any():
            self._last = None
            return self._last
        # if there is some activity
        # convert entropy to grayscale
        self.gray = CoaxialPlane.frame2gray(ent)
        # if thresholding the grayscale entropy
        if self._thresh:
            _,thresh = cv2.threshold(self.gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # find which pixels are 255
            self._rw,self._cw = np.where(thresh==MAX_8_BIT)
        # if using the entropy as weights
        else:
            # convert gray to 0-1
            norm = self.gray.astype("float32")/255
            # find where norm is greater than 0
            self._rw,self._cw = np.where(norm>0)
            # get the corresponding weight values
            self._w = norm[self._rw,self._cw].flatten()
        # if the data is less than min length
        # set to None and return None
        if (self._rw.shape[0]/self._segs**2) < self._minsegs:
            self._exp = None
            return self._exp
        # if there are masked regions
        if self._rw.shape[0]!=0:
            # if the user wants the coordinates trimmed to speed up time
            if self._trim:
                self._w = self._w[(self._cw%self._segs)==0]
                self._rw = self._rw[(self._rw%self._segs)==0]
                self._cw = self._cw[(self._cw%self._segs)==0]
            # fit curve
            # curve_fit doesn"t take weights
            self._exp,_ = curve_fit(CoaxialPlane._scaledSine,self._cw,self._rw,bounds=((1e-5,1,0),(2,464,348)),maxfev=int(10e3))
        # return the fitted parameters
        return self._exp

    def drawLastExperimental(self,show_marks:bool=False,on_source:bool=False):  # noqa: FBT001, FBT002
        # make an image to draw the line on
        # either the previous source data colormapped
        if on_source:
            if self._source is None:
                msg = "No prior source!"
                raise ValueError(msg)
            gray = CoaxialPlane.frame2gray(self._source)
            draw = cv2.applyColorMap(gray,cv2.COLORMAP_HOT)
        # or the grayscaled entropy mask
        else:
            if self.gray is None:
                msg = "No previous entropy grayscale!"
                raise ValueError(msg)
            draw = np.dstack(3*(self.gray,))
        # if a line hasn"t been fitted or failed to fit
        # return the the drawing unchanged
        if (self._exp is None) or (len(self._exp) ==0):
            return draw
        # if the user wants the coordinates to be used in the fitting marked
        if show_marks:
            # form into coordinate pairs
            coords = list(zip(self._cw,self._rw))
            # mark with green crosses
            for cc in coords:
                cv2.drawMarker(draw,cc,(0,255,0),cv2.MARKER_CROSS,5,3)
        # create X coordinates covering the x coordinates of fitted points
        if self._llims == 0:
            px = np.arange(self._cw.min(),self._cw.max(),1,dtype="int16")
        # create X coordinates coverinng the width of the image
        elif self._llims == 1:
            px = np.arange(0,self._shape[1],1,dtype="int16")
        # create X coordinates covering the width of the image minus 1
        # the last point is fitted to the middle of the image/
        elif self._llims == 2:
            px = np.arange(0,self._shape[1]-1,1,dtype="int16")
        # evaluate values
        py = CoaxialPlane._scaledSine(px,*self._exp)
        # form into coordinate pairs
        coords = [(int(cc),int(rr)) for cc,rr in zip(px,py)]
        # iterate over coordinates drawing a blue line between points
        for cA,cB in zip(coords,coords[1:]):
            draw = cv2.line(draw,cA,cB,(255,0,0),3)
        # if line mode is 2 when draw the final point at the middle row of the image on the right hand side
        if self._llims == 2:
            draw = cv2.line(draw,cB,(self._shape[1]-1,(self._shape[0]//2)-1),(255,0,0),3)
        return draw


def clipMaskPlasma(frame: np.ndarray,thresh: float = 1300,fill: str = "min") -> np.ndarray:
    """Iterate from right to left of the frame and fill columns until the max temperature is below the target threshold.

    The fill parameter can either be a float, int or a supported string.
    Supported strings:
        min : Initial frame min
        max : Initial frame max

    The goal of the fill value is to replace the plasma with a value that can be easily identified later on.

    Inputs:
        frame : Floating point temperature array
        thresh : Temperature threshold
        fill : Fill value for each column
    """
    # if the frame max temperature is already below the threshold
    # return the the frame is
    if frame.max()<thresh:
        # set fill value
        if isinstance(fill,(int,float)):
            frame[...] = fill
        elif fill == "min":
            frame[...] = frame.min()
        elif fill == "max":
            frame[...] = frame.max()
        else:
            msg = f"Unsupported fill value {fill}! Has to be either a float, int or supported string"
            raise ValueError(msg)
        return frame
    # column index starting from the far right column
    i=-1
    # while the max temperature is above the thereshold
    # and the index isn"t out of bounds
    while frame.max()>=thresh and (abs(i)<frame.shape[1]):
        # set fill value
        if isinstance(fill,(int,float)):
            frame[:,i] = fill
        elif fill == "min":
            frame[:,i] = frame.min()
        elif fill == "max":
            frame[:,i] = frame.max()
        else:
            msg = f"Unsupported fill value {fill}! Has to be either a float, int or supported string"
            raise ValueError(msg)
        # move one column to the left
        i -= 1
    return frame


def coaxialPlaneSlider(fn:str,k:int=2,filt=clipMaskPlasma,**kwargs):  # noqa: ANN003
    """Fit coaxial plane to the data at a fixed order.

    The user controls a slider to control which frame is being processed

    When filt is a list/tuple, the functions are executed un order and passed to the next
    function

    Inputs:
        fn : File path to NPZ
        k : Poly order
        filt : Single function or list/tuple of functions that pre-process the data before
                fitting coaxial plane.
        min_segs : Minimum number of segments. Default 8.
    """
    data = np.load(fn)["arr_0"]
    nf,r,c = data.shape
    cp = CoaxialPlane(k=k,trim=False,line_lims=1,min_segs=kwargs.get("min_segs",8),adjust_min_segs=kwargs.get("adjust_min_segs", False), threshold=True)
    cv2.namedWindow("CP")
    if filt:
        cv2.namedWindow("Filtered")

    def on_changed(val) -> None:
        ni = int(cv2.getTrackbarPos("Frame","CP"))
        frame = data[ni,:,:]
        source = CoaxialPlane.frame2gray(frame)
        source = cv2.applyColorMap(source,cv2.COLORMAP_HOT)
        # if there is a filtering function
        if filt:
            # if the filtering function is an iterable collection
            if isinstance(filt,(list,tuple)):
                for f in filt:
                    frame = f(frame)
            # if it fails try applying it as a single function
            else:
                frame = filt(frame)
            # colourmap the filtered frame
            norm = frame2gray(frame)
            proc = cv2.applyColorMap(norm,cv2.COLORMAP_HOT)
            # stack them together
            combo = np.hstack((source,proc))
            cv2.imshow("Filtered",combo)
        # fit the line and draw the result
        cp.estPlane(frame)
        draw = cp.drawLastLine(on_source=True,line_col=(76,255,0),line_width=5)
        draw = np.hstack((source,np.dstack(3*(frame2gray(cp.gray),)),draw))
        cv2.imshow("CP",draw)
    cv2.createTrackbar("Frame","CP",0,nf-1,on_changed)
    on_changed(0)
    cv2.waitKey(0)


def coaxialPlaneParams(fn:str,k:int=2,filt=None,**kwargs):
    """Plot the coaxial plane parameters for the target parameter.

    When filt is a list/tuple, the functions are executed un order and passed to the next
    function

    Inputs:
        fn : File path to NPZ
        k : Poly order
        filt : Single function or list/tuple of functions that pre-process the data before
                fitting coaxial plane.
        theta_offset : Offset given to CoaxialParams.estTheta method. Default -np.pi/2.
        no_plot : Flag to not plot. Returned figures will be None, Default False
        min_segs : Minimum number of segments. Default 8.
        adjust_min_segs : Update min_segs

    Returns a matplotlib figure of the plotted coefficient, plot of coaxial angle and dataframe of coeffs and stats collected
    """
    # load data
    data = np.load(fn)["arr_0"]
    # get max temp
    maxt = data.max(axis=(1,2))
    # get filename
    fname = Path(fn).stem
    # get size of the data
    nf,r,c = data.shape
    # setup class using the wanted settings
    cp = CoaxialPlane(k=k,trim=False,line_lims=1,min_segs=kwargs.get("min_segs",8),threshold=False)
    ## lists of metrics
    hist = []
    theta = []
    temp_stats = []
    temp_stats_cols = ["Min Temperature (deg C)","Max Temperature (deg C)","Mean Temperature (deg C)","Std Dev Temperature (deg C)","Variance Temperature (deg C)","Entropy","Width (pix)","Height (pix)","Aspect Ratio (Width/Height)"]
    index = []
    time = []
    # iterate over each frame
    for ni in range(nf):
        frame = data[ni,:,:]
        # filter if given
        if isinstance(filt,(list,tuple)):
            for f in filt:
                frame = f(frame)
        elif filt is not None:
            frame = filt(frame)
        #get parameters
        pp = cp.estPlane(frame)
        temp_stats.append(cp.getTempStats())
        # if it didn't fit
        # then add a series of -1s
        if pp is None:
            hist.append((k+1)*[0])
            theta.append(0)
        else:
            hist.append(pp)
            theta.append(cp.estTheta(kwargs.get("theta_offset",-np.pi/2)))
        index.append(ni)
        time.append(ni/30)
    # form into array for easier indexing
    hist = np.vstack(hist)
    temp_stats = np.vstack(temp_stats)
    # make the columns for the poly coefficients
    poly_cols = [f"x^{i}" if i>0 else "offset" for i in range(hist.shape[1]-1,-1,-1)]
    coaxialdf = pd.DataFrame({"Frame Index":index,"Time (s)":time} |
                      {c : temp_stats[:,i] for i,c in zip(range(temp_stats.shape[1]),temp_stats_cols)} |
                      {c : hist[:,i] for i,c in zip(range(hist.shape[1]),poly_cols)} |
                      {"Theta (rads)" : theta, "Theta (degrees)" : np.rad2deg(theta)})
    if not kwargs.get("no_plot",False):
        # create axis
        ww,hh = plt.rcParams["figure.figsize"]
        f,ax = plt.subplots(nrows=hist.shape[1]+1,sharex=True,constrained_layout=True,figsize=[ww,hh*1.5*k])
        x = np.arange(nf)
        ax[0].plot(x,maxt)
        ax[0].set(xlabel="Frame Number",ylabel="Max Frame Temperature (C)",title="Max Frame Temperature")
        # suffix for the first three numbers
        first_suf = ["st","nd","rd"]
        # iterate over coefficients
        for pp,aa in zip(range(hist.shape[1]),ax[1:]):
            # get suffix
            suff = first_suf[pp] if pp in [0, 1, 2] else "th"
            param = hist[:,pp]
            # plot data
            aa.plot(x,param,"b-")
            # set labels
            aa.set(xlabel="Frame Number",ylabel="Parameter Value",title=f"{pp+1}{suff} Order Coefficient")
        f.suptitle(f"{fname} Coaxial Plane k={k}")
        f.savefig(f"{Path(fn).stem}-coaxial-plane-params-k{k}.png")

        ft,axt = plt.subplots(constrained_layout=True,figsize=[ww,hh])
        axt.plot(x,theta)
        axt.set(xlabel="Frame Number",ylabel="Theta (rads)",title="Angle of Powder Travel (rads)")
        ft.savefig(f"{Path(fn).stem}-coaxial-theta.png")
        return f,ft,coaxialdf
    return None,None,coaxialdf


def plasmaGaussianSlider(path:str,padding:int=10,**kwargs):  # noqa: ANN003
    """Scroll through a file using a slider applying PlasmaGaussian.fit_guess to each frame and drawing the results.

    The input padding controls the amount of white space between the source and results.
    Makes it easier to screenshot. If padding is True, then a pad of 10 pizels is used

    Inputs:
        path : Path to input NPZ temperature file
        padding : Integer width of whitespace padding
    """
    # load the data file
    data = np.load(path)["arr_0"] if isinstance(path, str) else path
    nf,r,c = data.shape
    # make the image to draw on according
    if isinstance(padding,bool) and padding:
        pad = 255*np.ones((r,10,3),np.uint8)
    elif isinstance(padding,int):
        pad = 255*np.ones((r,padding,3),np.uint8)
    # make a named window to update
    cv2.namedWindow("Gaussian")
    # make class for fitting
    gf = PlasmaGaussian(kwargs.get("lim",0.9))
    # trackbar callback function
    def on_changed(val) -> None:
        # get target frame
        ni = int(cv2.getTrackbarPos("Frame","Gaussian"))
        frame = data[ni,:,:]
        gf.fit_guess(frame)
        gray = frame2gray(frame)
        cv2.imshow("Gaussian",np.hstack([cv2.applyColorMap(gray,cv2.COLORMAP_HOT),pad,gf.draw(True,False,False,False,**kwargs),pad,gf.draw(False,True,True,True,**kwargs)]))

    cv2.createTrackbar("Frame","Gaussian",0,nf-1,on_changed)
    on_changed(0)
    cv2.waitKey(0)


def plasmaGaussianParams(fn:str,transpose:int=2,**kwargs):
    """Load a temperature NPZ file and find the gaussian parameters from each frame then plot it.

    Creates a PlasmaGaussian class and applies it to each frame. If the opts is None,
    then 0s are added in it"s place. The results are then plotted on a series of graphs.
    The graphs are max temperature, height and standard deviation. The max temperature
    plot is to provide some context to the activity happening in each image,.

    The transpose flag controls whether the plots are horizontal or vertical.
    When transpose is:
        - 0 : Plots are stacked as rows
        - 1 : Plots are stacked as cols
        - ELSE : 2x2 array

    Inputs:
        fn : Path to temperature NPZ file
        transpose : Flag to transpose the plots to be horizontal rather than vertical. Default 2.
        filt_hist : Function to filter the history.
        xfsize : X-axis tick font size
        yfsize : Y-axis tick font size
        title_size : Figure title size
        fill_value : Value used when it fails to find the plasma. Default 0
        rmse_fill_value : Value used for RMSE when it fails to find plasma. Different from fill value as 0 RMSE implies perfect fitting. Default 0

    Returns the matplotlib figure
    """
    # load data
    data = np.load(fn)["arr_0"]
    # get max temp
    maxt = data.max(axis=(1,2))
    # get filename
    fname = Path(fn).stem
    # get size of the data
    nf,r,c = data.shape
    # setup class using the wanted settings
    cp = PlasmaGaussian(lim=kwargs.get("lim",0.9))
    hist = []
    rmse = []
    symmangle = []
    area_ratio = []
    skew = []
    kfish = []
    kurt = []
    index = []
    time = []
    area = []
    mid_row = []
    gauss_peak = []
    # iterate over each frame
    for ni in range(nf):
        time.append(ni/30)
        index.append(ni)
        frame = data[ni,:,:]
        opts = cp.fit_guess(frame)
        # if it didn't fit
        # then add a series of 0s
        if opts is None:
            hist.append(3*[kwargs.get("fill_value",0)])
            rmse.append(kwargs.get("rmse_fill_value",0))
            symmangle.append(kwargs.get("fill_value",0))
            area_ratio.append(kwargs.get("fill_value",0))
            skew.append(kwargs.get("fill_value",0))
            kfish.append(kwargs.get("fill_value",0))
            kurt.append(kwargs.get("fill_value",0))
            area.append(kwargs.get("fill_value",0))
            mid_row.append(kwargs.get("fill_value",0))
            gauss_peak.append([kwargs.get("fill_value",0),kwargs.get("fill_value",0)])
        else:
            hist.append(opts)
            rmse.append(cp.estRMSE())
            symmangle.append(cp.estSymmAngle(kwargs.get("angle_ref",169)))
            mid_row.append(cp._mid)
            area_ratio.append(cp.estAreaSplit())
            sk,kf,kk = cp.estSkewKurtosis()
            skew.append(sk)
            kfish.append(kf)
            kurt.append(kk)
            area.append(cp.estCtArea())
            gauss_peak.append(cp._plasmaPk)

    # form into array for easier indexing
    hist = np.vstack(hist)
    if "filt_hist" in kwargs:
        hist = kwargs.get("filt_hist")(hist)
        # stack the data into a giant array
    gauss_df = pd.DataFrame({"Frame Index":index,"Time (s)":time,"Height (pixels)":hist[:,0],"Mean (pixels)":hist[:,1],"Std Dev (pixels)":hist[:,2],
                       "RMSE":rmse,"Asymmetrical Angle (rads)":symmangle,"Skew":skew,"Kurtosis (Fisher)":kfish,"Kurtosis (Pearson)":kurt,"Contour Area (pix^2)":area,
                       "Midpoint Row (pixels)": mid_row, "Plasma Peak (row,col)" : gauss_peak})
    # get default figure size
    ww,hh = plt.rcParams["figure.figsize"]
    # create subplots scaling the image by number of required plots
    if int(transpose) in [0,1]:
        # if transpose is 1 then the subplots are arranged horizontally
        if transpose == 1:
            nc = hist.shape[1]+1
            nr = 1
            width = ww*hist.shape[1]+1
            height = hh
        # else if transpose is 0 then the subplots are arranged vertically
        elif transpose == 0:
            nc = 1
            nr = hist.shape[1]+1
            width = ww
            height = hh*hist.shape[1]+1
        f,ax = plt.subplots(ncols=nc,nrows=nr,sharex=True,constrained_layout=True,figsize=[width,height])
    else:
        f,ax = plt.subplots(ncols=2,nrows=2,sharex=True,constrained_layout=True,figsize=[ww*1.5*2,hh*1.5*2])
        ax = ax.flatten()
    # create x axis
    x = np.arange(nf)
    # plot max temperature
    ax[0].plot(x,maxt,color=kwargs.get("plot_color","magenta"))
    # set label
    ax[0].set(xlabel="Frame Number",ylabel="Max Frame Temperature (C)",title="Max Frame Temperature")
    if "xfsize" in kwargs:
        ax[0].tick_params(axis="x",labelsize=kwargs.get("xfsize"))
    if "yfsize" in kwargs:
        ax[0].tick_params(axis="y",labelsize=kwargs.get("yfsize"))
    # iterate over the other parameters, axes and titles
    for pp,aa,title in zip(range(hist.shape[1]),ax[1:],["Height (pixels)","Mean (pixels)","Std Dev (pixels)"]):
        param = hist[:,pp]
        aa.plot(x,param,color=kwargs.get("plot_color","magenta"))
        aa.set(xlabel="Frame Number",ylabel=title,title=title)
        if "xfsize" in kwargs:
            aa[0].tick_params(axis="x",labelsize=kwargs.get("xfsize"))
        if "yfsize" in kwargs:
            aa[0].tick_params(axis="y",labelsize=kwargs.get("yfsize"))
    f.suptitle(f"{fname} Gaussian Plasma",fontsize=kwargs.get("title_size",plt.rcParams["figure.titlesize"]))
    f.savefig(f"{Path(fn).stem}-gaussian-model-parameters.png")
    plt.close(f)

    ## distance from origin
    fd,axd = plt.subplots(constrained_layout=True,figsize=[ww,hh])
    # extract the height and std dev
    hist = hist[:,[0,2]]
    # split into coordinate pairs
    hist_s = np.vsplit(hist,hist.shape[0])
    # compute distance to origin
    dists = [euclidean(x.flatten(),len(x)*[0]) for x in hist_s]
    axd.plot(dists,color=kwargs.get("plot_color","magenta"))
    axd.set(xlabel="Frame Number",ylabel="Euclidean Distance to Origin",title="Parameter Euclidean Distance to Origin")
    fd.savefig(f"{Path(fn).stem}-parameter-euclidean-distance.png")
    plt.close(fd)

    ## neighbouring distances
    fd,axd = plt.subplots(constrained_layout=True,figsize=[ww,hh])
    # iterate over the history as pairs
    ff = np.arange(len(hist_s)-1)
    dists = [euclidean(a[0].flatten(),a[1].flatten()) for a in zip(hist_s[::2],hist_s[1::2])]
    axd.plot(ff[::2],dists,color=kwargs.get("plot_color","magenta"))
    axd.set(xlabel="Frame Number",ylabel="Euclidean Distance",title="Parameter Euclidean Distance betwen Neighbours")
    fd.savefig(f"{Path(fn).stem}-parameter-euclidean-distance-neighbours-p-2.png")
    plt.close(fd)

    ## windowed max distances
    # put the distances into a df
    win_dist = pd.Series([euclidean(a[0].flatten(),a[1].flatten()) for a in zip(hist_s,hist_s[1:])])
    for w in [2,3,5]:
        # find the max distance to origin over windows of 3 samples
        win_roll_max = win_dist.rolling(w).max()
        fd,axd = plt.subplots(constrained_layout=True,figsize=[ww,hh])
        axd = win_roll_max.plot(ax=axd,color=kwargs.get("plot_color","magenta"))
        axd.set(xlabel="Frame Number",ylabel="Rolling Max Distance",title=f"Max Rolling Parameter Euclidean\nDistance between Neighbours, w={w}")
        fd.savefig(f"{Path(fn).stem}-windowed-parameter-euclidean-distance-neighbours-p-{w}.png")
        plt.close(fd)

        fd,axd = plt.subplots(constrained_layout=True,figsize=[ww,hh])
        Dp = win_roll_max[win_roll_max>0]
        if Dp.shape[0]==0:
            warn(f"Skipping log scale for w={w} as all values are zero", stacklevel=2)
            continue
        axd = win_roll_max.plot(ax=axd,color=kwargs.get("plot_color","magenta"))
        axd.set_yscale("log")
        axd.set(xlabel="Frame Number",ylabel="Rolling Max Distance",title=f"Max Rolling Parameter Euclidean\nDistance between Neighbours, w={w}")
        fd.savefig(f"{Path(fn).stem}-windowed-parameter-euclidean-distance-neighbours-p-{w}-pve-log.png")
        plt.close(fd)

    # plot the RMSE between the contour and structure 
    fe,axe = plt.subplots()
    axe.plot(rmse,color=kwargs.get("plot_color","magenta"))
    axe.set(xlabel="Frame Number",ylabel="RMSE (Contour vs Gaussian)",title=f"{Path(fn).stem}\nRMSE between Contour and Gaussian Fitting")
    fe.savefig(f"{Path(fn).stem}-rmse.png")
    plt.close(fe)

    # plot the symmetric angle
    fs,axs = plt.subplots()
    axs.plot(symmangle,color=kwargs.get("plot_color","magenta"))
    axs.set(xlabel="Frame Number",title=f"{Path(fn).stem}\nAngle Between Peak and Mid Point (rads)",ylabel="Angle (rads)")
    fs.savefig(f"{Path(fn).stem}-symmangle.png")
    plt.close(fs)

    # plot the symmetric angle
    fs,axs = plt.subplots()
    axs.plot(area_ratio,color=kwargs.get("plot_color","magenta"))
    axs.set(xlabel="Frame Number",title=f"{Path(fn).stem}\nRatio between Midpoint Areas",ylabel="Ratio")
    fs.savefig(f"{Path(fn).stem}-area-ratio.png")
    plt.close(fs)

    # plot skew
    fs,axs = plt.subplots()
    axs.plot(skew,color=kwargs.get("plot_color","magenta"))
    axs.set(xlabel="Frame Number",title=f"{Path(fn).stem}\nContour Gaussian Skew",ylabel="Skew")
    fs.savefig(f"{Path(fn).stem}-skew.png")
    plt.close(fs)

    # plot Fisher Kurtosis
    fs,axs = plt.subplots()
    axs.plot(kfish,color=kwargs.get("plot_color","magenta"))
    axs.set(xlabel="Frame Number",title=f"{Path(fn).stem}\nContour Gaussian Kurtosis (Fisher)",ylabel="Kurtosis")
    fs.savefig(f"{Path(fn).stem}-kurtosis-fisher.png")
    plt.close(fs)

    # plot skew
    fs,axs = plt.subplots()
    axs.plot(kurt,color=kwargs.get("plot_color","magenta"))
    axs.set(xlabel="Frame Number",title=f"{Path(fn).stem}\nContour Gaussian Kurtosis (Pearson)",ylabel="Kurtosis")
    fs.savefig(f"{Path(fn).stem}-kurtosis-pearson.png")
    plt.close(fs)

    plt.close("all")
    return f,gauss_df

def plasmaAllGaussianParams(path:str,transpose:int=2,**kwargs) -> plt.Figure:
    """Load ALL temperature NPZ file and find the gaussian parameters from each frame then plot it.

    Creates a PlasmaGaussian class and applies it to each frame. If the opts is None,
    then 0s are added in it"s place. The results are then plotted on a series of graphs.
    The graphs are max temperature, height and standard deviation. The max temperature
    plot is to provide some context to the activity happening in each image,.

    The transpose flag controls whether the plots are horizontal or vertical.
    When transpose is:
        - 0 : Plots are stacked as rows
        - 1 : Plots are stacked as cols
        - ELSE : 2x2 array

    Inputs:
        path : Wildcard path to NPZ
        transpose : Flag to transpose the plots to be horizontal rather than vertical. Default 2.
        filt_hist : Function to filter the history.
        xfsize : X-axis tick font size
        yfsize : Y-axis tick font size
        title_size : Figure title size
        plot_labels : Legend label for each file path

    Returns the matplotlib figure
    """
    # setup class using the wanted settings
    cp = PlasmaGaussian(lim=kwargs.get("lim",0.9))
    # get default figure size
    ww,hh = plt.rcParams["figure.figsize"]
    # create subplots scaling the image by number of required plots
    if int(transpose) in [0,1]:
        # if transpose is 1 then the subplots are arranged horizontally
        if transpose == 1:
            nc = 3+1
            nr = 1
            width = ww*1.5*3
            height = hh
        # else if transpose is 0 then the subplots are arranged vertically
        elif transpose == 0:
            nc = 1
            nr = 3+1
            width = ww
            height = hh*1.5*3
        f,ax = plt.subplots(ncols=nc,nrows=nr,sharex=True,constrained_layout=True,figsize=[width,height])
    else:
        f,ax = plt.subplots(ncols=2,nrows=2,sharex=True,constrained_layout=True,figsize=[ww*1.5*2,hh*1.5*2])
        ax = ax.flatten()
    plot_labels = kwargs.get("plot_labels",None)
    # if plot labels weren"t specified
    # generate plot labels from file names
    if plot_labels is None:
        plot_labels = [Path(fn).stem for fn in glob(path)]  # noqa: PTH207
    # if the number of labels does not match the number of files
    # raise an error
    if len(plot_labels) != len(glob(path)):   # noqa: PTH207
        msg = f"Number of plot labels does not match the number of files! {len(plot_labels)} vs. {len(glob(path))}"
        raise ValueError(msg)   # noqa: PTH207

    for aa in ax:
        if "xfsize" in kwargs:
            aa.tick_params(axis="x",labelsize=kwargs.get("xfsize"))
        if "yfsize" in kwargs:
            aa.tick_params(axis="y",labelsize=kwargs.get("yfsize"))
    # iterate over file paths and labels
    for fn,plabel in zip(glob(path),plot_labels):   # noqa: PTH207
        # load data
        data = np.load(fn)["arr_0"]
        # get max temp
        maxt = data.max(axis=(1,2))
        # get filename
        # get size of the data
        nf,r,c = data.shape
        # clear history
        hist = []
        # iterate over each frame
        for ni in range(nf):
            frame = data[ni,:,:]
            opts = cp.fit_guess(frame)
            # if it didn't fit
            # then add a series of -1s
            if opts is None:
                hist.append(3*[0])
            else:
                hist.append(opts)
        # form into array for easier indexing
        hist = np.vstack(hist)
        if "filt_hist" in kwargs:
            hist = kwargs.get("filt_hist")(hist)
        # create x axis
        x = np.arange(nf)
        # plot max temperature
        ax[0].plot(x,maxt,label=plabel)
        # set label
        ax[0].set(xlabel="Frame Number",ylabel="Max Frame Temperature (C)",title="Max Frame Temperature")
        # iterate over the other parameters, axes and titles
        for pp,aa,title in zip(range(hist.shape[1]),ax[1:],["Height (pixels)","Mean (pixels)","Std Dev (pixels)"]):
            param = hist[:,pp]
            aa.plot(x,param,"-",label=plabel)
            aa.set(xlabel="Frame Number",ylabel=title,title=title)
    for aa in ax:
        aa.legend()
    f.suptitle("Gaussian Plasma Parameters",fontsize=kwargs.get("title_size",plt.rcParams["figure.titlesize"]))
    return f

def plasmaGaussianBoxPlot(path:str,transpose:int=1,**kwargs) -> plt.Figure:
    """Plot the gaussian parameter history as a Violin plot.

    The transpose flag controls whether the plots are horizontal or vertical.
    When transpose is:
        - 0 : Plots are stacked as rows
        - 1 : Plots are stacked as cols
        - ELSE : 2x2 array

    Inputs:
        path : Wildcard path to NPZ
        transpose : Flag to transpose the plots to be horizontal rather than vertical. Default 2.
        filt_hist : Function to filter the history.
        xfsize : X-axis tick font size
        yfsize : Y-axis tick font size
        title_size : Figure title size
        plot_labels : Legend label for each file path

    Returns the matplotlib figure
    """
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    # setup class using the wanted settings
    cp = PlasmaGaussian(lim=kwargs.get("lim",0.9))
    # get default figure size
    ww,hh = plt.rcParams["figure.figsize"]
    # create subplots scaling the image by number of required plots
    if int(transpose) in [0,1]:
        # if transpose is 1 then the subplots are arranged horizontally
        if transpose == 1:
            nc = 2
            nr = 1
            width = ww*1.5*nc
            height = hh
        # else if transpose is 0 then the subplots are arranged vertically
        elif transpose == 0:
            nc = 1
            nr = 2
            width = ww
            height = hh*1.5*nc
        f,ax = plt.subplots(ncols=nc,nrows=nr,sharex=True,constrained_layout=True,figsize=[width,height])
    plot_labels = kwargs.get("plot_labels",None)
    # if plot labels weren"t specified
    # generate plot labels from file names
    if plot_labels is None:
        plot_labels = [Path(fn).stem for fn in glob(path)]   # noqa: PTH207
    # if the number of labels does not match the number of files
    # raise an error
    if len(plot_labels) != len(glob(path)):   # noqa: PTH207
        msg = f"Number of plot labels does not match the number of files! {len(plot_labels)} vs. {len(glob(path))}"
        raise ValueError(msg)   # noqa: PTH207

    plot_labels = dict(zip([Path(fn).stem for fn in glob(path)],plot_labels))   # noqa: PTH207
    for aa in ax:
        if "xfsize" in kwargs:
            aa.tick_params(axis="x",labelsize=kwargs.get("xfsize"))
        if "yfsize" in kwargs:
            aa.tick_params(axis="y",labelsize=kwargs.get("yfsize"))

    height_hist = {}
    mean_hist = {}
    std_hist = {}

    fc,axc = plt.subplots(constrained_layout=True,sharex=True)

    # iterate over file paths and labels
    for fn,plabel in zip(glob(path),plot_labels):   # noqa: PTH207
        # load data
        data = np.load(fn)["arr_0"]
        # get filename
        fname = Path(fn).stem
        # get size of the data
        nf,r,c = data.shape
        # clear history
        hist = []
        # iterate over each frame
        for ni in range(nf):
            frame = data[ni,:,:]
            opts = cp.fit_guess(frame)
            # if it didn't fit
            # then add a series of -1s
            if opts is None:
                hist.append(3*[0])
            else:
                hist.append(opts)
        # form into array for easier indexing
        hist = np.vstack(hist)

        axc.plot(hist[:,0],hist[:,2],"x",label=plabel)
        # add to history
        for c,d in zip(np.hsplit(hist,hist.shape[1]),[height_hist,mean_hist,std_hist]):
            h = c[c>0]
            if len(h)>0:
                d[fname] = h

    for pp,aa,title in zip([height_hist,std_hist],ax,["Height (pixels)","Std Dev (pixels)"]):
        # stack history
        parts = aa.violinplot(tuple(pp.values()), showmeans=False, showmedians=False,showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_edgecolor("black")
            pc.set_alpha(1)

        # from https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        quartile1 = []
        medians  = []
        quartile3 = []
        for v in pp.values():
            q1,m,q3 = np.percentile(v, [25, 50, 75])
            quartile1.append(q1)
            medians.append(m)
            quartile3.append(q3)

        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        
        inds = np.arange(1, len(medians) + 1)
        aa.scatter(inds, medians, marker="o", color="white", s=30, zorder=3)
        aa.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)

        aa.vlines(inds, whiskers_min.max(1), whiskers_max.max(1), color="k", linestyle="-", lw=1)
            
        aa.set(xlabel="Plasma Gas Flow",ylabel=title,title=title)
        aa.set_xticks(np.arange(1,len(pp.values())+1),labels=[plot_labels[k] for k in pp])
        aa.set_xlim(0.25, len(pp.values()) + 0.75)
    f.suptitle(kwargs.get("ftitle","Violin Plot of Gas Flow Rate vs Gaussian Model Parameters"))
                  
    axc.set(xlabel="Height (pixels)",ylabel="Std Dev (pixels)",title="Height vs Std Dev")
    axc.legend()
    return ax,axc

def temperatureStatsAboveLim(path:str,lim:float=1300.0,transpose:int=1,**kwargs) -> plt.Figure:  # noqa: ANN003
    # get default figure size
    ww,hh = plt.rcParams["figure.figsize"]
    # create subplots scaling the image by number of required plots
    if int(transpose) in [0,1]:
        # if transpose is 1 then the subplots are arranged horizontally
        if transpose == 1:
            nc = 2
            nr = 1
            width = ww*1.5*nc
            height = hh
        # else if transpose is 0 then the subplots are arranged vertically
        elif transpose == 0:
            nc = 1
            nr = 2
            width = ww
            height = hh*1.5*nc
        f,ax = plt.subplots(ncols=nc,nrows=nr,sharex=True,constrained_layout=True,figsize=[width,height])
    plot_labels = kwargs.get("plot_labels",None)
    # if plot labels weren"t specified
    # generate plot labels from file names
    if plot_labels is None:
        plot_labels = [Path(fn).stem for fn in glob(path)]   # noqa: PTH207
    # if the number of labels does not match the number of files
    # raise an error
    if len(plot_labels) != len(glob(path)):   # noqa: PTH207
        msg = f"Number of plot labels does not match the number of files! {len(plot_labels)} vs. {len(glob(path))}"
        raise ValueError(msg)   # noqa: PTH207

    plot_labels = dict(zip([Path(fn).stem for fn in glob(path)],plot_labels))   # noqa: PTH207

    for aa in ax:
        if "xfsize" in kwargs:
            aa.tick_params(axis="x",labelsize=kwargs.get("xfsize"))
        if "yfsize" in kwargs:
            aa.tick_params(axis="y",labelsize=kwargs.get("yfsize"))

    mean_hist = {}
    var_hist = {}
    # iterate over file paths and labels
    for fn in glob(path):   # noqa: PTH207
        # load data
        data = np.load(fn)["arr_0"]
        # get filename
        fname = Path(fn).stem
        # get size of the data
        nf,r,c = data.shape
        mean = []
        var = []
        for fi in range(nf):
            frame = data[fi,...]
            # get values above lim
            vals = frame[frame>lim]
            if vals.shape[0]>0:
                mean.append(np.mean(vals))
                var.append(np.var(vals))
        if len(mean)>0:
            mean_hist[fname] = np.max(mean)
            var_hist[fname] = np.max(var)

    for aa,pp,dd in zip(ax.flatten(),[mean_hist,var_hist],["Mean Temperature (C)","Variance (C)"]):
        aa.scatter(range(len(pp.values())),pp.values(),s=70,facecolors="r",edgecolors="r")
        aa.set_xticks(np.arange(1,len(pp.values())+1),labels=[plot_labels[k] for k in pp])
        aa.set_xlim(0.25, len(pp.values()) + 0.75)
        aa.set(xlabel="Plasma Gas Flow Rate",ylabel=dd,title=dd)
    f.suptitle(f"Mean and Temperature of pixels above {lim}")
    return f


def plasmaGaussianVideo(path:str,opath:str | None=None,**kwargs) -> None:  # noqa: ANN003
    """Fit gaussian parameters and draw the results on a frame. The drawings are saves to a video.

    Inputs:
        path : Path to NPZ file
        opath : Output path of video file
    """
    # load data
    if isinstance(path,str):
        data = np.load(path)["arr_0"]
    # get size of the data
    nf,r,c = data.shape
    # setup class using the wanted settings
    cp = PlasmaGaussian(lim=kwargs.get("lim",0.9))
    # make the output video
    if opath is None:
        opath = Path(path).stem+"-gaussian-params.avi"
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    all_writer = cv2.VideoWriter(opath,fourcc,30.0,(c,r),1)
    # iterate over each frame
    for ni in range(nf):
        frame = data[ni,:,:]
        cp.fit_guess(frame)
        draw = cp.draw(False,True,True,**kwargs)  # noqa: FBT003
        all_writer.write(draw)
    all_writer.release()


def stackPlasmaGaussianParams(path:str,transpose:int=2,**kwargs) -> pd.DataFrame:  # noqa: ANN003
    """Iterate over several NPZ files, find the gaussian parameters from each frame then plot it on the same axis.

    Stacking the results

    Creates a PlasmaGaussian class and applies it to each frame. If the opts is None,
    then 0s are added in it"s place. The results are then plotted on a series of graphs.
    The graphs are max temperature, height and standard deviation. The max temperature
    plot is to provide some context to the activity happening in each image,.

    The transpose flag controls whether the plots are horizontal or vertical.
    When transpose is:
        - 0 : Plots are stacked as rows
        - 1 : Plots are stacked as cols
        - ELSE : 2x2 array

    Inputs:
        fn : Path to temperature NPZ file
        transpose : Flag to transpose the plots to be horizontal rather than vertical. Default 2.
        filt_hist : Function to filter the history.
        xfsize : X-axis tick font size
        yfsize : Y-axis tick font size
        title_size : Figure title size

    Returns the matplotlib figure
    """
    # get default figure size
    ww,hh = plt.rcParams["figure.figsize"]
    # create subplots scaling the image by number of required plots
    if int(transpose) in [0,1]:
        f,ax = plt.subplots(ncols=4 if transpose else 1,
                            nrows=4 if not transpose else 1,
                            sharex=True,
                            constrained_layout=True,
                            figsize=[ww if not transpose else ww*1.5*3,hh*1.5*3 if not transpose else hh])
    else:
        f,ax = plt.subplots(ncols=2,nrows=2,sharex=True,constrained_layout=True,figsize=[ww*1.5*2,hh*1.5*2])
        ax = ax.flatten()
    search = glob(path) if isinstance(path, str) else path   # noqa: PTH207
    # initialize class
    cp = PlasmaGaussian()
    for fn in search:
        data = np.load(fn)["arr_0"]
        # get max temp
        maxt = data.max(axis=(1,2))
        # get filename
        fname = Path(fn).stem
        # get size of the data
        nf,r,c = data.shape
        hist = []
        # iterate over each frame
        for ni in range(nf):
            frame = data[ni,:,:]
            opts = cp.fit_guess(frame)
            # if it didn't fit
            # then add a series of -1s
            if opts is None:
                hist.append(3*[0])
            else:
                hist.append(opts)
        # form into array for easier indexing
        hist = np.vstack(hist)
        if "filt_hist" in kwargs:
            hist = kwargs.get("filt_hist")(hist)
        # create normalized x axis
        x = np.linspace(0,1,nf)
        # plot the max temperature
        ax[0].plot(x,maxt,label=fname)
        # plot the history
        for pp,aa in zip(range(hist.shape[1]),ax[1:]):
            param = hist[:,pp]
            aa.plot(x,param)
    ax[0].set(xlabel="Frame Number",ylabel="Max Frame Temperature (C)",title="Max Frame Temperature")
    if "xfsize" in kwargs:
        ax[0].tick_params(axis="x",labelsize=kwargs.get("xfsize"))
    if "yfsize" in kwargs:
        ax[0].tick_params(axis="y",labelsize=kwargs.get("yfsize"))
    for aa,title in zip(ax[1:],["Height (pixels)","Mean (pixels)","Std Dev (pixels)"]):
        aa.set(xlabel="Frame Number",ylabel=title,title=title)
        if "xfsize" in kwargs:
            aa[0].tick_params(axis="x",labelsize=kwargs.get("xfsize"))
        if "yfsize" in kwargs:
            aa[0].tick_params(axis="y",labelsize=kwargs.get("yfsize"))
    f.suptitle(kwargs.get("title","Multiple Gaussian Plasma"),fontsize=kwargs.get("title_size",plt.rcParams["figure.titlesize"]))
    return f


def getGPS(fn:str) -> tuple[list, float]:
    # load file
    data = np.load(fn)["arr_0"]
    # get size
    nf,r,c = data.shape
    cp = PlasmaGaussian()
    hist = []
    for ni in range(nf):
        frame = data[ni,:,:]
        opts = cp.fit_guess(frame)
        if opts is None:
            hist.append(3*[0])
        else:
            hist.append(opts)
    return hist,data.max(axis=(1,2))


def plasmaGaussianScatterParamMP(path:str,target:int=0,**kwargs) -> plt.Figure:  # noqa: ANN003
    import multiprocessing as mp
    if isinstance(path,str):
        path = glob(path)   # noqa: PTH207
    f,ax = plt.subplots()
    labels = ["Height","Mean","Std Dev"]
    if target in (0,2):
        msg = f"Invalid target parameter index! Received {target}!"
        raise ValueError(msg)
    with mp.Pool(3) as pool:
        hist = pool.map(getGPS,path)
    for fn,hh in zip(path,hist):
        ax.plot(hh[0],[x[target] for x in hh[1]],"x",label=Path(fn).stem)
    ax.legend()
    ax.set(xlabel="Max Temperature (C)",ylabel=labels[target],title=kwargs.get("title",f"Max Temperature vs {labels[target]}"))
    return f


def frame2gray(frame:np.ndarray) -> np.ndarray:
    """Convert frame to 0-255 grayscale."""
    return (255*((frame-frame.min())/np.abs(frame.max()-frame.min()))).astype("uint8")


def distanceToHotSlider(path:str):
    """Display GUI for showing distance to non-zero hottest spot."""
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    # create coordinate grid for calculating distance
    xx = np.arange(0,c,1)
    yy = np.arange(0,r,1)
    xx,yy = np.meshgrid(xx,yy)
    # create named window
    cv2.namedWindow("Hot Distance")
    # function for handling when the slider has been moved
    def on_changed(val) -> None:
        # get slider value
        ni = int(cv2.getTrackbarPos("Frame","Hot Distance"))
        # extract frame
        frame = data[ni,:,:]
        # find hottest point
        hi = np.argmax(frame,axis=(0,1))
        hr,hc = np.unravel_index(hi,(r,c))
        # when there is nothing going on in the temperature frame
        # the hottest point is at (0,0) for some reason
        if (hr!=0) and (hc!=0):
            dist = ((xx-hc)**2 + (yy-hr)**2)**0.5
            # normalize and colormap
            dist_norm =frame2gray(dist)
        # when the point is at (0,0) show blank matrix
        else:
            dist_norm = np.zeros((r,c),np.uint8)
        # colormap the distance
        dist_map = cv2.applyColorMap(dist_norm,cv2.COLORMAP_HOT)
        # colormap the temperature
        frame_gray = frame2gray(frame)
        frame_map = cv2.applyColorMap(frame_gray,cv2.COLORMAP_HOT)
        # display side by side
        cv2.imshow("Hot Distance",cv2.hconcat([frame_map,dist_map]))

    cv2.createTrackbar("Frame","Hot Distance",0,nf-1,on_changed)

    on_changed(0)
    cv2.waitKey(0)

def coaxialPlaneSliderBlur(fn:str,k:int=2) -> None:
    """Fit coaxial plane to the data at a fixed order applying a guassian blur to source.

    Two OpenCV sliders are added to control what frame is being shown and the sigma of the
    Gaussian blur.

    Inputs:
        fn : Path to NPZ file
        k : Polynomial order to fit
    """
    from scipy.ndimage import gaussian_filter
    # load the target file
    data = np.load(fn)["arr_0"]
    nf,r,c = data.shape
    # create the coaxial plane object
    cp = CoaxialPlane(k=k,trim=False)
    # create a named window
    cv2.namedWindow("CP")
    def on_changed(val) -> None:
        # get the target frame
        ni = int(cv2.getTrackbarPos("Frame","CP"))
        # get gaussian blur sigma
        sigma = float(cv2.getTrackbarPos("GB","CP"))
        frame = data[ni,:,:]
        # apple gaussian blur to the temperature frame
        frame = gaussian_filter(frame,sigma=sigma)
        # fit and draw
        cp.estPlane(frame)
        draw = cp.drawLastLine(on_source=False)
        source = CoaxialPlane.frame2gray(frame)

        source = cv2.applyColorMap(source,cv2.COLORMAP_HOT)
        cv2.imshow("CP",np.column_stack((source,draw)))
    # make trackbars and assign update
    cv2.createTrackbar("Frame","CP",0,nf-1,on_changed)
    cv2.createTrackbar("GB","CP",0,10,on_changed)
    on_changed(0)
    cv2.waitKey(0)

def videoCoaxialPlane(fn:str,k:int=2,opath:str | None=None,filt=None,mode:int=1,**kwargs):  # noqa: ANN003
    """Perform coaxial plane prediction on the target data and draw to a video file.

    The mode parameter controls what is written to the video.
    Supported parameters:
        min, 0 : Line drawn on the source
        default,1 : Source colour mapped along side greyscale entropy with line drawn on it
        grey,2 : Line drawn on grayscale entropy
        all, 3 : Source, greyscale entropy and line drawn on source side by side

    Inputs:
        fn : Path to NPZ file
        k : Polynomial to fit
        opath : Output path for video
        filt : Filter to apply to frame before fitting. Default None.
        mode : Parameter that controls what is written to the video
    """
    data = np.load(fn)["arr_0"]
    nf,r,c = data.shape
    # if output path is None
    # set it based on the mode
    if opath is None:
        if mode in ("min", 0):
            opath = Path(fn).stem+f"_coaxial_plane_k_{k}_min{"" if filt is None else "-filt"}.avi"
            vc = c
            vr = r
        elif mode in ("default", 1):
            opath = Path(fn).stem+f"_coaxial_plane_k_{k}{"" if filt is None else "-filt"}.avi"
            vc = c*2
            vr = r
        elif mode in ("grey", 2):
            opath = Path(fn).stem+f"_coaxial_plane_k_{k}_grey-only{"" if filt is None else "-filt"}.avi"
            vc = c
            vr = r
        elif mode in ("all", 3):
            opath = Path(fn).stem+f"_coaxial_plane_k_{k}_all{"" if filt is None else "-filt"}.avi"
            vc = c*3
            vr = r
    
    # create the video writer using shape of frame
    writer = cv2.VideoWriter(opath,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(vc,vr),1)
    # check that the outfile has opened    
    if not writer.isOpened():
        msg = f"Failed to open output file {opath}!"
        raise OSError(msg)
    cp = CoaxialPlane(k=k,trim=False,line_lims=1,min_segs=8,threshold=False)
    for ni in range(nf):
        frame = data[ni,:,:]
        if isinstance(filt,(list,tuple)):
            for f in filt:
                frame = f(frame)
        elif filt is not None:
            frame = filt(frame)
        cp.estPlane(frame)
        source = CoaxialPlane.frame2gray(frame)
        source = cv2.applyColorMap(source,cv2.COLORMAP_HOT)
        # line drawn on source
        if mode in ("min", 0):
            draw = cp.drawLastLine(on_source=True,**kwargs)
        # default
        elif mode in ("default", 1):
            draw = cp.drawLastLine(on_source=False,**kwargs)
            draw = np.column_stack((source,draw))
        # greyscale entropy
        elif mode in ("grey", 2):
            draw = cp.drawLastLine(on_source=False,**kwargs)
        # full stages
        elif mode in ("all", 3):
            draw = cp.drawLastLine(on_source=True,**kwargs)
            draw = np.column_stack((source,np.dstack(3*(cp.gray,)),draw))
        # write to imave
        writer.write(draw)
    writer.release()

def videoCoaxialPlaneExperimental(fn:str,opath:str | None=None,filt=None,mode:int=1):
    """Perform coaxial plane prediction on the target data and draw to a video file.

    The mode parameter controls what is written to the video.
    Supported parameters:
        min, 0 : Line drawn on the source
        default,1 : Source colour mapped along side greyscale entropy with line drawn on it
        grey,2 : Line drawn on grayscale entropy
        all, 3 : Source, greyscale entropy and line drawn on source side by side

    Inputs:
        fn : Path to NPZ file
        k : Polynomial to fit
        opath : Output path for video
        filt : Filter to apply to frame before fitting. Default None
    """
    data = np.load(fn)["arr_0"]
    nf,r,c = data.shape
    # if output path is None
    # set it based on the mode
    if opath is None:
        if mode in ("min", 0):
            opath = Path(fn).stem+"_coaxial_plane_exp_min.avi"
            vc = c
            vr = r
        elif mode in ("default", 1):
            opath = Path(fn).stem+"_coaxial_plane_exp.avi"
            vc = c*2
            vr = r
        elif mode in ("grey", 2):
            opath = Path(fn).stem+"_coaxial_plane_exp_grey-only.avi"
            vc = c
            vr = r
        elif mode in ("all", 3):
            opath = Path(fn).stem+"_coaxial_plane_exp_all.avi"
            vc = c*3
            vr = r
    # create the video writer using shape of frame
    writer = cv2.VideoWriter(opath,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(vc,vr),1)
    # check that the outfile has opened    
    if not writer.isOpened():
        msg = f"Failed to open output file {opath}!"
        raise OSError(msg)
    cp = CoaxialPlane(k=2,trim=False,line_lims=1,min_segs=8,threshold=False)
    for ni in range(nf):
        frame = data[ni,:,:]
        if isinstance(filt,(list,tuple)):
            for f in filt:
                frame = f(frame)
        elif filt is not None:
            frame = filt(frame)
        cp.estPlaneExperimental(frame)
        source = CoaxialPlane.frame2gray(frame)
        source = cv2.applyColorMap(source,cv2.COLORMAP_HOT)
        # line drawn on source
        if mode in ("min", 0):
            draw = cp.drawLastExperimental(on_source=True)
        # default
        elif mode in ("default", 1):
            draw = cp.drawLastExperimental(on_source=False)
            draw = np.column_stack((source,draw))
        # greyscale entropy
        elif mode in ("grey", 2):
            draw = cp.drawLastExperimental(on_source=False)
        # full stages
        elif mode in ("all", 3):
            draw = cp.drawLastExperimental(on_source=True)
            draw = np.column_stack((source,np.dstack(3*(cp.gray,)),draw))
        # write to imave
        writer.write(draw)
    writer.release()

def findSumTotalObjSize(data:np.ndarray,units:str="pixels",**kwargs) -> float:
    """Find the sum of the sizes of the objects in the given temperature frame and return the area.

    Applies findObjectInTemp, finds the area of each contour and then returns it. If the user wants
    the area in metres squared, then the angles for each direction needs to be given.

    If the user wants the area in metres squared, they need to supply the angle in each direction.

    Inputs:
        data : 2D temperature frame
        units : What units to return the sum in.
            px, pixels : Pixels squared
            m2, metres_sq : Metres squared
        angleWidth : Angle Width field of view. Required if units is set to metres squared
        angleHeight : Angle height field of view. Required if units is set to metres squared
        FLIR:ObjectDistance, dist : Distance to objects. From CSQ metadata.

    Returns total area in the specified units
    """
    # find all objects in the image
    cts = findObjectsInTemp(data)
    if len(cts)==0:
        return 0.0
    # iterate over each contour
    # if the user wants the area in pixels^2
    if units in ["px","pixels"]:
        return sum(cv2.contourArea(cc) for cc in cts)
    # if the user wants it in metres squared
    if units in ["m2","metres_sq"]:
        # get angles from keywords
        angwidth = kwargs["angleWidth"]
        angheight = kwargs["angleHeight"]
        # get distance to target
        # attempt to get via FLIR Tag
        try:
            dist = kwargs["FLIR:ObjectDistance"]
        except KeyError:
            dist = kwargs["dist"]
        # get size of the image
        _,height,width = data.shape
        # use angle to convert image height to m
        height_m = dist*np.tan(angheight)
        # use angle to convert image width to m
        width_m = dist*np.tan(angwidth)
        # conversion factor m2/p2
        pconv = (height_m * width_m)/(width*height)
        # convert each area to m2
        return sum(pconv*cv2.contourArea(cc) for cc in cts)
    return None


def recolormapData(temp,tmin:float | None = None,tmax:float | None = None,cmap:str="hot",mode:str="linear"):
    """Colormap data using either a matplotlib or opencv colormap.

    This function is designed to mimic Flir Studios colormapping functionality

    If the colormap is specified by an integer, it is checked against the cv2 module attributes containing COLORMAP_.
    If it"s not in the found attributes, then a ValueError is raised. If the colormap is specifeid by a
    string, it is retrieved using plt.get_cmap.

    Inputs:
        temp : 2D or 3D temperature array
        tmin : Minimum value for normalization. If None, global minimum is used. Default None.
        tmax : Maximum value for normalization. If None, global maximum is used. Default None.
        cmap : Colormap used. If an integer, then it is assumed to be a cv2 colormap. If a string,
                then it is assumed to be a matplotlib colormap.
        mode : Normalization mode. If linear then Normalize is used. If log then LogNorm is used.
                Used to normalize the data before applying the colormap.
    """
    # if the temperature range isn"t given.
    # get from data
    if tmin is None:
        tmin = temp.min()
    if tmax is None:
        tmax = temp.max()
    # create norm
    if mode == "linear":
        norm = Normalize(vmin=tmin,vmax=tmax)
    elif mode == "log":
        norm = LogNorm(vmin=tmin,vmax=tmax)
    else:
        msg = f"Unsupported normalization mode {mode}!"
        raise ValueError(msg)
    # normalize data
    temp_norm = norm(temp)
    # if the cmap is specified as an integer, attempt to retreive using cv2
    if isinstance(cmap,int):
        # get list of supported colormaps in opencv
        cmaps_cv2 = dict(filter(lambda x : "COLORMAP_" in x[0],cv2.__dict__.items()))
        # check if the integer is in the dictionary
        if cmap not in list(cmaps_cv2.values()):
            msg = f"Integer colormap code {cmap} not supported in cv2!"
            raise ValueError(msg)
        color_range = cmap
    # if the cmap is a string
    # search matplotlib
    elif isinstance(cmap,str):
        # get colormap
        cmap = plt.get_cmap(cmap)
        # create mappable
        sm = plt.cm.ScalarMappable(norm,cmap)
        # create linear color range
        color_range = sm.to_rgba(np.linspace(0,1,256),bytes=True)[:,2::-1]
    # scale up
    temp_norm *= 255.0
    temp_norm = temp_norm.astype("uint8")
    # if the given data is 2D
    if len(temp.shape)==2:
        # apply colormap to frame
        return cv2.applyColorMap(temp_norm,color_range)
    # if the given data is 3D
    if len(temp.shape)==3:
        # create empty array to hold results
        cmapped = np.empty([*list(temp.shape), 3])
        # create multiprocessing 
        with mp.Pool(processes=4) as pool:
            res = pool.map(partial(cv2.applyColorMap,color_range),(temp[z,:,:] for z in range(temp.shape[0])))
            for z in range(temp.shape[0]):
                cmapped[z,:,:,:] = res[z]
        return cmapped
    return None


def maskMaxTempVideo(inf: str,outf: str|None = None,lim: float = 0.9):
    # if outf is None create it from the inf
    if outf is None:
        outf = Path(inf).stem+".avi"
    # load input file
    data = np.load(inf)["arr_0"]
    nf,r,c = data.shape
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),0)
    # check that it"s opened
    if not writer.isOpened():
        msg = f"Failed to open video file {outf}!"
        raise ValueError(msg)
    # iterate over frames
    for i in range(data.shape[0]):
        # get the max temperature
        maxt = data[i,:,:].max()
        # make blank frame
        blank = np.zeros((r,c),dtype="uint8")
        # set pixels where the max temperature is above lim times max
        blank[data[i,:,:]>=(0.9*maxt)] = 255
        # if the entire frame is white then tehre isn"t a target object
        # so reset it back to zeros        
        if (blank==MAX_8_BIT).all():
            blank = np.zeros((r,c),dtype="uint8")
            writer.write(blank)
            continue
        # write frames to video
        writer.write(blank)
    # release video writer
    writer.release()


def maskMaxTempSlider(path: str):
    """Load a temperature NPZ file and use sliders to clip the temperature to identify hot areas.

    One slider handles Frame index and the other is the lower temperature
    limit. The colormapped clipped frame is on the right and the original
    is on the left

    Inputs:
        path : File path to NPZ file
    """
    from matplotlib.widgets import Slider
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    f,ax = plt.subplots(ncols=2)
    
    axfreq = f.add_axes([0.15, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(np.argmax(data),(nf,r,c))[0],
        valstep=1,
    )

    axulim = f.add_axes([0.15, 0.05, 0.65, 0.03])
    lim_slider = Slider(
        ax=axulim,
        label="Lower Limit",
        valmin=0,
        valmax=1500,
        valinit=300,
        valstep=0.1,
    )
    def update(val):
        ni = int(freq_slider.val)
        frame = data[ni,:,:]
        ax[0].cla()
        ax[1].cla()
        # show on first axis
        ax[0].contourf(frame,cmap="hot",vmin=frame.min(),vmax=frame.max())
        # clip data to target
        ax[1].contourf(frame,cmap="hot",vmin=lim_slider.val,vmax=1800.0)
        f.canvas.draw_idle()
    
    freq_slider.on_changed(update)
    lim_slider.on_changed(update)
    update(0)
    plt.show()


def drawLargestObjVideo(inf: str,outf: str|None = None,lim: float = 0.9):
     # if outf is None create it from the inf
    if outf is None:
        outf = Path(inf).stem+"_lobj.avi"
    # load input file
    data = np.load(inf)["arr_0"]
    nf,r,c = data.shape
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),1)
    # check that it"s opened
    if not writer.isOpened():
        msg = f"Failed to open video file {outf}!"
        raise ValueError(msg)
    # iterate over frames
    for i in range(data.shape[0]):
        # get the max temperature
        maxt = data[i,:,:].max()
        # make blank frame
        blank = np.zeros((r,c),dtype="uint8")
        # set pixels where the max temperature is above lim times max
        blank[data[i,:,:]>=(lim*maxt)] = 255
        # if the entire frame is white then tehre isn"t a target object
        # so reset it back to zeros        
        if (blank==MAX_8_BIT).all():
            blank = np.zeros((r,c,3),dtype="uint8")
            writer.write(blank)
            continue
        # find contours
        cts = cv2.findContours(blank,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        maxct = max(cts,key=cv2.contourArea)
        img = np.dstack(3*(blank,))
        # draw contour
        cv2.drawContours(img,[maxct],0,(0,255,0),3)
        # write frames to video
        writer.write(img)
    # release video writer
    writer.release()


def drawExtBoundaryObj(inf: str,outf: str|None = None,lim: float = 0.9):
     # if outf is None create it from the inf
    if outf is None:
        outf = Path(inf).stem+"_extct.avi"
    # load input file
    data = np.load(inf)["arr_0"]
    nf,r,c = data.shape
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),1)
    # check that it"s opened
    if not writer.isOpened():
        msg = f"Failed to open video file {outf}!"
        raise ValueError(msg)
    # iterate over frames
    for i in range(data.shape[0]):
        # get the max temperature
        maxt = data[i,:,:].max()
        # make blank frame
        blank = np.zeros((r,c),dtype="uint8")
        # set pixels where the max temperature is above lim times max
        blank[data[i,:,:]>=(lim*maxt)] = 255
        # if the entire frame is white then tehre isn"t a target object
        # so reset it back to zeros        
        if (blank==MAX_8_BIT).all():
            blank = np.zeros((r,c,3),dtype="uint8")
            writer.write(blank)
            continue
        # find contours
        cts = cv2.findContours(blank,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        img = np.dstack(3*(blank,))
        # draw contour
        cv2.drawContours(img,cts,-1,(0,255,0),3)
        # write frames to video
        writer.write(img)
    # release video writer
    writer.release()


def plotExtBoundaryArea(inf: str,outf: str|None = None,lim: float = 0.9):
     # if outf is None create it from the inf
    if outf is None:
        outf = Path(inf).stem+"_extct.png"
    # load input file
    data = np.load(inf)["arr_0"]
    nf,r,c = data.shape
    area = []
    # iterate over frames
    for i in range(data.shape[0]):
        # get the max temperature
        maxt = data[i,:,:].max()
        # make blank frame
        blank = np.zeros((r,c),dtype="uint8")
        # set pixels where the max temperature is above lim times max
        blank[data[i,:,:]>=(lim*maxt)] = 255
        # if the entire frame is white then tehre isn"t a target object
        # so reset it back to zeros        
        if (blank==MAX_8_BIT).all():
            area.append(0.0)
            continue
        # find contours
        cts = cv2.findContours(blank,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        area.append(sum([cv2.contourArea(x) for x in cts]))
        
    f,ax = plt.subplots()
    ax.plot(area)
    ax.set(xlabel="Frame",ylabel="Outer Contour Area",title=f"{Path(inf).stem} Sum External Area")
    f.savefig(outf)
    plt.close(f)


def slicTemperature(frame: np.ndarray,segs: int = 100) -> np.ndarray:
    """Perform SLIC on a single temperature frame."""
    from skimage.segmentation import slic
    
    # rescale temperature to 0 1 
    frame -= frame.min()
    frame /= frame.max()
    # perfom slic
    segments = slic(frame,n_segments=segs,sigma=5)
    # create mask
    mask = np.zeros(frame.shape)
    for i,v in enumerate(np.unique(segments),start=1):
        mask[segments == v] = i
    return mask

def slicTemperatureSlider(path: str):
    from skimage.segmentation import slic
    
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape

    def frameNorm(frame):
        return (frame-frame.min())/abs(frame.max()-frame.min())

    def on_change(val) -> None:
        ni = cv2.getTrackbarPos("frame","SLIC")
        segs = max(1,cv2.getTrackbarPos("segs","SLIC"))
        sigma = max(1,cv2.getTrackbarPos("sigma","SLIC"))
        frame = data[int(ni),:,:]
        gray = frameNorm(frame)
        segments = slic(gray,n_segments=segs,sigma=sigma)
        mask = np.zeros((r,c),dtype="uint8")
        for i,v in enumerate(np.unique(segments),start=1):
            mask[segments == v] = i
        cv2.imshow("SLIC",np.column_stack((gray,mask.astype("uint8"))))
    
    # create window to update
    cv2.namedWindow("SLIC")
    cv2.createTrackbar("frame","SLIC",0,nf-1,on_change)
    cv2.createTrackbar("segs","SLIC",0,100,on_change)
    cv2.createTrackbar("sigma","SLIC",0,10,on_change)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def slicTemperatureColor(frame: str,segs: int = 100):
    """Perform SLIC on a single temperature frame."""
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    frame = img_as_float(frame)
    # perfom slic
    segments = slic(frame,n_segments=segs)
    # create mask
    mask = np.zeros(frame.shape)
    for i,v in enumerate(np.unique(segments),start=1):
        mask[segments == v] = i
    return mask


def slicTemperatureVideo(path: str,outf: str|None = None,segs: int = 1000):
    """Iterate over each frame in a temperature NPZ file, perform SLIC to make a mask and write mask to video file.

    Inputs:
        path : Path to NPZ file
        outf : Output file for video
        segs : Number of segments to search for
    """
     # if outf is None create it from the inf
    if outf is None:
        outf = Path(path).stem+f"_slic_segs_{segs}.avi"
    # load input file
    data = np.load(path)["arr_0"] if isinstance(path, str) else path
    # get shape
    nf,r,c = data.shape
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),0)
    # check that it"s opened
    if not writer.isOpened():
        msg = f"Failed to open video file {outf}!"
        raise ValueError(msg)
    for i in range(nf):
        mask = slicTemperature(data[i,:,:],segs)
        mask = mask.astype("float32")
        mask *= (255.0/mask.max())
        mask = mask.astype("uint8")
        writer.write(mask)
    writer.release()


def slicTemperatureColorVideo(path:str,outf:str|None=None,segs:int=1000) -> None:
    """Iterate over each frame in a temperature NPZ file, perform SLIC to make a mask and write mask to video file.

    Inputs:
        path : Path to video
        segs : Number of segments
    """
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        msg = f"Unable to open video file {path}"
        raise ValueError(msg)
    if outf is None:
        outf = Path(path).stem+f"_color_slic_segs_{segs}.avi"
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(464,348),0)
    # check that it"s opened
    if not writer.isOpened():
        msg = f"Failed to open video file {outf}!"
        raise ValueError(msg)
    while vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            break
        mask = slicTemperatureColor(frame,segs)
        mask = mask.astype("float32")
        mask *= (255.0/mask.max())
        mask = mask.astype("uint8")
        writer.write(mask)
    writer.release()


def denseOpticalFlow(path: str,outf: str|None = None,*args):  # noqa: ANN002
    """Apply dense optical flow to each frame in a video and save the result.

    The drawn result is based on the following code

        flow = cv2.calcOpticalFlowFarneback(prevs, now, None, *args)
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    Inputs:
        path : Path to colour mapped video file
        outf : Path for output video file
        *args : Params for calcOpticalFlowFarneback
    """
    if len(args)==0:
        args = (20/100,1,1,10,1,2.7,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    # open soruce file
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        msg = f"Failed to open file {path}!"
        raise ValueError(msg)
    # read first frame
    ret,frame = cap.read()
    if not ret:
        msg = f"Failed to read first frame from file {path}!"
        raise ValueError(msg)
    prevs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#
    if outf is None:
        outf = Path(path).stem+"_dense_optical_flow.avi"
    r,c,_ = frame.shape
    hsv = np.zeros((r,c,3),dtype="uint8")
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),1)
    # loop collecting frames
    while cap.isOpened():
        # load a 2nd frame
        ret,frame = cap.read()
        if not ret:
            break
        # convert to gray
        now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # perform optical flow looking at differences between current and prev frame
        flow = cv2.calcOpticalFlowFarneback(prevs, now, None, *args)
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prevs = now
        writer.write(bgr)
    writer.release()


def denseOpticalFlowSlider(path: str,filt: list|None = None,winname: str = "optical-flow"):
    """OpenCV slider application to apply dense optical flow and plot the result.

    Inputs:
        path : Input path to video file (AVI)
        filt : List of filtering functions to apply to the frame before displaying. Must accept a 3D array as input
        winname : Name of the window for displaying. Default optical-flow
    """
    # open soruce file
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        msg = f"Failed to open file {path}!"
        raise ValueError(msg)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # read first frame
    ret,frame = cap.read()
    if not ret:
        msg = f"Failed to read first frame from file {path}!"
        raise ValueError(msg)

    r,c,_ = frame.shape
    hsv = np.zeros((r,c,3),dtype="uint8")
    # create a window
    cv2.namedWindow(winname)

    def on_trackbar(val) -> None:
        # get target frame
        fi = int(cv2.getTrackbarPos("Frame",winname))
        # move frame pointer
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        # read target frame and previous frame
        ret,frame = cap.read()
        if not ret:
            warn(f"Failed to read frame {fi}!", stacklevel=2)
            return
        # apply given filters to frame
        if filt is not None:
            try:
                for f in filt:
                    frame = f(frame)
            except TypeError:
                frame = filt(frame)
        ret,frame_next = cap.read()
        if not ret:
            warn(f"Failed to read frame {fi+1}!", stacklevel=2)
            return
        # apply given filters to frame
        if filt is not None:
            try:
                for f in filt:
                    frame = f(frame)
            except TypeError:
                frame = filt(frame)
        # convert to gray scale
        prevs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        now = cv2.cvtColor(frame_next,cv2.COLOR_BGR2GRAY)
        # get pyramid scale
        pyrc = cv2.getTrackbarPos("Pyramid Scale",winname)
        if pyrc == 0:
            pyrc += 1
        pyrc /= 100
        # get number of levels
        lvl = int(cv2.getTrackbarPos("No. Levels",winname))
        # average window size
        win = int(cv2.getTrackbarPos("Win Size",winname))
        if win == 0:
            win = 4
        # iterations
        its = int(cv2.getTrackbarPos("Iter",winname))
        # perform dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prevs, now, None,pyrc,lvl,win,its, 5, 1.2, 0)
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow(winname,np.hstack((frame,bgr)))

    cv2.createTrackbar("Frame",winname,1,nf,on_trackbar)
    # slider for pyramid scale
    cv2.createTrackbar("Pyramid Scale",winname,0,100,on_trackbar)
    # slider for number of pyramid layers
    cv2.createTrackbar("No. Levels",winname,2,10,on_trackbar)
    # slider for averaging window size
    cv2.createTrackbar("Win Size",winname,4,30,on_trackbar)
    # slider for iterations
    cv2.createTrackbar("Iter",winname,1,100,on_trackbar)

    on_trackbar(0)
    cv2.waitKey()


def trackHotPixel(path: str,outf: str|None = None, thresh: float=1300):
    """Track the hottest pixel in each frame and mark it with a green cross.

    Each drawn frame is written to a video file specified by the user.

    If the output file path is not specified it is based off the name of the input

    Path(path).stem+"_hot-pixel-marked.avi"

    Inputs:
        path : Input path to NPZ temprature file
        outf : Fully qualified output filename for video file
    """
    # open soruce file
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    if outf is None:
        outf = Path(path).stem+"_hot-pixel-marked.avi"
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),1)

    for ni in range(nf):
        frame = data[ni,:,:]
        gray = frame2gray(frame)
        hot = cv2.applyColorMap(gray,cv2.COLORMAP_HOT)
        # find max location
        rm,cm = np.unravel_index(np.argmax(frame),(r,c))
        # only draw location if value is above 1300 C
        if frame[rm,cm]>thresh:
            cv2.drawMarker(hot,(cm,rm),(0,255,0),cv2.MARKER_CROSS,10,5)
        writer.write(hot)
    writer.release()


def trackHotPixelHistory(path: str, thresh:float=1300) -> plt.Figure:
    """Load and track the location of the hottest pixel in each frame.

    Plot the result and return the figure

    Inputs:
        path : Path to NPZ file

    Returnf figure
    """
    # open soruce file
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    rhist = []
    chist = []
    for ni in range(nf):
        frame = data[ni,:,:]
        rm,cm = np.unravel_index(np.argmax(frame),(r,c))
        
        rhist.append(rm if frame[rm,cm]>thresh else None)
        chist.append(cm if frame[rm,cm]>thresh else None)
    f,ax = plt.subplots()
    ax.plot(chist,rhist,"r-")
    ax.set(xlabel="X Position",ylabel="Y Position")
    f.suptitle(f"{Path(path).stem}, Hot Pixel Loc")
    return f

def denseOpticalFlowTemperature(path: str,outf: str|None = None,cmap: str = "hsv",crop_plasma: bool = True,*args):  # noqa: ANN002, FBT001, FBT002
    """Apply dense optical flow to each frame in the specified file, colormap it and write to a video.

    The parameters for dense optical flow are specified as a vector given in the *args parameter.
    See calcOpticalFlowFarneback for details

    Supported colormaps:
        hsv : The angle forms the hue component and magnitude the value
        hsv_r : The magnitude forms the hue component and the angle the value
        mag_only : Normalized magnitude forms both hue and value
        yellow : Normalized magnitude is translated to a shade of yellow
        uv : U components forms the hue and V the value
        gray : Normalized magnitude is grayscale

    Each frame of the video is the colormapped temperature using the OpenCV HOT colormap stacked next to the colormapped
    dense optical flow

    The crop_plasma flag. Crops the columns to 360 ignoring where the plasma tends to form. This is useful as the movement and
    temperature of the plasma can easily swamp the movement of the powder. This allows the powder to be the focus.

    Inputs:
        path : Input path to temperature NPZ file
        outf : Output filepath. If None, then it;s based off the input filepath. Default None.
        cmap : Applied colormap. Default hsv
        crop_plasma : Crop the data to [:,:,:360] to ignore the plasma. Default True.
        *args : Vector of parameters for calcOpticalFlowFarneback
    """
    # open soruce file
    data = np.load(path)["arr_0"]
    if crop_plasma:
        data = data[:,:,:360]
    nf,r,c = data.shape
    hsv = np.zeros((r,c,3),dtype="uint8")
    hsv[..., 1] = 255
    if outf is None:
        outf = Path(path).stem+f"_dense_optical_flow_temp-{cmap}-{"-cropped-plasma" if crop_plasma else ""}.avi"
    # create the video writer
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c*2, r),1)
    # loop collecting frames
    for i in range(1,nf):
        # read first frame
        frame = data[i-1,:,:]
        prvs = frame2gray(frame)
        frame_next = data[i,:,:]
        now = frame2gray(frame_next)
        # perform optical flow looking at differences between current and prev frame
        flow = cv2.calcOpticalFlowFarneback(prvs, now, None, *args)
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # map mag and angle to H and V respectfully
        if cmap == "hsv":
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # map mag and angle to V and H respectfully
        elif cmap == "hsv_r":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = ang*180/np.pi/2
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # only map mag to H and V
        elif cmap == "mag_only":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # map mag to the Blue channel and angle to the Green channel
        elif cmap == "yellow":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = 0
            bgr = hsv
        # map X (U) component to Hue channel and Y (V) component to the S channel        
        elif cmap == "uv":
            hsv[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,2] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 1] = 255
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # map mag to a grayscale value
        elif cmap == "gray":
            hsv[...,0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = hsv

        im = cv2.hconcat([cv2.applyColorMap(now,cv2.COLORMAP_HOT),bgr])
        writer.write(im)
    writer.release()

def denseOpticalFlowTemperatureQuiver(path: str,outf: str|None = None,*args):  # noqa: ANN002
    from matplotlib.animation import FuncAnimation
    # open soruce file
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    # create meshgrid of X and Y
    xx = np.arange(c)
    yy = np.arange(r)
    xmat,ymat = np.meshgrid(xx,yy)
    umat = np.zeros_like(xmat)
    vmat = np.zeros_like(ymat)

    # make axes
    f,ax = plt.subplots(ncols=2)
    # set title
    f.suptitle(Path(path).stem)
    # create initial one to set aspect ratio
    ax[0].imshow(np.zeros_like(umat,dtype="uint8"),cmap="hot")
    ax[1].set_aspect(ax[0].get_aspect())
    # set labels
    ax[0].set_title("Temperature (C)")
    ax[1].set_title("Optical Flow")
    # create quiver object to update
    qmat = ax[0].quiver(xmat,ymat,umat,vmat,pivot="mid",scale=1e4,cmap="hot")

    # loop collecting frames
    def update_quiver(num,Q,X,Y):
        frame = data[max(0,num-1),:,:]
        prvs = frame2gray(frame)
        frame_next = data[num,:,:]
        now = frame2gray(frame_next)
        # perform optical flow looking at differences between current and prev frame
        flow = cv2.calcOpticalFlowFarneback(prvs, now, None, *args)
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        Q.set_UVC(umat,vmat)
        ax[0].imshow(frame_next,cmap="hot")
        return qmat
    if outf is None:
        outf = Path(path).stem+"_dense_optical_flow_temp-plt.avi"
    f.tight_layout()
    anim = FuncAnimation(f,update_quiver,frames=np.arange(nf),fargs=(qmat,xmat,ymat),interval=int((1/30)*1000),blit=False)
    anim.save(outf)

def denseOpticalFlowTemperatureQuiverSlider(path: str,hide_temp: bool = False,clip_plasma: bool = True,flow_args: list |None= None):  # noqa: FBT001, FBT002
    # open soruce file
    if flow_args is None:
        flow_args = []
    data = np.load(path)["arr_0"]
    if clip_plasma:
        data = data[:,:,:360]
    nf,r,c = data.shape
    # create meshgrid of X and Y
    xx = np.arange(c)
    yy = np.arange(r)
    xmat,ymat = np.meshgrid(xx,yy)
    umat = np.zeros_like(xmat)
    vmat = np.zeros_like(ymat)

    cp = CoaxialPlane()
    cp._shape = (r,c)

    # make axes
    f,ax = plt.subplots(ncols=3 if hide_temp else 4)
    # set title
    f.suptitle(Path(path).stem)
    # create initial one to set aspect ratio
    if not hide_temp:
        ax[0].imshow(np.zeros_like(umat,dtype="uint8"))
        # set labels
        ax[0].set_title("Temperature (C)")
        ax[1].set_title("Optical Flow")
        
        ax[1].set_xlim(0,c)
        ax[1].set_ylim(0,r)
        ax[1].invert_yaxis()
        ax[1].set_aspect("equal")
        Q = ax[1].quiver(xmat,ymat,umat,vmat,pivot="mid",scale=1e2)
        ax[2].contourf(np.zeros((r,c)))
        ax[2].set_title("U")
        ax[3].contourf(np.zeros((r,c)))
        ax[3].set_title("V")
    else:
        ax[0].set_title("Optical Flow")
        for aa in ax:
            aa.set_xlim(0,c)
            aa.set_ylim(0,r)
            aa.invert_yaxis()
            aa.set_aspect("equal")
        # create quiver object to update
        Q = ax[0].quiver(xmat,ymat,umat,vmat,pivot="mid",scale=1e2,cmap="hot")
        ax[1].contourf(np.zeros((r,c)))
        ax[1].set_title("U")
        ax[2].contourf(np.zeros((r,c)))
        ax[2].set_title("V")

    # adjust the main plot to make room for the sliders
    f.subplots_adjust(left=0.1,bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = f.add_axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(np.argmax(data),(nf,r,c))[0],
        valstep=1,
    )
    # loop collecting frames
    def update_quiver(val):
        num = int(freq_slider.val)
        frame = data[max(0,num-1),:,:]
        prvs = frame2gray(frame)
        frame = data[num,:,:]
        now = frame2gray(frame)
        # perform optical flow looking at differences between current and prev frame
        flow = cv2.calcOpticalFlowFarneback(prvs, now, None, *flow_args)
        umat = flow[...,0]
        vmat = flow[...,1]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ## mask values according to entropy
        ## mask and norm
        umat[umat==0] = None
        umat[vmat==0] = None
        # direction is converted to hue and magnitude is converted to value
        mag, ang = cv2.cartToPolar(umat, vmat)
        Q.set_UVC(umat,vmat,mag)
        if not hide_temp:
            ax[0].imshow(frame,cmap="hot")
            for coll in ax[2].collections:
                coll.remove()
            ax[2].contourf(umat,cmap="hot")
            for coll in ax[3].collections:
                ax[3].collections.remove(coll)
            ax[3].contourf(vmat,cmap="hot")
        else:
            for coll in ax[1].collections:
                coll.remove()
            ax[1].contourf(umat,cmap="hot")
            for coll in ax[2].collections:
                coll.remove()
            ax[2].contourf(vmat,cmap="hot")
        f.canvas.draw_idle()

    freq_slider.on_changed(update_quiver)
    plt.show()
        
def denseOpticalFlowTemperatureStreamSlider(path: str,hide_temp: bool = False,flow_args: list|None=None):  # noqa: FBT001, FBT002
    # open soruce file
    if flow_args is None:
        flow_args = []
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    # create meshgrid of X and Y
    xx = np.arange(c)
    yy = np.arange(r)
    xmat,ymat = np.meshgrid(xx,yy)

    cp = CoaxialPlane()
    cp._shape = (r,c)

    # make axes
    f,ax = plt.subplots(ncols=2 if not hide_temp else 1)
    # set title
    f.suptitle(Path(path).stem)
    # create initial one to set aspect ratio
    if not hide_temp:
        ax[0].imshow(np.zeros_like(xmat,dtype="uint8"))
        # set labels
        ax[0].set_title("Temperature (C)")
        ax[1].set_title("Optical Flow")
        
        ax[1].set_xlim(0,c)
        ax[1].set_ylim(0,r)
        ax[1].invert_yaxis()
        ax[1].set_aspect("equal")
    else:
        ax.set_title("Optical Flow")
        ax.set_xlim(0,c)
        ax.set_ylim(0,r)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        
    
    # adjust the main plot to make room for the sliders
    f.subplots_adjust(left=0.1,bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = f.add_axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(np.argmax(data),(nf,r,c))[0],
        valstep=1,
    )
    # loop collecting frames
    def update_quiver(val):
        num = int(freq_slider.val)
        frame = data[max(0,num-1),:,:]
        prvs = frame2gray(frame)
        frame = data[num,:,:]
        now = frame2gray(frame)
        # perform optical flow looking at differences between current and prev frame
        flow = cv2.calcOpticalFlowFarneback(prvs, now, None, *flow_args)
        umat = flow[...,0]
        vmat = flow[...,1]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ## mask and norm
        umat[umat==0] = None
        vmat[vmat==0] = None
        umat[umat>0] = None
        if hide_temp:
            ax.cla()
            # create quiver object to update
            ax.streamplot(xmat,ymat,umat,vmat)
        else:
            ax[1].cla()
            # create quiver object to update
            ax[1].streamplot(xmat,ymat,umat,vmat)
        if not hide_temp:
            ax[0].imshow(frame,cmap="hot")
        f.canvas.draw_idle()

    freq_slider.on_changed(update_quiver)
    plt.show()
    
def denseOpticalFlowTemperatureSlider(path: str,filt: list | None = None,crop_plasma: bool = False,**kwargs):  # noqa: ANN003, FBT001, FBT002
    """Perform dense optical flow analysis on the temperature data inside an OpenCV trackbar based GUI.

    Trackbars allows the user to scan across frames and tweak parameters

    Inputs:
        path : Path to NPZ file
        filt : Filtering function or list of filters to apply
        crop_plasma : Flag to crop the plasma using using hardcoded column 360
    """
    data = np.load(path)["arr_0"]
    if crop_plasma:
        data = data[:,:,:360]
    nf,r,c = data.shape
    hsv = np.zeros((r,c,3),dtype="uint8")
    hsv[..., 1] = 255

    cv2.namedWindow("dense-optical-temp")

    def on_change(val) -> None:
        # get frame position
        ni = int(cv2.getTrackbarPos("Frame","dense-optical-temp"))
        # ensure non-zero
        ni = max(1,ni)
        # get two frames
        prvs = data[ni-1,:,:]
        now = data[ni,:,:]
        # apply given filters to frame
        if filt is not None:
            try:
                for f in filt:
                    prvs = f(prvs)
            except TypeError:
                prvs = filt(prvs)
        # apply given filters to frame
        if filt is not None:
            try:
                for f in filt:
                    now = f(now)
            except TypeError:
                now = filt(now)
        # convert to gray
        prvs = frame2gray(prvs)
        now = frame2gray(now)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, now, None,
                   max(0.01,cv2.getTrackbarPos("Scale","dense-optical-temp")/100),
                   max(1,int(cv2.getTrackbarPos("Levels","dense-optical-temp"))),
                   max(1,int(cv2.getTrackbarPos("Winsz","dense-optical-temp"))),
                   max(1,int(cv2.getTrackbarPos("Its","dense-optical-temp"))),
                   max(1,int(cv2.getTrackbarPos("N","dense-optical-temp"))),
                   max(1.0,cv2.getTrackbarPos("sigma","dense-optical-temp")/10),
                   cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if kwargs.get("cmap","hsv") == "hsv":
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif kwargs.get("cmap","hsv") == "hsv_r":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = ang*180/np.pi/2
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif kwargs.get("cmap","hsv") == "mag_only":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif kwargs.get("cmap","hsv") == "yellow":
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = 0
            bgr = hsv
        elif kwargs.get("cmap","hsv") == "uv":
            hsv[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = 255
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif kwargs.get("cmap","hsv") == "gray":
            hsv[...,0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = hsv
        
        cv2.imshow("dense-optical-temp",cv2.hconcat([cv2.applyColorMap(now,cv2.COLORMAP_HOT),bgr]))
        cv2.waitKey(1)

    cv2.createTrackbar("Frame","dense-optical-temp",0,nf-1,on_change)

    cv2.createTrackbar("Scale","dense-optical-temp",0,99,on_change)
    cv2.createTrackbar("Levels","dense-optical-temp",0,10,on_change)
    cv2.createTrackbar("Winsz","dense-optical-temp",0,500,on_change)
    cv2.createTrackbar("Its","dense-optical-temp",0,10,on_change)
    cv2.createTrackbar("N","dense-optical-temp",0,9,on_change)
    cv2.createTrackbar("sigma","dense-optical-temp",10,30,on_change)
    cv2.createTrackbar("Frame","dense-optical-temp",0,nf-1,on_change)

    on_change(0)
    cv2.waitKey(0)

def videoTemperatureFFT(path: str,outf: str|None = None,use_log_mag: bool = True):  # noqa: FBT001, FBT002
    from scipy.fft import fft2, fftshift
    # load temperature
    temp = np.load(path)["arr_0"]
    nf,r,c = temp.shape
    if outf is None:
        outf = Path(path).stem+"_fft_temp.avi"
    # read first frame
    # perform fft on the frame
    out = rfft2(temp[0,:,:])
    mag = np.abs(out)
    phase = np.angle(out)

    # convert amplitude to grayscale
    mag_norm = np.zeros((r,c),dtype="uint8")
    mag_norm = cv2.normalize(mag,mag_norm,0,255,cv2.NORM_MINMAX).astype("uint8")
    if use_log_mag:
        cc = 255 / np.log(1 + np.max(mag_norm))
        mag_norm = cc * (np.log(mag_norm + 1))
        mag_norm = mag_norm.astype("uint8")

    # convert phase to grayscale        
    phase_norm = np.zeros((r,c),dtype="uint8")
    phase_norm = cv2.normalize(phase,phase_norm,0,255,cv2.NORM_MINMAX).astype("uint8")

    # stack to form image
    frame = np.hstack((mag_norm,phase_norm))
    # get shape
    r,c = frame.shape
    # create the video writer using shape of frame
    writer = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc(*"mjpg"),30.0,(c,r),0)
    if not writer.isOpened():
        msg = f"Failed to open output file {outf}!"
        raise OSError(msg)

    writer.write(frame)
    # iterate over each frame
    for f in range(1,nf):
        out = fft2(temp[f,:,:])
        out = fftshift(out)
        mag = np.abs(out)
        phase = np.angle(out)

        # convert amplitude to grayscale
        mag_norm = np.zeros((r,c),dtype="uint8")
        mag_norm = cv2.normalize(mag,mag_norm,0,255,cv2.NORM_MINMAX).astype("uint8")
        if use_log_mag:
            cc = 255 / np.log(1 + np.max(mag_norm))
            mag_norm = cc * (np.log(mag_norm + 1)).astype("uint8")
            mag_norm = mag_norm.astype("uint8")
        # convert phase to grayscale        
        phase_norm = np.zeros((r,c),dtype="uint8")
        phase_norm = cv2.normalize(phase,phase_norm,0,255,cv2.NORM_MINMAX).astype("uint8")

        frame = np.hstack((mag_norm,phase_norm))
        writer.write(frame)
    writer.release()

def plotTemperatureFFT(path: str,**kwargs) -> tuple[plt.Figure, plt.Figure]:  # noqa: ANN003
    from scipy.fft import rfft2
    # load temperature
    temp = np.load(path)["arr_0"]
    # get temp shape
    nf,r,c = temp.shape
    # set of stats for magnitude
    mag_min = []
    mag_max = []
    mag_mean = []
    mag_var = []
    # set of status for phase
    phase_min = []
    phase_max = []
    phase_mean = []
    phase_var = []
    # iterate over the frames
    for f in range(0,nf):
        # perform fft ignoring half the signal due to symmetry
        out = rfft2(temp[f,:,:])
        # find magnitude
        mag = np.abs(out)
        # find phase
        phase = np.angle(out)
        # add mag stats to lists
        mag_min.append(mag.min())
        mag_max.append(mag.max())
        mag_mean.append(mag.mean())
        mag_var.append(mag.var())
        # add phase stats to lists
        phase_min.append(phase.min())
        phase_max.append(phase.max())
        phase_mean.append(phase.mean())
        phase_var.append(phase.var())

    fname = Path(path).stem
    # plot mag stats
    fmag,ax = plt.subplots(nrows=2,ncols=2,sharex=True,constrained_layout=True)
    ax[0,0].plot(mag_min)
    ax[0,1].plot(mag_max)
    ax[1,0].plot(mag_mean)
    ax[1,1].plot(mag_var)
    ax[0,0].set(xlabel="Frame",ylabel="Min FFT Mag",title="Minimum FFT Magnitude")
    ax[0,1].set(xlabel="Frame",ylabel="Max FFT Mag",title="Maximum FFT Magnitude")
    ax[1,0].set(xlabel="Frame",ylabel="Mean FFT Mag",title="Average FFT Magnitude")
    ax[1,1].set(xlabel="Frame",ylabel="Var FFT Mag",title="Variance FFT Magnitude")
    fmag.suptitle(kwargs.get("title",f"{fname} FFT Magnitude"))
    # plot phase stats
    fphase,ax = plt.subplots(nrows=2,ncols=2,sharex=True,constrained_layout=True)
    ax[0,0].plot(phase_min)
    ax[0,1].plot(phase_max)
    ax[1,0].plot(phase_mean)
    ax[1,1].plot(phase_var)
    ax[0,0].set(xlabel="Frame",ylabel="Min FFT Phase",title="Minimum FFT Phase")
    ax[0,1].set(xlabel="Frame",ylabel="Max FFT Phase",title="Maximum FFT Phase")
    ax[1,0].set(xlabel="Frame",ylabel="Mean FFT Phase",title="Average FFT Phase")
    ax[1,1].set(xlabel="Frame",ylabel="Var FFT Phase",title="Variance FFT Phase")
    fphase.suptitle(kwargs.get("title",f"{fname} FFT Phase"))
    
    return fmag, fphase

def sliderTemperatureFFT(path:str) -> None:
    from matplotlib.widgets import Slider
    from matplotlib.colors import LogNorm
    from scipy.fft import fft2, fftshift
    # load data
    temp = np.load(path)["arr_0"]
    # get number of frames
    nf,r,c = temp.shape
    # create 3 aaxes
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    # hide the axes for easier viewing
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)
    # plot first frame
    ax[0].imshow(temp[0,:,:],cmap="hot")
    asp = ax[0].get_aspect()
    # force all axes to
    ax[1].set_aspect(asp)
    ax[2].set_aspect(asp)
    # set labels
    ax[0].set_title("Temperature")
    ax[1].set_title("Magnitude")
    ax[2].set_title("Phase")
    f.suptitle(Path(path).stem)
    # create axis for slider
    axidx = plt.axes([0.25, 0.1, 0.65, 0.03])
    # slider is for selecting a target frame
    # initially set to the frame with the highest temperature to save scrolling
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(temp.argmax(),temp.shape)[0],
        valstep=1,
        valfmt="%d",
    )
    # update function for slider
    def on_changed(val) -> None:
        # get target frame
        ni = int(idx_slider.val)
        frame = temp[ni,:,:]
        # perform fft on the temperature
        out = fft2(frame)
        # shift it so the 0-mag is in the centre
        out = fftshift(out)
        # get the magnitude
        mag = np.abs(out)
        # get the phase        
        phase = np.angle(out)
        # show temperature using hot colormap
        ax[0].imshow(frame,cmap="hot")
        # plot the mag and phase on log scale
        ax[1].pcolormesh(mag,cmap="gray",norm=LogNorm())
        ax[2].pcolormesh(phase,cmap="gray",norm=LogNorm())
    # set callback on slider
    idx_slider.on_changed(on_changed)
    # show initial frames
    on_changed(0)
    plt.show()

def sliderTemperatureFFTMask(path: str,mtype: int = cv2.MARKER_CROSS,size: int = 20):
    from matplotlib.widgets import Slider
    from matplotlib.colors import LogNorm
    from scipy.fft import fft2, fftshift, ifft2
    # load data
    temp = np.load(path)["arr_0"]
    # get number of frames
    nf,r,c = temp.shape
    # create 3 aaxes
    f,ax = plt.subplots(ncols=4,constrained_layout=True)
    # hide the axes for easier viewing
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)
    # plot first frame
    ax[0].imshow(temp[0,:,:],cmap="hot")
    asp = ax[0].get_aspect()
    # force all axes to
    ax[1].set_aspect(asp)
    ax[2].set_aspect(asp)
    # set labels
    ax[0].set_title("Temperature")
    ax[1].set_title("Magnitude")
    ax[2].set_title("Phase")
    f.suptitle(Path(path).stem)
    # create axis for slider
    axidx = plt.axes([0.25, 0.1, 0.65, 0.03])
    # slider is for selecting a target frame
    # initially set to the frame with the highest temperature to save scrolling
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(temp.argmax(),temp.shape)[0],
        valstep=1,
        valfmt="%d",
    )

    axsize = plt.axes([0.25, 0.05, 0.65, 0.03])

    size_slider = Slider(
        ax=axsize,
        label="size",
        valmin=0,
        valmax=max([r,c])//2,
        valinit=10,
        valstep=1,
        valfmt="%d",
    )
    
    # update function for slider
    def on_changed(val) -> None:
        # get target frame
        ni = int(idx_slider.val)
        frame = temp[ni,:,:]
        # make mask
        mask = np.zeros((r,c),np.uint8)
        mask = cv2.drawMarker(mask,((c-1)//2,(r-1)//2),255,mtype,int(size_slider.val),int(size_slider.val)//2)
        cv2.imshow("mask",mask)
        # perform fft on the temperature
        out = fft2(frame)
        # shift it so the 0-mag is in the centre
        out = fftshift(out)
        # mask array
        out[mask!=MAX_8_BIT] = 0
        # get the magnitude
        mag = np.abs(out)
        # get the phase        
        phase = np.angle(out)
        # show temperature using hot colormap
        ax[0].imshow(frame,cmap="hot")
        # plot the mag and phase on log scale
        ax[1].imshow(mag,cmap="gray",norm=LogNorm())
        ax[2].imshow(phase,cmap="gray",norm=LogNorm())
        ax[3].imshow(np.abs(ifft2(out)),cmap="hot")
    # set callback on slider
    idx_slider.on_changed(on_changed)
    size_slider.on_changed(on_changed)
    # show initial frames
    on_changed(0)
    plt.show()

def sliderTemperatureEntropy(path: str,segs: int = 29):
    """Matplotlib GUI for testing and showing entropy masking.

    Each temperature frame axis is broken into segs number of sections.
    The entropy of each section is calculated and the resulting matrix is grayscaled.
    The image is then masked using Otsu threshold and applied to the source image.
    The masked temperature frame is them recolormapped.

    The GUI contains a series of axes to show the different stages of the process.
    The slider moves to a different frame in the source dataset. The slider
    is set to start at the frame with the highest temperature to save scanning
    for the active sections

    Inputs:
        path : Filepath to NPZ file
        segs : Number of sections along each axes to use
    """
    # load data
    temp = np.load(path)["arr_0"]
    # get number of frames
    nf,r,c = temp.shape
    # create 3 aaxes
    f,ax = plt.subplots(ncols=4,constrained_layout=True)
    # plot first frame
    ax[0].imshow(temp[0,:,:],cmap="hot")
    asp = ax[0].get_aspect()
    # hide the axes for easier viewing
    for aa in ax:
        aa.xaxis.set_visible(False)
        aa.yaxis.set_visible(False)
        aa.set_aspect(asp)

    # set labels
    ax[0].set_title("Temperature")
    ax[1].set_title("Entropy")
    ax[2].set_title("Entropy Mask")
    ax[3].set_title("Masked")
    f.suptitle(Path(path).stem)
    # create axis for slider
    axidx = plt.axes([0.25, 0.1, 0.65, 0.03])
    # slider is for selecting a target frame
    # initially set to the frame with the highest temperature to save scrolling
    idx_slider = Slider(
        ax=axidx,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valinit=np.unravel_index(temp.argmax(),temp.shape)[0],
        valstep=1,
        valfmt="%d",
    )

    def frame_entropy(frame,segs=segs):
        r,c = frame.shape
        # break image into sections
        ent = np.zeros((r,c),dtype="float32")
        for ri in range(r//segs):
            for ci in range(c//segs):
                ent[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)] = entropy(np.unique(frame[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)],return_counts=True)[1],base=None)
        return ent

    # update function for slider
    def on_changed(val) -> None:
        # get target frame
        ni = int(idx_slider.val)
        frame = temp[ni,:,:]
        ent = frame_entropy(frame)
        # show temperature using hot colormap
        ax[0].imshow(frame,cmap="hot")
        # plot the mag and phase on log scale
        im = ax[1].imshow(ent,cmap="gray")
        ccmap = im.get_cmap()
        im_rgb = ccmap(im._A)[:,:,0]  # noqa: SLF001
        # convert to 0-255
        img_rgb = (255.0*im_rgb).astype("uint8")
        # threshold
        _,thresh = cv2.threshold(img_rgb,120,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # show mask
        ax[2].imshow(thresh,cmap="gray", vmin=0,vmax=255)
        # apply mask to data replacing black pixels with min frame value
        frame_mask = frame.copy()
        frame_mask[thresh==0] = frame_mask.min()
        ax[3].imshow(frame_mask,cmap="hot",norm=LogNorm())
    # set callback on slider
    idx_slider.on_changed(on_changed)
    # show initial frames
    on_changed(0)
    plt.show()

def videoTemperatureEntropy(path: str,opath: str|None = None,segs: int = 29):
    """Break the video frames into segs x segs sections and calculate entropy.

    Entropy is colour mapped using opencv hot colormap

    Inputs:
        path : NPZ file path
        opath : Output path for video
        segs : Number of segments along each axis
    """
    from scipy.stats import entropy
    if opath is None:
        opath = Path(path).stem+"_entropy_mask.avi"
    # load data
    temp = np.load(path)["arr_0"]
    # get number of frames
    nf,r,c = temp.shape
    # create 3 aaxes
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    out = cv2.VideoWriter(opath,fourcc,30.0,(int(c*4),r),1)
    if not out.isOpened():
        msg = f"Failed to open output file at {opath}!"
        raise ValueError(msg)

    def frame_entropy(frame,segs=segs):
        r,c = frame.shape
        # break image into sections
        ent = np.zeros((r,c),dtype="float32")
        for ri in range(r//segs):
            for ci in range(c//segs):
                ent[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)] = entropy(np.unique(frame[segs*ri:segs*(ri+1),segs*ci:segs*(ci+1)],return_counts=True)[1],base=None)
        return ent

    norm = plt.Normalize()
    # update function for slider
    for ni in range(nf):
        frame = temp[int(ni),:,:]
        frame_rgb = cv2.applyColorMap(frame2gray(frame),cv2.COLORMAP_HOT, norm=norm)
        ent = frame_entropy(frame)
        # convert entropy to grayscale
        ent_gray = frame2gray(ent)
        # stack to form RGB image
        if len(ent_gray.shape)<3:
            ent_stack = np.dstack(3*(ent_gray,))
        else:
            ent_stack = ent_gray.copy()
        # threshold
        _,thresh = cv2.threshold(ent_gray,120,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_stack = np.dstack(3*(thresh,))
        # apply mask to data replacing black pixels with min frame value
        frame_mask = frame.copy()
        frame_mask[thresh==0] = frame_mask.min()
        # convert masked value to gray and then apply colormap
        frame_mask = cv2.applyColorMap(frame2gray(frame_mask),cv2.COLORMAP_HOT)
        # place the original, entropy, threshold mask and masked image side by side
        out.write(np.column_stack((frame_rgb,ent_stack,thresh_stack,frame_mask)))
    out.release()

def findOrderedStripes(path: str,order: str = "top-bottom",as_ct: bool = True) -> list:  # noqa: FBT001, FBT002
    """Load stripe mask and find the edges of each white blob in the image.

    The input order controls the stripes order.
    Supported orders:
        + top-bottom : From top to bottom
        + bottom-top : From bottom to top

    The flag as_ct is to control how the mask is returned. If as_ct is True,
    then the contours of each stripe is returned. When False, separate masks
    are generated for each stripe

    Inputs:
        path : JPG mask file
        order : Order in which to return the stripe
        as_ct : Flag to return the stripes as contours. Default True

    List of contours if as_ct is True else list of binary images for each stripe
    """
    # load mask
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if isinstance(path, str) else path
    # additional thresholding to deal with encoding errors
    _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    r,c = mask.shape
    # search for contours in edge mask
    cts,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # check if any contours have been found
    if len(cts)==0:
        msg = "Failed to find contours!"
        raise ValueError(msg)
    # search for where hier is -1
    ci = np.where(hier[0,:,-1]==-1)[0]
    # check if any match
    if ci.shape[0]==0:
        msg = "Cannot find any contours whose hierarchy is -1!"
        raise ValueError(msg)
    # get contours
    cts = [cts[ii] for ii in ci]
    # sort by y-position in image in the required order
    if order == "top-bottom":
        cts.sort(key=lambda x : x[:,0,-1].min())
    elif order == "bottom-top":
        cts.sort(key=lambda x : x[:,0,-1].max())
    else:
        msg = f"Unsupported target order! Received {order}!"
        raise ValueError(msg)
    # return the ordered stripes as contour paths
    if as_ct:
        return cts
    # convert contours to masks to use
    masks = []
    for cc in cts:
        # create 3 channel image to draw on
        mask_cc = np.zeros((r,c,3),mask.dtype)
        # fill in contour area and add first channel to list
        masks.append(cv2.drawContours(mask_cc,[cc],-1,(255,255,255),-1)[:,:,0])
    return masks

def plotMaxStripeTemperature(path: str,mask: str,**kwargs) -> plt.Figure:  # noqa: ANN003
    """Plot the max temperature of each stripe in the target file using specified mask.

    The mask is a binary image where the white pixels are the location of the stripes.
    The function findOrderedStripes splits the mask into the individual stripes

    The mask temperature within each stripe is plotted on the same axis

    Inputs:
        path : Path to NPZ file
        mask : Path to mask JPG file
        title : Figure title. Defailt {filename} stripe temperatures
        stripes_labels : Labels for each stripe. By default it"s Stripe {stripe index}
        no_all : Flag to not plot all. Default False.
        clip : Clip the signal to either side of the activity using find_peaks. Default False.
    Returns figure
    """
    from scipy.signal import find_peaks
    masks_list = findOrderedStripes(mask,as_ct=False)
    f,ax = plt.subplots(constrained_layout=True)
    data = np.load(path)["arr_0"]
    nf = data.shape[0]
    x = np.arange(0,nf,1,dtype="int32")
    # plot the global frame temperature for reference
    if not kwargs.get("no_all",True):
        maxt = data.max((1,2))
        ax.plot(maxt,label="All")
    # build stripe label list
    stripe_labels = kwargs.get("stripe_labels",[f"Stripe {mi+1}" for mi in range(len(masks_list))])
    pks = []
    for mask,label in zip(masks_list,stripe_labels):
        data_mask = data.copy()
        # brute force masking
        data_mask[:,mask!=MAX_8_BIT] = 0
        # get max temperature within region
        maxt = data_mask.max((1,2))
        ax.plot(x,maxt,label=label)
        if kwargs.get("clip",False):
            pks+= find_peaks(maxt,height=0.9*maxt.max())[0].tolist()
    ax.legend()
    ax.set(xlabel="Frame Number",ylabel=r"Max Frame Temperature ($^\circ$C)")
    # attempt to clip to activity
    if kwargs.get("clip",False):
        ax.set_xlim(min(pks)-100,max(pks)+100)
    f.suptitle(kwargs.get("title",f"{Path(path).stem} Stripe Temperatures"))
    return f

def gaussianMixtureHist(path: str,nb: int = 60):
    """Attempt to classify each frame using a Gaussian mixture model.

    The results are displayed as part of an OpenCV GUI application

    Inputs:
        path : Path to input NPZ file
        nb : Number of bins for histogram (not used)
    """
    from sklearn.mixture import GaussianMixture
    # load npz file
    data = np.load(path)["arr_0"]
    nf,r,c = data.shape
    # create named window to show results
    cv2.namedWindow("GMM")
    def on_changed(val) -> None:
        ni = int(cv2.getTrackbarPos("Frame","GMM"))
        # get frame
        frame = data[ni,:,:]
        # binary classify the image
        classif = GaussianMixture(n_components=2)
        classif.fit(frame.reshape((frame.size, 1)))
        # find threshold in menas
        threshold = np.mean(classif.means_)
        # binarize image
        binary_img = frame > threshold
        binary_img = np.dstack(3*(255*binary_img.astype("uint8"),))
        frame_gray = frame2gray(frame)
        frame_rgb = cv2.applyColorMap(frame_gray,cv2.COLORMAP_HOT)
        # also try with 3 classes
        classif = GaussianMixture(n_components=3)
        img_three = classif.fit_predict(frame.reshape((frame.size, 1))).reshape(r,c)
        img_three = img_three.astype("float16")
        img_three *= (255/img_three.max())
        # stack 2 classes and 3 classes together
        cv2.imshow("GMM",np.column_stack((frame_rgb,binary_img,np.dstack(3*(img_three.astype("uint8"),)))))
    cv2.createTrackbar("Frame","GMM",0,nf-1,on_changed)
    on_changed(0)
    cv2.waitKey(0)

def temperatureFlatten(path: str):
    from matplotlib.widgets import Slider
    # load numpy file
    data = np.load(path)["arr_0"]
    # get global min and max
    tmin = data.min()
    tmax = data.max()
    nf,r,c = data.shape
    # make 2D grid for plotting
    xx = np.arange(0,c,1,dtype="uint16")
    yy = np.arange(0,r,1,dtype="uint16")
    xmat,ymat = np.meshgrid(xx,yy)

    # setup figure
    fig,ax = plt.subplots(ncols=2)
    fig.suptitle(Path(path).stem)
    # plot first frame as a contour plot
    ax[0].contourf(xmat,ymat,data[0,:,:],cmap="hot")
    # flatten frame and plot as a line
    line, = ax[1].plot(data[0,:,:].flatten(),"b-")
    ax[1].set_ylim(tmin,tmax)
    # add a slider for changing frame
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label="Frame",
        valmin=0,
        valmax=nf-1,
        valstep=1,
    )

    def update(val):
        ni = int(freq_slider.val)
        # get target frame + update contour
        frame = data[ni,:,:]
        ax[0].contourf(xmat,ymat,frame,cmap="hot")
        # update line plot
        line.set_ydata(frame.flatten())
        fig.canvas.draw_idle()

    freq_slider.on_changed(update)
    plt.show()


def morphCorrectPlate(img: np.ndarray,corners: list | None = None) -> np.ndarray:
    """Attempt to correct the perspective of the plate using a warping.

    Corners are provided by the user

    Inputs:
        img : RGB image to process
        corners : List of XY corner locations

    Returns result of warping
    """
    if corners is None:
        corners = [[162,52],[376,0],[376,348],[154,297]]
    if isinstance(img,str):
        img = cv2.imread(img)
    # targets if corners were moved diagonally to nearest edge
    maxw = max([x[0] for x in corners])
    maxh = max([x[1] for x in corners])
    targets = np.float32([(0,0),(maxw-1,0),(maxw-1,maxh-1),(0,maxh-1)])
    # get perspective transform
    warp = cv2.getPerspectiveTransform(np.float32(corners),np.float32(targets))
    # warp setting target size based on targets
    return cv2.warpPerspective(img,warp,(maxh,maxw),flags=cv2.INTER_LINEAR)


def morphCorrectPlateVideo(path: str,opath: str|None = None,corners: list | None = None):
    """Attempt to correct the perspective of the plate using a warping and write the result to a video.

    Depends on morphCorrectPlate

    Corners are provided by the user

    Inputs:
        path : Input path to video file to process
        opath : Output path to write the result to
        corners : List of XY corner locations
    """
    if corners is None:
        corners = [[162,52],[376,0],[376,348],[152,310]]
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        msg = f"Failed to open file {path}!"
        raise ValueError(msg)
    if opath is None:
        opath = Path(path).stem+"-plate-warp.avi"
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # create video writer
    out = None
    ret,frame = video.read()
    while ret:
        morph = morphCorrectPlate(frame,corners)
        if out is None:
            rows,width,_=morph.shape
            out = cv2.VideoWriter(opath,fourcc,30.0,(width,rows),1)
        out.write(morph)
        ret,frame = video.read()
    if out is not None:
        out.release()


def convolveColor(path: str,force_odd: bool = True, kernel = None):
    """OpenCV slider application that applies a supplied kernel to the data and shows the result.

    The user is presented with sliders for the target frame index, weight of the kernel, and the size

    If kernel is None, an array of weights set to the value of the Weight slider is used.
    The size of the kernel is controlled by the sliders.

    Inputs:
        path : Input path to video file
        force_odd : Flag to force the kernel size to be odd shaped
        kernel : Function used to create kernel. Default None.
    """
    cv2.namedWindow("convolve")
    # open color image
    source = cv2.VideoCapture(path)
    if not source.isOpened():
        msg = f"Cannot open source file {path}!"
        raise ValueError(msg)
    nf = int(source.get(cv2.CAP_PROP_FRAME_COUNT ))
    # define function for updating
    def on_changed(val) -> None:
        # get frame count
        ni = int(cv2.getTrackbarPos("frame","convolve"))
        # get frame
        source.set(cv2.CAP_PROP_POS_FRAMES,ni)
        ret,frame = source.read()
        if not ret:
            warn(f"Failed to read frame {ni}!", stacklevel=2)
            return
        # get filter size
        r = max(1,int(cv2.getTrackbarPos("rows","convolve")))
        if force_odd and ((r%2)==0):
            r += 1
        c = max(1,int(cv2.getTrackbarPos("cols","convolve")))
        if force_odd and ((c%2)==0):
            c += 1

        weight = max(1,cv2.getTrackbarPos("weight","convolve"))
        if kernel is None:
            win = np.ones((r,c),np.float32)/weight
        else:
            try:
                win = kernel(r,c,weight)
            except TypeError:
                win = kernel/weight

        frame = cv2.blur(frame,(r,c))
        res = cv2.filter2D(frame,-1,win)
        th = np.dstack(3*(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),))
        cv2.imshow("convolve",cv2.hconcat((frame,cv2.applyColorMap(res,cv2.COLORMAP_HOT),th)))
    # make trackbars
    cv2.createTrackbar("frame","convolve",0,nf-1,on_changed)
    cv2.createTrackbar("rows","convolve",1,20,on_changed)
    cv2.createTrackbar("cols","convolve",1,20,on_changed)
    cv2.createTrackbar("weight","convolve",1,256,on_changed)

    on_changed(0)
    cv2.waitKey(0)
    source.release()


def convolveTemperature(path: str, force_odd: bool = True, kernel: np.ndarray = None):
    """Load NPZ file of 2D arrays and investigate convolving a 2D kernel with the data.

    Provides a Figure with trackbars to change the kernel

    Inputs:
        path : Numpy array to load
        force_odd: Force odd sized kernels
        kernel : 2D kernel to convolve. See scipy.signal.convolve2d
    """
    cv2.namedWindow("convolve")
    # open color image
    source = np.load(path)["arr_0"]
    nf,r,c = source.shape
    # define function for updating
    def on_changed(val) -> None:
        # get frame count
        ni = int(cv2.getTrackbarPos("frame","convolve"))
        # get frame
        frame = source[ni,:,:]
        # get filter size
        r = max(1,int(cv2.getTrackbarPos("rows","convolve")))
        if force_odd and ((r%2)==0):
            r += 1
        c = max(1,int(cv2.getTrackbarPos("cols","convolve")))
        if force_odd and ((c%2)==0):
            c += 1
        weight = max(1,cv2.getTrackbarPos("weight","convolve"))
        if kernel is None:
            win = np.ones((r,c),np.float32)/weight
        else:
            try:
                win = kernel(r,c,weight)
            except TypeError:
                win = kernel/weight
        # convolve image with filter
        res = convolve2d(frame,win,boundary="symm", mode="same")
        # colormap
        res = cv2.applyColorMap(frame2gray(res),cv2.COLORMAP_HOT)
        # concat results together
        cv2.imshow("convolve",cv2.hconcat((cv2.applyColorMap(frame2gray(frame),cv2.COLORMAP_HOT),cv2.applyColorMap(frame2gray(res),cv2.COLORMAP_HOT))))
    # make trackbars
    cv2.createTrackbar("frame","convolve",0,nf-1,on_changed)
    cv2.createTrackbar("rows","convolve",1,5,on_changed)
    cv2.createTrackbar("cols","convolve",1,5,on_changed)
    cv2.createTrackbar("weight","convolve",1,100,on_changed)
    plt.show()


def formatEstAllPlasmaSizeToDF(data: dict) -> pd.DataFrame:
    """Format the nested dict from estAllPlasmaSize into a pandas dataframe.

    The dict is arranged by source filename and contains a sub-dict for each
    temperature threshold applied to it

    Filename -> Temperature threshold -> label : User defined label
                                            data : 2D array of stats collected
                                            cols : Column labels for data

    Each of the temp threshold dicts are converted to a dataframe with
    the data columns and the additional columns Label, File and Threshold (C).

    These dfs are then concatenated together to form a giant dataframe with
    ignore_index to False and is returned

    Inputs:
        data : Dict structure from estAllPlasmaSize

    Returns combined pandas dataframe
    """
    dfs_to_concat = []
    for k,v in data.items():
        for t,d in v.items():
            temp = pd.DataFrame(d["data"],columns=d["cols"])
            temp["Label"] = d["label"]
            temp["File"] = k
            temp["Threshold (C)"] = float(t)
            dfs_to_concat.append(temp)
    return pd.concat(dfs_to_concat,ignore_index=True)    


def stackContourImgs(imgs: dict) -> dict:
    new_dict = {}
    # iterate over filenames and sub dicts
    for fn,v in imgs.items():
        idict = {}
        # sub dicts organised by threshold and contains a list of images
        for th,ii in v.items():
            # stack images into a numpy array to make it easier to handle
            # new dtype is to handle ops that would push the limit beyong 255
            stack = np.dstack(ii)
            r,c,nf = stack.shape
            # res is the final result
            res = np.zeros((r,c),np.uint8)
            # combine them all together via bitwise_or
            for n in range(nf):
                res = cv2.bitwise_or(res,stack[...,n])
            idict[th] = res
        new_dict[fn] = idict
    return new_dict


def plotMaxContourLW(data_df: pd.DataFrame,th: str = "all",**kwargs) -> list[plt.Figure]:
    from matplotlib.lines import Line2D
    # ensure target thresholds are something iterable
    if isinstance(th,(int,float)):
        th = [th]
    elif th == "all":
        th = list(data_df["Threshold (C)"].unique())
    # iterate over thresholds
    figs = []
    for t in th:
        # filter and find max
        filt = data_df[data_df["Threshold (C)"]==t]
        mx = filt.groupby("Label").max()
        # initially plot the width/length of the contour
        f,ax = plt.subplots()
        ax = sns.lineplot(mx,x=mx.index,y="Width (pix)",ax=ax)
        ax.set(xlabel=kwargs.get("ax_xlabel","Plasma Gas Flow Rate"),ylabel=kwargs.get("ax_ylabel","Max Contour Length (pix)"))
        ax.grid(False)  # noqa: FBT003
        # make a twin axis to plot height/width
        tax = ax.twinx()
        tax.plot(mx["Height (pix)"],"r")
        tax.set_ylabel(kwargs.get("tax_ylabel","Max Contour Width (pix)"))
        tax.grid(False)  # noqa: FBT003
        # set title
        ax.figure.suptitle(rf"Max Plasma Contour Length and Width (Tlim$\geq$ {t} $^\circ$C)")
        # make legend
        custom_lines = [Line2D([0],[0],color="b",lw=2),Line2D([0],[0],color="r",lw=2)]
        ax.legend(custom_lines,["Length","Width"],loc=kwargs.get("legend_loc",6))
        figs.append(ax.figure)
    return figs


def plotContourAreaVsTime(data_df: pd.DataFrame | list[np.ndarray], **kwargs) -> list[plt.Figure]:
    """Plot the Contour Area for different temperature thresholdsd against time.

    If a dataframe is supplied, it"s assumed to be created by one of the other methods so
    each unique value of Threshold (C) is iterated over and used to set the title for each figure

    If a list of numpy arrays is supplied, then each of those are used and no custom labelling is applied

    Inputs:
        data_df : Pandas dataframe or list of numpy array
        labels : List of labels corresponding to each array when a list of arrays is given

    Returns list of figures for each temperature threshold
    """
    figs = []
    if isinstance(data_df, pd.DataFrame):
        for th in data_df["Threshold (C)"].unique():
            filt = data_df[data_df["Threshold (C)"]==th]
            f,ax = plt.subplots()
            sns.lineplot(filt,x="Time (s)",y="Contour Area ($pix^2$)",hue="Label",ax=ax)
            f.suptitle(rf"Estimated Plasma Area vs Time (Tlim$\geq$ {th} $^\circ$C)")
            figs.append(f)
    elif isinstance(data_df, list):
        for filt, label in zip(data_df, kwargs.get("labels", [f"Array {i}" for i in range(len(data_df))])):
            f,ax = plt.subplots()
            ax.plot(np.arange(len(filt)/30.0), filt)
            ax.set(xlabel="Time [s]", ylabel="Contour Area ($pix^2$)")
            f.suptitle(f"Estimated Plasma Area vs Time {label}")
            figs.append(f)
    return figs


def plotAsymmetricalAngle(plasma_all_params: pd.DataFrame,agg_fn=np.nanmean,drop_file: int|str|None= None,group: str = "File") -> plt.Figure:
    """Plot the Asymmetrical Angle against time.

    Input:
        plasma_all_params: Dataframe of all plasma parameters generated by a previous function
        agg_fn : Aggregation function used for plotting. Default np.nanmean
        drop_file : Specific file in array by specific index or by name. This was a cheap way to deal with a bad file. Default -1.
        group : Column to group the variables by. Default File

    Returns plotted figure
    """
    # drop the target file
    if drop_file is not None:
        # drop the specific index
        if isinstance(drop_file,int):
            plasma_all_params = plasma_all_params[plasma_all_params.File != plasma_all_params.File.unique()[drop_file]]
        # drop file by name
        else:
            plasma_all_params = plasma_all_params[plasma_all_params.File != drop_file]
    # group and aggregate data using target function
    group_agg = plasma_all_params.groupby(plasma_all_params[group]).agg(agg_fn)
    ax= sns.lineplot(group_agg,x=group_agg.index,y="Asymmetrical Angle (degrees)")
    ax.set(ylabel="Plasma Gas Feed Rate (SL/min)")
    ax.figure.suptitle("Average Asymmetrical Angle (degrees) for Different Flow Rates")
    return ax

def _filterGroup(gp: pd.DataFrame,nstd: int = 3) -> pd.DataFrame:
    for c in gp.columns:
        if c in ["Frame Index","Time (s)","File"]:
            continue
        col = gp[c]
        col[col.gt(col.mean()+(nstd*col.std()))] = np.nan
        gp[c] = col
    return gp


def filterPlasmaParams(plasma_all_params: pd.DataFrame):
    """Group data by file and for each column replace the data that is greater than 3 std from mean with np.nan.

    Skips columns Frame Index, Time (s) and File.

    Input:
        plasma_all_params : Dataframe of information

    Returns filtered dataframe
    """
    return plasma_all_params.groupby("File").apply(_filterGroup)


def plotPlasmaParams(plasma_all_params: pd.DataFrame,opath: str = "."):
    """Iterate over each column in the dataframe, plot and save figures to a target folder.

    Designed for dataframes generated by collectPlasmaStats and plasmaGaussianParams

    It first plots the columns setting Hue based on column File.
    It then groups the data via File and then plots each column again. In this case,
    the saved files have the source filename added as a prefix.

    Inputs:
        plasma_all_params : Pandas dataframe
        opath : Output folder to save the plots in
    """
    for c in plasma_all_params.columns:
        if c in ["Frame Index","Time (s)","File","Flow Rate (SL/MIN)"]:
            continue
        c_title = c
        if "Aspect Ratio" in c:
            c_title = "Aspect Ratio"
        ax = sns.lineplot(plasma_all_params,x="Time (s)",y=c,hue="File")
        ax.figure.suptitle(c_title)
        ax.figure.savefig(Path(opath) / f"{c_title}.png")
        plt.close("all")

    for fn,gp in plasma_all_params.groupby("File"):
        for c in gp.columns:
            if c in ["Frame Index","Time (s)","File","Flow Rate (SL/MIN)"]:
                continue
            c_title = c
            if "Aspect Ratio" in c:
                c_title = "Aspect Ratio"
            ax = sns.lineplot(gp,x="Time (s)",y=c,color="magenta")
            ax.figure.suptitle(f"{fn}\n{c_title}")
            ax.figure.savefig(Path(opath) / f"{fn}-{c_title}.png")
            plt.close("all")


def plotAreaAspectRatio(plasma_all_params: pd.DataFrame,opath: str = ".", **kwargs):
    """Plot the aspect ratio and contour area against time along two separate axis.

    This was a requested plot

    The supplied dataframe must have the following columns at least:
        File : Filename used to generate the data
        Aspect Ratio (w/h) : Aspect Ratio between the width and height of the gaussian distribution
        Contour Area (pix^2) : Area of the contour found to describe the plasma

    All filenames follow the format

        {fn}-area-aspect-ratio.png

    where fn is the source filename.

    Inputs:
        plasma_all_params : Dataframe with at least the required columns
        opath : Output folder to save the images to
        file_replace : Replace the filename in the title with the corresponding string
    """
    from matplotlib.lines import Line2D
    file_replace = kwargs.get("file_replace",None)
    # check if user wants to replace the filename with something else
    if file_replace is None:
        file_replace = {fn:fn for fn in plasma_all_params.File.unique()}
    for fn,gp in plasma_all_params.groupby("File"):
        f,ax = plt.subplots()
        sns.lineplot(gp,x="Time (s)",y="Aspect Ratio (Width/Height)",color="magenta",ax=ax)
        tax = ax.twinx()
        sns.lineplot(gp,x="Time (s)",y="Contour Area (pix^2)",color="black",ax=tax)
        ax.set(xlabel="Time (s)",ylabel="Gaussian Aspect Ratio (height/std. dev)")
        tax.set_ylabel("Contour Area (pixels squared)")
        f.suptitle(f"{file_replace[fn]}\nGaussian Aspect Ratio (height/std. dev) & Contour Area ($pix^2$)")
        custom_lines = [Line2D([0],[0],color=(1.0,0,1.0),lw=4),Line2D([0],[0],color=(0,0,0),lw=4)]
        ax.legend(custom_lines,["G.A.R. (height/std. dev)","Ct. Area ($pix^2$)"],loc="lower right",facecolor="white")
        f.savefig(Path(opath) /f"{fn}-area-aspect-ratio.png")
        plt.close(f)


def plotAverageContourArea(pstack: pd.DataFrame, x: str="Flow Rate (SL/MIN)",threshold: float = 1000.0) -> plt.Figure:
    """Find the average contour area for each flow rate and plot it.

    Checks if the specified user column exists, if not it defaults to the column File for grouping
    and warns the user as such.

    The input threshold is used for setting the title of the plot to indicate the temperature threshold used

    Inputs:
        pstack : Dataframe
        x : Column used for grouping and plotting on the x-axis. Default Flow Rate (SL/MIN).
        threshold : Temperature threshold used for generating the data. Used in title. Default 1000.0

    Returns matplotlib figure
    """
    if "Flow Rate (SL/MIN)" not in pstack.columns:
        warn("Cannot find Flow Rate! Default to File", stacklevel=2)
        x="File"
    # group by target x parameter
    mean_arr = pstack.groupby(x).apply(lambda x : x["Contour Area (pix^2)"].mean())
    # make a separate axis to ensure that it"s not overriding an existing axis
    f,ax = plt.subplots()
    ax = sns.lineplot(mean_arr,color="magenta",ax=ax)
    # set labels
    ax.set(xlabel="Flow Rate (SL/MIN)",ylabel="Average Contour Area (pixels squared)")
    ax.figure.suptitle(f"Average Contour Area per Flow Rate (SL/MIN)\nThreshold {threshold}°C")
    return f

def maskTemp(x:np.ndarray,lim:float=0.15) -> np.ndarray:
    x[x>=(lim*x.max())]=x.min()
    return x


def clipShape(x:np.ndarray,cclip:int | None = None,rclip:int | None = None) -> np.ndarray:
    r,c = x.shape
    if cclip:
        x = x[:,:min([cclip,c])]
    if rclip:
        x = x[:min([rclip,r]),:]
    return x


def maskShape(x:np.ndarray,cclip:int | None = None,rclip:int | None = None) -> np.ndarray:
    r,c = x.shape
    xmin = x.min()
    if cclip:
        x[:,min([cclip,c]):] = xmin
    if rclip:
        x[min([rclip,r]):,:] = xmin
    return x


def threshTemp(x:np.ndarray,tt:float=600.0) -> np.ndarray:
    x[x>=tt] = x.min()
    return x

def filt(hist):
    ii = np.where(hist[:,1]<75)[0]
    hist[ii,:] = 0
    return hist


def blockFill(x: np.ndarray,shape: tuple,fill: str = "min") -> np.ndarray:
    if len(shape)==4:
        ra,rb,ca,cb = shape
        if isinstance(fill,(int,float)):
            x[ra:rb,ca:cb] = fill
        elif fill == "min":
            x[ra:rb,ca:cb] = x.min()
        elif fill == "max":
            x[ra:rb,ca:cb] = x.max()
        return x
    if len(shape)==2:
        ra,ca = shape
        if isinstance(fill,(int,float)):
            x[ra:,ca:] = fill
        elif fill == "min":
            x[ra:,ca:] = x.min()
        elif fill == "max":
            x[ra:,ca:] = x.max()
        return x
    return None


def collectFolderAllPlasmaParams(path: str,tlim: float = 1000.0,**kwargs) -> pd.DataFrame:
    """Apply plasmaGaussianParams and collectPlasmaStats for each file found in path.

    Each found dataframe is stacked together into a single dataframe

    Inputs:
        path : Wildcard path to a folder of NPZ files
        tlim : Masking temperature. Default 1000.0
        **kwargs : See plasmaGaussianParams

    Returns a single dataframe
    """
    plasma_params_dfs = []
    plasma_other_params_dfs = []
    if isinstance(path,str):
        path = glob(path)
    for fn in path:
        f,df = plasmaGaussianParams(fn,lim=tlim,**kwargs)
        df["File"] = Path(fn).stem
        plasma_params_dfs.append(df)
        df = collectPlasmaStats(fn,tlim)
        df["File"] = Path(fn).stem
        plasma_other_params_dfs.append(df)

    plasma_all_params = pd.concat(plasma_params_dfs,ignore_index=True)
    plasma_all_params["Asymmetrical Angle (degrees)"] = np.degrees(plasma_all_params["Asymmetrical Angle (rads)"])
    # remove nans
    plasma_all_params.dropna(inplace=True)  # noqa: PD002

    plasma_other_params = pd.concat(plasma_other_params_dfs,ignore_index=True)
    plasma_other_params.dropna(inplace=True)  # noqa: PD002
    return pd.concat([plasma_all_params,plasma_other_params])

if __name__ == "__main__":
    import scienceplots  # noqa: F401
    # set plotting style
    ##plt.style.use(["science","no-latex"])
    import seaborn as sns
    sns.set_theme()
    flag = False
    b=0
    # laptop path doe-npz-em-01\npz\powder_plasma\has_plasma\*.npz
    # desktop path powder_plasma_npz\npz\has_plasma\sheffield_doe_flowrate_gasrate_000*.npz
##    plasma_df=combinePlasmaStats(glob(r"powder_plasma_npz\npz\has_plasma\*.npz")[:-1],Tlim=1000.0,labels=["40 SL/min","60 SL/min","70 SL/min","80 SL/min"])
##    plasma_true = findTrueArea(plasma_df)
##    f,data,imgs = estAllPlasmaSize(glob(r"doe-npz-em-01\npz\powder_plasma\has_plasma\*.npz")[:-1],Tlim=[650,1125],
##                              labels=["40 SL/min","60 SL/min","70 SL/min","80 SL/min"],return_data=True)
##    imgs_stack = stackContourImgs(imgs)
##    # format to a dataframe
##    data_df = formatEstAllPlasmaSizeToDF(data)
##    # fit the poly to the target areas
##    poly_figures, poly_data = fitPolyToPlasmaArea(data,area=[0.15,0.54])

##    all_dfs = []
##    for fn,label in zip(glob(r"D:\Plasma-Spray-iCoating\scripts\trenchcoat\src\doe-npz-em-01\npz\powder_plasma\has_plasma\sheffield_doe_flowrate_gasrate_000*.npz")[:-1],["40 SL/min","60 SL/min","70 SL/min","80 SL/min"]):
##        df = collectPlasmaStats(fn)
##        df["File Path"] = fn
##        df["Label"] = label
##        all_dfs.append(df)
##    combined = pd.concat(all_dfs,ignore_index=True)
        
    #kk = list(data.keys())
    #f,imgs = estPlasmaSize(kk[0])
    coaxialPlaneSlider(glob(r"D:\Git Clones\Plasma-Spray-iCoating\scripts\trenchcoat\notebooks\tool-head-calibrated\npz\*.npz")[1],
                        k=1, # order of line fitting
                        min_segs=0, # min number of fitting
                        filt=clipMaskPlasma,
                        )
    # laptop path: doe-npz-em-01\npz\powder_plasma\good\*.npz
    # pc path: powder_plasma_npz/npz/good/*.npz
    #for fn,flag,b in zip(glob(r"doe-npz-em-01\npz\powder_plasma\good\*.npz"),[True,False,False,False],[50,0,0,0]):
    
##    plasmaGaussianBoxPlot(r"doe-npz-em-01\npz\powder_plasma\*.npz",lim=1300.0,
##                          plot_labels=["40 (SL/min)","40 (SL/min)","50 (SL/min)","60 (SL/min)","70 (SL/min)","80 (SL/min)","80 (SL/min)","80 (SL/min)"])

    #temperatureStatsAboveLim(r"doe-npz-em-01\npz\powder_plasma\*.npz",lim=1300.0,
    #                      plot_labels=["40 (SL/min)","40 (SL/min)","50 (SL/min)","60 (SL/min)","70 (SL/min)","80 (SL/min)","80 (SL/min)","80 (SL/min)"])


    # laptop path doe-npz-em-01\npz\powder_plasma\has_plasma\*.npz
    # desktop path powder_plasma_npz\npz\has_plasma\*.npz
##    plasma_params_dfs = []
##    plasma_other_params_dfs = []
##    for fn in glob(r"tool-head-calibrated/npz/*.npz")[:-1]:
##        f,df = plasmaGaussianParams(fn,lim=1000.0)
##        df["File"] = Path(fn).stem
##        #df["Label"] = label
##        plasma_params_dfs.append(df)
##        df = collectPlasmaStats(fn,1000.0)
##        df["File"] = Path(fn).stem
##        plasma_other_params_dfs.append(df)
##
##    plasma_all_params = pd.concat(plasma_params_dfs,ignore_index=True)
##    plasma_all_params["Asymmetrical Angle (degrees)"] = np.degrees(plasma_all_params["Asymmetrical Angle (rads)"])
##    # remove nans
##    plasma_all_params.dropna(inplace=True)
##
##    plasma_other_params = pd.concat(plasma_other_params_dfs,ignore_index=True)
##    plasma_other_params.dropna(inplace=True)
##    pstack = pd.concat([plasma_all_params,plasma_other_params])
    
##
##    coaxial_dfs = []
##    for fn in glob(r"powder_plasma_npz\npz\*.npz"):
##        f,ft,df = coaxialPlaneParams(fn,k=1,filt=clipMaskPlasma)
##        df["File"] = Path(fn).stem
##        coaxial_dfs.append(df)
##        #filt=[lambda x,b=b, : blockFill(x,(0,464-b)),clipMaskPlasma],
##        videoCoaxialPlane(fn,k=1,filt=clipMaskPlasma,mode="all",line_col=(0,255,0))
##        #videoCoaxialPlane(fn,k=2,filt=clipMaskPlasma,mode=0,line_col=(0,255,0))
##        #videoCoaxialPlane(fn,k=3,filt=clipMaskPlasma,mode=0,line_col=(0,255,0))
##    if len(coaxial_dfs)>0:
##        coaxial_all_dfs = pd.concat(coaxial_dfs,ignore_index=True)
##    coaxial_all_dfs.dropna(inplace=True)
        
##        break
##        f.savefig(f"{Path(fn).stem}-gaussian-plasma-params.png")
##        plt.close(f)
##        plasmaGaussianVideo(fn,text_format=lambda h,m,std : f"H: {h} M: {m} STD: {std}",gauss_col=(255,0,255),lim=1300.0)
        #denseOpticalFlowTemperatureSlider(fn,crop_plasma=False,cmap="mag_only")
        #denseOpticalFlowTemperatureQuiverSlider(fn,hide_temp=True,clip_plasma=False,flow_args=(20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN))
##        if flag:
##            coaxialPlaneSlider(fn,3,filt=[lambda x,b=b, : blockFill(x,(0,464-b)),clipMaskPlasma])
##        else:
##            coaxialPlaneSlider(fn,3,filt=lambda x,b=b, : blockFill(x,(0,464-b)))
##        plasmaGaussianSlider(fn,text_format=lambda h,m,std : f"H: {h} M: {m} STD: {std}",gauss_col=(255,0,255),ct_col=(255,0,0),lim=1300.0)

        #videoCoaxialPlaneExperimental(fn,mode="all")
##        if flag:
##            f,ft = coaxialPlaneParams(fn,3,filt=[lambda x,b=b, : blockFill(x,(0,464-b)),clipMaskPlasma])
##            f.savefig(f"{Path(fn).stem}-coaxial-plane.png")
##            plt.close(f)
##            ft.savefig(f"{Path(fn).stem}-coaxial-theta.png")
##            plt.close(ft)
##            ft.axes[0].set_yscale("log")
##            ft.savefig(f"{Path(fn).stem}-coaxial-theta-log.png")
##            videoCoaxialPlane(fn,1,filt=[lambda x,b=b, : blockFill(x,(0,464-b)),clipMaskPlasma],mode="all")
##        else:
##            f,ft = coaxialPlaneParams(fn,3,filt=lambda x,b=b, : blockFill(x,(0,464-b)))
##            f.savefig(f"{Path(fn).stem}-coaxial-plane.png")
##            plt.close(f)
##            ft.savefig(f"{Path(fn).stem}-coaxial-theta.png")
##            ft.axes[0].set_yscale("log")
##            ft.savefig(f"{Path(fn).stem}-coaxial-theta-log.png")
##            plt.close(ft)
##            videoCoaxialPlane(fn,1,filt=lambda x,b=b, : blockFill(x,(0,464-b)),mode="all")

        #denseOpticalFlowTemperature(fn,None,"mag_only",crop,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        #if crop:
        #    denseOpticalFlowTemperature(fn,None,"mag_only",not crop,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        #denseOpticalFlowTemperatureQuiver(fn,None,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #plasmaAllGaussianParams(r"doe-npz-em-01\npz\powder_plasma\has_plasma\*.npz",plot_labels=["40 SL/MIN","50 SL/MIN","60 SL/MIN","70 SL/MIN","80 SL/MIN"],lim=1300.0)
    #f = plasmaGaussianScatterParam(glob("lsbu-doe-plasma-powder/npz/*.npz")[1:])
    #sliderTemperatureFFTMask(glob("lsbu-doe-stripe/npz/*.npz")[1],cv2.MARKER_STAR)

    #sns.set_theme("paper")
    # stripe labels per recording
##    stripe_labels = {"lsbu_doe_powder_stripes_0002":["15 (G/MIN)","20 (G/MIN)","25 (G/MIN)","30 (G/MIN)","35 (G/MIN)"],
##                     "lsbu_doe_powder_stripes_0003":["15 (G/MIN)","20 (G/MIN)","25 (G/MIN)","30 (G/MIN)","35 (G/MIN)","15 (G/MIN)","20 (G/MIN)","25 (G/MIN)","30 (G/MIN)","35 (G/MIN)"],
##                     "lsbu_doe_powder_stripes_0004":["PULSING"]}
##    for fn in glob("stripes_npz/em01/npz/stripes/*.npz"):
##        mask = glob(f"stripes_masks/cleanup{Path(fn).stem}*stripes.jpg")
##        if len(mask)>0:
##            f = plotMaxStripeTemperature(fn,mask[0],stripe_labels=stripe_labels[Path(fn).stem],title=f"Stripe Temperature ($^\circ$C)")
##            f.savefig(f"{Path(fn).stem}-stripe-max-temps.png")
##            plt.close(f)
##            f = plotMaxStripeTemperature(fn,mask[0],clip=True,title=f"Stripe Temperature ($^\circ$C)",stripe_labels=stripe_labels[Path(fn).stem])
##            f.savefig(f"{Path(fn).stem}-stripe-max-temps-clipped.png")
##            plt.close(f)
##    plt.show()
    #maskMaxTempSlider(r"D:\FLIR Studio Output\20221116 171052\lsbu_plasma_coating_plate_1.npz")
##    for fn in glob("powder_plasma_npz/npz/*.npz"):
##        if Path(fn).stem[-1] == "1":
##            continue
##        f = plasmaGaussianParams(fn,True)
##        f.savefig(f"{Path(fn).stem}-gaussian-plasma-params-flip.png")
##        plt.close(f)
##    search = glob("powder_plasma_npz/npz/*.npz")
##    # remove specific files we know not to contain plasma
##    # 1,3
##    for fi,fn in enumerate(search):
##        if Path(fn).stem[-1] in ["1","3"]:
##            search.remove(fn)
##    f = stackPlasmaGaussianParams(search,True)
##    for aa in f.axes:
##        aa.legend()
##    f.savefig("powder-plasma-npz-gaussian-params-stack.png")
#    for fn,crop in zip(glob("lsbu-doe-plasma-powder/npz/*.npz"),[False,True,False,False,True,False,True,True]):
        #gaussianMixtureHist(fn)
##        print(fn)
##        #denseOpticalFlowTemperatureSlider(fn,crop_plasma=False,cmap="mag_only")
#        denseOpticalFlowTemperatureQuiverSlider(fn,hide_temp=True,clip_plasma=crop,flow_args=(20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN))
##
##        #denseOpticalFlowTemperature(fn,None,"mag_only",crop,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
##        #if crop:
##        #    denseOpticalFlowTemperature(fn,None,"mag_only",not crop,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
##        denseOpticalFlowTemperatureQuiver(fn,None,20/100,1,21,1,1,1.0,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    #for fn in glob("lsbu-doe-stripe/npz/*.npz"):
    #    trackHotPixel(fn)
    #    f = trackHotPixelHistory(fn)
        #f.savefig(f"{Path(fn).stem}-hot-pixel-location.png")
        #plt.close(f)
