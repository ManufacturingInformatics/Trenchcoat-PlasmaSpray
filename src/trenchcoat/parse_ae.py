from nptdms import TdmsFile
from nptdms.reader import TdmsReader
from nptdms.common import toc_properties
import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import stft,butter,sosfiltfilt,sosfreqz
import os
import struct
import logging
import warnings

from collections import OrderedDict

_struct_unpack = struct.unpack

# setup logging file for more details on the custom tdms reader
# tends to produce very large log files after a while so is recommended to delete/clear them once in a while
fh = logging.FileHandler(r"tdms.log", "w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
log = logging.getLogger("nptdms.reader")
log.setLevel(logging.DEBUG)
if fh not in log.handlers:
    log.addHandler(fh)

# time periods where the stripes occur in each file
CLIP_PERIOD = {'sheff_lsbu_stripe_coating_1' : [120.0,180.0],
               'sheff_lsbu_stripe_coating_2' : [100.0,350.0],
               'sheff_lsbu_stripe_coating_3_pulsing' : [100.0,180.0]}

# stripe locations 0-1000
STRIPE_PERIOD = {'sheff_lsbu_stripe_coating_1': {'stripe_1':[120, 125],
                                                'stripe_2':[127, 133],
                                                'stripe_3':[141, 146],
                                                'stripe_4':[151, 156],
                                                'stripe_5':[162.5, 167.5],
                                                'stripe_6':[175, 180]}}
                                                 # first pass
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {'stripe_1_1':[107, 112],
                                                'stripe_2_1':[118, 123],
                                                'stripe_3_1':[127, 133],
                                                'stripe_4_1':[137, 143],
                                                'stripe_5_1':[146, 152],
                                                # 2nd pass
                                                'stripe_1_2':[157, 163], 
                                                'stripe_2_2':[169, 175],
                                                'stripe_3_2':[180, 185.5],
                                                'stripe_4_2':[190, 195],
                                                'stripe_5_2':[199, 205],
                                                # 3rd pass
                                                'stripe_1_3':[215, 221],
                                                'stripe_2_3':[226, 231], 
                                                'stripe_3_3':[237, 243],
                                                'stripe_4_3':[249, 255],
                                                'stripe_5_3':[260, 266],
                                                # unknown
                                                'stripe_1_4':[277, 282],
                                                'stripe_2_4':[290, 296],
                                                'stripe_3_4':[304, 310],
                                                'stripe_4_4':[314, 320],
                                                'stripe_5_4':[327,333],
                                                 }}

STRIPE_PERIOD_3 = {'sheff_lsbu_stripe_coating_3_pulsing':{'stripe_1':[105, 110],
                                                'stripe_2':[112, 117],
                                                'stripe_3':[119, 124],
                                                'stripe_4':[125, 131],
                                                'stripe_5':[134, 139],
                                                'stripe_6':[141, 147],
                                                'stripe_7':[148, 154],
                                                'stripe_8':[157, 163],
                                                'stripe_9':[164, 170],
                                                'stripe_10':[172, 178],
                                                }}
PERIODS = [STRIPE_PERIOD,STRIPE_PERIOD_2,STRIPE_PERIOD_3]

## dict for mapping stripe number to feed rate
FEED_RATE_1 = {'sheff_lsbu_stripe_coating_1':   {'stripe_1':'15 G/MIN',
                                                'stripe_2':'15 G/MIN',
                                                'stripe_3':'20 G/MIN',
                                                'stripe_4':'25 G/MIN',
                                                'stripe_5':'30 G/MIN',
                                                'stripe_6':'35 G/MIN'}}

FEED_RATE_2 = {'sheff_lsbu_stripe_coating_2': {'stripe_1_1':'15 G/MIN',
                                                'stripe_2_1':'20 G/MIN',
                                                'stripe_3_1':'25 G/MIN',
                                                'stripe_4_1':'30 G/MIN',
                                                'stripe_5_1':'35 G/MIN',
                                                # 2nd pass
                                                'stripe_1_2':'15 G/MIN (2)',
                                                'stripe_2_2':'20 G/MIN (2)',
                                                'stripe_3_2':'25 G/MIN (2)',
                                                'stripe_4_2':'30 G/MIN (2)',
                                                'stripe_5_2':'35 G/MIN (2)',
                                                # 3rd pass
                                                'stripe_1_3':'15 G/MIN (3)',
                                                'stripe_2_3':'20 G/MIN (3)',
                                                'stripe_3_3':'25 G/MIN (3)',
                                                'stripe_4_3':'30 G/MIN (3)',
                                                'stripe_5_3':'35 G/MIN (3)',
                                                # unknown
                                                'stripe_1_4':'15 G/MIN (4)',
                                                'stripe_2_4':'20 G/MIN (4)',
                                                'stripe_3_4':'25 G/MIN (4)',
                                                'stripe_4_4':'30 G/MIN (4)',
                                                'stripe_5_4':'35 G/MIN (4)',
                                                 }}

FEED_RATE = [FEED_RATE_1, FEED_RATE_2]

class TdmsReaderExt(TdmsReader):
    """
        Modified version of TdmsReader to deal with corrupt/incorrect header data

        In the case of invalid header information, the original reader would throw and exception and fail.
        This version iterates foward to find the location of the next segment tag and adjusts the header values to
        compensate
    """
    HEADER_SIZE = 28
    # modified version of reading in the header data segment
    def _read_lead_in(self, file, segment_position, is_index_file=False):
        log = logging.getLogger("nptdms.reader")
        log.debug(f"Current segments {self._segments}")
        lead_in_bytes = file.read(self.HEADER_SIZE)
        # if it read fewer bytes than expected
        # then we've reached the EOF
        if len(lead_in_bytes) < self.HEADER_SIZE:
            raise EOFError
        self.correct_positions = []
        # check that it's a valid tag
        expected_tag = b'TDSh' if is_index_file else b'TDSm'
        tag = lead_in_bytes[:4]
        if tag != expected_tag:
            extra_bytes = 0
            # loop until TDSm is found
            prev_segment = self._segments[-1]
            log.warning(f"Previous segment from pos 0x{prev_segment.position:X} metadata is incorrect as the set next_segment_pos (0x{prev_segment.next_segment_pos:X}) did not lead to a valid tag. Attempting to correct position")
            while True:
                try:
                    bytes = file.read(8)
                    # if the tag has been found
                    idx = bytes.find(expected_tag)
                    if idx != -1:
                        # move the cursor back
                        file.seek(-(8-idx), 1)
                        extra_bytes += 8-idx
                        segment_position = file.tell()
                        # re read segment header
                        lead_in_bytes = file.read(28)
                        # recheck the tag
                        tag = lead_in_bytes[:4]
                        if tag == expected_tag:
                            break
                        else:
                            raise ValueError("Failed to re-read tag from corrected position!")
                    extra_bytes += 8
                except EOFError as e:
                    log.error("Failed to find starting tag of next segment!")
                    raise e
            log.debug(f"TdmsFileExt debug head lead_in_bytes {lead_in_bytes[:10]}, extra bytes {extra_bytes}")
            log.debug(f"Moved segment position to 0x{segment_position:X}")
            log.debug("Updating last segment next_segment_pos value")
            # correct prev segment
            self._segments[-1].next_segment_pos = segment_position

        # Next four bytes are table of contents mask
        toc_mask = _struct_unpack('<l', lead_in_bytes[4:8])[0]
        # log the properties of the segment
        log.debug(f"Reading segment at 0x{segment_position:X}")
        for prop_name, prop_mask in toc_properties.items():
            prop_is_set = (toc_mask & prop_mask) != 0
            log.debug(f"Property {prop_name} is {prop_is_set}")

        endianness = '>' if (toc_mask & toc_properties['kTocBigEndian']) else '<'

        # Next four bytes are version number, then 8 bytes each for the offset values
        (version, next_segment_offset, raw_data_offset) = _struct_unpack(endianness + 'lQQ', lead_in_bytes[8:28])
        log.debug(f"Property {prop_name} version is {version}")

        if self.tdms_version is None:
            if version not in (4712, 4713):
                log.warning("Unrecognised version number: %d" % version)
            self.tdms_version = version
        elif self.tdms_version != version:
            log.warning("Segment version mismatch, %d != %d" % (version, self.tdms_version))

        # Calculate data and next segment position
        lead_size = self.HEADER_SIZE
        data_position = segment_position + lead_size + raw_data_offset
        segment_incomplete = next_segment_offset == 0xFFFFFFFFFFFFFFFF
        if segment_incomplete:
            # Segment size is unknown. This can happen if LabVIEW crashes.
            next_segment_pos = self._get_data_file_size()
            if next_segment_pos < data_position:
                # Metadata wasn't completely written and don't have any data in this segment,
                # don't try to read any metadata
                log.warning("Last segment metadata is incomplete")
                raise EOFError
            # Try to read until the end of the file if we have complete metadata
            log.warning(
                "Last segment of file has unknown size, "
                "will attempt to read to the end of the file")
        else:
            log.debug("Next segment offset = %d, raw data offset = %d, data size = %d b",
                      next_segment_offset, raw_data_offset, next_segment_offset - raw_data_offset)
            next_segment_pos = (
                    segment_position + next_segment_offset + lead_size)

        log.debug(f"Next segment offset = 0x{next_segment_offset:X}, raw data offset = 0x{raw_data_offset:X}, expected data size = {next_segment_offset - raw_data_offset} b, actual data size = {next_segment_pos - data_position} b")

        return segment_position, toc_mask, data_position, next_segment_pos, segment_incomplete


class TdmsFileExt(TdmsFile):
    @staticmethod
    def read(file: str, raw_timestamps:bool=False, memmap_dir=None, clear_log: bool=True):
        """ 
            Creates a new TdmsFileExt object and reads all data in the file

            Input:
                file: Path to the tdms file
                raw_timestamps: Flag to read either TDMS timestamps as numpy datetime64 (False) or
                                as custom TdmsTimestamp type (True).
                memmap_dir: The directory to store memory mapped data files in,
                    or None to read data into memory. The data files are created
                    as temporary files and are deleted when the channel data is no
                    longer used. tempfile.gettempdir() can be used to get the default
                    temporary file directory.
                clear_log : Flag to clear the log file created for debugging the metadata. Default True.
        """
        return TdmsFileExt(file, raw_timestamps=raw_timestamps, memmap_dir=memmap_dir, clear_log=clear_log)

    @staticmethod
    def open(file: str, raw_timestamps: bool=False, memmap_dir=None, clear_log: bool=True):
        """ 
            Creates a new TdmsFileExt object and reads the metadata and keeps the file open for reading the data later

            Input:
                file: Path to the tdms file
                raw_timestamps: Flag to read either TDMS timestamps as numpy datetime64 (False) or
                                as custom TdmsTimestamp type (True).
                memmap_dir: The directory to store memory mapped data files in,
                    or None to read data into memory. The data files are created
                    as temporary files and are deleted when the channel data is no
                    longer used. tempfile.gettempdir() can be used to get the default
                    temporary file directory.
                clear_log : Flag to clear the log file created for debugging the metadata. Default True.
        """
        return TdmsFileExt(
            file, raw_timestamps=raw_timestamps, memmap_dir=memmap_dir, read_metadata_only=True, keep_open=True, clear_log=clear_log)

    @staticmethod
    def read_metadata(file, raw_timestamps=False, clear_log=True):
        """ Creates a new TdmsFile object and only reads the metadata

        :param file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param raw_timestamps: By default TDMS timestamps are read as numpy datetime64
            but this loses some precision.
            Setting this to true will read timestamps as a custom TdmsTimestamp type.
        """
        return TdmsFileExt(file, raw_timestamps=raw_timestamps, read_metadata_only=True, clear_log=clear_log)
    
    def __init__(self, file: str, raw_timestamps: bool=False, memmap_dir=None, read_metadata_only: bool=False, keep_open: bool=False, clear_log: bool=True):
        """
            Initialise a new TdmsFileExt object

            Inputs:
                file: Path to TDMS file
                raw_timestamps: By default TDMS timestamps are read as numpy datetime64
                    but this loses some precision.
                    Setting this to true will read timestamps as a custom TdmsTimestamp type.
                memmap_dir: The directory to store memory mapped data files in,
                    or None to read data into memory. The data files are created
                    as temporary files and are deleted when the channel data is no
                    longer used. tempfile.gettempdir() can be used to get the default
                    temporary file directory.
                read_metadata_only: If True, only the metadata of the TDMS file will read.
                keep_open: Keeps the file open so data can be read if only metadata
                    is read initially.
                clear_log : Flag to initialise and clear the log file. Recommended to be True.
        """

        self._memmap_dir = memmap_dir
        self._raw_timestamps = raw_timestamps
        self._groups = OrderedDict()
        self._properties = OrderedDict()
        self._channel_data = {}
        self._tdms_version = 0
        self.data_read = False

        # reset filehandler to essentially clear the log file
        if clear_log:
            fh = logging.FileHandler(r"tdms.log", "w")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            for h in log.handlers:
                if isinstance(h, logging.FileHandler):
                    if h.baseFilename == r"tdms.log":
                        h = fh

        self._reader = TdmsReaderExt(file)

        try:
            self._read_file(
                self._reader,
                read_metadata_only if not self._reader.is_index_file_only() else True,
                keep_open
            )
        finally:
            if not keep_open:
                self._reader.close()
    
    # find good channels by identifying the channels whose data is within the plot limits properties
    def find_goodchannels(self, th: float = 0.3) -> dict:
        """
            Find the channels whose values are within the limit -th <= value <= th.

            This was added to check for corrupt values that went to NAN or INF

            Inputs:
                th : Value representing the abs value limits. Default 0.3.

            Returns a dict of channel paths for the channels that are valid
        """
        good_chanels = {}
        # set the limits
        plot_min = -1*th
        plot_max = th
        for g in list(self._groups.values()):
            for c in g.channels():
                # load data
                data = c.read_data()
                # check that the data is within range
                if (data.min() >= plot_min) and (data.max() <= plot_max):
                    good_chanels[c.path] = c
        return good_chanels
    
    # modified version of as_dataframe to only extract the data
    def as_dataframe(self, th: float = 0.3 ,time_index: bool=False, absolute_time: bool=False, scaled_data: bool=True, arrow_dtypes: bool=False) -> pd.DataFrame:
        """ 
            Modified version of as_dataframe method that only exports channels whose min/max values are within a certain limit

            A channel is regarded as good if
                (data.min() >= -th) and (data.max() <= th)

            The idea is to ignore channels that have dummy/bad values

            Inputs:
                th : Floating point threshold for checking
                time_index: Whether to include a time index for the dataframe.
                absolute_time: If time_index is true, whether the time index
                    values are absolute times or relative to the start time.
                scaled_data: By default the scaled data will be used.
                    Set to False to use raw unscaled data.
                    For DAQmx data, there will be one column per DAQmx raw scaler and column names will include the scale id.
                arrow_dtypes: Use PyArrow data types in the DataFrame.
            
            Returns dataframe of obtained data
        """
        from nptdms.export.pandas_export import _channels_to_dataframe
        # if the threshold is None then just use the default method
        if th is None:
            return super().as_dataframe(time_index, absolute_time, scaled_data, arrow_dtypes)
        # find the good channels
        good_channels = self.find_goodchannels(th)
        # if it failed to find any good channels
        if len(good_channels) == 0:
            raise ValueError("Failed to find good channels that satisfied PlotMin and PlotMax conditions!")
        # get the dataframe using only the found channels
        return _channels_to_dataframe(good_channels, time_index, absolute_time, scaled_data, arrow_dtypes)


def clean_dict(mdict : dict) -> dict:
    """
        Replaces datetime objects with UTC timestamp string
    """
    for k in mdict.keys():
        if isinstance(mdict[k], np.datetime64):
            mdict[k]  = np.datetime_as_string(mdict[k], timezone="UTC")
            if isinstance(mdict[k], OrderedDict):
                mdict[k] = clean_dict(mdict[k])
    return mdict

def stack_metadata(file: str) -> dict:
    """
        Read a TDMS file's metadata and stack it into a single dictionary
    """
    meta_all = {}
    # attempt to read metadata
    try:
        md = TdmsFile.read_metadata(file)
    except ValueError:
        warnings.warn(f"Failed to read metadata from {file} using regular TdmsFile. Trying with TdmsFileExt")
        md = TdmsFileExt.read_metadata(file)
    # stack into a single dict
    meta_all.update(md.properties)
    for g in md.groups():
        meta_all[g.name] = g.properties
        for c in g.channels():
            c_md = c.properties
            #c_md = clean_dict(c_md)
            meta_all[g.name][c.name] = c_md
    return meta_all


## from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# band pass
def butter_bandpass(lowcut: float, highcut: float, order: int=5) -> np.ndarray:
    """ Create a Butterworth bandpass filter SOS digital params """
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    print(low, high)
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, order: int=5) -> np.ndarray:
    """
        Apply a Butterwoth bandpass filter to the specified data

        Inputs:
            data : numpy array to filter
            lowcut : Lower freq band
            highcut: Higher freq band
            order : Model order

        Returns filtered numpy array
    """
    sos = butter_bandpass(lowcut, highcut, order=order)
    return sosfiltfilt(sos, data)

# band stop
def butter_bandstop(lowcut: float, highcut: float, order: int=5) -> np.ndarray:
    """ Create a Butterworth bandstop filter SOS digital params """
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandstop', analog=False, output='sos')


def butter_bandstop_filter(data: np.ndarray, lowcut: float, highcut: float, order: int=5) -> np.ndarray:
    """
        Apply a Butterwoth bandstop filter to the specified data

        Inputs:
            data : numpy array to filter
            lowcut : Lower freq band
            highcut: Higher freq band
            order : Model order

        Returns filtered numpy array
    """
    sos = butter_bandstop(lowcut, highcut, 1e6, order=order)
    return sosfiltfilt(sos, data)

# general band filter
def butter_band(lowcut: float, highcut: float, btype: str, order: int=5) -> np.ndarray:
    """ Create a Butterworth filter SOS digital params """
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype=btype, analog=False, output='sos')


def butter_bfilter(data: np.ndarray, lowcut: float, highcut: float, btype: str, order: int=5) -> np.ndarray:
    """
        Apply a Butterwoth bandpass filter to the specified data

        Inputs:
            data : numpy array to filter
            lowcut : Lower freq band
            highcut: Higher freq band
            btype : Type of filter to apply e.g. highpass, lowpass, bandstop etc.
            order : Model order

        Returns filtered numpy array
    """
    sos = butter_band(lowcut, highcut, btype, order)
    return sosfiltfilt(sos, data)


def applyFilters(data: np.ndarray,freq: float | list,mode: str | list,**kwargs):
    """
        Apply a series of filters to the given daa

        freq is either a single cutoff frequency or a 2-element collection for band filters
        mode is either a single string or list of strings if multiple filters are specified in mode.

        Inputs:
            data : Array of values
            freq : Single value or list of frequencies
            mode : String or list of strings defining type of filters

        Returns filtered data
    """
    # ensure frequencies and modes are lists
    if isinstance(freq,(int,float)):
        freq = [freq,]
    # if mode is a single string them treat all filters as that mode
    if isinstance(mode,str):
        modes = len(freq)*[mode,]
    elif len(freq) != len(mode):
        raise ValueError("Number of modes must match the number of filters!")
    else:
        modes = list(mode)
    # iterate over each filter and mode
    for c,m in zip(freq,modes):
        # if it's a single value then it's a highpass/lowpass filter
        if isinstance(c,(int,float)):
            sos = butter(kwargs.get("order",10), c/1e6, m, fs=1e6, output='sos',analog=False)
            data = sosfiltfilt(sos, data)
        # if it's a list/tuple then it's a bandpass filter
        elif isinstance(c,(tuple,list)):
            if m == "bandpass":
                data = butter_bandpass_filter(data,c[0],c[1],kwargs.get("order",10))
            elif m == "bandstop":
                data = butter_bandstop_filter(data,c[0],c[1],kwargs.get("order",10))
    return data


def plotFreqResponse(freq: float | tuple[float, float],btype: str,pts: int =int(1e6/4),**kwargs) -> plt.Figure:
    """
        Butterworth filter frequency response

        freq is either a single cutoff frequency or a 2-element collection for band filters
        pts is the number of points used for plotting. Passed to worN parameter in sosfreqz.

        Inputs:
            freq : Single or 2-element collection of cutoff frequencies
            btype : String for frequency type. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html.
            pts : Number of points for plotting. Default 1e6/4
            order : Filter model order

        Returns figure
    """
    # digital filters require freq be between 0 and 1
    if isinstance(freq,(float,int)):
        freq_norm = freq/(1e6/2)
    else:
        freq_norm = freq
    # if low or highpass
    if btype in ["lowpass","highpass"]:
        sos = butter(kwargs.get("order",10),freq_norm,btype,analog=False,output='sos')
    # for band filters
    elif btype == "bandpass":
        sos = butter_bandpass(freq[0],freq[1],1e6,kwargs.get("order",10))
    elif btype == "bandstop":
        sos = butter_bandstop(freq[0],freq[1],1e6,kwargs.get("order",10))
    # generate freq response
    w,h = sosfreqz(sos,worN=pts,fs=1e6)
    fig,ax = plt.subplots(nrows=2,constrained_layout=True)
    ax[0].plot(w,np.abs(h))
    ax[1].plot(w,np.angle(h))
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Gain")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (radians)")
    ax[0].vlines(freq,0,1,color='purple')
    ax[1].vlines(freq,-2*np.pi,2*np.pi,color='purple')
    fig.suptitle("Butterworth Frequency Response")
    return fig


def loadTDMSData(path: str) -> pd.DataFrame:
    """
        Load in the TDMS data and add a column for time

        All the functions were built around there being a Time columns rather than using index as time

        Columns are renamed from their full paths to Input 0 and Input 1

        Returns pandas dataframe
    """
    data = TdmsFile(path).as_dataframe(time_index=False)
    data['Time (s)'] = np.arange(data.shape[0])/1e6
    data.rename(columns={c:c.split('/')[-1].strip("'") for c in data.columns},inplace=True)
    return data


def loadTDMSSub(path: str,chunk: tuple[int|float, int|float],is_time: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Load in a sub-section of TDMS data

        Helps avoid loading in the entire file
        
        Inputs:
            path : Input file path to TDMS file
            chunk: Index or time range to load
            is_time : Flag indicating of the chunk range is index or time range

        Returns tuple of specified chunk in time vector, Input 0 and Input 1
    """
    with TdmsFile(path) as tdms_file:
        # find group that contains recording
        group = list(filter(lambda x : "Recording" in x.name,tdms_file.groups()))[0]
        nf=group['Input 0'].data.shape[0]
        #mt = nf*1e-6
        # convert to index
        if is_time:
            chunk = [max(0,min(nf,int(c*1e6))) for c in chunk]
        nf = group['Input 0'][chunk[0]:chunk[1]].shape[0]
        time = np.arange(nf)*1e-6
        time += chunk[0]*1e-6
        return time,group['Input 0'][chunk[0]:chunk[1]],group['Input 1'][chunk[0]:chunk[1]]
        

def replotAE(path: str,clip: bool = True,ds : int =100) -> tuple[plt.Figure, plt.Figure]:
    """
        Replot the TDMS Acoutic Emission files

        If the file is supported in CLIP_PERIOD dictionary at the top of the file, then  it is cropped to the target period else plotted at full res

        Inputs:
            path : Input file path to TDMS file
            clip : Flag to clip the data. If True, then CLIP_PERIOD is referenced. If a tuple, then it's taken as the time period to clip to

        Returns figures for Input 0 and Input 1 respectively
    """
    sns.set_theme("talk")
    data = loadTDMSData(path)
    # clip to known activity
    if clip:
        if any([os.path.splitext(os.path.basename(path))[0] in k for k in CLIP_PERIOD.keys()]):
            data = data[(data['Time (s)'] >= CLIP_PERIOD[os.path.splitext(os.path.basename(path))[0]][0]) & (data['Time (s)'] <= CLIP_PERIOD[os.path.splitext(os.path.basename(path))[0]][1])]
        elif isinstance(clip,(tuple,list)):
            data = data[(data['Time (s)'] >= clip[0]) & (data['Time (s)'] <= clip[1])]
        else:
            print(f"Failed to clip file {path}!")

    if isinstance(ds,(int,float)) and (ds is not None):
        data = data.iloc[::ds]
    # plot both channels
    f0,ax0 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 0',ax=ax0)
    ax0.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 0")
    
    f1,ax1 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 1',ax=ax1)
    ax1.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 1")
    return f0,f1


def plotRaster(path: str,nbins: int = 1000,**kwargs):
    """
        Plot the large dataset by rasterising into colours

        The parameter nbins controls the number of colours used

        Creates an image that's faster to produce than plotting the time series.

        The plot is saved to the same location as the source data

        Inputs:
            path : TDMS file path
            nbins : Number of colour bins
            cmap : Matplotlib colour map to use to convert data to colours
            bk : Background colour of plot
    """
    import datashader as ds
    import colorcet as cc
    from fast_histogram import histogram2d
    import matplotlib.colors as colors
    data = loadTDMSData(path)
    for i,c in enumerate(data.columns):
        if c == "Time (s)":
            continue
        # from https://towardsdatascience.com/how-to-create-fast-and-accurate-scatter-plots-with-lots-of-data-in-python-a1d3f578e551
        cvs = ds.Canvas(plot_width=1000, plot_height=500)  # auto range or provide the `bounds` argument
        agg = cvs.points(data, 'Time (s)', 'Input 0')  # this is the histogram
        ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), kwargs.get("bk","black")).to_pil()  # create a rasterized imageplt.imshow(img)
        # stack first column and time column
        X = np.hstack((data.values[:,i].reshape(-1,1),data.values[:,2].reshape(-1,1)))
        cmap = cc.cm[kwargs.get("cmap","fire")].copy()
        cmap.set_bad(cmap.get_under())  # set the color for 0 to avoid log(0)=inf
        # get bounds for axis
        bounds = [[X[:, 0].min(), X[:, 0].max()], [X[:, 1].min(), X[:, 1].max()]]
        # calculate 2d histogram for colour levels and let matplotlib handle the shading
        h = histogram2d(X[:, 0], X[:, 1], range=bounds, bins=nbins)
        f,ax = plt.subplots(constrained_layout=True)
        X = ax.imshow(h, norm=colors.LogNorm(vmin=1, vmax=h.max()), cmap=cmap)
        ax.axis('off')
        plt.colorbar(X)
        f.savefig(f"{os.path.splitext(path)[0]}-{c}.png")
        plt.close(f)


def plotLombScargle(path: str,freqs: np.array=np.arange(300e3,350e3,1e3),normalize: bool = True,**kwargs) -> tuple[plt.Figure, plt.Figure]:
    """
        Plot Lomb-Scargle periodgram of the given file between the target frequencies

        Input 0 and Input 1 are plotted on two separate figures and returned

        Inputs:
            path : TDMS path
            freqs : Array of frequencies to evaluate at
            normalize : Flag to normalize response
            tclip : 2-element time period to clip to
            input0_title : Figure title used on plot for Input 0
            input1_title : Figure title used on plot for Input 1

        Returns generated figures
    """
    from scipy.signal import lombscargle
    time,i0,i1 = loadTDMSData(path)
    # convert freq to rad/s
    rads = freqs/(2*np.pi)
    # clip to time period if desired
    if kwargs.get("tclip",None) is not None:
        tclip = kwargs.get("tclip",None)
        i0 = i0[(time >= tclip[0]) & (time <= tclip[1])]
        i1 = i1[(time >= tclip[0]) & (time <= tclip[1])]
    # calculate
    pgram = lombscargle(i0, time, rads, normalize=normalize)
    f0,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(x=freqs,y=pgram,ax=ax)
    ax.set(xlabel="Frequency (rad/s)",ylabel="Normalized Amplitude",title="Input 0")
    f0.suptitle(kwargs.get("input0_title","Acoustic Lomb-Scargle Periodogram"))

    pgram = lombscargle(i1,time, rads, normalize=normalize)
    f1,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(x=freqs,y=pgram,ax=ax)
    ax.set(xlabel="Frequency (rad/s)",ylabel="Normalized Amplitude",title="Input 1")
    f1.suptitle(kwargs.get("input1_title","Acoustic Lomb-Scargle Periodogram"))
    return f0,f1


def plotSTFTSB(signal: np.ndarray,nperseg: int =256,fclip: float =None,**kwargs) -> plt.Figure:
    """
        Plot the STFT of the target signal using Seaborn

        fclip is for clipping the freq to above the set threshold
        as contrast over the full range can be an issue

        Inputs:
            signal : Numpy array to perform stft on
            nperseg : Number of segments. See scipy.signal.stft
            fclip : Freq clipping threshold
            theme : Seaborn threshold to use

        Returns generated figure
    """
    import pandas as pd
    sns.set_theme(kwargs.get("theme","paper"))
    f, t, Zxx = stft(signal, 1e6, nperseg=nperseg)
    if fclip:
        Zxx = Zxx[f>=fclip,:]
        f = f[f>=fclip]
    # convert data into dataframe
    data = pd.DataFrame(np.abs(Zxx),index=f,columns=t)
    ax = sns.heatmap(data)
    ax.invert_yaxis()
    return plt.gcf()


def plotSTFTPLT(signal: np.ndarray,nperseg: int =256,fclip: float =None,use_log: bool =True,**kwargs) -> plt.Figure:
    """
        Plot the STFT of the target signal using Matplotlib

        fclip is for clipping the freq to above the set threshold
        as contrast over the full range can be an issue

        Inputs:
            signal : Numpy array to perform stft on
            nperseg : Number of segments. See scipy.signal.stft
            fclip : Freq clipping threshold. Default None
            use_log : Flag to use log yaxis. Default True,
            theme : Seaborn threshold to use

        Returns generated figure
    """
    import matplotlib.colors as colors
    # set seaborne theme
    sns.set_theme(kwargs.get("theme","paper"))
    # perform STFT at the set parameters
    f, t, Zxx = stft(signal, 1e6, nperseg=nperseg)
    # offset time vector for plotting
    t += kwargs.get("tmin",0.0)
    # if user wants the frequency clipped above a particular point
    if fclip:
        Zxx = Zxx[f>=fclip,:]
        f = f[f>=fclip]
    fig,ax = plt.subplots(constrained_layout=True)
    # if using colormap log Norm
    if use_log:
        X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*0.01, vmax=np.abs(Zxx).max()),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    else:
        X = ax.pcolormesh(t, f, np.abs(Zxx),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    # set title
    ax.set_title(kwargs.get("title",'STFT Magnitude'))
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    # set colorbar
    plt.colorbar(X)
    return fig


def STFTWithLombScargle(path: str,span: float=0.4,n_bins: int=100, grid_size: int=100,min_freq:float=300e3,max_freq:float=400e3) -> plt.Figure:
    # see https://stackoverflow.com/a/65843574
    import fitwrap as fw

    if isinstance(path,str):
        data = loadTDMSData(path)
    else:
        data = path
    # extract columns
    signaldata = data.values[:,0]
    t = data.values[:,-1]
    # time bins
    x_bins = np.linspace(t.min()+span, t.max()-span, n_bins)
    # area for spectrogram
    spectrogram = np.zeros([grid_size, x_bins.shape[0]])
    # iterate over bins
    for index, x_bin in enumerate(x_bins):
        # build mask to isolate sequence
        mask = np.logical_and((x_bin-span)<=t, (x_bin+span)>=t)
        # perform periodigram over period
        frequency_grid, lombscargle_spectrum = fw.lomb_spectrum(t[mask], signaldata[mask],
                        frequency_span=[min_freq, max_freq], grid_size=grid_size)
        # update spectrogram using results
        spectrogram[:, index] = lombscargle_spectrum

    # plot results
    plt.imshow(spectrogram, aspect='auto', extent=[x_bins[0],x_bins[-1],
                frequency_grid[0],frequency_grid[-1]], origin='lower') 
    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    return plt.gcf()


def plotSignalEnergyFreqBands(data: str| pd.DataFrame,fstep: float,tclip:list[float, float]|None=None,fmin: float=0.1) -> plt.Figure:
    """
        Plot the signal energy within frequency bands

        Going from fmin to 1e6/2 in steps of fstep
        eg. [0,fstep],[fstep,2*step] etc.

        Frequencies are limited to these ranges using bandpass Butterworth filter

        tclip is for specifying a specific time period to look at. Useful for targeting
        a specifc stripe or for at least limiting it to the activity to manage memory better

        Inputs:
            data : File path or result of loadTDMSData or loadTDMSSub
            fstep : Frequency steps
            tclip : Time period to clip to. Default None
            fmin : Min frequency to start at. Has to be non-zero. Default 0.1 Hz

        Return generated figure
    """
    if isinstance(data,str):
        data = loadTDMSData(data)
    if tclip:
        data = data[(data['Time (s)'] >= tclip[0]) & (data['Time (s)'] <= tclip[1])]
    fsteps = np.arange(fmin,1e6/2,fstep)
    print(fsteps.shape[0])
    energy = {'Input 0':[],'Input 1':[]}
    # make an axis for each signal
    fig,ax = plt.subplots(ncols=data.shape[1]-1,constrained_layout=True,figsize=(7*data.shape[1]-1,6))
    for c in ['Input 0','Input 1']:
        signal = data[c]
        # for each freq band
        for fA,fB in zip(fsteps,fsteps[1:]):
            yf = butter_bandpass_filter(signal,fA,fB,order=10)
            energy[c].append(np.sum(yf**2))
    # calculate the bar locations
    locs = [fA+fstep/2 for fA in fsteps[:-1]]
    for aa,(cc,ee) in zip(ax,energy.items()):
        aa.bar(locs,ee,width=fstep,align='center')
    return fig


def plotSTFTStripes(path: str,nperseg: int=256,fclip: float=None,use_log: bool=True,**kwargs):
    """
        Plot the STFT of each stripe in each channel

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported

        fclip is the lower frequency to threshold above. This is to avoid the noise floor of signals and focus non the interesting stuff.

        When use_log is True, matplotlib.colors.LogNorm is being used

        Figures are saved to the same location at the source file

        Input:
            path : TDMS path
            nperseg : Number of points per segment
            fclip : Frequency to clip the STFT above
            use_log : Use log y-axis
            **kwargs: Keyword arguments passed to plotSTFTPLT
    """
    # for each tdms
    for fn in glob(path):
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # perform stft for each channel
                f=plotSTFTPLT(i0,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT",tmin=chunk[0],**kwargs)
                if fclip:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}-clip-{fclip}.png")
                else:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}.png")
                plt.close(f)

                # perform stft for each channel
                f=plotSTFTPLT(i1,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname} STFT",tmin=chunk[0],**kwargs)
                if fclip:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}-clip-{fclip}.png")
                else:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def plotStripes(path: str,ds: float=None,**kwargs):
    """
        Plot the STFT of each stripe in each channel

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported

        fclip is the lower frequency to threshold above. This is to avoid the noise floor of signals and focus non the interesting stuff.

        When use_log is True, matplotlib.colors.LogNorm is being used

        Figures are saved to the same location at the source file

        Input:
            path : TDMS path
            ds : Rate of downsampling
    """
    sns.set_theme("paper")
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
                ax.plot(time,i0)
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 0")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}.png")
                plt.close(f)

                # plot period
                f,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
                ax.plot(time,i1)
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 1")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def plotStripesLimits(path: str,**kwargs) -> plt.Figure:
    """
        Plot the max value of each stripe for each tool.

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported.

        A filter can be applied to the signal using the mode, order and and cutoff_freq keywords.
        The cutoff_freq keyword is the cutoff frequency of the filter.
        Can either be:
            - Single value representing a highpass/lowpass filter
            - 2-element Tuple/list for a bandpass filter.
            - List of values or list/tuples for a series of filters applied sequentially

        The mode parameter is to specify whether the filters are lowpass ("lp") or highpass ("hp"). If it's a single string, then it's applied
        to all non-bandpass filters in cutoff_freq. If it's a list, then it MUST be the same length as the number of filters

        The max value of each stripe is stored and plotted on a set of axis.

        Inputs:
            path : TDMS file path
            cutoff_freq : Float or tuple/list representing a cutoff frequency. Default None.
            mode : String stating whether it's a lowpass or highpass respectively. Default lp.
            order : Butterworth model order. Default 10.

        Returns a plot with max signal data
    """
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    #mode_dict = {'lp': "Low Pass",'hp':"High Pass",'bp':"Bandpass","lowpass":"Low Pass","highpass":"High Pass","bandpass":"Bandpass"}
    fname = os.path.splitext(os.path.basename(path))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        i0_max = []
        i0_min = []
        i1_max = []
        i1_min = []
        if kwargs.get("cutoff_freq",None):
            cf = kwargs.get("cutoff_freq",None)
            # if it's a single value, ensure it's a list
            if isinstance(cf,(int,float)):
                cf = [cf,]
            # if mode is a single string them treat all filters as that mode
            if isinstance(kwargs.get("mode","lowpass"),str):
                modes = len(cf)*[kwargs["mode"],]
            elif len(cf) != len(kwargs.get("mode","lowpass")):
                raise ValueError("Number of modes must match the number of filters!")
            else:
                modes = kwargs.get("mode","lowpass")
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            _,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            # iterate over each filter and mode
            for c,m in zip(cf,modes):
                print(c,m)
                # if it's a single value then it's a highpass/lowpass filter
                if isinstance(c,(int,float)):
                    sos = butter(kwargs.get("order",10), c/(1e6/2), m, fs=1e6, output='sos',analog=False)
                    i0 = sosfiltfilt(sos, i0)
                    i1 = sosfiltfilt(sos, i1)
                # if it's a list/tuple then it's a bandpass filter
                elif isinstance(c,(tuple,list)):
                    if m == "bandpass":
                        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
                    elif m == "bandstop":
                        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
            i0_max.append(i0.max())
            i0_min.append(i0.min())

            i1_max.append(i1.max())
            i1_min.append(i1.min())
        # combine modes together
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in cf])
        freq_plot_string = ','.join([str(c) for c in cf])
        # get number of values
        nf = len(i0_max)
        # get plotting mode
        pmode = kwargs.get("plot_mode","both")
        # plot both min and max signal amplitudes
        if pmode == "both":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(16,8))
            # ensure that the axis ticks are integer
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            # plot max
            ax[0].plot(range(nf),i0_max,'r-')
            ax[0].set_xticks(range(nf))
            # make twin axis for min
            tax = ax[0].twinx()
            # plot min
            tax.plot(range(nf),i0_min,'b-')
            # set labels based on if the user gave a cutoff freq
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            tax.set_ylabel("Min Voltage (V)")
            # make patches for legend
            patches = [mpatches.Patch(color='blue', label="Min Voltage"),mpatches.Patch(color='red', label="Max Voltage")]
            ax[0].legend(handles=patches)
            # plot Input 1 max
            ax[1].plot(range(nf),i1_max,'r-')
            ax[1].set_xticks(range(nf))
            tax = ax[1].twinx()
            # plot Input 1 min
            tax.plot(range(nf),i1_min,'b-')
            tax.set_ylabel("Min Voltage (V)")
            # make patches for ax[1]
            patches = [mpatches.Patch(color='blue', label="Min Voltage"),mpatches.Patch(color='red', label="Max Voltage")]
            ax[1].legend(handles=patches)
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            if kwargs.get("cutoff_freq",None):
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            else:
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits.png")
        # plot ONLY max
        elif pmode == "max":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(16,8))
            # ensure that the axis ticks are integer
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].plot(range(nf),i0_max,'r-')
            ax[1].plot(range(nf),i1_max,'r-')
            ax[0].set_xticks(range(nf))
            ax[1].set_xticks(range(nf))
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            if kwargs.get("cutoff_freq",None):
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}-max-only.png")
            else:
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-max-only.png")
        # plot ONLY min
        elif pmode == "min":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].plot(range(nf),i0_min,'b-')
            ax[1].plot(range(nf),i1_min,'b-')
            ax[0].set_xticks(range(nf))
            ax[1].set_xticks(range(nf))
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Min Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Min Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Min Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Min Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            if kwargs.get("cutoff_freq",None):
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}-min-only.png")
            else:
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-min-only.png")
            plt.close(f)
        return f
    else:
        print(f"Unsupported file {path}!")


def drawEdgeAroundStripe(path: str,dist: int=int(50e3),mode:str="separate",**kwargs) -> plt.Figure:
    """
        Draw around the edge of the stripe AE data using the peaks

        The +ve and -ve edges of Input 0 and Input 1 are identified and plotted
        The idea is to get a sense of how the area of each input differs over the same time period

        The mode input controls how the edges are plotted
            separate : Plot the signal and edges on separate axis
            overlay : Plot the edges on the same axis with NO signal

        This is designed for a SINGLE stripe as opposed to all of them

        The dist parameter is the min distance between peaks and is passed to find_peaks. Treat
        like a smoothing paramter of sorts

        Inputs:
            path : Path to TDMS file
            dist : Distance between peaks. See scipy.signal.find_peaks
            mode : Plotting mode. Default separate.
            
            Method of defining what stripe to process (see loadTDMSSub)
                time_period : Two-element iterable of time period to look at
                index_period: Two-element iterable of index period to look at
                stripe_ref : String or index reference of stripe.

        Returns generated figure
    """
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    fname = os.path.splitext(os.path.basename(path))[0]
    add_label = ""
    # check how the user is identifying the stripe
    if kwargs.get("time_period",None):
        time,i0,i1 = loadTDMSSub(path,kwargs.get("time_period",None),is_time=True)
        add_label=f" Time={kwargs.get('time_period',None)}"
    elif kwargs.get("index_period",None):
        time,i0,i1 = loadTDMSSub(path,kwargs.get("index_period",None),is_time=False)
        
    elif "stripe_ref" in kwargs:
        periods = list(filter(lambda x : fname in x,PERIODS))
        if len(periods)==0:
            raise ValueError(f"Path {path} has no supported periods!")
        periods = periods[0][fname]
        # if referencing
        if isinstance(kwargs.get("stripe_ref",None),str):
            if kwargs.get("stripe_ref",None) not in periods:
                raise ValueError(f"Stripe reference {kwargs.get('stripe_ref',None)} is invalid!")
            ref = kwargs.get("stripe_ref",None)
        elif isinstance(kwargs.get("stripe_ref",None),int):
            if kwargs.get("stripe_ref",None) > len(periods):
                raise ValueError("Out of bounds stripe index!")
            ref = list(periods.keys())[int(kwargs.get("stripe_ref"))]
        chunk = periods[ref]
        add_label=f" Stripe {ref}"
        # load time period
        time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
    else:
        raise ValueError("Missing reference to target period!")

    if mode == "separate":
        ## find edge of Input 0
        # mask signal to +ve
        mask_filt = i0.copy()
        mask_filt[i0<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find +ve edge in Input 0!")
        # plot the signal and +ve edge
        f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
        ax[0].plot(time,i0,'b-')
        ax[0].plot(time[pks],i0[pks],'r-')
        
        # mask signal to -ve
        mask_filt = i0.copy()
        mask_filt[i0>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]

        if len(pks)==0:
            raise ValueError("Failed to find -ve edge in Input 0!")
        ax[0].plot(time[pks],i0[pks],'r-')

        patches = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
        ax[0].legend(handles=patches)
        ax[0].set(xlabel="Time (s)",ylabel="Input 0",title=f"Input 0{add_label}")

        ## find edge of Input 1
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find +ve edge in Input 1!")
        # plot the signal and +ve edge
        ax[1].plot(time,i1,'b-')
        ax[1].plot(time[pks],i1[pks],'r-')
        
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]

        if len(pks)==0:
            raise ValueError("Failed to find -ve edge in Input 1!")
        ax[1].plot(time[pks],i1[pks],'r-')

        patches = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
        ax[1].legend(handles=patches)
        ax[1].set(xlabel="Time (s)",ylabel="Input 0",title=f"Input 1{add_label}")
    elif mode == "overlay":
        ## find edge of Input 0
        # mask signal to +ve
        mask_filt = i0.copy()
        mask_filt[i0<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find +ve edge in Input 0!")

        f,ax = plt.subplots(constrained_layout=True)
        ax.plot(time[pks],i0[pks],'r-')
        
        # mask signal to -ve
        mask_filt = i0.copy()
        mask_filt[i0>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find -ve edge in Input 0!")

        ax.plot(time[pks],i0[pks],'r-')

        ## find edge of Input 1
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find +ve edge in Input 1!")
        tax = ax.twinx()
        # plot the signal and +ve edge
        tax.plot(time[pks],i1[pks],'b-')
        
        # mask signal to -ve
        mask_filt = i1.copy()
        mask_filt[i1>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]
        if len(pks)==0:
            raise ValueError("Failed to find -ve edge in Input 1!")

        tax.plot(time[pks],i1[pks],'b-')

        patches = [mpatches.Patch(color='red', label="Input 0"),mpatches.Patch(color='blue', label="Input 1")]
        ax.legend(handles=patches)
        ax.set(xlabel="Time (s)",ylabel="Input 0 Voltage (V)",title=f"{fname} {add_label}")
        tax.set_ylabel("Input 1 Voltage (V)")
    f.suptitle(fname)
    return f

def plotAllStripeEdges(path: str,dist: int=int(50e3),**kwargs) -> plt.Figure:
    """
        Trace the edges of each stripe and draw the edges for Input 0 and Input 1 on the same axis

        The edges are traced using find_peaks where the user sets the dist parameter.

        It is recommended that dist be an important frequency like 50e3 as it traces over the strongest
        frequencies well. Increasing it acts as a smoothing parameter whilst decreasing means it picks up the valleys
        more

        All stripe edges are drawn on the SAME axis

        Inputs:
            path : TDMS file path
            dist : Distance between peaks. See find_peaks

        Returns figure
    """
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from matplotlib.pyplot import cm
    fname = os.path.splitext(os.path.basename(path))[0]
    # check if supported
    periods = list(filter(lambda x,fname=fname : fname in x,PERIODS))
    if periods:
        periods = periods[0][os.path.splitext(os.path.basename(path))[0]]
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        color = cm.rainbow(np.linspace(0, 1, len(periods)))
        # filter to time periods dict
        for (sname,chunk),c in zip(periods.items(),color):
            time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            time = (time-time.min())/(time.max()-time.max())
            ## find edge of Input 0
            # mask signal to +ve
            mask_filt = i0.copy()
            mask_filt[i0<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find +ve edge in Input 0!")

            ax[0].plot(time[pks],i0[pks],c)
            
            # mask signal to -ve
            mask_filt = i0.copy()
            mask_filt[i0>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find -ve edge in Input 0!")

            ax[0].plot(time[pks],i0[pks],c)

            ## find edge of Input 1
            # mask signal to +ve
            mask_filt = i1.copy()
            mask_filt[i1<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find +ve edge in Input 1!")
            # plot the signal and +ve edge
            ax[1].plot(time[pks],i1[pks],c)
            
            # mask signal to -ve
            mask_filt = i1.copy()
            mask_filt[i1>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find -ve edge in Input 1!")

            ax[1].plot(time[pks],i1[pks],c)

        patches = [mpatches.Patch(color=c, label=sname) for sname,c in zip(periods.keys(),color)]
        ax[0].legend(handles=patches)
        ax[1].legend(handles=patches)
        ax[0].set(xlabel="Time (s)",ylabel="Input 0 Voltage (V)",title="Input 0")
        ax[1].set(xlabel="Time (s)",ylabel="Input 1 Voltage (V)",title="Input 1")
        return f

# from https://stackoverflow.com/a/30408825
# shoelace formula
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def calcStripeAreas(path: str,dist: int=int(50e3),**kwargs) -> plt.Figure:
    """
        Find the edges of the stripe and estimate the area treating the edges as a complicated polygon

        The edges are traced using find_peaks where the user sets the dist parameter.

        It is recommended that dist be an important frequency like 50e3 as it traces over the strongest
        frequencies well. Increasing it acts as a smoothing parameter whilst decreasing means it picks up the valleys
        more

        All stripe edges are drawn on the SAME axis.

        The area of each stripe for Input 0 and Input 1 is plotted on separate exes

        Inputs:
            path : TDMS path
            dist : Distance between peaks. See find_peaks

        Returns figure
    """
    from scipy.signal import find_peaks
    from matplotlib.pyplot import cm
    sns.set_theme("paper")
    fname = os.path.splitext(os.path.basename(path))[0]
    # check if supported
    periods = list(filter(lambda x,fname=fname : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        
        color = cm.rainbow(np.linspace(0, 1, len(periods)))
        time_pts_i0 = []
        v_pts_i0 = []
        area_i0 = []

        time_pts_i1 = []
        v_pts_i1 = []
        area_i1 = []
        # filter to time periods dict
        for (sname,chunk),c in zip(periods.items(),color):
            time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            ## find edge of Input 0
            # mask signal to +ve
            mask_filt = i0.copy()
            mask_filt[i0<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find +ve edge in Input 0!")

            time_pts_i0.extend(time[pks].tolist())
            v_pts_i0.extend(i0[pks].tolist())
            
            # mask signal to -ve
            mask_filt = i0.copy()
            mask_filt[i0>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find -ve edge in Input 0!")

            # add the points so that it's in clockwise order
            time_pts_i0.extend(time[pks].tolist()[::-1])
            v_pts_i0.extend(i0[pks].tolist()[::-1])
            area_i0.append(PolyArea(time_pts_i0,v_pts_i0))
            
            ## find edge of Input 1
            # mask signal to +ve
            mask_filt = i1.copy()
            mask_filt[i1<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find +ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist())
            v_pts_i1.extend(i0[pks].tolist())

            # mask signal to -ve
            mask_filt = i1.copy()
            mask_filt[i1>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError("Failed to find -ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist()[::-1])
            v_pts_i1.extend(i0[pks].tolist()[::-1])
            area_i1.append(PolyArea(time_pts_i1,v_pts_i1))

            time_pts_i0.clear()
            v_pts_i0.clear()
            
            time_pts_i1.clear()
            v_pts_i1.clear()
        # plot
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        ax[0].plot(area_i0)
        ax[0].set(xlabel="Stripe Index",ylabel="Area (V*t)",title="Input 0")
        ax[1].plot(area_i1)
        ax[1].set(xlabel="Stripe Index",ylabel="Area (V*t)",title="Input 1")
        f.suptitle(os.path.splitext(os.path.basename(path))[0]+" Shoelace Area between Edges")
        return f

def filterStripes(fn: str,freq:float=50e3,mode: str='highpass',**kwargs):
    """
        Apply filters to each identified stripe in the file

        A filter can be applied to the signal using the mode, order and and cutoff_freq keywords.
        The cutoff_freq keyword is the cutoff frequency of the filter.
        Can either be:
            - Single value representing a highpass/lowpass filter
            - 2-element Tuple/list for a bandpass filter.
            - List of values or list/tuples for a series of filters applied sequentially

        The mode parameter is to specify whether the filters are lowpass ("lp") or highpass ("hp"). If it's a single string, then it's applied
        to all non-bandpass filters in cutoff_freq. If it's a list, then it MUST be the same length as the number of filters

        Generated plots are saved in the same location as the source file

        This is intended to remove the noise floor.

        Inputs:
            fn : TDMS path
            freq : Cutoff freq. Float, list of floats or list of tuples/lists. Default 50e3.
            order : Filter order. Default 10.
    """
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # ensure frequencies and modes are lists
        if isinstance(freq,(int,float)):
            freq = [freq,]
        # if mode is a single string them treat all filters as that mode
        if isinstance(mode,str):
            modes = len(freq)*[mode,]
        elif len(freq) != len(mode):
            raise ValueError("Number of modes must match the number of filters!")
        else:
            modes = list(mode)
        # format frequencies and modes into strings for plotting titles and saving
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in freq])
        freq_plot_string = ','.join([str(c) for c in freq])
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            # plot period
            f_i0,ax_i0 = plt.subplots(constrained_layout=True)
            ax_i0.plot(time,i0,'b-',label="Original")
            # plot period
            f_i1,ax_i1 = plt.subplots(constrained_layout=True)
            ax_i1.plot(time,i0,'b-',label="Original")
            # iterate over each filter and mode
            for c,m in zip(freq,modes):
                print(sname,c,m)
                # if it's a single value then it's a highpass/lowpass filter
                if isinstance(c,(int,float)):
                    print("bing!")
                    sos = butter(kwargs.get("order",10), c, m, fs=1e6, output='sos',analog=False)
                    i0 = sosfiltfilt(sos, i0)
                    i1 = sosfiltfilt(sos, i1)
                # if it's a list/tuple then it's a bandpass filter
                elif isinstance(c,(tuple,list)):
                    print("bong!")
                    if m == "bandpass":
                        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
                    elif m == "bandstop":
                        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
            # plot data
            ax_i0.plot(time,i0,'r-',label="Filtered")
            ax_i0.legend()
            ax_i0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            plt.close(f_i0)
            # plot data
            ax_i1.plot(time,i1,'r-',label="Filtered")
            ax_i1.legend()
            ax_i1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 1, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            plt.close(f_i1)
    else:
        print(f"Unsupported file {fn}!")


def stripeSpectrogram(fn: str,shift: bool=False) -> plt.Figure:
    """
        Plot the spectrogram of each stripe in the target file

        Each file is 

        Inputs:
            fn : TDMS path

        Returns generated figure
    """
    from scipy.signal import spectrogram
    import matplotlib.colors as colors 
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)

            f,t,Zxx = spectrogram(i0,1e6)
            fig,ax = plt.subplots()
            if shift:
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
            else:
                plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname}")
            fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            plt.close(fig)

            f,t,Zxx = spectrogram(i1,1e6)
            fig,ax = plt.subplots(constrained_layout=True)
            if shift:
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
            else:
                plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname}")
            fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            plt.close(fig)


def stackFilterStripes(path: str,freq: float=50e3,mode: str='highpass',**kwargs):
    """
        Apply a highpass filter to each identified stripe in the file

        A Butterworth filter set to high-pass is applied to the target stripe.
        The original and filtered signal are plotted on the same axis and saved

        The order of the filter is set using order keyword.

        Generated plots are saved in the same location as the source file

        This is intended to remove the noise floor.

        Inputs:
            path : TDMS path
            freq : Cutoff freq. Default 50 KHz
            order : Filter order. Default 10.
    """
    from scipy import signal
    sns.set_theme("paper")
    mode_dict = {'lp': "Low Pass",'hp':"High Pass",'bp':"Bandpass",'lowpass': "Low Pass",'highpass':"High Pass",'bandpass':"Bandpass"}
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # make filter
            sos = signal.butter(kwargs.get("order",10), freq/(1e6/2), mode, fs=1e6, output='sos',analog=False)
            # plot period
            f_i0,ax_i0 = plt.subplots(constrained_layout=True)
            f_i1,ax_i1 = plt.subplots(constrained_layout=True)
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # filter data
                filtered = signal.sosfiltfilt(sos, i0)
                ax_i0.plot(time,filtered,label=sname)
                # filter data
                filtered = signal.sosfiltfilt(sos, i1)
                ax_i1.plot(time,filtered,label=sname)

            ax_i0.legend(loc='upper left')
            ax_i0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}, {mode_dict[mode]}, {freq:.2E}Hz")
            f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-overlap-filtered-{mode}-freq-{freq}.png")
            plt.close(f_i0)

            ax_i1.legend(loc='upper left')
            ax_i1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}, {mode_dict[mode]}, {freq:.2E}Hz")
            f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-overlap-filtered-{mode}-freq-{freq}.png")
            plt.close(f_i1)
        else:
            print(f"Unsupported file {fn}!")

def periodogramStripes(path: str,**kwargs):
    """ 
        Plot the periodigram of the target signal and saves the plots locally

        Currently hard coded to Input 0 and Input 1.

        Loads file using loadTDMSSub

        Inputs:
            path : Input file path
    """
    from scipy import signal
    sns.set_theme("paper")
    fname = os.path.splitext(path)[0]
    ftitle = os.path.basename(os.path.splitext(path)[0])
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.periodogram(i0, kwargs.get("sr",1e6))
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 {sname} Power Spectral Density")
                fig.savefig(fr"{fname}-Input 0-{sname}-psd.png")
                plt.close(fig)
        
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.periodogram(i1, 1e6)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 {sname} Power Spectral Density")
                fig.savefig(fr"{fname}-Input 1-{sname}-psd.png")
                plt.close(fig)
        else:
            print(f"Unsupported file {fn}!")


def welchStripes(path: str,**kwargs):
    """
        Plot the Welch PSD of each stripe in the target file

        Each stripe is saved separately to the same location as the source file

        When freq_clip is used, the user controls the y axis limit by setting the yspace parameter.
        The y-axis lim is the max value within freq_clip + yspace x max value

        Inputs:
            path : TDMS fpath
            freq_clip : Frequency range to clip the plot to
            yspace : Fraction of max value to add to the top. Default 0.1.
    """
    from scipy import signal
    sns.set_theme("paper")
    fname = os.path.splitext(path)[0]
    ftitle = os.path.basename(os.path.splitext(path)[0])
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x,fname=fname : fname in x,PERIODS))
        if periods:
            periods = periods[0][fname]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.welch(i0, 1e6,nperseg=1024)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 {sname} Power Spectral Density")
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    ax.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    ax.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
                    fig.savefig(fr"{fname}-Input 0-{sname}-welch-freq-clip-{f0}-{f1}.png")
                else:
                    fig.savefig(fr"{fname}-Input 0-{sname}-welch.png")
                plt.close(fig)
        
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.welch(i1, 1e6,nperseg=1024)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 {sname} Power Spectral Density")
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    ax.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    ax.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
                    fig.savefig(fr"{fname}-Input 1-{sname}-welch-freq-clip-{f0}-{f1}.png")
                else:
                    fig.savefig(fr"{fname}-Input 1-{sname}-welch.png")
                plt.close(fig)
        else:
            print(f"Unsupported file {fn}!")


def welchStripesOverlap(path: str,**kwargs):
    """
        Plot the Welch PSD of each stripe in the target file and OVERLAP all the plots on the same axis

        Each stripe is saved separately to the same location as the source file

        When freq_clip is used, the user controls the y axis limit by setting the yspace parameter.
        The y-axis lim is the max value within freq_clip + yspace x max value

        Inputs:
            path : TDMS fpath
            freq_clip : Frequency range to clip the plot to
            yspace : Fraction of max value to add to the top. Default 0.1.
    """
    from scipy import signal
    sns.set_theme("paper")
    fname = os.path.splitext(path)[0]
    ftitle = os.path.splitext(os.path.basename(path))[0]
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,FEED_RATE))
        if feedrate:
            feedrate = feedrate[0][ftitle]
        if periods:
            periods = periods[0][ftitle]
            fig_i0,ax_i0 = plt.subplots(constrained_layout=True)
            fig_i1,ax_i1 = plt.subplots(constrained_layout=True)
            Pmax_i0 = 0
            Pmax_i1 = 0
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f, Pxx_i0 = signal.welch(i0, 1e6,nperseg=1024)
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    Pmax_i0 = max(Pmax_i0,Pxx_i0[(f>=f0)&(f<=f1)].max())

                if kwargs.get("use_fr",False) and feedrate:
                    ax_i0.plot(f,Pxx_i0,label=feedrate[sname])
                else:
                    ax_i0.plot(f,Pxx_i0,label=sname)

                f, Pxx_i1 = signal.welch(i1, 1e6,nperseg=1024)
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    Pmax_i1 = max(Pmax_i1,Pxx_i1[(f>=f0)&(f<=f1)].max())

                if kwargs.get("use_fr",False) and feedrate:
                    ax_i1.plot(f,Pxx_i1,label=feedrate[sname])
                else:
                    ax_i1.plot(f,Pxx_i1,label=sname)

            ax_i0.legend()
            ax_i0.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 Power Spectral Density")
            if kwargs.get("freq_clip",None):
                f0,f1 = kwargs.get("freq_clip",None)
                ax_i0.set_xlim(f0,f1)
                ax_i0.set_ylim(0,Pmax_i0+kwargs.get("yspace",0.10)*Pmax_i0)
                fig_i0.savefig(fr"{fname}-Input 0-overlap-welch-freq-clip-{f0}-{f1}.png")
            else:
                fig_i0.savefig(fr"{fname}-Input 0-overlap-welch.png")

            ax_i1.legend()
            ax_i1.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 Power Spectral Density")
            if kwargs.get("freq_clip",None):
                f0,f1 = kwargs.get("freq_clip",None)
                ax_i1.set_xlim(f0,f1)
                ax_i1.set_ylim(0,Pmax_i1+kwargs.get("yspace",0.10)*Pmax_i1)
                fig_i1.savefig(fr"{fname}-Input 1-overlap-welch-freq-clip-{f0}-{f1}.png")
            else:
                fig_i1.savefig(fr"{fname}-Input 1-overlap-welch.png")

            plt.close('all')
        else:
            print(f"Unsupported file {fn}!")


def filterStripesProportion(path: str,minfreq: float=10e3,**kwargs):
    """
        For each stripe, change the cut off frequency and record the ratio between the filtered max value and the unfiltered max value.

        The minfreq value is the minimum cutoff freq. Go too low and the gain will skyrocket due to close to 0 freqs being removed.

        The order of the filter is set using order keyword.

        Inputs:
            path : TDMS path
            minfreq : Minimum cutoff frequency. Default 10e3
            order : Filter order. Default 10.
            res : Resolution of the jumps. Default 1e3.
    """
    from scipy import signal
    sns.set_theme("paper")
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            f,ax = plt.subplots(ncols=2,constrained_layout=True)
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                ## setup arrays for plotting
                # cutoff frequencies
                freq = [1e6/2,]
                # gain and set reference for Input 0
                gain_i0 = [1.0,]
                ref_i0 = i0.max()
                # gain and set reference for Input 1
                gain_i1 = [1.0,]
                ref_i1 = i1.max()
                for filt in np.arange(minfreq,1e6/2,kwargs.get("res",1e3))[::-1]:
                    # make filter
                    sos = signal.butter(kwargs.get("order",10), filt/(1e6/2), 'lowpass', fs=1e6, output='sos', analog=False)
                    # filter data
                    filtered = signal.sosfiltfilt(sos, i0)
                    gain_i0.append(filtered.max()/ref_i0.max())
                    # filter data
                    filtered = signal.sosfiltfilt(sos, i1)
                    gain_i1.append(filtered.max()/ref_i1.max())
                    # append cutoff freq
                    freq.append(filt)
                ax[0].plot(freq,gain_i0,label=sname)
                ax[1].plot(freq,gain_i1,label=sname)
            # setup legend
            ax[0].legend()
            ax[1].legend()
            # set labels
            ax[0].set(xlabel="Cuttoff Freq (Hz)",ylabel="Gain",title="Input 0")
            ax[1].set(xlabel="Cuttoff Freq (Hz)",ylabel="Gain",title="Input 1")
            f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}\nGain at different cuttoff freq")
            plt.close(f)
        else:
            print(f"Unsupported file {fn}!")


def filterStripesBP(path: str,lowcut:float=300e3,highcut:float=350e3,**kwargs):
    """
        Apply a bandpass butterworth filter to each stripe in the target file and plot the result

        A Butterworth bandpass filter is applied to the signal.
        The original and filtered signals are plotted as separate traces on the same axis

        Generated plots are saved in the same location as the source file

        This is intended to investigate specific frequency bandsidentified in the STFT

        Inputs:
            path : TDMS file path
            lowcut : Low cut off frequency. Default 300 kHz
            highcut : High cut off frequency. Default 350 kHz
            order : Order of filter. Default 6.
    """
    for fn in glob(path):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # check if supported
        periods = list(filter(lambda x,fname=fname : fname in x,PERIODS))
        if periods:
            periods = periods[0][fname]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f,ax = plt.subplots(constrained_layout=True)
                ax.plot(time,i0,'b-',label="Original")
                # make filter
                filtered = butter_bandpass_filter(i0, lowcut, highcut, order=kwargs.get("order",6))
                
                ax.plot(time,filtered,'r-',label="Filtered")
                ax.legend()
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 0")
                # set title
                ax.set_title(f"{fname}, Input 0, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-{lowcut}-{highcut}.png")
                plt.close(f)

                # plot period
                f,ax = plt.subplots(constrained_layout=True)
                ax.plot(time,i1,'b-',label="Original")
                filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                ax.plot(time,filtered,'r-',label="Filtered")
                ax.legend()
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 1")
                # set title
                ax.set_title(f"{fname}, Input 1, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-{lowcut}-{highcut}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")


def stackFilterStripesBP(path: str,freq: list[tuple[float, float]],**kwargs):
    """
        Apply a bandpass butterworth filter to each stripe in the target file and plot the result
        ON THE SAME AXIS

        A Butterworth bandpass filter is applied to the signal.
        The original and filtered signals are plotted as separate traces on the same axis

        Generated plots are saved in the same location as the source file

        This is intended to investigate specific frequency bands identified in the STFT

        Inputs:
            path : TDMS file path
            freq : List of freq pairs representing each bandpass filter to test
            order : Order of filter. Default 6.
    """
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                print(sname)
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f0,ax0 = plt.subplots(constrained_layout=True)
                ax0.plot(time,i0,label="Original")

                # plot period
                f1,ax1 = plt.subplots(constrained_layout=True)
                ax1.plot(time,i1,label="Original")
                # make filter
                for lowcut,highcut in freq:
                    print(lowcut,highcut)
                    filtered = butter_bandpass_filter(i0, lowcut, highcut, order=kwargs.get("order",6))
                    ax0.plot(time,filtered,label=f"{lowcut},{highcut} Hz")

                    filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                    ax1.plot(time,filtered,label=f"{lowcut},{highcut} Hz")
                ax0.set_ylim(i0.min(),i0.max())
                ax0.legend()
                ax0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                plt.show()
                f0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f0)

                ax1.set_ylim(i1.min(),i1.max())
                ax1.legend()
                ax1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f1)
        else:
            print(f"Unsupported file {fn}!")


def findEdgePoly3D(time: np.ndarray, i0: np.ndarray, dist: float) -> list[np.ndarray, np.ndarray]:
    """
        Find the edges of signal in the +ve and negativ e
    """
    from scipy.signal import find_peaks
    tt = []
    V = []
    #ff = []

    ## find the peaks in the +ve half of the signal
    # mask signal to +ve
    mask_filt = i0.copy()
    mask_filt[i0<0]=0
    # find peaks in the signal
    pks = find_peaks(mask_filt,distance=dist)[0]
    if len(pks)==0:
        raise ValueError("Failed to find +ve edge in data!")
    
    #pts.extend([[time[pk],mask_filt[pk],freq] for pk in pks])
    tt.extend(time[pks].tolist())
    V.extend(mask_filt[pks].tolist())
    #ff.extend(len(pks)*[freq,])

    ## find the peaks in the -ve half of the signal
    # mask signal to -ve
    mask_filt = i0.copy()
    mask_filt[i0>0]=0
    # find peaks in the signal
    pks = find_peaks(mask_filt,distance=dist)[0]
    if len(pks)==0:
        raise ValueError("Failed to find -ve edge in data!")
    pks = pks[::-1]
    #pts.extend([[time[pk],mask_filt[pk],freq] for pk in pks[::-1]])
    tt.extend(time[pks].tolist())
    V.extend(mask_filt[pks].tolist())
    #ff.extend(len(pks)*[freq,])
    return list(zip(tt,V))


def stackFilterEdgesBP3D(path: str,freq,dist=int(50e3),**kwargs):
    from matplotlib.collections import PolyCollection
    for fn in glob(path):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # check if supported
        periods = list(filter(lambda x : fname in x,PERIODS))
        if periods:
            periods = periods[0][fname]
            # filter to time periods dict
            for sname,chunk in periods.items():
                print(sname)
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                #f0,ax0 = plt.subplots(constrained_layout=True)
                f0 = plt.figure()
                ax0 = f0.add_subplot(projection="3d")
                # find the edges in 3d
                verts = findEdgePoly3D(time,i0,dist)
                print(np.array(verts).shape)
                # draw the edges as a 3d shape
                poly = PolyCollection([verts],facecolors=[plt.colormaps[kwargs.get("cmap","viridis_r")](0.1),],alpha=.7)
                ax0.add_collection3d(poly,zs=0,zdir='y')
                plt.show()

                # plot period
                f1,ax1 = plt.subplots(constrained_layout=True)
                ax1.plot(time,i1,label="Original")
                # make filter
                bp_30 = butter_bandpass_filter(30e3,"lowpass")
                for lowcut,highcut in freq:
                    print(lowcut,highcut)
                    # apply a first filter to below 30k and then apply the target freq?
                    # can't remember why
                    filtered = bp_30(i0, lowcut, highcut, order=kwargs.get("order",6))
                    ax0.plot(time,filtered,label=f"{lowcut},{highcut} Hz")

                    filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                    ax1.plot(time,filtered,label=f"{lowcut},{highcut} Hz")
                # set axis limits based on signal values
                ax0.set_ylim(i0.min(),i0.max())
                ax0.legend()
                ax0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 0, {sname}")
                plt.show()
                f0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f0)

                ax1.set_ylim(i1.min(),i1.max())
                ax1.legend()
                ax1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 1, {sname}")
                f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f1)
        else:
            print(f"Unsupported file {fn}!")


def get_channel_shapes(path: str, use_metadata: bool = False) -> list:
    """
        Get the shape of each channel

        Inputs:
            path : File path
            use_metadata : Flag to get the shape via metadata instead of the data. Can be inaccurate. Default False

        Returns list of channel paths and shapes
    """
    channel_shapes = []
    with TdmsFile(path, read_metadata_only=use_metadata, keep_open=True) as file:
        for g in file.groups():
            for r in g.channels():
                if use_metadata:
                    props = r.properties
                    channel_shapes.append((r.name, (int(props["wf_samples"]),)))
                else:
                    channel_shapes.append((r.name, r.data.shape))
    return channel_shapes


def stack_tdms_metadata(path: str) -> dict:
    """
        Stack the metadata in the TDMS files into a single nested dictionary

        The metadata is stored as OrderedDicts in the properties attribute of the channels

        The dict is organsied by group and then channel

        Inputs:
            file : Path to TDMS file

        Returns dict
    """
    meta_all = {}
    md = TdmsFile.read_metadata(path)
    
    meta_all.update(md.properties)
    for g in md.groups():
        meta_all[g.name] = g.properties
        for c in g.channels():
            c_md = c.properties
            meta_all[g.name][c.name] = c_md
    return meta_all


def find_sampling_rate(path: str, use_metadata: bool = False) -> float:
    """
        Find the sampling rate of the data

        The use_metadata flag is to control where it retrieves the data from. If True,
        only the metadata is loaded and the sampling rate is extracted from the wf_increment property

        If False, the sampling rate is calculated by finding using the difference between the first and second
        value of the time track.

        Inputs:
            path : Path to TDMS file
            use_metadata : Flag to control where the data is extracted from

        Returns floating point sampling rate
    """
    with TdmsFile(path, read_metadata_only=use_metadata) as file:
        for g in file.groups():
            if "Default" not in g.name:
                break
        channels = g.channels()
        if use_metadata:
            return 1./float(channels.properties["wf_increment"])
        else:
            return 1/(channels[0].time_track()[1] - channels[0].time_track()[0])


def convert_to_parquet(path: str, opath: str = None):
    """
        Convert target TDMS signal to parquet file

        The file is read in as a pandas dataframe. The columns are cleaned up to only be the channel names
        and the data types is changed using pd.to_numeric.

        Inputs:
            path : Path to TDMS
            opath : Output path for parquet file. If None, then the same filename and location is used. Default None
    """
    if opath is None:
        opath = os.path.splitex(os.path.basename(path))[0]

    df = TdmsFileExt(path, keep_open=True).as_dataframe(time_index=True, absolute_time=False)
    # change index to be called timestamp
    df.index.set_names(["Timestamp (s)"])
    # move index to column
    df.reset_index(inplace=True)
    cols = df.columns
    # force columns to be smallest datatype to save memory
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # simplify the column names
    df.rename(columns=lambda x : x.split("'/'")[-1][:-1], inplace=True)
    # convert and save
    df.to_parquet(opath)