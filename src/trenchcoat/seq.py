import numpy as np
from warnings import warn
from exiftool.exceptions import ExifToolException
import os
import cv2
import re
import jpeg_ls
import json

LOADED_EXIFTOOL = False
try:
    import exiftool
    LOADED_EXIFTOOL = True
except ImportError:
    LOADED_EXIFTOOL = False


class SEQCustomFile:
    """
        Custom class for reading the contents of a SEQ file

        This assumes that the file contains JPEG-LS image data

        On creation is scans and maps out where the image data is located and the user
        can access the specific images via index

        Example 1
        # load image and map the dat
        seq = SEQCustomFile(r"D:\thermal_recording_CSQ_files\2024-visit-1\plate-1-1.seq")
        # print number of frames
        print(seq.numframes())
        # get first frame
        seq[0]

    """
    def __init__(self, path, **kwargs):
        """ 
            Read SEQ file and get data about the file

            Inputs:
                path : Full path to SEQ file
                skip_tags : A flag to skip loading in the tags. Default False
        """
        self._path = path
        fname = os.path.splitext(os.path.basename(self._path))[0]
        self._tags = None
        # useful parameters for calculations
        self.B = None
        self.R1 = None
        self.R2 = None
        self.F = None
        self.O = None
        self.E = None
        self.T_refl = None
        self.Raw_refl = None
        self.fshape = None
        self.imgtype = None
        # extract tags using exiftool
        # and update the parameters
        if not kwargs.get("skip_tags", False):
            try:
                # if exiftool was imported
                if LOADED_EXIFTOOL:
                    with exiftool.ExifToolHelper() as et:
                        self._tags = et.get_metadata(path)[0]
                # else fallback to JSON backup
                else:
                    self._tags = json.load(open(os.path.join("seq_tags", f"{fname}-tags.json"), 'r'))
            except ExifToolException as e:
                warn(f"Failed to read tags and extract parameters! Reason: {e}")
                warn("Attempting fallback log")
                self._tags = json.load(open(os.path.join("seq_tags", f"{fname}-tags.json"), 'r'))
                self.B = self._tags["FLIR:PlanckB"]
                self.R1 = self._tags["FLIR:PlanckR1"]
                self.R2 = self._tags["FLIR:PlanckR2"]
                self.F = self._tags["FLIR:PlanckF"]
                self.O = self._tags["FLIR:PlanckO"]
                self.E = self._tags["FLIR:Emissivity"]
                self.T_refl = self._tags['FLIR:ReflectedApparentTemperature']
                self.Raw_refl = self._tags["FLIR:PlanckR1"]/(self._tags["FLIR:PlanckR2"]*np.exp(self._tags["FLIR:PlanckB"]/self._tags['FLIR:ReflectedApparentTemperature'])-self._tags["FLIR:PlanckF"])-self._tags["FLIR:PlanckO"]
                self.fshape = (self._tags['FLIR:RawThermalImageHeight'], self._tags['FLIR:RawThermalImageWidth'])
                self.imgtype = self._tags["FLIR:RawThermalImageType"]
                self.OD = self._tags["FLIR:ObjectDistance"]
                self.ATemp = self._tags["FLIR:AtmosphericTemperature"]
                self.IRWTemp = self._tags["FLIR:IRWindowTemperature"]
                self.IRT = self._tags["FLIR:IRWindowTransmission"]
                self.RH = self._tags["FLIR:RelativeHumidity"]

                # for atmospheric attenuation
                ATA1 = float(self._tags["FLIR:AtmosphericTransAlpha1"])
                ATA2 = float(self._tags["FLIR:AtmosphericTransAlpha2"])
                ATB1 = float(self._tags["FLIR:AtmosphericTransBeta1"])
                ATB2 = float(self._tags["FLIR:AtmosphericTransBeta2"])
                ATX = self._tags["FLIR:AtmosphericTransX"]

                emiss_wind = 1 - self.IRT
                refl_wind = 0

                # transmission through the air
                # from https://github.com/ManishSahu53/read_thermal_temperature/blob/master/flir_image_extractor.py
                h2o = (self.RH / 100) * np.exp(1.5587 + 0.06939 * (self.ATemp) - 0.00027816 * (self.ATemp) ** 2 + 0.00000068455 * (self.ATemp) ** 3)
                self.tau1 = ATX * np.exp(-np.sqrt(self.OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
                    -np.sqrt(self.OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))
                self.tau2 = ATX * np.exp(-np.sqrt(self.OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
                    -np.sqrt(self.OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))
                
                # radiance from the environment
                raw_refl1 = self.R1 / (self.R2 * (np.exp(self.B / (self.T_refl + 273.15)) - self.F)) - self.O
                self.raw_refl1_attn = (1 - self.E) / self.E * raw_refl1
                raw_atm1 = self.R1 / (self.R2 * (np.exp(self.B / (self.ATemp + 273.15)) - self.F)) - self.O
                self.raw_atm1_attn = (1 - self.tau1) / self.E / self.tau1 * raw_atm1
                raw_wind = self.R1 / (self.R2 * (np.exp(self.B / (self.IRWTemp + 273.15)) - self.F)) - self.O
                self.raw_wind_attn = emiss_wind / self.E / self.tau1 / self.IRT * raw_wind
                raw_refl2 = self.R1 / (self.R2 * (np.exp(self.B / (self.T_refl + 273.15)) - self.F)) - self.O
                self.raw_refl2_attn = refl_wind / self.E / self.tau1 / self.IRT * raw_refl2
                raw_atm2 = self.R1 / (self.R2 * (np.exp(self.B / (self.ATemp + 273.15)) - self.F)) - self.O
                self.raw_atm2_attn = (1 - self.tau2) / self.E / self.tau1 / self.IRT / self.tau2 * raw_atm2
            else:
                self.B = self._tags["FLIR:PlanckB"]
                self.R1 = self._tags["FLIR:PlanckR1"]
                self.R2 = self._tags["FLIR:PlanckR2"]
                self.F = self._tags["FLIR:PlanckF"]
                self.O = self._tags["FLIR:PlanckO"]
                self.E = self._tags["FLIR:Emissivity"]
                self.T_refl = self._tags['FLIR:ReflectedApparentTemperature']
                self.Raw_refl = self._tags["FLIR:PlanckR1"]/(self._tags["FLIR:PlanckR2"]*np.exp(self._tags["FLIR:PlanckB"]/self._tags['FLIR:ReflectedApparentTemperature'])-self._tags["FLIR:PlanckF"])-self._tags["FLIR:PlanckO"]
                self.fshape = (self._tags['FLIR:RawThermalImageHeight'], self._tags['FLIR:RawThermalImageWidth'])
                self.imgtype = self._tags["FLIR:RawThermalImageType"]
                self.OD = self._tags["FLIR:ObjectDistance"]
                self.ATemp = self._tags["FLIR:AtmosphericTemperature"]
                self.IRWTemp = self._tags["FLIR:IRWindowTemperature"]
                self.IRT = self._tags["FLIR:IRWindowTransmission"]
                self.RH = self._tags["FLIR:RelativeHumidity"]

                # for atmospheric attenuation
                ATA1 = float(self._tags["FLIR:AtmosphericTransAlpha1"])
                ATA2 = float(self._tags["FLIR:AtmosphericTransAlpha2"])
                ATB1 = float(self._tags["FLIR:AtmosphericTransBeta1"])
                ATB2 = float(self._tags["FLIR:AtmosphericTransBeta2"])
                ATX = self._tags["FLIR:AtmosphericTransX"]

                emiss_wind = 1 - self.IRT
                refl_wind = 0

                # transmission through the air
                # from https://github.com/ManishSahu53/read_thermal_temperature/blob/master/flir_image_extractor.py
                h2o = (self.RH / 100) * np.exp(1.5587 + 0.06939 * (self.ATemp) - 0.00027816 * (self.ATemp) ** 2 + 0.00000068455 * (self.ATemp) ** 3)
                self.tau1 = ATX * np.exp(-np.sqrt(self.OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
                    -np.sqrt(self.OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))
                self.tau2 = ATX * np.exp(-np.sqrt(self.OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) + (1 - ATX) * np.exp(
                    -np.sqrt(self.OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o)))
                
                # radiance from the environment
                raw_refl1 = self.R1 / (self.R2 * (np.exp(self.B / (self.T_refl + 273.15)) - self.F)) - self.O
                self.raw_refl1_attn = (1 - self.E) / self.E * raw_refl1
                raw_atm1 = self.R1 / (self.R2 * (np.exp(self.B / (self.ATemp + 273.15)) - self.F)) - self.O
                self.raw_atm1_attn = (1 - self.tau1) / self.E / self.tau1 * raw_atm1
                raw_wind = self.R1 / (self.R2 * (np.exp(self.B / (self.IRWTemp + 273.15)) - self.F)) - self.O
                self.raw_wind_attn = emiss_wind / self.E / self.tau1 / self.IRT * raw_wind
                raw_refl2 = self.R1 / (self.R2 * (np.exp(self.B / (self.T_refl + 273.15)) - self.F)) - self.O
                self.raw_refl2_attn = refl_wind / self.E / self.tau1 / self.IRT * raw_refl2
                raw_atm2 = self.R1 / (self.R2 * (np.exp(self.B / (self.ATemp + 273.15)) - self.F)) - self.O
                self.raw_atm2_attn = (1 - self.tau2) / self.E / self.tau1 / self.IRT / self.tau2 * raw_atm2

        # list of locations for FF FF FF 00
        self._mlocs = []
        # loc for start of JEG-LS
        self._jlocs = []
        # setup regular expression for searching for FF FF FF 00
        fff_search = re.compile(b"\x46\x46\x46\x00")
        # start + end bytes of a JPEG-LS
        if self.imgtype == "JPG":
            jpeg_start = re.compile(b"\xff\xd8\xff\xf7")
            jpeg_end = re.compile(b"\xFF\xD9")
        else:
            raise ValueError("Unsupported image type "+ self.imgtype)
        # read in file
        data = open(self._path,"rb").read()
        # search for FF FF FF 00 markers
        for seg in fff_search.finditer(data):
            self._mlocs.append(seg.span()[0])
            # starting from here
            start = jpeg_start.search(data, seg.span()[1])
            end = jpeg_end.search(data, start.span()[1])
            self._jlocs.append((start.span()[0], end.span()[1]))
        # record the number of frames
        self.__numframes = len(self._jlocs)
        # open the file to avoid re-opening the file
        self._data = open(self._path,"rb")


    def get_flir_tags(self):
        return {k:v for k,v in self._tags.items() if "FLIR:" in k}


    def numframes(self):
        return self.__numframes


    def __len__(self):
        return self.__numframes


    def __str__(self):
        return f"SEQCustomFile {self._path}, ({self.fshape[0]},{self.fshape[1]}, {self.__numframes})"


    def shape(self):
        return (*self.fshape, self.__numframes)


    def close(self):
        self._data.close()


    def reopen(self):
        self._data = open(self._path,'rb')


    def fname(self):
        return os.path.splitext(os.path.basename(self._path))[0]


    def frame_rate(self) -> float:
        return float(self._tags['FLIR:FrameRate'])
    
    # extract images by index
    # if it fails to decode the image then it returns None
    def __getitem__(self, index):
        # check index
        if not isinstance(index, int):
            raise IndexError("Index has to be an integer!")
        if index >= self.__numframes:
            raise IndexError("Index out of bounds!")
        start,end = self._jlocs[index]
        self._data.seek(start)
        # load data
        data_img = self._data.read(abs(end-start))
        data_img_arr = np.array(data_img)
        try:
            # attempt to convert to valid image
            im = jpeg_ls.decode(data_img_arr)
            # update frame shape if not set
            if self.fshape is None:
                self.fshape = im.shape
            return im
        # on encoding error return None
        except RuntimeError:
            blank_arr = np.empty(self.fshape)
            blank_arr.fill(np.nan)
            return blank_arr


    def get_jpeg_data(self, index):
        """
            Extract the raw JPEG data for the specific frame index

            On creation a JPEG lookup table is made to simplify searching for the image data.
            The lookup table is a list of the identified start and end bytes of the JPEG data.
            Invalid images can occur typically due to the identified end of the image being too
            far away from the header (according to the header data) causing it to fail.

            See flir-seq-file-specification.md for mroe details

            Intended as a debug method to investigate invalid or failed images

            Inputs:
                index : Target frame index

            Returns bytearray
        """
        start,end = self._jlocs[index]
        self._data.seek(start)
        data_img = self._data.read(abs(end-start))
        return data_img
    

    def get_max_frame(self):
        """
            Get the frame with the highest max value

            The file is scanned for the valid frame which has the highest raw max value.
            The max value is calulated using np.nanmax

            Invalid frames or those that cause an error are skipped

            Returns identified decoded image
        """
        return max(self, key=lambda x : np.nanmax(x) if not np.isnan(np.nanmax(x)) else -1)


    def get_max_frame_idx(self):
        """
            Get index of frame with the highest max value

            The file is scanned for the valid frame which has the highest raw max value.
            The max value is calulated using np.nanmax

            Invalid frames or those that cause an error are skipped

            Returns index of frame with highest raw max value
        """
        return int(np.argmax(list(map(lambda x : np.nanmax(x) if not np.isnan(np.nanmax(x)) else -1, self))))


    def apply_each_frame(self, proc, frame_type, **kwargs):
        """ 
            Apply the supplied function to each valid frame in the file and store the results in a list

            Example: Finding mean of each frame
                seq = SEQCustomFile("path to file")
                res = seq.apply_each_frame(lambda x : x.mean())

            The input frame_type is for stating the type of data that needs to be processed.
            The currently supported impots are
            raw : Raw JPEG data
            radiance, rad : Raw converted to radiance values
            temp, temperature : Raw converted to temperature data

            Inputs:
                proc : Function to apply to each decoded frame
                frame_type : Type of frame to process
                **kwargs : See temprad or tempiter

            Returns a list of the results returned by the function
        """
        if frame_type not in ["raw", "radiance", "rad", "temp", "temperature"]:
            raise ValueError(f'Unsupported frame_type {frame_type}! Supported frame types are ["raw", "radiance", "rad", "temp", "temperature"]')
        # if sending
        if frame_type in ["raw"]:
            return list(map(proc, self))
        # else if sending radiance values
        elif frame_type in ["radiance", "rad"]:
            return list(map(proc, self.temprad(**kwargs)))
        # else if sending temperature
        elif frame_type in ["temperature", "temp"]:
            return list(map(proc, self.tempiter(**kwargs)))

    # convert raw 16 bit value to radiance
    def _frame2rad(self, frame_raw, E = None):
        """ 
            Convert Raw 16 bit frame to radiance values

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            Used as intermediate step to convert to temperature

            Inputs:
                frame_raw : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None

            Returns converted array
        """
        if E is None:
            E = self.E
        # convert 16-bit FLIR RAW to radiance of measured object
        #return (frame-(1-E)*Raw_refl)/E
        frame_raw = frame_raw.astype("float64")
        return (frame_raw / E / self.tau1 / self.IRT / self.tau2 - self.raw_atm1_attn -
                    self.raw_atm2_attn - self.raw_wind_attn - self.raw_refl1_attn - self.raw_refl2_attn)
    
    # single function to convert raw 16 bit to temperature
    def frame2temp(self, frame_raw, E = None, units = "C"):
        """ 
            Convert Raw 16 bit frame to temperature

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            The units input states what temperature units the returned frame should be.
            The supported units are as follows:
                K : Kelvin
                F : Farenheit
                C : Celcius

            Anything else is not supported

            Inputs:
                frame : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as

            Returns array of temperature values in the requested units
        """
        if E is None:
            E = self.E
        if units not in ["K", "C", "F"]:
            raise ValueError("Received unsupported units string!")
        # convert radiance values to temperature
        #kelvin = self.B/np.log(self.R1/(self.R2*(frame_float+self.O))+self.F)
        kelvin = self.B / np.log(self.R1 / (self.R2 * (self._frame2rad(frame_raw) + self.O)) + self.F)
        # convert to appropriate units
        if units == "K":
            return kelvin
        elif units == "C":
            return kelvin - 273.15
        elif units == "F":
            return (kelvin * 1.8) - 459.67
        
    # iterator for getting the frames as temperature
    def tempiter(self, E = None, units= "C", start_at = 0):
        """
            Alternative iterator that retrieve frames as temperature rather than raw

            Example:
                for frame in SEQCustomFile("path to file").tempiter(units="C"):
                    max_temp = frame.max()

            Inputs:
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as
                start_at : Start at the following frame index 
        """
        # check parameters
        if E is None:
            E = self.E
        if units not in ["K", "C", "F"]:
            raise ValueError("Received unsupported units string!")
        # iterate over the mapped locations
        for start, end in self._jlocs[start_at:]:
            # go to location
            self._data.seek(start)
            # read data
            data_img = self._data.read(abs(end-start))
            data_img_arr = np.array(data_img)
            # attempt to decode it
            try:
                raw = jpeg_ls.decode(data_img_arr)
                if self.fshape is None:
                    self.fshape = raw.shape
                yield self.frame2temp(raw, E, units)
            # if it failed then return an empty frame
            except RuntimeError:
                blank_arr = np.empty(self.fshape)
                blank_arr.fill(np.nan)
                yield blank_arr


    def temprad(self, E = None, start_at = 0, **kwargs):
        """
            Alternative iterator that retrieve frames as radiance rather than raw

            Example:
                for frame in SEQCustomFile("path to file").temprad():
                    max_rad = frame.max()

            Emissivity can also be specified as a list of emissivities to be applied across different
            regions of the data. This requires an input called emask which is an array the same shape
            as the frames where the values correspond to the index to reference the emissivity array

            So: 
                Where emask is 0, the 0th element of the emissivity list is used
                Where emask is 1, the 1st element of the emissivity list is used
                Where emask is 2, the 2nd element of the emissivity list is used
            etc:

            Example 2:
                # create an array of values
                seq = SEQCustomFile(path)
                # create array of values and populate it
                emask = np.zeros(seq.fshape, "float64")
                emask[50:100, 50:100] = 1
                emask[200:, 200:] = 2
                # list of emissivities to apply
                ems = [0.96, 0.8, 0.6]
                # iterate over frames passing list of emissivity values + mask
                for frame in seq.temprad(ems, emask=emask):
                    ### do something with frame ###

            Inputs:
                E : Emissivity value or values to use. If None, use value in recording. Default None
                start_at : Start at the following frame index
                emask : Array of values used to index the emissivity array

            Yields array of radiance values
        """
        # check parameters
        if E is None:
            E = self.E
        # iterate over the mapped locations
        for start, end in self._jlocs[start_at:]:
            # go to location
            self._data.seek(start)
            # read data
            data_img = self._data.read(abs(end-start))
            data_img_arr = np.array(data_img)
            # attempt to decode it
            try:
                raw = jpeg_ls.decode(data_img_arr)
                if self.fshape is None:
                    self.fshape = raw.shape
                # if emissivity is a single value
                if isinstance(E, float):
                    yield self._frame2rad(raw, E)
                # if it's a list of values
                else:
                    emasks = kwargs.get("emask", None)
                    if (emasks is None):
                        raise ValueError("emask must not be None!")
                    res = np.zeros(self.fshape, "float64")
                    # replace certain section of the 
                    for ei, e in enumerate(E):
                        res[emasks == ei] = self._frame2rad(raw, e)[emasks == ei]
                    yield res
            # if it failed then return an empty frame
            except RuntimeError:
                blank_arr = np.empty(self.fshape)
                blank_arr.fill(np.nan)
                yield blank_arr


    def get_frame_temp(self, idx, E = None, units = "C", **kwargs):
        """
            Get specific frame in terms of temeprature

            Emissivity can also be specified as a list of emissivities to be applied across different
            regions of the data. This requires an input called emask which is an array the same shape
            as the frames where the values correspond to the index to reference the emissivity array

            So: 
                Where emask is 0, the 0th element of the emissivity list is used
                Where emask is 1, the 1st element of the emissivity list is used
                Where emask is 2, the 2nd element of the emissivity list is used
            etc:

            Example 2:
                # create an array of values
                seq = SEQCustomFile(path)
                # create array of values and populate it
                emask = np.zeros(seq.fshape, "float64")
                emask[50:100, 50:100] = 1
                emask[200:, 200:] = 2
                # list of emissivities to apply
                ems = [0.96, 0.8, 0.6]
                # iterate over frames passing list of emissivity values + mask
                for frame in seq.temprad(ems, emask=emask):
                    ### do something with frame ###

            Inputs:
                idx : Target frame index
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as
                emask : Array of values used to index the emissivity array

            Yield array of temperature values in the specified units
        """
        if E is None:
            E = self.E
        if units not in ["K", "C", "F"]:
            raise ValueError("Received unsupported units string!")
        # get the data range
        start, end = self._jlocs[idx]
        # load the data
        self._data.seek(start)
        data_img = self._data.read(abs(end-start))
        data_img_arr = np.array(data_img)
        # attempt to decode it
        try:
            raw = jpeg_ls.decode(data_img_arr)
            if self.fshape is None:
                self.fshape = raw.shape
            # if emissivity is a single value
            if isinstance(E, float):
                return self.frame2temp(raw, E, units)
            # if it's a list of values
            else:
                emasks = kwargs.get("emask", None)
                if (emasks is None):
                    raise ValueError("emask must not be None!")
                res = np.zeros(self.fshape, "float64")
                # replace certain section of the 
                for ei, e in enumerate(E):
                    res[emasks == ei] = self.frame2temp(raw, e, units)[emasks == ei]
                return res
        # if it failed then return an empty frame
        except RuntimeError:
            blank_arr = np.empty(self.fshape)
            blank_arr.fill(np.nan)
            return blank_arr


    def get_frame_rad(self, idx, E = None):
        """
            Get specific frame in terms of radiance

            Inputs:
                idx : Target frame index
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as

            Return frame
        """
        if E is None:
            E = self.E
        # get the data range
        start, end = self._jlocs[idx]
        # load the data
        self._data.seek(start)
        data_img = self._data.read(abs(end-start))
        data_img_arr = np.array(data_img)
        # attempt to decode it
        try:
            raw = jpeg_ls.decode(data_img_arr)
            if self.fshape is None:
                self.fshape = raw.shape
            return self._frame2rad(raw, E)
        # if it failed then return an empty frame
        except RuntimeError:
            blank_arr = np.empty(self.fshape)
            blank_arr.fill(np.nan)
            return blank_arr
        

    def export_to_npz(self, opath = None, E = None, units = "C", skip_bad_frames = False):
        """ 
            Load and export the temperature data to a NPZ file at the target location

            NPZ is a compressed numpy array that's a good way to save space and avoid having to go to CSV

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            The units input states what temperature units the returned frame should be.
            The supported units are as follows:
                K : Kelvin
                F : Farenheit
                C : Celcius

            Anything else is not supported.

            If the flag skip_bad_frames is set to True, the bad frames are not added to the stack.
            If skip_bad_frames is False, a frame of NaNs is added instead. This can be useful for ensuring consistent sizes.

            Inputs:
                opath : Output path for the NPZ file. If None, the path is set to the same folder as the source recording. Default None
                frame : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as
                skip_bad_frames : Flag to skip bad frames. Default False
        """
        if E is None:
            E = self.E
        if units not in ["K", "C", "F"]:
            raise ValueError("Received unsupported units string!")
        # if output path is not set then make one using the original path of the recording
        if opath is None:
            opath = os.path.splitext(os.path.basename(self._path))[0] + ".npz"
        frames = [frame for frame in self.tempiter(E, units)]
        # filter bad values
        if skip_bad_frames:
            frames = list(filter(lambda fr : not np.isnan(fr).all(), frames))
        frames_arr = np.dstack(frames)
        np.savez_compressed(opath, frames_arr)

    # construct a time track for the frames
    def make_time_track(self):
        """ 
            Construct an array of time values useful for plotting

            The number of frames 
        """
        frame_rate = self._tags['FLIR:FrameRate']
        return np.arange(self.__numframes)/float(frame_rate)
    
    # identify the indices of bad frames
    def find_invalid_indices(self):
        """ 
            Scan the file and find the indices for the INVALID frames

            An frame is invalid/bad if it cannot be decoded correctly

            Returns list of indices corresponding to these bad frames
        """
        invalid_idx = []
        for i in range(self.__numframes):
            try:
                self[i]
            except RuntimeError:
                invalid_idx.append(i)
                continue
        return invalid_idx

    # identify the indices of good frames
    def find_valid_indices(self):
        """ 
            Scan the file and find the indices for the VALID frames

            An frame is valid/good if it can be decoded correctly

            Returns list of indices corresponding to these good frames
        """
        valid_idx = []
        for i in range(self.__numframes):
            try:
                self[i]
                valid_idx.append(i)
            except RuntimeError:
                continue
        return valid_idx

    # Get the shape of each frame and return the shape
    def get_shapes(self):
        """ 
            Get the shape of each frame and store in a list

            Useful as a debugging function for checking size of read in frames

            Returns set of frame shapes
        """
        return set(map(lambda frame: frame.shape, self))
    
    # iterate over each frame and save as a 16-bit grayscale PNG
    def export_images(self, opath: str, mode: str = "raw", **kwargs):
        fname = os.path.splitext(os.path.basename(opath))[0]

        if mode == "raw":
            frame_iter = self
        elif mode == "temp":
            frame_iter = self.tempiter(kwargs.get("E", None))

        for i, frame in enumerate(frame_iter):
            if mode == "raw":
                frame = frame.astype("uint16")
                cv2.imwrite(os.path.join(opath,f"{fname}_frame_{i:06}.png"), frame)
            if mode == "temp":
                frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                frame_cmap = cv2.applyColorMap(frame_norm, kwargs.get("cmap", cv2.COLORMAP_HOT))
                cv2.imwrite(os.path.join(opath,f"{fname}_frame_{i:06}.png"), frame_cmap)