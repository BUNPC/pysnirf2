from abc import ABC, abstractmethod
import h5py
import os
import sys
import numpy as np
from warnings import warn
from collections import MutableSequence
from tempfile import TemporaryFile
import logging

def _create_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# Package wide logger that can superseded in files
_logger = _create_logger('pysnirf2', 'pysnirf2.log')

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')

# -- methods to cast data prior to writing to and after reading from h5py interfaces------

_varlen_str_type = h5py.string_dtype(encoding='ascii', length=None)  # Length=None creates HDF5 variable length string
_DTYPE_FLOAT = 'f8'
_DTYPE_INT = 'i4'
_DTYPE_STR = 'S'  # Not sure how robust this is, but fixed length strings will always at least contain S


def _create_dataset_dataset(file, name, data):
    """
    Formats the input to satisfy the SNIRF specification as well as the HDF5 format. Determines appropriate write
    function from the type of data. The returned value should be passed to h5py.create_dataset() as kwarg "data" to be
    saved to an HDF5 file.

    Returns None if input is invalid and an h5py.Dataset instance if successful.

    Supported dtype str values: 'O', _DTYPE_FLOAT, _DTYPE_INT
    """
    try:
        if len(data) > 1:
            data = np.array(data)  # Cast to array
            dtype = data[0].dtype
            if any([dtype is t for t in [int, np.int32, np.int64]]):  # int
                return _create_dataset_int_array(file, name, data)
            elif any([dtype is t for t in [float, np.float, np.float64]]):  # float
                return _create_dataset_float_array(file, name, data)
            elif any([dtype is t for t in [str, np.string_]]):  # string
                return _create_dataset_string_array(file, name, data)
    except TypeError:  # data has no len()
        dtype = type(data)
    if any([dtype is t for t in [int, np.int32, np.int64]]):  # int
        return _create_dataset_int(file, name, data)
    elif any([dtype is t for t in [float, np.float, np.float64]]):  # float
        return _create_dataset_float(file, name, data)
    elif any([dtype is t for t in [str, np.string_]]):  # string
        return _create_dataset_string(file, name, data)
    raise TypeError('Unrecognized data type' + str(dtype)
                    + '. Please provide an int, float, or str, or an iterable of these.')


def _create_dataset_string(file: h5py.File, name: str, data: str):
    return file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=str(data))


def _create_dataset_int(file: h5py.File, name: str, data: int):
    return file.create_dataset(name, dtype=_DTYPE_INT, data=int(data))


def _create_dataset_float(file: h5py.File, name: str, data: float):
    return file.create_dataset(name, dtype=_DTYPE_FLOAT, data=float(data))


def _create_dataset_string_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype('O')
    if data.size is 0 or data.any() is None:
        array = AbsentDataset()  # Do not save empty or "None" NumPy arrays
    return file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=array)


def _create_dataset_int_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype(int)
    if data.size is 0 or data.any() is None:
        array = AbsentDataset()
    return file.create_dataset(name, dtype=_DTYPE_INT, data=array)


def _create_dataset_float_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype(float)
    if data.size is 0 or data.any() is None:
        array = AbsentDataset()
    return file.create_dataset(name, dtype=_DTYPE_FLOAT, data=array)


# -----------------------------------------


def _read_dataset(h_dataset: h5py.Dataset):
    """
    Reads data from a SNIRF file using the HDF5 interface and casts it to a Pythonic type or numpy array. Determines
    the appropriate read function from the properties of h_dataset
    """
    if type(h_dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if h_dataset.size > 1:
        if _DTYPE_STR in h_dataset.dtype:
            return _read_string_array(h_dataset)
        elif _DTYPE_INT in h_dataset.dtype:
            return _read_int_array(h_dataset)
        elif _DTYPE_FLOAT in h_dataset.dtype:
            return _read_float_array(h_dataset)
    else:
        if _DTYPE_STR in h_dataset.dtype:
            return _read_string(h_dataset)
        elif _DTYPE_INT in h_dataset.dtype:
            return _read_int(h_dataset)
        elif _DTYPE_FLOAT in h_dataset.dtype:
            return _read_float(h_dataset)
    raise TypeError("Dataset dtype='" + str(h_dataset.dtype)
    + "' not recognized. Expecting dtype to contain one of these: " + str([_DTYPE_STR, _DTYPE_INT, _DTYPE_FLOAT]))


def _read_string(dataset: h5py.Dataset):
    # Because many SNIRF files are saved with string values in length 1 arrays
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return dataset[0].decode('ascii')
    else:
        return dataset[()].decode('ascii')


def _read_int(dataset: h5py.Dataset):
    # Because many SNIRF files are saved with string values in length 1 arrays
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return int(dataset[0])
    else:
        return int(dataset[()])


def _read_float(dataset: h5py.Dataset):
    # Because many SNIRF files are saved with string values in length 1 arrays
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return float(dataset[0])
    else:
        return float(dataset[()])


def _read_string_array(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(str)


def _read_int_array(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(int)


def _read_float_array(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(float)


# -----------------------------------------


class SnirfFormatError(Exception):
    pass


class SnirfConfig:
    dynamic_loading: bool = False  # If False, data is loaded in the constructor, if True, data is loaded on access


class AbsentDataset():    
    def __repr__(self):
        return str(None)


class AbsentGroup():
    def __repr__(self):
        return str(None)


class Group(ABC):

    def __init__(self, varg, cfg: SnirfConfig):
        """
        Wrapper for an HDF5 Group element defined by SNIRF. Must be created with a
        Group ID or string specifying a complete path relative to file root--in
        the latter case, the wrapper will not correspond to a real HDF5 group on
        disk until _save() (with no arguments) is executed for the first time
        """
        self._cfg = cfg
        if type(varg) is str:  # If a Group wrapper is created prior to a save to HDF Group object
            self._h = {}
            self._location = varg
        elif isinstance(varg, h5py.h5g.GroupID):  # If Group is created based on an HDF Group object
            self._h = h5py.Group(varg)
            self._location = self._h.name
        else:
            raise TypeError('must initialize ' + self.__class__.__name__ + ' with a Group ID or string, not ' + str(type(varg)))

    def save(self, *args):
        """
        Entry to Group-level save
        """
        if len(args) > 0:
            if type(args[0]) is h5py.File:
                self._save(args[0])
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError("No such SNIRF file '" + path + "'. Create a SNIRF file before attempting to save a Group to it.")
                self._save(file)
                file.close()
        else:
            if self._h != {}:
                file = self._h.file
            self._save(file)
            
    @property
    def filename(self):
        """
        Returns None if the wrapper is not associated with a Group on disk        
        """
        if self._h != {}:
            return self._h.file.filename
        else:
            return None

    @property
    def location(self):
        if self._h != {}:
            return self._h.name
        else:
            return self._location
    
    @abstractmethod
    def _save(self, *args):
        """
        args is path or empty
        """
        raise NotImplementedError('_save is an abstract method')
        
    def __repr__(self):
        props = [p for p in dir(self) if ('_' not in p and not callable(getattr(self, p)))]
        out = str(self.__class__.__name__) + ' at ' + str(self.location) + '\n'
        for prop in props:
            attr = getattr(self, prop)
            out += prop + ': '
            if type(attr) is np.ndarray or type(attr) is list:
                if np.size(attr) > 32:
                    out += '<' + str(np.shape(attr)) + ' array of ' + str(attr.dtype) + '>'
                else:
                    out += str(attr)
            else:
                prepr = str(attr)
                if len(prepr) < 64:
                    out += prepr
                else:
                    out += '\n' + prepr
            out += '\n'
        return out[:-1]

    def __getitem__(self, key):
        if self._h != {}:
            if key in self._h:
                return self._h[key]
        else:
            return None

    def __contains__(self, key):
        return key in self._h;


class IndexedGroup(MutableSequence, ABC):
    """
    Represents the "indexed group" which is defined by v1.0 of the SNIRF
    specification as:
        If a data element is an HDF5 group and contains multiple sub-groups,
        it is referred to as an indexed group. Each element of the sub-group
        is uniquely identified by appending a string-formatted index (starting
        from 1, with no preceding zeros) in the name, for example, /.../name1
        denotes the first sub-group of data element name, and /.../name2
        denotes the 2nd element, and so on.
    """
    
    _name: str = ''  # The specified prefix to this indexed group's members, i.e. nirs, data, stim, aux, measurementList
    _element: Group = None  # The type of Group which belongs to this IndexedGroup

    def __init__(self, parent: Group, cfg: SnirfConfig):
        # Because the indexed group is not a true HDF5 group but rather an
        # iterable list of HDF5 groups, it takes a base group or file and
        # searches its keys, appending the appropriate elements to itself
        # in order
        self._parent = parent
        self._cfg = cfg
        self._populate_list()
    
    @property
    def filename(self):
        return self._parent.filename

    def __len__(self): return len(self._list)

    def __getitem__(self, i): return self._list[i]

    def __delitem__(self, i): del self._list[i]
    
    def __setitem__(self, i, item):
        self._check_type(item)
        if not item.location in [e.location for e in self._list]:
            self._list[i] = item
        else:
            raise SnirfFormatError(item.location + ' already an element of ' + self.__class__.__name__)

    def __getattr__(self, name):
        # If user tries to access an element's properties, raise informative exception
        if name in [p for p in dir(self._element) if ('_' not in p and not callable(getattr(self._element, p)))]:
            raise AttributeError(self.__class__.__name__ + ' is an interable list of '
                                + str(len(self)) + ' ' + str(self._element)
                                + ', access these with an index i.e. '
                                + str(self._name) + '[0].' + name
                                )

    def __repr__(self):
        return str('<' + 'iterable of ' + str(len(self._list)) + ' ' + str(self._element) + '>')

    def insert(self, i, item):
        self._check_type(item)
        self._list.insert(i, item)

    def append(self, item):
        self._check_type(item)
        self._list.append(item)
    
    def save(self, *args):
        if len(args) > 0:
            if type(args[0]) is h5py.File:
                self._save(args[0])
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError("No such SNIRF file '" + path + "'. Create a SNIRF file before attempting to save an IndexedGroup to it.")
                self._save(file)
                file.close()
        else:
            if self._parent._h != {}:
                file = self._parent._h.file
            self._save(file)

    def appendGroup(self):
        'Adds a group to the end of the list'
        location = self._parent.location + '/' + self._name + str(len(self._list) + 1)
        self._list.append(self._element(location, self._cfg))
    
    def _populate_list(self):
        """
        Add all the appropriate groups in parent to the list
        """
        self._list = list()
        names = self._get_matching_keys()
        for name in names:
            if name in self._parent._h:
                self._list.append(self._element(self._parent[name].id, self._cfg))
    
    def _check_type(self, item):
        if type(item) is not self._element:
            raise TypeError('elements of ' + str(self.__class__.__name__) +
                            ' must be ' + str(self._element) + ', not ' +
                            str(type(item))
                            )
        
    def _order_names(self, h=None):
        '''
        Enforce the format of the names of HDF5 groups within a group or file on disk. i.e. IndexedGroup stim's elements
        will be renamed, in order, /stim1, /stim2, /stim3. This is expensive but can be avoided by save()ing individual groups
        within the IndexedGroup
        '''
        if h is None:
            h = self._parent._h
        print([e.location for e in self._list])
        if all([len(e.location.split('/' + self._name)[-1]) > 0 for e in self._list]):
            if not [int(e.location.split('/' + self._name)[-1]) for e in self._list] == list(range(1, len(self._list) + 1)):
                # if list is not already ordered propertly
                for i, e in enumerate(self._list):
                    # To avoid assignment to an existing name, move all
                    h.move(e.location,
                           '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp')
    #                print(e.location, '--->',
    #                      '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp')
                for i, e in enumerate(self._list):
                    h.move('/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp',
                           '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1))
    #                print('/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp', '--->',
    #                      '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1))
        
    def _get_matching_keys(self, h=None):
        '''
        Return sorted list of a group or file's keys which match this IndexedList's _name format
        '''
        if h is None:
            h = self._parent._h
        unordered = []
        indices = []
        for key in h:
            numsplit = key.split(self._name)
            if len(numsplit) > 1 and len(numsplit[1]) > 0:
                if len(numsplit[1]) == len(str(int(numsplit[1]))):
                    unordered.append(key)
                    indices.append(int(numsplit[1]))
            elif key.endswith(self._name):
                unordered.append(key)
                indices.append(0)
        order = np.argsort(indices)
        return [unordered[i] for i in order]
        
    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            h = args[0]
        else:
            if self._parent._h != {}:
                h = self._parent._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance')
        names_in_file = self._get_matching_keys(h=h)  # List of all names in the file on disk
        names_to_save = [e.location.split('/')[-1] for e in self._list]  # List of names in the wrapper
        print('Saving indexed group', self.__class__.__name__)
        print('names to save: ', names_to_save)
        print('names in file: ', names_in_file)
        # Remove groups which remain on disk after being removed from the wrapper
        for name in names_in_file:
            if name not in names_to_save:
                print('Deleting', self._parent.name + '/' + name, 'while overwriting indexed group', self.__class__.__name__, 'as it has been deleted from the file')
                del h[self._parent.name + '/' + name]  # Remove the actual data from the hdf5 file.
        for e in self._list:
            e._save(*args)  # Group save functions handle the write to disk
        self._order_names(h=h)  # Enforce order in the group names



# generated by sstucker on 2021-11-29
# version 1.0 SNIRF specification parsed from https://raw.githubusercontent.com/fNIRS/snirf/v1.0/snirf_specification.md


class MetaDataTags(Group):

    _SubjectID = AbsentDataset()  # "s"*
    _MeasurementDate = AbsentDataset()  # "s"*
    _MeasurementTime = AbsentDataset()  # "s"*
    _LengthUnit = AbsentDataset()  # "s"*
    _TimeUnit = AbsentDataset()  # "s"*
    _FrequencyUnit = AbsentDataset()  # "s"*


    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self.__snirfnames = ['SubjectID', 'MeasurementDate', 'MeasurementTime', 'LengthUnit', 'TimeUnit', 'FrequencyUnit', ]
        if 'SubjectID' in self._h:
            if not self._cfg.dynamic_loading:
                self._SubjectID = _read_string(self._h['SubjectID'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"SubjectID"')
        if 'MeasurementDate' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementDate = _read_string(self._h['MeasurementDate'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"MeasurementDate"')
        if 'MeasurementTime' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementTime = _read_string(self._h['MeasurementTime'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"MeasurementTime"')
        if 'LengthUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._LengthUnit = _read_string(self._h['LengthUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"LengthUnit"')
        if 'TimeUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._TimeUnit = _read_string(self._h['TimeUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"TimeUnit"')
        if 'FrequencyUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._FrequencyUnit = _read_string(self._h['FrequencyUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"FrequencyUnit"')

    @property
    def SubjectID(self):
        if type(self._SubjectID) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'SubjectID' in self._h:
                    return _read_string(self._h['SubjectID'])
            else:
                return None
        else:
            return self._SubjectID

    @SubjectID.setter
    def SubjectID(self, value):
        self._SubjectID = value

    @SubjectID.deleter
    def SubjectID(self):
        self._SubjectID = AbsentDataset()

    @property
    def MeasurementDate(self):
        if type(self._MeasurementDate) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'MeasurementDate' in self._h:
                    return _read_string(self._h['MeasurementDate'])
            else:
                return None
        else:
            return self._MeasurementDate

    @MeasurementDate.setter
    def MeasurementDate(self, value):
        self._MeasurementDate = value

    @MeasurementDate.deleter
    def MeasurementDate(self):
        self._MeasurementDate = AbsentDataset()

    @property
    def MeasurementTime(self):
        if type(self._MeasurementTime) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'MeasurementTime' in self._h:
                    return _read_string(self._h['MeasurementTime'])
            else:
                return None
        else:
            return self._MeasurementTime

    @MeasurementTime.setter
    def MeasurementTime(self, value):
        self._MeasurementTime = value

    @MeasurementTime.deleter
    def MeasurementTime(self):
        self._MeasurementTime = AbsentDataset()

    @property
    def LengthUnit(self):
        if type(self._LengthUnit) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'LengthUnit' in self._h:
                    return _read_string(self._h['LengthUnit'])
            else:
                return None
        else:
            return self._LengthUnit

    @LengthUnit.setter
    def LengthUnit(self, value):
        self._LengthUnit = value

    @LengthUnit.deleter
    def LengthUnit(self):
        self._LengthUnit = AbsentDataset()

    @property
    def TimeUnit(self):
        if type(self._TimeUnit) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'TimeUnit' in self._h:
                    return _read_string(self._h['TimeUnit'])
            else:
                return None
        else:
            return self._TimeUnit

    @TimeUnit.setter
    def TimeUnit(self, value):
        self._TimeUnit = value

    @TimeUnit.deleter
    def TimeUnit(self):
        self._TimeUnit = AbsentDataset()

    @property
    def FrequencyUnit(self):
        if type(self._FrequencyUnit) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'FrequencyUnit' in self._h:
                    return _read_string(self._h['FrequencyUnit'])
            else:
                return None
        else:
            return self._FrequencyUnit

    @FrequencyUnit.setter
    def FrequencyUnit(self, value):
        self._FrequencyUnit = value

    @FrequencyUnit.deleter
    def FrequencyUnit(self):
        self._FrequencyUnit = AbsentDataset()


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/SubjectID'
        data = self.SubjectID
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/MeasurementDate'
        data = self.MeasurementDate
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/MeasurementTime'
        data = self.MeasurementTime
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/LengthUnit'
        data = self.LengthUnit
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/TimeUnit'
        data = self.TimeUnit
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/FrequencyUnit'
        data = self.FrequencyUnit
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)



class Probe(Group):

    _wavelengths = AbsentDataset()  # [<f>,...]*
    _wavelengthsEmission = AbsentDataset()  # [<f>,...]
    _sourcePos2D = AbsentDataset()  # [[<f>,...]]*1
    _sourcePos3D = AbsentDataset()  # [[<f>,...]]*1
    _detectorPos2D = AbsentDataset()  # [[<f>,...]]*2
    _detectorPos3D = AbsentDataset()  # [[<f>,...]]*2
    _frequencies = AbsentDataset()  # [<f>,...]
    _timeDelays = AbsentDataset()  # [<f>,...]
    _timeDelayWidths = AbsentDataset()  # [<f>,...]
    _momentOrders = AbsentDataset()  # [<f>,...]
    _correlationTimeDelays = AbsentDataset()  # [<f>,...]
    _correlationTimeDelayWidths = AbsentDataset()  # [<f>,...]
    _sourceLabels = AbsentDataset()  # ["s",...]
    _detectorLabels = AbsentDataset()  # ["s",...]
    _landmarkPos2D = AbsentDataset()  # [[<f>,...]]
    _landmarkPos3D = AbsentDataset()  # [[<f>,...]]
    _landmarkLabels = AbsentDataset()  # ["s",...]
    _useLocalIndex = AbsentDataset()  # <i>


    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self.__snirfnames = ['wavelengths', 'wavelengthsEmission', 'sourcePos2D', 'sourcePos3D', 'detectorPos2D', 'detectorPos3D', 'frequencies', 'timeDelays', 'timeDelayWidths', 'momentOrders', 'correlationTimeDelays', 'correlationTimeDelayWidths', 'sourceLabels', 'detectorLabels', 'landmarkPos2D', 'landmarkPos3D', 'landmarkLabels', 'useLocalIndex', ]
        if 'wavelengths' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengths = _read_float_array(self._h['wavelengths'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"wavelengths"')
        if 'wavelengthsEmission' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthsEmission = _read_float_array(self._h['wavelengthsEmission'])
        if 'sourcePos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos2D = _read_float_array(self._h['sourcePos2D'])
        if 'sourcePos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos3D = _read_float_array(self._h['sourcePos3D'])
        if 'detectorPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos2D = _read_float_array(self._h['detectorPos2D'])
        if 'detectorPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos3D = _read_float_array(self._h['detectorPos3D'])
        if 'frequencies' in self._h:
            if not self._cfg.dynamic_loading:
                self._frequencies = _read_float_array(self._h['frequencies'])
        if 'timeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelays = _read_float_array(self._h['timeDelays'])
        if 'timeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelayWidths = _read_float_array(self._h['timeDelayWidths'])
        if 'momentOrders' in self._h:
            if not self._cfg.dynamic_loading:
                self._momentOrders = _read_float_array(self._h['momentOrders'])
        if 'correlationTimeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelays = _read_float_array(self._h['correlationTimeDelays'])
        if 'correlationTimeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelayWidths = _read_float_array(self._h['correlationTimeDelayWidths'])
        if 'sourceLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceLabels = _read_string_array(self._h['sourceLabels'])
        if 'detectorLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorLabels = _read_string_array(self._h['detectorLabels'])
        if 'landmarkPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos2D = _read_float_array(self._h['landmarkPos2D'])
        if 'landmarkPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos3D = _read_float_array(self._h['landmarkPos3D'])
        if 'landmarkLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkLabels = _read_string_array(self._h['landmarkLabels'])
        if 'useLocalIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._useLocalIndex = _read_int(self._h['useLocalIndex'])

    @property
    def wavelengths(self):
        if type(self._wavelengths) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'wavelengths' in self._h:
                    return np.array(self._h['wavelengths']).astype(float)  # Array
            else:
                return None
        else:
            return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    @wavelengths.deleter
    def wavelengths(self):
        self._wavelengths = AbsentDataset()

    @property
    def wavelengthsEmission(self):
        if type(self._wavelengthsEmission) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'wavelengthsEmission' in self._h:
                    return np.array(self._h['wavelengthsEmission']).astype(float)  # Array
            else:
                return None
        else:
            return self._wavelengthsEmission

    @wavelengthsEmission.setter
    def wavelengthsEmission(self, value):
        self._wavelengthsEmission = value

    @wavelengthsEmission.deleter
    def wavelengthsEmission(self):
        self._wavelengthsEmission = AbsentDataset()

    @property
    def sourcePos2D(self):
        if type(self._sourcePos2D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourcePos2D' in self._h:
                    return np.array(self._h['sourcePos2D']).astype(float)  # Array
            else:
                return None
        else:
            return self._sourcePos2D

    @sourcePos2D.setter
    def sourcePos2D(self, value):
        self._sourcePos2D = value

    @sourcePos2D.deleter
    def sourcePos2D(self):
        self._sourcePos2D = AbsentDataset()

    @property
    def sourcePos3D(self):
        if type(self._sourcePos3D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourcePos3D' in self._h:
                    return np.array(self._h['sourcePos3D']).astype(float)  # Array
            else:
                return None
        else:
            return self._sourcePos3D

    @sourcePos3D.setter
    def sourcePos3D(self, value):
        self._sourcePos3D = value

    @sourcePos3D.deleter
    def sourcePos3D(self):
        self._sourcePos3D = AbsentDataset()

    @property
    def detectorPos2D(self):
        if type(self._detectorPos2D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorPos2D' in self._h:
                    return np.array(self._h['detectorPos2D']).astype(float)  # Array
            else:
                return None
        else:
            return self._detectorPos2D

    @detectorPos2D.setter
    def detectorPos2D(self, value):
        self._detectorPos2D = value

    @detectorPos2D.deleter
    def detectorPos2D(self):
        self._detectorPos2D = AbsentDataset()

    @property
    def detectorPos3D(self):
        if type(self._detectorPos3D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorPos3D' in self._h:
                    return np.array(self._h['detectorPos3D']).astype(float)  # Array
            else:
                return None
        else:
            return self._detectorPos3D

    @detectorPos3D.setter
    def detectorPos3D(self, value):
        self._detectorPos3D = value

    @detectorPos3D.deleter
    def detectorPos3D(self):
        self._detectorPos3D = AbsentDataset()

    @property
    def frequencies(self):
        if type(self._frequencies) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'frequencies' in self._h:
                    return np.array(self._h['frequencies']).astype(float)  # Array
            else:
                return None
        else:
            return self._frequencies

    @frequencies.setter
    def frequencies(self, value):
        self._frequencies = value

    @frequencies.deleter
    def frequencies(self):
        self._frequencies = AbsentDataset()

    @property
    def timeDelays(self):
        if type(self._timeDelays) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'timeDelays' in self._h:
                    return np.array(self._h['timeDelays']).astype(float)  # Array
            else:
                return None
        else:
            return self._timeDelays

    @timeDelays.setter
    def timeDelays(self, value):
        self._timeDelays = value

    @timeDelays.deleter
    def timeDelays(self):
        self._timeDelays = AbsentDataset()

    @property
    def timeDelayWidths(self):
        if type(self._timeDelayWidths) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'timeDelayWidths' in self._h:
                    return np.array(self._h['timeDelayWidths']).astype(float)  # Array
            else:
                return None
        else:
            return self._timeDelayWidths

    @timeDelayWidths.setter
    def timeDelayWidths(self, value):
        self._timeDelayWidths = value

    @timeDelayWidths.deleter
    def timeDelayWidths(self):
        self._timeDelayWidths = AbsentDataset()

    @property
    def momentOrders(self):
        if type(self._momentOrders) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'momentOrders' in self._h:
                    return np.array(self._h['momentOrders']).astype(float)  # Array
            else:
                return None
        else:
            return self._momentOrders

    @momentOrders.setter
    def momentOrders(self, value):
        self._momentOrders = value

    @momentOrders.deleter
    def momentOrders(self):
        self._momentOrders = AbsentDataset()

    @property
    def correlationTimeDelays(self):
        if type(self._correlationTimeDelays) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'correlationTimeDelays' in self._h:
                    return np.array(self._h['correlationTimeDelays']).astype(float)  # Array
            else:
                return None
        else:
            return self._correlationTimeDelays

    @correlationTimeDelays.setter
    def correlationTimeDelays(self, value):
        self._correlationTimeDelays = value

    @correlationTimeDelays.deleter
    def correlationTimeDelays(self):
        self._correlationTimeDelays = AbsentDataset()

    @property
    def correlationTimeDelayWidths(self):
        if type(self._correlationTimeDelayWidths) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'correlationTimeDelayWidths' in self._h:
                    return np.array(self._h['correlationTimeDelayWidths']).astype(float)  # Array
            else:
                return None
        else:
            return self._correlationTimeDelayWidths

    @correlationTimeDelayWidths.setter
    def correlationTimeDelayWidths(self, value):
        self._correlationTimeDelayWidths = value

    @correlationTimeDelayWidths.deleter
    def correlationTimeDelayWidths(self):
        self._correlationTimeDelayWidths = AbsentDataset()

    @property
    def sourceLabels(self):
        if type(self._sourceLabels) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourceLabels' in self._h:
                    return np.array(self._h['sourceLabels']).astype(str)  # Array
            else:
                return None
        else:
            return self._sourceLabels

    @sourceLabels.setter
    def sourceLabels(self, value):
        self._sourceLabels = value

    @sourceLabels.deleter
    def sourceLabels(self):
        self._sourceLabels = AbsentDataset()

    @property
    def detectorLabels(self):
        if type(self._detectorLabels) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorLabels' in self._h:
                    return np.array(self._h['detectorLabels']).astype(str)  # Array
            else:
                return None
        else:
            return self._detectorLabels

    @detectorLabels.setter
    def detectorLabels(self, value):
        self._detectorLabels = value

    @detectorLabels.deleter
    def detectorLabels(self):
        self._detectorLabels = AbsentDataset()

    @property
    def landmarkPos2D(self):
        if type(self._landmarkPos2D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'landmarkPos2D' in self._h:
                    return np.array(self._h['landmarkPos2D']).astype(float)  # Array
            else:
                return None
        else:
            return self._landmarkPos2D

    @landmarkPos2D.setter
    def landmarkPos2D(self, value):
        self._landmarkPos2D = value

    @landmarkPos2D.deleter
    def landmarkPos2D(self):
        self._landmarkPos2D = AbsentDataset()

    @property
    def landmarkPos3D(self):
        if type(self._landmarkPos3D) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'landmarkPos3D' in self._h:
                    return np.array(self._h['landmarkPos3D']).astype(float)  # Array
            else:
                return None
        else:
            return self._landmarkPos3D

    @landmarkPos3D.setter
    def landmarkPos3D(self, value):
        self._landmarkPos3D = value

    @landmarkPos3D.deleter
    def landmarkPos3D(self):
        self._landmarkPos3D = AbsentDataset()

    @property
    def landmarkLabels(self):
        if type(self._landmarkLabels) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'landmarkLabels' in self._h:
                    return np.array(self._h['landmarkLabels']).astype(str)  # Array
            else:
                return None
        else:
            return self._landmarkLabels

    @landmarkLabels.setter
    def landmarkLabels(self, value):
        self._landmarkLabels = value

    @landmarkLabels.deleter
    def landmarkLabels(self):
        self._landmarkLabels = AbsentDataset()

    @property
    def useLocalIndex(self):
        if type(self._useLocalIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'useLocalIndex' in self._h:
                    return int(self._h['useLocalIndex'][()])
            else:
                return None
        else:
            return self._useLocalIndex

    @useLocalIndex.setter
    def useLocalIndex(self, value):
        self._useLocalIndex = value

    @useLocalIndex.deleter
    def useLocalIndex(self):
        self._useLocalIndex = AbsentDataset()


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/wavelengths'
        data = np.array(self.wavelengths)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/wavelengthsEmission'
        data = np.array(self.wavelengthsEmission)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/sourcePos2D'
        data = np.array(self.sourcePos2D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/sourcePos3D'
        data = np.array(self.sourcePos3D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/detectorPos2D'
        data = np.array(self.detectorPos2D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/detectorPos3D'
        data = np.array(self.detectorPos3D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/frequencies'
        data = np.array(self.frequencies)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/timeDelays'
        data = np.array(self.timeDelays)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/timeDelayWidths'
        data = np.array(self.timeDelayWidths)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/momentOrders'
        data = np.array(self.momentOrders)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/correlationTimeDelays'
        data = np.array(self.correlationTimeDelays)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/correlationTimeDelayWidths'
        data = np.array(self.correlationTimeDelayWidths)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/sourceLabels'
        data = np.array(self.sourceLabels).astype('O')
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/detectorLabels'
        data = np.array(self.detectorLabels).astype('O')
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/landmarkPos2D'
        data = np.array(self.landmarkPos2D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/landmarkPos3D'
        data = np.array(self.landmarkPos3D)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/landmarkLabels'
        data = np.array(self.landmarkLabels).astype('O')
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/useLocalIndex'
        data = self.useLocalIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)



class NirsElement(Group):

    _metaDataTags = AbsentDataset()  # {.}*
    _data = AbsentDataset()  # {i}*
    _stim = AbsentDataset()  # {i}
    _probe = AbsentDataset()  # {.}*
    _aux = AbsentDataset()  # {i}


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self.__snirfnames = ['metaDataTags', 'data', 'stim', 'probe', 'aux', ]
        if 'metaDataTags' in self._h:
            self._metaDataTags = MetaDataTags(self._h['metaDataTags'].id, self._cfg)  # Group
        else:
            self._metaDataTags = MetaDataTags(self.location + '/' + 'metaDataTags', self._cfg)  # Anonymous group (wrapper only)
        self.data = Data(self, self._cfg)  # Indexed group
        self.stim = Stim(self, self._cfg)  # Indexed group
        if 'probe' in self._h:
            self._probe = Probe(self._h['probe'].id, self._cfg)  # Group
        else:
            self._probe = Probe(self.location + '/' + 'probe', self._cfg)  # Anonymous group (wrapper only)
        self.aux = Aux(self, self._cfg)  # Indexed group

    @property
    def metaDataTags(self):
        return self._metaDataTags

    @metaDataTags.setter
    def metaDataTags(self, value):
        self._metaDataTags = value

    @metaDataTags.deleter
    def metaDataTags(self):
        self._metaDataTags = AbsentGroup()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @data.deleter
    def data(self):
        raise AttributeError('IndexedGroup ' + str(type(self._data)) + ' cannot be deleted')

    @property
    def stim(self):
        return self._stim

    @stim.setter
    def stim(self, value):
        self._stim = value

    @stim.deleter
    def stim(self):
        raise AttributeError('IndexedGroup ' + str(type(self._stim)) + ' cannot be deleted')

    @property
    def probe(self):
        return self._probe

    @probe.setter
    def probe(self, value):
        self._probe = value

    @probe.deleter
    def probe(self):
        self._probe = AbsentGroup()

    @property
    def aux(self):
        return self._aux

    @aux.setter
    def aux(self, value):
        self._aux = value

    @aux.deleter
    def aux(self):
        raise AttributeError('IndexedGroup ' + str(type(self._aux)) + ' cannot be deleted')


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        if type(self._metaDataTags) is AbsentGroup:
            if 'metaDataTags' in file:
                # print('Deleting group metaDataTags from', file)
                del file['metaDataTags']
        else:
            self.metaDataTags._save(*args)
        self.data._save(*args)
        self.stim._save(*args)
        if type(self._probe) is AbsentGroup:
            if 'probe' in file:
                # print('Deleting group probe from', file)
                del file['probe']
        else:
            self.probe._save(*args)
        self.aux._save(*args)


class Nirs(IndexedGroup):

    _name: str = 'nirs'
    _element: Group = NirsElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class DataElement(Group):

    _dataTimeSeries = AbsentDataset()  # [[<f>,...]]*
    _time = AbsentDataset()  # [<f>,...]*
    _measurementList = AbsentDataset()  # {i}*


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self.__snirfnames = ['dataTimeSeries', 'time', 'measurementList', ]
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(self._h['dataTimeSeries'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataTimeSeries"')
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"time"')
        self.measurementList = MeasurementList(self, self._cfg)  # Indexed group

    @property
    def dataTimeSeries(self):
        if type(self._dataTimeSeries) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataTimeSeries' in self._h:
                    return np.array(self._h['dataTimeSeries']).astype(float)  # Array
            else:
                return None
        else:
            return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = AbsentDataset()

    @property
    def time(self):
        if type(self._time) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'time' in self._h:
                    return np.array(self._h['time']).astype(float)  # Array
            else:
                return None
        else:
            return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @time.deleter
    def time(self):
        self._time = AbsentDataset()

    @property
    def measurementList(self):
        return self._measurementList

    @measurementList.setter
    def measurementList(self, value):
        self._measurementList = value

    @measurementList.deleter
    def measurementList(self):
        raise AttributeError('IndexedGroup ' + str(type(self._measurementList)) + ' cannot be deleted')


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/dataTimeSeries'
        data = np.array(self.dataTimeSeries)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/time'
        data = np.array(self.time)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        self.measurementList._save(*args)


class Data(IndexedGroup):

    _name: str = 'data'
    _element: Group = DataElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class MeasurementListElement(Group):

    _sourceIndex = AbsentDataset()  # <i>*
    _detectorIndex = AbsentDataset()  # <i>*
    _wavelengthIndex = AbsentDataset()  # <i>*
    _wavelengthActual = AbsentDataset()  # <f>
    _wavelengthEmissionActual = AbsentDataset()  # <f>
    _dataType = AbsentDataset()  # <i>*
    _dataTypeLabel = AbsentDataset()  # "s"
    _dataTypeIndex = AbsentDataset()  # <i>*
    _sourcePower = AbsentDataset()  # <f>
    _detectorGain = AbsentDataset()  # <f>
    _moduleIndex = AbsentDataset()  # <i>
    _sourceModuleIndex = AbsentDataset()  # <i>
    _detectorModuleIndex = AbsentDataset()  # <i>


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self.__snirfnames = ['sourceIndex', 'detectorIndex', 'wavelengthIndex', 'wavelengthActual', 'wavelengthEmissionActual', 'dataType', 'dataTypeLabel', 'dataTypeIndex', 'sourcePower', 'detectorGain', 'moduleIndex', 'sourceModuleIndex', 'detectorModuleIndex', ]
        if 'sourceIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceIndex = _read_int(self._h['sourceIndex'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"sourceIndex"')
        if 'detectorIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorIndex = _read_int(self._h['detectorIndex'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"detectorIndex"')
        if 'wavelengthIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthIndex = _read_int(self._h['wavelengthIndex'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"wavelengthIndex"')
        if 'wavelengthActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthActual = _read_float(self._h['wavelengthActual'])
        if 'wavelengthEmissionActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthEmissionActual = _read_float(self._h['wavelengthEmissionActual'])
        if 'dataType' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataType = _read_int(self._h['dataType'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataType"')
        if 'dataTypeLabel' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeLabel = _read_string(self._h['dataTypeLabel'])
        if 'dataTypeIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeIndex = _read_int(self._h['dataTypeIndex'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataTypeIndex"')
        if 'sourcePower' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePower = _read_float(self._h['sourcePower'])
        if 'detectorGain' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorGain = _read_float(self._h['detectorGain'])
        if 'moduleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._moduleIndex = _read_int(self._h['moduleIndex'])
        if 'sourceModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceModuleIndex = _read_int(self._h['sourceModuleIndex'])
        if 'detectorModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorModuleIndex = _read_int(self._h['detectorModuleIndex'])

    @property
    def sourceIndex(self):
        if type(self._sourceIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourceIndex' in self._h:
                    return int(self._h['sourceIndex'][()])
            else:
                return None
        else:
            return self._sourceIndex

    @sourceIndex.setter
    def sourceIndex(self, value):
        self._sourceIndex = value

    @sourceIndex.deleter
    def sourceIndex(self):
        self._sourceIndex = AbsentDataset()

    @property
    def detectorIndex(self):
        if type(self._detectorIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorIndex' in self._h:
                    return int(self._h['detectorIndex'][()])
            else:
                return None
        else:
            return self._detectorIndex

    @detectorIndex.setter
    def detectorIndex(self, value):
        self._detectorIndex = value

    @detectorIndex.deleter
    def detectorIndex(self):
        self._detectorIndex = AbsentDataset()

    @property
    def wavelengthIndex(self):
        if type(self._wavelengthIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'wavelengthIndex' in self._h:
                    return int(self._h['wavelengthIndex'][()])
            else:
                return None
        else:
            return self._wavelengthIndex

    @wavelengthIndex.setter
    def wavelengthIndex(self, value):
        self._wavelengthIndex = value

    @wavelengthIndex.deleter
    def wavelengthIndex(self):
        self._wavelengthIndex = AbsentDataset()

    @property
    def wavelengthActual(self):
        if type(self._wavelengthActual) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'wavelengthActual' in self._h:
                    return float(self._h['wavelengthActual'][()])
            else:
                return None
        else:
            return self._wavelengthActual

    @wavelengthActual.setter
    def wavelengthActual(self, value):
        self._wavelengthActual = value

    @wavelengthActual.deleter
    def wavelengthActual(self):
        self._wavelengthActual = AbsentDataset()

    @property
    def wavelengthEmissionActual(self):
        if type(self._wavelengthEmissionActual) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'wavelengthEmissionActual' in self._h:
                    return float(self._h['wavelengthEmissionActual'][()])
            else:
                return None
        else:
            return self._wavelengthEmissionActual

    @wavelengthEmissionActual.setter
    def wavelengthEmissionActual(self, value):
        self._wavelengthEmissionActual = value

    @wavelengthEmissionActual.deleter
    def wavelengthEmissionActual(self):
        self._wavelengthEmissionActual = AbsentDataset()

    @property
    def dataType(self):
        if type(self._dataType) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataType' in self._h:
                    return int(self._h['dataType'][()])
            else:
                return None
        else:
            return self._dataType

    @dataType.setter
    def dataType(self, value):
        self._dataType = value

    @dataType.deleter
    def dataType(self):
        self._dataType = AbsentDataset()

    @property
    def dataTypeLabel(self):
        if type(self._dataTypeLabel) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataTypeLabel' in self._h:
                    return _read_string(self._h['dataTypeLabel'])
            else:
                return None
        else:
            return self._dataTypeLabel

    @dataTypeLabel.setter
    def dataTypeLabel(self, value):
        self._dataTypeLabel = value

    @dataTypeLabel.deleter
    def dataTypeLabel(self):
        self._dataTypeLabel = AbsentDataset()

    @property
    def dataTypeIndex(self):
        if type(self._dataTypeIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataTypeIndex' in self._h:
                    return int(self._h['dataTypeIndex'][()])
            else:
                return None
        else:
            return self._dataTypeIndex

    @dataTypeIndex.setter
    def dataTypeIndex(self, value):
        self._dataTypeIndex = value

    @dataTypeIndex.deleter
    def dataTypeIndex(self):
        self._dataTypeIndex = AbsentDataset()

    @property
    def sourcePower(self):
        if type(self._sourcePower) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourcePower' in self._h:
                    return float(self._h['sourcePower'][()])
            else:
                return None
        else:
            return self._sourcePower

    @sourcePower.setter
    def sourcePower(self, value):
        self._sourcePower = value

    @sourcePower.deleter
    def sourcePower(self):
        self._sourcePower = AbsentDataset()

    @property
    def detectorGain(self):
        if type(self._detectorGain) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorGain' in self._h:
                    return float(self._h['detectorGain'][()])
            else:
                return None
        else:
            return self._detectorGain

    @detectorGain.setter
    def detectorGain(self, value):
        self._detectorGain = value

    @detectorGain.deleter
    def detectorGain(self):
        self._detectorGain = AbsentDataset()

    @property
    def moduleIndex(self):
        if type(self._moduleIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'moduleIndex' in self._h:
                    return int(self._h['moduleIndex'][()])
            else:
                return None
        else:
            return self._moduleIndex

    @moduleIndex.setter
    def moduleIndex(self, value):
        self._moduleIndex = value

    @moduleIndex.deleter
    def moduleIndex(self):
        self._moduleIndex = AbsentDataset()

    @property
    def sourceModuleIndex(self):
        if type(self._sourceModuleIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'sourceModuleIndex' in self._h:
                    return int(self._h['sourceModuleIndex'][()])
            else:
                return None
        else:
            return self._sourceModuleIndex

    @sourceModuleIndex.setter
    def sourceModuleIndex(self, value):
        self._sourceModuleIndex = value

    @sourceModuleIndex.deleter
    def sourceModuleIndex(self):
        self._sourceModuleIndex = AbsentDataset()

    @property
    def detectorModuleIndex(self):
        if type(self._detectorModuleIndex) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'detectorModuleIndex' in self._h:
                    return int(self._h['detectorModuleIndex'][()])
            else:
                return None
        else:
            return self._detectorModuleIndex

    @detectorModuleIndex.setter
    def detectorModuleIndex(self, value):
        self._detectorModuleIndex = value

    @detectorModuleIndex.deleter
    def detectorModuleIndex(self):
        self._detectorModuleIndex = AbsentDataset()


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/sourceIndex'
        data = self.sourceIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/detectorIndex'
        data = self.detectorIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/wavelengthIndex'
        data = self.wavelengthIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/wavelengthActual'
        data = self.wavelengthActual
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/wavelengthEmissionActual'
        data = self.wavelengthEmissionActual
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/dataType'
        data = self.dataType
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/dataTypeLabel'
        data = self.dataTypeLabel
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/dataTypeIndex'
        data = self.dataTypeIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/sourcePower'
        data = self.sourcePower
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/detectorGain'
        data = self.detectorGain
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/moduleIndex'
        data = self.moduleIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/sourceModuleIndex'
        data = self.sourceModuleIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)
        name = self.location + '/detectorModuleIndex'
        data = self.detectorModuleIndex
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'i4')
            file.create_dataset(name, dtype='i4', data=data)


class MeasurementList(IndexedGroup):

    _name: str = 'measurementList'
    _element: Group = MeasurementListElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class StimElement(Group):

    _name = AbsentDataset()  # "s"+
    _data = AbsentDataset()  # [<f>,...]+
    _dataLabels = AbsentDataset()  # ["s",...]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self.__snirfnames = ['name', 'data', 'dataLabels', ]
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
        if 'data' in self._h:
            if not self._cfg.dynamic_loading:
                self._data = _read_float_array(self._h['data'])
        if 'dataLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataLabels = _read_string_array(self._h['dataLabels'])

    @property
    def name(self):
        if type(self._name) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'name' in self._h:
                    return _read_string(self._h['name'])
            else:
                return None
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.deleter
    def name(self):
        self._name = AbsentDataset()

    @property
    def data(self):
        if type(self._data) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'data' in self._h:
                    return np.array(self._h['data']).astype(float)  # Array
            else:
                return None
        else:
            return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @data.deleter
    def data(self):
        self._data = AbsentDataset()

    @property
    def dataLabels(self):
        if type(self._dataLabels) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataLabels' in self._h:
                    return np.array(self._h['dataLabels']).astype(str)  # Array
            else:
                return None
        else:
            return self._dataLabels

    @dataLabels.setter
    def dataLabels(self, value):
        self._dataLabels = value

    @dataLabels.deleter
    def dataLabels(self):
        self._dataLabels = AbsentDataset()


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/name'
        data = self.name
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/data'
        data = np.array(self.data)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/dataLabels'
        data = np.array(self.dataLabels).astype('O')
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)


class Stim(IndexedGroup):

    _name: str = 'stim'
    _element: Group = StimElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class AuxElement(Group):

    _name = AbsentDataset()  # "s"+
    _dataTimeSeries = AbsentDataset()  # [[<f>,...]]+
    _time = AbsentDataset()  # [<f>,...]+
    _timeOffset = AbsentDataset()  # [<f>,...]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self.__snirfnames = ['name', 'dataTimeSeries', 'time', 'timeOffset', ]
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(self._h['dataTimeSeries'])
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
        if 'timeOffset' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeOffset = _read_float_array(self._h['timeOffset'])

    @property
    def name(self):
        if type(self._name) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'name' in self._h:
                    return _read_string(self._h['name'])
            else:
                return None
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.deleter
    def name(self):
        self._name = AbsentDataset()

    @property
    def dataTimeSeries(self):
        if type(self._dataTimeSeries) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'dataTimeSeries' in self._h:
                    return np.array(self._h['dataTimeSeries']).astype(float)  # Array
            else:
                return None
        else:
            return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = AbsentDataset()

    @property
    def time(self):
        if type(self._time) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'time' in self._h:
                    return np.array(self._h['time']).astype(float)  # Array
            else:
                return None
        else:
            return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @time.deleter
    def time(self):
        self._time = AbsentDataset()

    @property
    def timeOffset(self):
        if type(self._timeOffset) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'timeOffset' in self._h:
                    return np.array(self._h['timeOffset']).astype(float)  # Array
            else:
                return None
        else:
            return self._timeOffset

    @timeOffset.setter
    def timeOffset(self, value):
        self._timeOffset = value

    @timeOffset.deleter
    def timeOffset(self):
        self._timeOffset = AbsentDataset()


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/name'
        data = self.name
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        name = self.location + '/dataTimeSeries'
        data = np.array(self.dataTimeSeries)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/time'
        data = np.array(self.time)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)
        name = self.location + '/timeOffset'
        data = np.array(self.timeOffset)
        if data.size is 0 or data.any() is None:
            data = AbsentDataset()  # Do not save empty or "None" NumPy arrays
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', 'f8')
            file.create_dataset(name, dtype='f8', data=data)


class Aux(IndexedGroup):

    _name: str = 'aux'
    _element: Group = AuxElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class Snirf():
    
    _name = '/'
    _formatVersion = AbsentDataset()  # "s"*
    _nirs = AbsentDataset()  # {i}*


    def __init__(self, *args, dynamic_loading: bool = False, logfile: bool = False):
        self._cfg = SnirfConfig()
        self._cfg.dynamic_loading = dynamic_loading
        if len(args) > 0:
            path = args[0]
            if type(path) is str:
                if not path.endswith('.snirf'):
                    path = path + '.snirf'
                if logfile:
                    self._cfg.logger = _create_logger(path, path.split('.')[0] + '.log')
                else:
                    self._cfg.logger = _logger  # Use global logger
                if os.path.exists(path):
                    self._cfg.logger.info('Loading from file ' + path)
                    self._h = h5py.File(path, 'r+')
                else:
                    self._cfg.logger.info('Creating new file at ' + path)
                    self._h = h5py.File(path, 'w')
            else:
                raise TypeError(str(path) + ' is not a valid filename')
        else:
            self._cfg.logger = _logger
            self._cfg.logger.info('Created Snirf object based on tempfile')
            path = None
            self._h = h5py.File(TemporaryFile(), 'w')
        self.__snirfnames = ['formatVersion', 'nirs', ]
        if 'formatVersion' in self._h:
            if not self._cfg.dynamic_loading:
                self._formatVersion = _read_string(self._h['formatVersion'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"formatVersion"')
        self.nirs = Nirs(self, self._cfg)  # Indexed group

    @property
    def formatVersion(self):
        if type(self._formatVersion) is AbsentDataset:
            if self._cfg.dynamic_loading:
                if 'formatVersion' in self._h:
                    return _read_string(self._h['formatVersion'])
            else:
                return None
        else:
            return self._formatVersion

    @formatVersion.setter
    def formatVersion(self, value):
        self._formatVersion = value

    @formatVersion.deleter
    def formatVersion(self):
        self._formatVersion = AbsentDataset()

    @property
    def nirs(self):
        return self._nirs

    @nirs.setter
    def nirs(self, value):
        self._nirs = value

    @nirs.deleter
    def nirs(self):
        raise AttributeError('IndexedGroup ' + str(type(self._nirs)) + ' cannot be deleted')


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/formatVersion'
        data = self.formatVersion
        if name in file:
            del file[name]
        if type(data) is not AbsentDataset and data is not None:
            # # print('Saving', name, 'data is', type(data), data, 'being cast to', h5py.string_dtype(encoding='ascii', length=None))
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        self.nirs._save(*args)


    # overload
    def save(self, path: str):
        ...

    def save(self, *args):
        '''
        Save changes you have made to the Snirf object to disk. If a filepath is supplied, the changes will be
        'saved as' in a new file.
        '''
        if len(args) > 0 and type(args[0]) is str:
            path = args[0]
            if not path.endswith('.snirf'):
                path += '.snirf'
            new_file = h5py.File(path, 'w')
            self._save(new_file)
            new_file.close()
        else:
            self._save(self._h.file)

    @property
    def filename(self):
        return self._h.filename

    @property
    def location(self):
        return self._h.name

    def close(self):
        self._cfg.logger.handlers.clear()
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __del__(self):
        self.close()

    def __getitem__(self, key):
        if self._h != {}:
            if key in self._h:
                return self._h[key]
        else:
            return None

    def __contains__(self, key):
        return key in self._h;

    def __repr__(self):
        props = [p for p in dir(self) if ('_' not in p and not callable(getattr(self, p)))]
        out = str(self.__class__.__name__) + ' at /' + '\n'
        for prop in props:
            attr = getattr(self, prop)
            out += prop + ': '
            if type(attr) is np.ndarray or type(attr) is list:
                if np.size(attr) > 32:
                    out += '<' + str(np.shape(attr)) + ' array of ' + str(attr.dtype) + '>'
                else:
                    out += str(attr)
            else:
                prepr = str(attr)
                if len(prepr) < 64:
                    out += prepr
                else:
                    out += '\n' + prepr
            out += '\n'
        return out[:-1]

# Extend metaDataTags to support loading and saving of unspecified datasets

class MetaDataTags(MetaDataTags):
    
    def __init__(self, varg, cfg):
        super().__init__(varg, cfg)
        self.__other = []
        for key in self._h:
            if key not in self.__snirfnames:
                data = np.array(self._h[key])
                self.__other.append(key)
                setattr(self, key, data)
                
    def _save(self, *args):
        super()._save(*args)
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        for key in self.__other:
            print(self.__class__.__name__, 'writing misc dataset to', self.location + '/' + key, 'in', file, 'with value', getattr(self, key))
            if key in file:
                del file[key]
            file.create_dataset(self.location + '/' + key, data=getattr(self, key))
                