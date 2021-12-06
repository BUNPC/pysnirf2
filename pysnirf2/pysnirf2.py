from abc import ABC, abstractmethod
import h5py
import os
import sys
import numpy as np
from warnings import warn
from collections.abc import MutableSequence
from tempfile import TemporaryFile
import logging
import termcolor

# Colored prints for validator

printr = lambda x: termcolor.cprint(x, 'red')
printg = lambda x: termcolor.cprint(x, 'green')
printb = lambda x: termcolor.cprint(x, 'blue')
printm = lambda x: termcolor.cprint(x, 'magenta')

_loggers = {}
def _create_logger(name, log_file, level=logging.INFO):
    if name in _loggers.keys():
        return _loggers[name]
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    _loggers[name] = logger
    return logger


# Package-wide logger
_logger = _create_logger('pysnirf2', 'pysnirf2.log')

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')

# -- methods to cast data prior to writing to and after reading from h5py interfaces------

_varlen_str_type = h5py.string_dtype(encoding='ascii', length=None)  # Length=None creates HDF5 variable length string
_DTYPE_FLOAT32 = 'f4'
_DTYPE_FLOAT64 = 'f8'
_DTYPE_INT = 'i4'
_DTYPE_FIXED_LEN_STR = 'S'  # Not sure how robust this is, but fixed length strings will always at least contain S
_DTYPE_VAR_LEN_STR = 'O'  # Variable length string


# -- Dataset creators  ---------------------------------------


def _create_dataset(file, name, data):
    """
    Formats the input to satisfy the SNIRF specification as well as the HDF5 format. Determines appropriate write
    function from the type of data. The returned value should be passed to h5py.create_dataset() as kwarg "data" to be
    saved to an HDF5 file.

    Returns None if input is invalid and an h5py.Dataset instance if successful.
    """
    data = np.array(data)  # Cast to numpy type to identify
    try:
        if len(data) > 1:
            dtype = data[0].dtype
            if any([dtype is t for t in [int, np.int32, np.int64]]):  # int
                return _create_dataset_int_array(file, name, data)
            elif any([dtype is t for t in [float, np.float, np.float64]]):  # float
                return _create_dataset_float_array(file, name, data)
            elif any([dtype is t for t in [str, np.string_]]) or any([t in dtype.str for t in ['U', 'S']]):  # string
                return _create_dataset_string_array(file, name, data)
    except (TypeError, IndexError):  # data has no len() or a string was passed
        dtype = data.dtype
        if any([dtype is t for t in [int, np.int32, np.int64]]):  # int
            return _create_dataset_int(file, name, data)
        elif any([dtype is t for t in [float, np.float, np.float64]]):  # float
            return _create_dataset_float(file, name, data)
        elif any([dtype is t for t in [str, np.string_]]) or any([t in dtype.str for t in ['U', 'S']]):  # string
            return _create_dataset_string(file, name, data)
    raise TypeError('Unrecognized data type' + str(dtype)
                    + '. Please provide an int, float, or str, or an iterable of these.')


def _create_dataset_string(file: h5py.File, name: str, data: str):
    return file.create_dataset(name, dtype=_varlen_str_type, data=str(data))


def _create_dataset_int(file: h5py.File, name: str, data: int):
    return file.create_dataset(name, dtype=_DTYPE_INT, data=int(data))


def _create_dataset_float(file: h5py.File, name: str, data: float):
    return file.create_dataset(name, dtype=_DTYPE_FLOAT64, data=float(data))


def _create_dataset_string_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype('O')
    return file.create_dataset(name, dtype=_varlen_str_type, data=array)


def _create_dataset_int_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype(int)
    return file.create_dataset(name, dtype=_DTYPE_INT, data=array)


def _create_dataset_float_array(file: h5py.File, name: str, data: np.ndarray):
    array = np.array(data).astype(float)
    return file.create_dataset(name, dtype=_DTYPE_FLOAT64, data=array)


# -- Dataset readers  ---------------------------------------


def _read_dataset(h_dataset: h5py.Dataset):
    """
    Reads data from a SNIRF file using the HDF5 interface and casts it to a Pythonic type or numpy array. Determines
    the appropriate read function from the properties of h_dataset
    """
    if type(h_dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if h_dataset.size > 1:
        if _DTYPE_FIXED_LEN_STR in h_dataset.dtype or _DTYPE_VAR_LEN_STR in h_dataset.dtype.str:
            return _read_string_array(h_dataset)
        elif _DTYPE_INT in h_dataset.dtype.str:
            return _read_int_array(h_dataset)
        elif _DTYPE_FLOAT32 in h_dataset.dtype.str or _DTYPE_FLOAT64 in h_dataset.dtype.str:
            return _read_float_array(h_dataset)
    else:
        if _DTYPE_FIXED_LEN_STR in h_dataset.dtype.str or _DTYPE_VAR_LEN_STR in h_dataset.dtype.str:
            return _read_string(h_dataset)
        elif _DTYPE_INT in h_dataset.dtype.str:
            return _read_int(h_dataset)
        elif _DTYPE_FLOAT32 in h_dataset.dtype.str or _DTYPE_FLOAT64 in h_dataset.dtype.str:
            return _read_float(h_dataset)
    raise TypeError("Dataset dtype='" + str(h_dataset.dtype)
                    + "' not recognized. Expecting dtype to contain one of these: "
                    + str([_DTYPE_FIXED_LEN_STR, _DTYPE_VAR_LEN_STR, _DTYPE_INT, _DTYPE_FLOAT32, _DTYPE_FLOAT64]))


def _read_string(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    # Because many SNIRF files are saved with string values in length 1 arrays
    if dataset.ndim > 0:
        return str(dataset[0].decode('ascii'))
    else:
        return str(dataset[()].decode('ascii'))


def _read_int(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return int(dataset[0])
    else:
        return int(dataset[()])


def _read_float(dataset: h5py.Dataset):
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


# -- Validation types ---------------------------------------


class ValidationResult:
    
    _locations = {}  # HDF locations associated with errors or an additional validation result

    _SEVERITY_LEVELS = {
                        0: 'OK     ',
                        1: 'INFO   ',
                        2: 'WARNING',
                        3: 'FATAL  ',
                        }
    _CODES = {
            # Errors (Severity 1)
            'INVALID_FILE_NAME': (0 << 1, 3, 'Valid SNIRF files must end with .snirf'),
            'INVALID_FILE': (0 << 1, 3, 'The file could not be opened'),
            'REQUIRED_DATASET_MISSING': (0 << 1, 3, 'A required dataset is missing from the file'),
            'REQUIRED_GROUP_MISSING': (0 << 1, 3, 'A required Group is missing from the file'),
            'INVALID_DATASET_TYPE': (0 << 1, 3, 'An HDF5 Dataset is not stored in the specified format'),
            'INVALID_DATASET_SHAPE': (0 << 1, 3, 'An HDF5 Dataset is not stored in the specified shape. Strings and scalars should never be stored as arrays of length 1.'),
            'INVALID_MEASUREMENTLIST': (0 << 1, 3, 'The number of measurementList elements does not match the second dimension of dataTimeSeries'),
            'INVALID_TIME': (0 << 1, 3, 'The length of the data/time vector does not match the first dimension of data/dataTimeSeries'),
            'INVALID_INDEX': (0 << 1, 3, 'An index is negative'),
            'INVALID_SOURCE_INDEX': (0 << 1, 3, 'measurementList/sourceIndex exceeds probe/sourceLabels'),
            'INVALID_DETECTOR_INDEX': (0 << 1, 3, 'measurementList/detectorIndex exceeds probe/detectorLabels'),
            'INVALID_PROBE_LABEL': (0 << 1, 3, 'a duplicate sourceLabel or detectorLabel appears'),
            'INVALID_WAVELENGTH_INDEX': (0 << 1, 3, 'measurementList/waveLengthIndex exceeds probe/wavelengths, probe/wavelengthsEmission, or probe/sourceLabels'),
            'INVALID_DATATYPE_INDEX': (0 << 1, 3, 'measurementList/dataTypeIndex exceeds probe/frequencies, probe/timeDelays, probe/timeDelayWidths, probe/momentOrders, probe/correlationTimeDelayWidths or probe/correlationTimeDelays'),
            'INVALID_PROBE_MODULE_INDEX': (0 << 1, 3, 'sourceModuleIndex and detectorModuleIndex are used along with moduleIndex'),
    #            'INVALID_LANDMARKPOS': (0 << 1, 3, 'A value in the last column of landmarkPos2D or landmarkPos3D exceeds the length of'),
            'INVALID_STIM_DATALABELS': (0 << 1, 3, 'The length of stim/dataLabels exceeds the columns of stim/data'),
            # Warnings (Severity 2)
            'UNRECOGNIZED_GROUP': (0 << 1, 2, 'An unspecified Group is a part of the file'),
            'UNRECOGNIZED_DATASET': (0 << 1, 2, 'An unspecified Dataset is a part of the file in an unexpected place'),
            'UNRECOGNIZED_DATATYPELABEL': (0 << 1, 2, 'measurementList/dataTypeLabel is not one of the recognized values listed in the Appendix'),
            'UNRECOGNIZED_DATATYPE': (0 << 1, 2, 'measurementList/dataType is not one of the recognized values listed in the Appendix'),
            'INDEX_OF_ZERO': (0 << 1, 2, 'An index of zero is usually undefined'),
            'FIXED_LENGTH_STRING': (0 << 1, 2, 'The use of fixed-length strings is discouraged and may be banned by a future spec version. Rewrite this file with pysnirf2 to use variable length strings'),
            # Info (Severity 1)
            'OPTIONAL_GROUP_MISSING': (0 << 1, 1, 'OK (missing optional Group)'),
            'OPTIONAL_DATASET_MISSING': (0 << 1, 1, 'OK (missing optional Dataset)'),
            # OK (Severity 0)
            'OK': (0 << 1, 0, 'OK'),
            }

    def is_valid(self):
        for value in self._locations.values():
            if value[1][1] > 2:  # Severity
                return False
        return True

    def _add(self, location, key):
        if key not in self._CODES.keys():
            raise KeyError("Invalid code '" + key + "'")
        # Locations is nested tuple (code, (id, level, msg))
        self._locations[location] = (key, self._CODES[key])
        print('\n', location, key, self._CODES[key][0])
        
    def display(self, severity=1):
        longest_key = max([len(key) for key in self._locations.keys()])
        longest_code = max([len(key[0]) for key in self._locations.keys()])
        s = object.__repr__(self) + '\n'
        printed = [0, 0, 0, 0]
        for key in self._locations.keys():
            sev = self._locations[key][1][1]
            printed[sev] += 1
            if sev >= severity:
                s += key.ljust(longest_key) + ' ' + self._SEVERITY_LEVELS[sev] + ' ' + self._locations[key][0].ljust(longest_code) + '\n'
        print(s)
        for i in range(0, severity):
            [printg, printb, printm, printr][i]('Found ' + str(printed[i]) + ' ' + self._SEVERITY_LEVELS[i] + ' (hidden)')            
        for i in range(severity, 4):
            [printg, printb, printm, printr][i]('Found ' + str(printed[i]) + ' ' + self._SEVERITY_LEVELS[i])
        i = int(self.is_valid())
        [printr, printg][i]('\nFile is ' +['INVALID', 'VALID'][i])
        
        
    def __repr__(self):
        return object.__repr__(self) + ' is_valid ' + str(self.is_valid()) 
        
    
# -- Validation functions ---------------------------------------


def _validate_string(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1 or dataset.ndim > 0:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_VAR_LEN_STR in dataset.dtype.str:
        return 'OK'
    elif _DTYPE_FIXED_LEN_STR in dataset.dtype.str:
        return 'FIXED_LENGTH_STRING'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_int(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1 or dataset.ndim > 0:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_INT in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_float(dataset: h5py.Dataset):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1 or dataset.ndim > 0:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_string_array(dataset: h5py.Dataset, ndims=[1]):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim not in ndims:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_VAR_LEN_STR in dataset.dtype.str:
        return 'OK'
    elif _DTYPE_FIXED_LEN_STR in dataset.dtype.str:
        return 'FIXED_LENGTH_STRING'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_int_array(dataset: h5py.Dataset, ndims=[1]):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim not in ndims:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_INT in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'

def _validate_float_array(dataset: h5py.Dataset, ndims=[1]):
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim not in ndims:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'


# -----------------------------------------


class SnirfFormatError(Exception):
    pass


class SnirfConfig:
    logger: logging.Logger = _logger
    dynamic_loading: bool = False  # If False, data is loaded in the constructor, if True, data is loaded on access


#class Singleton(object):
#    __instance = None
#    def __new__(cls, val):
#        if Singleton.__instance is None:
#            Singleton.__instance = object.__new__(cls)
#        Singleton.__instance.val = val
#        return Singleton.__instance

# Placeholder for a Dataset that is not on disk or in memory
class AbsentDatasetType():
    pass


# Placeholder for a Group that is not on disk or in memory
class AbsentGroupType():
    pass


# Placeholder for a Dataset that is available only on disk in a dynamic_loading=True wrapper
class PresentDatasetType():
    pass


# Instantiate singletons
AbsentDataset = AbsentDatasetType()
AbsentGroup = AbsentGroupType()
PresentDataset = PresentDatasetType()


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
                self._cfg.logger.info('Group-level save of %s in %s', self.location, self.filename)
                self._save(args[0])
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError("No such SNIRF file '" + path + "'. Create a SNIRF file before attempting to save a Group to it.")
                self._cfg.logger.info('Group-level save of %s in %s to new file %s', self.location, self.filename, file)
                self._save(file)
                file.close()
        else:
            if self._h != {}:
                file = self._h.file
            self._cfg.logger.info('Group-level save of %s in %s', self.location, file)
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

    def is_empty(self):
        for name in self._snirf_names:
            attr = getattr(self, '_' + name)
            if isinstance(attr, Group) or isinstance(attr, IndexedGroup):
                if not attr.is_empty():
                    return False
            else:
                if not any([attr is a for a in [None, AbsentGroup, AbsentDataset]]):
                    return False
        return True

    @abstractmethod
    def _save(self, *args):
        """
        args is path or empty
        """
        raise NotImplementedError('_save is an abstract method')

    # @abstractmethod
    # def _validate(self, result: ValidationResult):
    #     raise NotImplementedError('_validate is an abstract method')

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

    # def __getitem__(self, key):
    #     if self._h != {}:
    #         if key in self._h:
    #             return self._h[key]
    #     else:
    #         return None

    def __contains__(self, key):
        return key in self._h


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
        self._cfg.logger.info('IndexedGroup %s at %s in %s initalized with %i instances of %s', self.__class__.__name__,
                              self._parent.location, self.filename, len(self._list), self._element)

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

    def __contains__(self, key):
        return any([key == e.location.split('/')[-1] for e in self._list])

    def is_empty(self):
        if len(self._list) > 0:
            for e in self._list:
                if not e.is_empty():
                    return False
        return True

    def insert(self, i, item):
        self._check_type(item)
        self._list.insert(i, item)
        self._cfg.logger.info('%i th element inserted into IndexedGroup %s at %s in %s at %i', len(self._list),
                              self.__class__.__name__, self._parent.location, self.filename, i)

    def append(self, item):
        self._check_type(item)
        self._list.append(item)
        self._cfg.logger.info('%i th element appended to IndexedGroup %s at %s in %s', len(self._list),
                              self.__class__.__name__, self._parent.location, self.filename)

    def save(self, *args):
        if len(args) > 0:
            if type(args[0]) is h5py.File:
                self._save(args[0])
                self._cfg.logger.info('IndexedGroup-level save of %s at %s in %s', self.__class__.__name__,
                      self._parent.location, self.filename)
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError("No such SNIRF file '" + path + "'. Create a SNIRF file before attempting to save an IndexedGroup to it.")
                self._cfg.logger.info('IndexedGroup-level save of %s at %s in %s to %s', self.__class__.__name__,
                                      self._parent.location, self.filename, file)
                self._save(file)
                file.close()
        else:
            if self._parent._h != {}:
                file = self._parent._h.file
            self._save(file)
            self._cfg.logger.info('IndexedGroup-level save of %s at %s in %s', self.__class__.__name__,
                      self._parent.location, self.filename)

    def appendGroup(self):
        'Adds a group to the end of the list'
        location = self._parent.location + '/' + self._name + str(len(self._list) + 1)
        self._list.append(self._element(location, self._cfg))
        self._cfg.logger.info('%i th %s appended to IndexedGroup %s at %s in %s', len(self._list),
                          self._element, self.__class__.__name__, self._parent.location, self.filename)

    def _populate_list(self):
        """
        Add all the appropriate groups in parent to the list
        """
        self._list = list()
        names = self._get_matching_keys()
        for name in names:
            if name in self._parent._h:
                self._list.append(self._element(self._parent._h[name].id, self._cfg))

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
        if all([len(e.location.split('/' + self._name)[-1]) > 0 for e in self._list]):
            if not [int(e.location.split('/' + self._name)[-1]) for e in self._list] == list(range(1, len(self._list) + 1)):
                self._cfg.logger.info('renaming elements of IndexedGroup ' + self.__class__.__name__ + ' at '
                                      + self._parent.location + ' in ' + self.filename + ' to agree with naming format')
                # if list is not already ordered propertly
                for i, e in enumerate(self._list):
                    # To avoid assignment to an existing name, move all
                    h.move(e.location,
                           '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp')
                    self._cfg.logger.info(e.location, '--->',
                                          '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp')
                for i, e in enumerate(self._list):
                    h.move('/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp',
                           '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1))
                    self._cfg.logger.info('/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1) + '_tmp',
                                          '--->', '/'.join(e.location.split('/')[:-1]) + '/' + self._name + str(i + 1))

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

    def _validate(self, result: ValidationResult):
        [e._validate(result) for e in self._list]

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
        # Remove groups which remain on disk after being removed from the wrapper
        for name in names_in_file:
            if name not in names_to_save:
                del h[self._parent.name + '/' + name]  # Remove the actual data from the hdf5 file.
        for e in self._list:
            e._save(*args)  # Group save functions handle the write to disk
        self._order_names(h=h)  # Enforce order in the group names


# generated by sstucker on 2021-12-06
# version v1.0.1-development SNIRF specification parsed from https://raw.githubusercontent.com/fNIRS/snirf/master/snirf_specification.md


class MetaDataTags(Group):

    _SubjectID = AbsentDataset  # "s"*
    _MeasurementDate = AbsentDataset  # "s"*
    _MeasurementTime = AbsentDataset  # "s"*
    _LengthUnit = AbsentDataset  # "s"*
    _TimeUnit = AbsentDataset  # "s"*
    _FrequencyUnit = AbsentDataset  # "s"*
    _snirf_names = ['SubjectID', 'MeasurementDate', 'MeasurementTime', 'LengthUnit', 'TimeUnit', 'FrequencyUnit', ]


    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self._indexed_groups = []
        if 'SubjectID' in self._h:
            if not self._cfg.dynamic_loading:
                self._SubjectID = _read_string(self._h['SubjectID'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._SubjectID = PresentDataset
        else:  # if the dataset is not found on disk
            self._SubjectID = AbsentDataset
        if 'MeasurementDate' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementDate = _read_string(self._h['MeasurementDate'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._MeasurementDate = PresentDataset
        else:  # if the dataset is not found on disk
            self._MeasurementDate = AbsentDataset
        if 'MeasurementTime' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementTime = _read_string(self._h['MeasurementTime'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._MeasurementTime = PresentDataset
        else:  # if the dataset is not found on disk
            self._MeasurementTime = AbsentDataset
        if 'LengthUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._LengthUnit = _read_string(self._h['LengthUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._LengthUnit = PresentDataset
        else:  # if the dataset is not found on disk
            self._LengthUnit = AbsentDataset
        if 'TimeUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._TimeUnit = _read_string(self._h['TimeUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._TimeUnit = PresentDataset
        else:  # if the dataset is not found on disk
            self._TimeUnit = AbsentDataset
        if 'FrequencyUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._FrequencyUnit = _read_string(self._h['FrequencyUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._FrequencyUnit = PresentDataset
        else:  # if the dataset is not found on disk
            self._FrequencyUnit = AbsentDataset
        self._unspecified_names = []
        # Unspecified datasets are not properties and unaffected by dynamic_loading
        for key in self._h.keys():
            # If the name isn't specified 
            if key not in self._snirf_names and not any([key in indexed_group for indexed_group in self._indexed_groups]):
                setattr(self, key, _read_dataset(self._h[key]))
                self._unspecified_names.append(key)
                

    @property
    def SubjectID(self):
        if type(self._SubjectID) is type(AbsentDataset):
            return None
        if type(self._SubjectID) is type(PresentDataset) and 'SubjectID' in self._h:
            return _read_string(self._h['SubjectID'])
            self._cfg.logger.info('Dynamically loaded %s/SubjectID from %s', self.location, self.filename)
        else:
            return self._SubjectID

    @SubjectID.setter
    def SubjectID(self, value):
        self._SubjectID = value
        # self._cfg.logger.info('Assignment to %s/SubjectID in %s', self.location, self.filename)

    @SubjectID.deleter
    def SubjectID(self):
        self._SubjectID = AbsentDataset
        self._cfg.logger.info('Deleted %s/SubjectID from %s', self.location, self.filename)

    @property
    def MeasurementDate(self):
        if type(self._MeasurementDate) is type(AbsentDataset):
            return None
        if type(self._MeasurementDate) is type(PresentDataset) and 'MeasurementDate' in self._h:
            return _read_string(self._h['MeasurementDate'])
            self._cfg.logger.info('Dynamically loaded %s/MeasurementDate from %s', self.location, self.filename)
        else:
            return self._MeasurementDate

    @MeasurementDate.setter
    def MeasurementDate(self, value):
        self._MeasurementDate = value
        # self._cfg.logger.info('Assignment to %s/MeasurementDate in %s', self.location, self.filename)

    @MeasurementDate.deleter
    def MeasurementDate(self):
        self._MeasurementDate = AbsentDataset
        self._cfg.logger.info('Deleted %s/MeasurementDate from %s', self.location, self.filename)

    @property
    def MeasurementTime(self):
        if type(self._MeasurementTime) is type(AbsentDataset):
            return None
        if type(self._MeasurementTime) is type(PresentDataset) and 'MeasurementTime' in self._h:
            return _read_string(self._h['MeasurementTime'])
            self._cfg.logger.info('Dynamically loaded %s/MeasurementTime from %s', self.location, self.filename)
        else:
            return self._MeasurementTime

    @MeasurementTime.setter
    def MeasurementTime(self, value):
        self._MeasurementTime = value
        # self._cfg.logger.info('Assignment to %s/MeasurementTime in %s', self.location, self.filename)

    @MeasurementTime.deleter
    def MeasurementTime(self):
        self._MeasurementTime = AbsentDataset
        self._cfg.logger.info('Deleted %s/MeasurementTime from %s', self.location, self.filename)

    @property
    def LengthUnit(self):
        if type(self._LengthUnit) is type(AbsentDataset):
            return None
        if type(self._LengthUnit) is type(PresentDataset) and 'LengthUnit' in self._h:
            return _read_string(self._h['LengthUnit'])
            self._cfg.logger.info('Dynamically loaded %s/LengthUnit from %s', self.location, self.filename)
        else:
            return self._LengthUnit

    @LengthUnit.setter
    def LengthUnit(self, value):
        self._LengthUnit = value
        # self._cfg.logger.info('Assignment to %s/LengthUnit in %s', self.location, self.filename)

    @LengthUnit.deleter
    def LengthUnit(self):
        self._LengthUnit = AbsentDataset
        self._cfg.logger.info('Deleted %s/LengthUnit from %s', self.location, self.filename)

    @property
    def TimeUnit(self):
        if type(self._TimeUnit) is type(AbsentDataset):
            return None
        if type(self._TimeUnit) is type(PresentDataset) and 'TimeUnit' in self._h:
            return _read_string(self._h['TimeUnit'])
            self._cfg.logger.info('Dynamically loaded %s/TimeUnit from %s', self.location, self.filename)
        else:
            return self._TimeUnit

    @TimeUnit.setter
    def TimeUnit(self, value):
        self._TimeUnit = value
        # self._cfg.logger.info('Assignment to %s/TimeUnit in %s', self.location, self.filename)

    @TimeUnit.deleter
    def TimeUnit(self):
        self._TimeUnit = AbsentDataset
        self._cfg.logger.info('Deleted %s/TimeUnit from %s', self.location, self.filename)

    @property
    def FrequencyUnit(self):
        if type(self._FrequencyUnit) is type(AbsentDataset):
            return None
        if type(self._FrequencyUnit) is type(PresentDataset) and 'FrequencyUnit' in self._h:
            return _read_string(self._h['FrequencyUnit'])
            self._cfg.logger.info('Dynamically loaded %s/FrequencyUnit from %s', self.location, self.filename)
        else:
            return self._FrequencyUnit

    @FrequencyUnit.setter
    def FrequencyUnit(self, value):
        self._FrequencyUnit = value
        # self._cfg.logger.info('Assignment to %s/FrequencyUnit in %s', self.location, self.filename)

    @FrequencyUnit.deleter
    def FrequencyUnit(self):
        self._FrequencyUnit = AbsentDataset
        self._cfg.logger.info('Deleted %s/FrequencyUnit from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/SubjectID'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.SubjectID
        if type(self._SubjectID) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/MeasurementDate'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.MeasurementDate
        if type(self._MeasurementDate) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/MeasurementTime'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.MeasurementTime
        if type(self._MeasurementTime) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/LengthUnit'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.LengthUnit
        if type(self._LengthUnit) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/TimeUnit'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.TimeUnit
        if type(self._TimeUnit) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/FrequencyUnit'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.FrequencyUnit
        if type(self._FrequencyUnit) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        for unspecified_name in self._unspecified_names:
            name = self.location + '/' + unspecified_name
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
            try:
                data = getattr(self, unspecified_name)
            except AttributeError:  # Dataset was deleted
                print('Attribute was deleted or could not be found... not saving')
                continue
            _create_dataset(file, name, data)
            

    def _validate(self, result: ValidationResult):
        name = self.location + '/SubjectID'
        if type(self._SubjectID) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._SubjectID) is type(PresentDataset):
                    dataset = self._h['SubjectID']
                else:
                    dataset = _create_dataset_string(tmp, 'SubjectID', self._SubjectID)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/MeasurementDate'
        if type(self._MeasurementDate) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._MeasurementDate) is type(PresentDataset):
                    dataset = self._h['MeasurementDate']
                else:
                    dataset = _create_dataset_string(tmp, 'MeasurementDate', self._MeasurementDate)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/MeasurementTime'
        if type(self._MeasurementTime) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._MeasurementTime) is type(PresentDataset):
                    dataset = self._h['MeasurementTime']
                else:
                    dataset = _create_dataset_string(tmp, 'MeasurementTime', self._MeasurementTime)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/LengthUnit'
        if type(self._LengthUnit) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._LengthUnit) is type(PresentDataset):
                    dataset = self._h['LengthUnit']
                else:
                    dataset = _create_dataset_string(tmp, 'LengthUnit', self._LengthUnit)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/TimeUnit'
        if type(self._TimeUnit) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._TimeUnit) is type(PresentDataset):
                    dataset = self._h['TimeUnit']
                else:
                    dataset = _create_dataset_string(tmp, 'TimeUnit', self._TimeUnit)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/FrequencyUnit'
        if type(self._FrequencyUnit) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._FrequencyUnit) is type(PresentDataset):
                    dataset = self._h['FrequencyUnit']
                else:
                    dataset = _create_dataset_string(tmp, 'FrequencyUnit', self._FrequencyUnit)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)



class Probe(Group):

    _wavelengths = AbsentDataset  # [<f>,...]*
    _wavelengthsEmission = AbsentDataset  # [<f>,...]
    _sourcePos2D = AbsentDataset  # [[<f>,...]]*1
    _sourcePos3D = AbsentDataset  # [[<f>,...]]*1
    _detectorPos2D = AbsentDataset  # [[<f>,...]]*2
    _detectorPos3D = AbsentDataset  # [[<f>,...]]*2
    _frequencies = AbsentDataset  # [<f>,...]
    _timeDelays = AbsentDataset  # [<f>,...]
    _timeDelayWidths = AbsentDataset  # [<f>,...]
    _momentOrders = AbsentDataset  # [<f>,...]
    _correlationTimeDelays = AbsentDataset  # [<f>,...]
    _correlationTimeDelayWidths = AbsentDataset  # [<f>,...]
    _sourceLabels = AbsentDataset  # ["s",...]
    _detectorLabels = AbsentDataset  # ["s",...]
    _landmarkPos2D = AbsentDataset  # [[<f>,...]]
    _landmarkPos3D = AbsentDataset  # [[<f>,...]]
    _landmarkLabels = AbsentDataset  # ["s",...]
    _useLocalIndex = AbsentDataset  # <i>
    _snirf_names = ['wavelengths', 'wavelengthsEmission', 'sourcePos2D', 'sourcePos3D', 'detectorPos2D', 'detectorPos3D', 'frequencies', 'timeDelays', 'timeDelayWidths', 'momentOrders', 'correlationTimeDelays', 'correlationTimeDelayWidths', 'sourceLabels', 'detectorLabels', 'landmarkPos2D', 'landmarkPos3D', 'landmarkLabels', 'useLocalIndex', ]


    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self._indexed_groups = []
        if 'wavelengths' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengths = _read_float_array(self._h['wavelengths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengths = PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengths = AbsentDataset
        if 'wavelengthsEmission' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthsEmission = _read_float_array(self._h['wavelengthsEmission'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthsEmission = PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthsEmission = AbsentDataset
        if 'sourcePos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos2D = _read_float_array(self._h['sourcePos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePos2D = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePos2D = AbsentDataset
        if 'sourcePos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos3D = _read_float_array(self._h['sourcePos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePos3D = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePos3D = AbsentDataset
        if 'detectorPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos2D = _read_float_array(self._h['detectorPos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorPos2D = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorPos2D = AbsentDataset
        if 'detectorPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos3D = _read_float_array(self._h['detectorPos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorPos3D = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorPos3D = AbsentDataset
        if 'frequencies' in self._h:
            if not self._cfg.dynamic_loading:
                self._frequencies = _read_float_array(self._h['frequencies'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._frequencies = PresentDataset
        else:  # if the dataset is not found on disk
            self._frequencies = AbsentDataset
        if 'timeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelays = _read_float_array(self._h['timeDelays'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeDelays = PresentDataset
        else:  # if the dataset is not found on disk
            self._timeDelays = AbsentDataset
        if 'timeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelayWidths = _read_float_array(self._h['timeDelayWidths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeDelayWidths = PresentDataset
        else:  # if the dataset is not found on disk
            self._timeDelayWidths = AbsentDataset
        if 'momentOrders' in self._h:
            if not self._cfg.dynamic_loading:
                self._momentOrders = _read_float_array(self._h['momentOrders'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._momentOrders = PresentDataset
        else:  # if the dataset is not found on disk
            self._momentOrders = AbsentDataset
        if 'correlationTimeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelays = _read_float_array(self._h['correlationTimeDelays'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._correlationTimeDelays = PresentDataset
        else:  # if the dataset is not found on disk
            self._correlationTimeDelays = AbsentDataset
        if 'correlationTimeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelayWidths = _read_float_array(self._h['correlationTimeDelayWidths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._correlationTimeDelayWidths = PresentDataset
        else:  # if the dataset is not found on disk
            self._correlationTimeDelayWidths = AbsentDataset
        if 'sourceLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceLabels = _read_string_array(self._h['sourceLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceLabels = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceLabels = AbsentDataset
        if 'detectorLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorLabels = _read_string_array(self._h['detectorLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorLabels = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorLabels = AbsentDataset
        if 'landmarkPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos2D = _read_float_array(self._h['landmarkPos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkPos2D = PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkPos2D = AbsentDataset
        if 'landmarkPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos3D = _read_float_array(self._h['landmarkPos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkPos3D = PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkPos3D = AbsentDataset
        if 'landmarkLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkLabels = _read_string_array(self._h['landmarkLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkLabels = PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkLabels = AbsentDataset
        if 'useLocalIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._useLocalIndex = _read_int(self._h['useLocalIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._useLocalIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._useLocalIndex = AbsentDataset
                

    @property
    def wavelengths(self):
        if type(self._wavelengths) is type(AbsentDataset):
            return None
        if type(self._wavelengths) is type(PresentDataset) and 'wavelengths' in self._h:
            return _read_float_array(self._h['wavelengths'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengths from %s', self.location, self.filename)
        else:
            return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        # self._cfg.logger.info('Assignment to %s/wavelengths in %s', self.location, self.filename)

    @wavelengths.deleter
    def wavelengths(self):
        self._wavelengths = AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengths from %s', self.location, self.filename)

    @property
    def wavelengthsEmission(self):
        if type(self._wavelengthsEmission) is type(AbsentDataset):
            return None
        if type(self._wavelengthsEmission) is type(PresentDataset) and 'wavelengthsEmission' in self._h:
            return _read_float_array(self._h['wavelengthsEmission'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengthsEmission from %s', self.location, self.filename)
        else:
            return self._wavelengthsEmission

    @wavelengthsEmission.setter
    def wavelengthsEmission(self, value):
        self._wavelengthsEmission = value
        # self._cfg.logger.info('Assignment to %s/wavelengthsEmission in %s', self.location, self.filename)

    @wavelengthsEmission.deleter
    def wavelengthsEmission(self):
        self._wavelengthsEmission = AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthsEmission from %s', self.location, self.filename)

    @property
    def sourcePos2D(self):
        if type(self._sourcePos2D) is type(AbsentDataset):
            return None
        if type(self._sourcePos2D) is type(PresentDataset) and 'sourcePos2D' in self._h:
            return _read_float_array(self._h['sourcePos2D'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePos2D from %s', self.location, self.filename)
        else:
            return self._sourcePos2D

    @sourcePos2D.setter
    def sourcePos2D(self, value):
        self._sourcePos2D = value
        # self._cfg.logger.info('Assignment to %s/sourcePos2D in %s', self.location, self.filename)

    @sourcePos2D.deleter
    def sourcePos2D(self):
        self._sourcePos2D = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePos2D from %s', self.location, self.filename)

    @property
    def sourcePos3D(self):
        if type(self._sourcePos3D) is type(AbsentDataset):
            return None
        if type(self._sourcePos3D) is type(PresentDataset) and 'sourcePos3D' in self._h:
            return _read_float_array(self._h['sourcePos3D'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePos3D from %s', self.location, self.filename)
        else:
            return self._sourcePos3D

    @sourcePos3D.setter
    def sourcePos3D(self, value):
        self._sourcePos3D = value
        # self._cfg.logger.info('Assignment to %s/sourcePos3D in %s', self.location, self.filename)

    @sourcePos3D.deleter
    def sourcePos3D(self):
        self._sourcePos3D = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePos3D from %s', self.location, self.filename)

    @property
    def detectorPos2D(self):
        if type(self._detectorPos2D) is type(AbsentDataset):
            return None
        if type(self._detectorPos2D) is type(PresentDataset) and 'detectorPos2D' in self._h:
            return _read_float_array(self._h['detectorPos2D'])
            self._cfg.logger.info('Dynamically loaded %s/detectorPos2D from %s', self.location, self.filename)
        else:
            return self._detectorPos2D

    @detectorPos2D.setter
    def detectorPos2D(self, value):
        self._detectorPos2D = value
        # self._cfg.logger.info('Assignment to %s/detectorPos2D in %s', self.location, self.filename)

    @detectorPos2D.deleter
    def detectorPos2D(self):
        self._detectorPos2D = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorPos2D from %s', self.location, self.filename)

    @property
    def detectorPos3D(self):
        if type(self._detectorPos3D) is type(AbsentDataset):
            return None
        if type(self._detectorPos3D) is type(PresentDataset) and 'detectorPos3D' in self._h:
            return _read_float_array(self._h['detectorPos3D'])
            self._cfg.logger.info('Dynamically loaded %s/detectorPos3D from %s', self.location, self.filename)
        else:
            return self._detectorPos3D

    @detectorPos3D.setter
    def detectorPos3D(self, value):
        self._detectorPos3D = value
        # self._cfg.logger.info('Assignment to %s/detectorPos3D in %s', self.location, self.filename)

    @detectorPos3D.deleter
    def detectorPos3D(self):
        self._detectorPos3D = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorPos3D from %s', self.location, self.filename)

    @property
    def frequencies(self):
        if type(self._frequencies) is type(AbsentDataset):
            return None
        if type(self._frequencies) is type(PresentDataset) and 'frequencies' in self._h:
            return _read_float_array(self._h['frequencies'])
            self._cfg.logger.info('Dynamically loaded %s/frequencies from %s', self.location, self.filename)
        else:
            return self._frequencies

    @frequencies.setter
    def frequencies(self, value):
        self._frequencies = value
        # self._cfg.logger.info('Assignment to %s/frequencies in %s', self.location, self.filename)

    @frequencies.deleter
    def frequencies(self):
        self._frequencies = AbsentDataset
        self._cfg.logger.info('Deleted %s/frequencies from %s', self.location, self.filename)

    @property
    def timeDelays(self):
        if type(self._timeDelays) is type(AbsentDataset):
            return None
        if type(self._timeDelays) is type(PresentDataset) and 'timeDelays' in self._h:
            return _read_float_array(self._h['timeDelays'])
            self._cfg.logger.info('Dynamically loaded %s/timeDelays from %s', self.location, self.filename)
        else:
            return self._timeDelays

    @timeDelays.setter
    def timeDelays(self, value):
        self._timeDelays = value
        # self._cfg.logger.info('Assignment to %s/timeDelays in %s', self.location, self.filename)

    @timeDelays.deleter
    def timeDelays(self):
        self._timeDelays = AbsentDataset
        self._cfg.logger.info('Deleted %s/timeDelays from %s', self.location, self.filename)

    @property
    def timeDelayWidths(self):
        if type(self._timeDelayWidths) is type(AbsentDataset):
            return None
        if type(self._timeDelayWidths) is type(PresentDataset) and 'timeDelayWidths' in self._h:
            return _read_float_array(self._h['timeDelayWidths'])
            self._cfg.logger.info('Dynamically loaded %s/timeDelayWidths from %s', self.location, self.filename)
        else:
            return self._timeDelayWidths

    @timeDelayWidths.setter
    def timeDelayWidths(self, value):
        self._timeDelayWidths = value
        # self._cfg.logger.info('Assignment to %s/timeDelayWidths in %s', self.location, self.filename)

    @timeDelayWidths.deleter
    def timeDelayWidths(self):
        self._timeDelayWidths = AbsentDataset
        self._cfg.logger.info('Deleted %s/timeDelayWidths from %s', self.location, self.filename)

    @property
    def momentOrders(self):
        if type(self._momentOrders) is type(AbsentDataset):
            return None
        if type(self._momentOrders) is type(PresentDataset) and 'momentOrders' in self._h:
            return _read_float_array(self._h['momentOrders'])
            self._cfg.logger.info('Dynamically loaded %s/momentOrders from %s', self.location, self.filename)
        else:
            return self._momentOrders

    @momentOrders.setter
    def momentOrders(self, value):
        self._momentOrders = value
        # self._cfg.logger.info('Assignment to %s/momentOrders in %s', self.location, self.filename)

    @momentOrders.deleter
    def momentOrders(self):
        self._momentOrders = AbsentDataset
        self._cfg.logger.info('Deleted %s/momentOrders from %s', self.location, self.filename)

    @property
    def correlationTimeDelays(self):
        if type(self._correlationTimeDelays) is type(AbsentDataset):
            return None
        if type(self._correlationTimeDelays) is type(PresentDataset) and 'correlationTimeDelays' in self._h:
            return _read_float_array(self._h['correlationTimeDelays'])
            self._cfg.logger.info('Dynamically loaded %s/correlationTimeDelays from %s', self.location, self.filename)
        else:
            return self._correlationTimeDelays

    @correlationTimeDelays.setter
    def correlationTimeDelays(self, value):
        self._correlationTimeDelays = value
        # self._cfg.logger.info('Assignment to %s/correlationTimeDelays in %s', self.location, self.filename)

    @correlationTimeDelays.deleter
    def correlationTimeDelays(self):
        self._correlationTimeDelays = AbsentDataset
        self._cfg.logger.info('Deleted %s/correlationTimeDelays from %s', self.location, self.filename)

    @property
    def correlationTimeDelayWidths(self):
        if type(self._correlationTimeDelayWidths) is type(AbsentDataset):
            return None
        if type(self._correlationTimeDelayWidths) is type(PresentDataset) and 'correlationTimeDelayWidths' in self._h:
            return _read_float_array(self._h['correlationTimeDelayWidths'])
            self._cfg.logger.info('Dynamically loaded %s/correlationTimeDelayWidths from %s', self.location, self.filename)
        else:
            return self._correlationTimeDelayWidths

    @correlationTimeDelayWidths.setter
    def correlationTimeDelayWidths(self, value):
        self._correlationTimeDelayWidths = value
        # self._cfg.logger.info('Assignment to %s/correlationTimeDelayWidths in %s', self.location, self.filename)

    @correlationTimeDelayWidths.deleter
    def correlationTimeDelayWidths(self):
        self._correlationTimeDelayWidths = AbsentDataset
        self._cfg.logger.info('Deleted %s/correlationTimeDelayWidths from %s', self.location, self.filename)

    @property
    def sourceLabels(self):
        if type(self._sourceLabels) is type(AbsentDataset):
            return None
        if type(self._sourceLabels) is type(PresentDataset) and 'sourceLabels' in self._h:
            return _read_string_array(self._h['sourceLabels'])
            self._cfg.logger.info('Dynamically loaded %s/sourceLabels from %s', self.location, self.filename)
        else:
            return self._sourceLabels

    @sourceLabels.setter
    def sourceLabels(self, value):
        self._sourceLabels = value
        # self._cfg.logger.info('Assignment to %s/sourceLabels in %s', self.location, self.filename)

    @sourceLabels.deleter
    def sourceLabels(self):
        self._sourceLabels = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceLabels from %s', self.location, self.filename)

    @property
    def detectorLabels(self):
        if type(self._detectorLabels) is type(AbsentDataset):
            return None
        if type(self._detectorLabels) is type(PresentDataset) and 'detectorLabels' in self._h:
            return _read_string_array(self._h['detectorLabels'])
            self._cfg.logger.info('Dynamically loaded %s/detectorLabels from %s', self.location, self.filename)
        else:
            return self._detectorLabels

    @detectorLabels.setter
    def detectorLabels(self, value):
        self._detectorLabels = value
        # self._cfg.logger.info('Assignment to %s/detectorLabels in %s', self.location, self.filename)

    @detectorLabels.deleter
    def detectorLabels(self):
        self._detectorLabels = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorLabels from %s', self.location, self.filename)

    @property
    def landmarkPos2D(self):
        if type(self._landmarkPos2D) is type(AbsentDataset):
            return None
        if type(self._landmarkPos2D) is type(PresentDataset) and 'landmarkPos2D' in self._h:
            return _read_float_array(self._h['landmarkPos2D'])
            self._cfg.logger.info('Dynamically loaded %s/landmarkPos2D from %s', self.location, self.filename)
        else:
            return self._landmarkPos2D

    @landmarkPos2D.setter
    def landmarkPos2D(self, value):
        self._landmarkPos2D = value
        # self._cfg.logger.info('Assignment to %s/landmarkPos2D in %s', self.location, self.filename)

    @landmarkPos2D.deleter
    def landmarkPos2D(self):
        self._landmarkPos2D = AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkPos2D from %s', self.location, self.filename)

    @property
    def landmarkPos3D(self):
        if type(self._landmarkPos3D) is type(AbsentDataset):
            return None
        if type(self._landmarkPos3D) is type(PresentDataset) and 'landmarkPos3D' in self._h:
            return _read_float_array(self._h['landmarkPos3D'])
            self._cfg.logger.info('Dynamically loaded %s/landmarkPos3D from %s', self.location, self.filename)
        else:
            return self._landmarkPos3D

    @landmarkPos3D.setter
    def landmarkPos3D(self, value):
        self._landmarkPos3D = value
        # self._cfg.logger.info('Assignment to %s/landmarkPos3D in %s', self.location, self.filename)

    @landmarkPos3D.deleter
    def landmarkPos3D(self):
        self._landmarkPos3D = AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkPos3D from %s', self.location, self.filename)

    @property
    def landmarkLabels(self):
        if type(self._landmarkLabels) is type(AbsentDataset):
            return None
        if type(self._landmarkLabels) is type(PresentDataset) and 'landmarkLabels' in self._h:
            return _read_string_array(self._h['landmarkLabels'])
            self._cfg.logger.info('Dynamically loaded %s/landmarkLabels from %s', self.location, self.filename)
        else:
            return self._landmarkLabels

    @landmarkLabels.setter
    def landmarkLabels(self, value):
        self._landmarkLabels = value
        # self._cfg.logger.info('Assignment to %s/landmarkLabels in %s', self.location, self.filename)

    @landmarkLabels.deleter
    def landmarkLabels(self):
        self._landmarkLabels = AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkLabels from %s', self.location, self.filename)

    @property
    def useLocalIndex(self):
        if type(self._useLocalIndex) is type(AbsentDataset):
            return None
        if type(self._useLocalIndex) is type(PresentDataset) and 'useLocalIndex' in self._h:
            return _read_int(self._h['useLocalIndex'])
            self._cfg.logger.info('Dynamically loaded %s/useLocalIndex from %s', self.location, self.filename)
        else:
            return self._useLocalIndex

    @useLocalIndex.setter
    def useLocalIndex(self, value):
        self._useLocalIndex = value
        # self._cfg.logger.info('Assignment to %s/useLocalIndex in %s', self.location, self.filename)

    @useLocalIndex.deleter
    def useLocalIndex(self):
        self._useLocalIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/useLocalIndex from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/wavelengths'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.wavelengths
        if type(self._wavelengths) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/wavelengthsEmission'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.wavelengthsEmission
        if type(self._wavelengthsEmission) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/sourcePos2D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourcePos2D
        if type(self._sourcePos2D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/sourcePos3D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourcePos3D
        if type(self._sourcePos3D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorPos2D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorPos2D
        if type(self._detectorPos2D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorPos3D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorPos3D
        if type(self._detectorPos3D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/frequencies'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.frequencies
        if type(self._frequencies) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/timeDelays'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.timeDelays
        if type(self._timeDelays) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/timeDelayWidths'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.timeDelayWidths
        if type(self._timeDelayWidths) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/momentOrders'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.momentOrders
        if type(self._momentOrders) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/correlationTimeDelays'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.correlationTimeDelays
        if type(self._correlationTimeDelays) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/correlationTimeDelayWidths'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.correlationTimeDelayWidths
        if type(self._correlationTimeDelayWidths) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/sourceLabels'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourceLabels
        if type(self._sourceLabels) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorLabels'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorLabels
        if type(self._detectorLabels) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/landmarkPos2D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.landmarkPos2D
        if type(self._landmarkPos2D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/landmarkPos3D'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.landmarkPos3D
        if type(self._landmarkPos3D) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/landmarkLabels'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.landmarkLabels
        if type(self._landmarkLabels) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/useLocalIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.useLocalIndex
        if type(self._useLocalIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)

    def _validate(self, result: ValidationResult):
        name = self.location + '/wavelengths'
        if type(self._wavelengths) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._wavelengths) is type(PresentDataset):
                    dataset = self._h['wavelengths']
                else:
                    dataset = _create_dataset_float_array(tmp, 'wavelengths', self._wavelengths)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/wavelengthsEmission'
        if type(self._wavelengthsEmission) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._wavelengthsEmission) is type(PresentDataset):
                    dataset = self._h['wavelengthsEmission']
                else:
                    dataset = _create_dataset_float_array(tmp, 'wavelengthsEmission', self._wavelengthsEmission)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/sourcePos2D'
        if type(self._sourcePos2D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourcePos2D) is type(PresentDataset):
                    dataset = self._h['sourcePos2D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'sourcePos2D', self._sourcePos2D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/sourcePos3D'
        if type(self._sourcePos3D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourcePos3D) is type(PresentDataset):
                    dataset = self._h['sourcePos3D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'sourcePos3D', self._sourcePos3D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorPos2D'
        if type(self._detectorPos2D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorPos2D) is type(PresentDataset):
                    dataset = self._h['detectorPos2D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'detectorPos2D', self._detectorPos2D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorPos3D'
        if type(self._detectorPos3D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorPos3D) is type(PresentDataset):
                    dataset = self._h['detectorPos3D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'detectorPos3D', self._detectorPos3D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/frequencies'
        if type(self._frequencies) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._frequencies) is type(PresentDataset):
                    dataset = self._h['frequencies']
                else:
                    dataset = _create_dataset_float_array(tmp, 'frequencies', self._frequencies)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/timeDelays'
        if type(self._timeDelays) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._timeDelays) is type(PresentDataset):
                    dataset = self._h['timeDelays']
                else:
                    dataset = _create_dataset_float_array(tmp, 'timeDelays', self._timeDelays)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/timeDelayWidths'
        if type(self._timeDelayWidths) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._timeDelayWidths) is type(PresentDataset):
                    dataset = self._h['timeDelayWidths']
                else:
                    dataset = _create_dataset_float_array(tmp, 'timeDelayWidths', self._timeDelayWidths)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/momentOrders'
        if type(self._momentOrders) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._momentOrders) is type(PresentDataset):
                    dataset = self._h['momentOrders']
                else:
                    dataset = _create_dataset_float_array(tmp, 'momentOrders', self._momentOrders)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/correlationTimeDelays'
        if type(self._correlationTimeDelays) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._correlationTimeDelays) is type(PresentDataset):
                    dataset = self._h['correlationTimeDelays']
                else:
                    dataset = _create_dataset_float_array(tmp, 'correlationTimeDelays', self._correlationTimeDelays)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/correlationTimeDelayWidths'
        if type(self._correlationTimeDelayWidths) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._correlationTimeDelayWidths) is type(PresentDataset):
                    dataset = self._h['correlationTimeDelayWidths']
                else:
                    dataset = _create_dataset_float_array(tmp, 'correlationTimeDelayWidths', self._correlationTimeDelayWidths)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/sourceLabels'
        if type(self._sourceLabels) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourceLabels) is type(PresentDataset):
                    dataset = self._h['sourceLabels']
                else:
                    dataset = _create_dataset_string_array(tmp, 'sourceLabels', self._sourceLabels)
                result._add(name, _validate_string_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorLabels'
        if type(self._detectorLabels) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorLabels) is type(PresentDataset):
                    dataset = self._h['detectorLabels']
                else:
                    dataset = _create_dataset_string_array(tmp, 'detectorLabels', self._detectorLabels)
                result._add(name, _validate_string_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/landmarkPos2D'
        if type(self._landmarkPos2D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._landmarkPos2D) is type(PresentDataset):
                    dataset = self._h['landmarkPos2D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'landmarkPos2D', self._landmarkPos2D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/landmarkPos3D'
        if type(self._landmarkPos3D) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._landmarkPos3D) is type(PresentDataset):
                    dataset = self._h['landmarkPos3D']
                else:
                    dataset = _create_dataset_float_array(tmp, 'landmarkPos3D', self._landmarkPos3D)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/landmarkLabels'
        if type(self._landmarkLabels) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._landmarkLabels) is type(PresentDataset):
                    dataset = self._h['landmarkLabels']
                else:
                    dataset = _create_dataset_string_array(tmp, 'landmarkLabels', self._landmarkLabels)
                result._add(name, _validate_string_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/useLocalIndex'
        if type(self._useLocalIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._useLocalIndex) is type(PresentDataset):
                    dataset = self._h['useLocalIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'useLocalIndex', self._useLocalIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')



class NirsElement(Group):

    _metaDataTags = AbsentGroup  # {.}*
    _data = AbsentDataset  # {i}*
    _stim = AbsentDataset  # {i}
    _probe = AbsentGroup  # {.}*
    _aux = AbsentDataset  # {i}
    _snirf_names = ['metaDataTags', 'data', 'stim', 'probe', 'aux', ]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._indexed_groups = []
        if 'metaDataTags' in self._h:
            self._metaDataTags = MetaDataTags(self._h['metaDataTags'].id, self._cfg)  # Group
        else:
            self._metaDataTags = MetaDataTags(self.location + '/' + 'metaDataTags', self._cfg)  # Anonymous group (wrapper only)
        self.data = Data(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.data)
        self.stim = Stim(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.stim)
        if 'probe' in self._h:
            self._probe = Probe(self._h['probe'].id, self._cfg)  # Group
        else:
            self._probe = Probe(self.location + '/' + 'probe', self._cfg)  # Anonymous group (wrapper only)
        self.aux = Aux(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.aux)
                

    @property
    def metaDataTags(self):
        if type(self._metaDataTags) is type(AbsentGroup):
            return None
        return self._metaDataTags

    @metaDataTags.setter
    def metaDataTags(self, value):
        self._metaDataTags = value
        # self._cfg.logger.info('Assignment to %s/metaDataTags in %s', self.location, self.filename)

    @metaDataTags.deleter
    def metaDataTags(self):
        self._metaDataTags = AbsentGroup
        self._cfg.logger.info('Deleted %s/metaDataTags from %s', self.location, self.filename)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # self._cfg.logger.info('Assignment to %s/data in %s', self.location, self.filename)

    @data.deleter
    def data(self):
        raise AttributeError('IndexedGroup ' + str(type(self._data)) + ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/data from %s', self.location, self.filename)

    @property
    def stim(self):
        return self._stim

    @stim.setter
    def stim(self, value):
        self._stim = value
        # self._cfg.logger.info('Assignment to %s/stim in %s', self.location, self.filename)

    @stim.deleter
    def stim(self):
        raise AttributeError('IndexedGroup ' + str(type(self._stim)) + ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/stim from %s', self.location, self.filename)

    @property
    def probe(self):
        if type(self._probe) is type(AbsentGroup):
            return None
        return self._probe

    @probe.setter
    def probe(self, value):
        self._probe = value
        # self._cfg.logger.info('Assignment to %s/probe in %s', self.location, self.filename)

    @probe.deleter
    def probe(self):
        self._probe = AbsentGroup
        self._cfg.logger.info('Deleted %s/probe from %s', self.location, self.filename)

    @property
    def aux(self):
        return self._aux

    @aux.setter
    def aux(self, value):
        self._aux = value
        # self._cfg.logger.info('Assignment to %s/aux in %s', self.location, self.filename)

    @aux.deleter
    def aux(self):
        raise AttributeError('IndexedGroup ' + str(type(self._aux)) + ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/aux from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        if type(self._metaDataTags) is type(AbsentGroup) or self._metaDataTags.is_empty():
            if 'metaDataTags' in file:
                del file['metaDataTags']
                self._cfg.logger.info('Deleted Group %s/metaDataTags from %s', self.location, file)
        else:
            self.metaDataTags._save(*args)
        self.data._save(*args)
        self.stim._save(*args)
        if type(self._probe) is type(AbsentGroup) or self._probe.is_empty():
            if 'probe' in file:
                del file['probe']
                self._cfg.logger.info('Deleted Group %s/probe from %s', self.location, file)
        else:
            self.probe._save(*args)
        self.aux._save(*args)

    def _validate(self, result: ValidationResult):
        name = self.location + '/metaDataTags'
        if type(self._metaDataTags) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_GROUP_MISSING')
            print(name, 'not found')
        else:
            self.metaDataTags._validate(result)
        name = self.location + '/data'
        self.data._validate(result)
        name = self.location + '/stim'
        self.stim._validate(result)
        name = self.location + '/probe'
        if type(self._probe) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_GROUP_MISSING')
            print(name, 'not found')
        else:
            self.probe._validate(result)
        name = self.location + '/aux'
        self.aux._validate(result)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


class Nirs(IndexedGroup):

    _name: str = 'nirs'
    _element: Group = NirsElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class DataElement(Group):

    _dataTimeSeries = AbsentDataset  # [[<f>,...]]*
    _time = AbsentDataset  # [<f>,...]*
    _measurementList = AbsentDataset  # {i}*
    _snirf_names = ['dataTimeSeries', 'time', 'measurementList', ]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._indexed_groups = []
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(self._h['dataTimeSeries'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTimeSeries = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTimeSeries = AbsentDataset
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._time = PresentDataset
        else:  # if the dataset is not found on disk
            self._time = AbsentDataset
        self.measurementList = MeasurementList(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.measurementList)
                

    @property
    def dataTimeSeries(self):
        if type(self._dataTimeSeries) is type(AbsentDataset):
            return None
        if type(self._dataTimeSeries) is type(PresentDataset) and 'dataTimeSeries' in self._h:
            return _read_float_array(self._h['dataTimeSeries'])
            self._cfg.logger.info('Dynamically loaded %s/dataTimeSeries from %s', self.location, self.filename)
        else:
            return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value
        # self._cfg.logger.info('Assignment to %s/dataTimeSeries in %s', self.location, self.filename)

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTimeSeries from %s', self.location, self.filename)

    @property
    def time(self):
        if type(self._time) is type(AbsentDataset):
            return None
        if type(self._time) is type(PresentDataset) and 'time' in self._h:
            return _read_float_array(self._h['time'])
            self._cfg.logger.info('Dynamically loaded %s/time from %s', self.location, self.filename)
        else:
            return self._time

    @time.setter
    def time(self, value):
        self._time = value
        # self._cfg.logger.info('Assignment to %s/time in %s', self.location, self.filename)

    @time.deleter
    def time(self):
        self._time = AbsentDataset
        self._cfg.logger.info('Deleted %s/time from %s', self.location, self.filename)

    @property
    def measurementList(self):
        return self._measurementList

    @measurementList.setter
    def measurementList(self, value):
        self._measurementList = value
        # self._cfg.logger.info('Assignment to %s/measurementList in %s', self.location, self.filename)

    @measurementList.deleter
    def measurementList(self):
        raise AttributeError('IndexedGroup ' + str(type(self._measurementList)) + ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/measurementList from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/dataTimeSeries'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataTimeSeries
        if type(self._dataTimeSeries) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/time'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.time
        if type(self._time) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        self.measurementList._save(*args)

    def _validate(self, result: ValidationResult):
        name = self.location + '/dataTimeSeries'
        if type(self._dataTimeSeries) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataTimeSeries) is type(PresentDataset):
                    dataset = self._h['dataTimeSeries']
                else:
                    dataset = _create_dataset_float_array(tmp, 'dataTimeSeries', self._dataTimeSeries)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/time'
        if type(self._time) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._time) is type(PresentDataset):
                    dataset = self._h['time']
                else:
                    dataset = _create_dataset_float_array(tmp, 'time', self._time)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/measurementList'
        self.measurementList._validate(result)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


class Data(IndexedGroup):

    _name: str = 'data'
    _element: Group = DataElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class MeasurementListElement(Group):

    _sourceIndex = AbsentDataset  # <i>*
    _detectorIndex = AbsentDataset  # <i>*
    _wavelengthIndex = AbsentDataset  # <i>*
    _wavelengthActual = AbsentDataset  # <f>
    _wavelengthEmissionActual = AbsentDataset  # <f>
    _dataType = AbsentDataset  # <i>*
    _dataUnit = AbsentDataset  # "s"
    _dataTypeLabel = AbsentDataset  # "s"
    _dataTypeIndex = AbsentDataset  # <i>*
    _sourcePower = AbsentDataset  # <f>
    _detectorGain = AbsentDataset  # <f>
    _moduleIndex = AbsentDataset  # <i>
    _sourceModuleIndex = AbsentDataset  # <i>
    _detectorModuleIndex = AbsentDataset  # <i>
    _snirf_names = ['sourceIndex', 'detectorIndex', 'wavelengthIndex', 'wavelengthActual', 'wavelengthEmissionActual', 'dataType', 'dataUnit', 'dataTypeLabel', 'dataTypeIndex', 'sourcePower', 'detectorGain', 'moduleIndex', 'sourceModuleIndex', 'detectorModuleIndex', ]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._indexed_groups = []
        if 'sourceIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceIndex = _read_int(self._h['sourceIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceIndex = AbsentDataset
        if 'detectorIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorIndex = _read_int(self._h['detectorIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorIndex = AbsentDataset
        if 'wavelengthIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthIndex = _read_int(self._h['wavelengthIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthIndex = AbsentDataset
        if 'wavelengthActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthActual = _read_float(self._h['wavelengthActual'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthActual = PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthActual = AbsentDataset
        if 'wavelengthEmissionActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthEmissionActual = _read_float(self._h['wavelengthEmissionActual'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthEmissionActual = PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthEmissionActual = AbsentDataset
        if 'dataType' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataType = _read_int(self._h['dataType'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataType = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataType = AbsentDataset
        if 'dataUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataUnit = _read_string(self._h['dataUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataUnit = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataUnit = AbsentDataset
        if 'dataTypeLabel' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeLabel = _read_string(self._h['dataTypeLabel'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTypeLabel = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTypeLabel = AbsentDataset
        if 'dataTypeIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeIndex = _read_int(self._h['dataTypeIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTypeIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTypeIndex = AbsentDataset
        if 'sourcePower' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePower = _read_float(self._h['sourcePower'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePower = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePower = AbsentDataset
        if 'detectorGain' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorGain = _read_float(self._h['detectorGain'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorGain = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorGain = AbsentDataset
        if 'moduleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._moduleIndex = _read_int(self._h['moduleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._moduleIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._moduleIndex = AbsentDataset
        if 'sourceModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceModuleIndex = _read_int(self._h['sourceModuleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceModuleIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceModuleIndex = AbsentDataset
        if 'detectorModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorModuleIndex = _read_int(self._h['detectorModuleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorModuleIndex = PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorModuleIndex = AbsentDataset
                

    @property
    def sourceIndex(self):
        if type(self._sourceIndex) is type(AbsentDataset):
            return None
        if type(self._sourceIndex) is type(PresentDataset) and 'sourceIndex' in self._h:
            return _read_int(self._h['sourceIndex'])
            self._cfg.logger.info('Dynamically loaded %s/sourceIndex from %s', self.location, self.filename)
        else:
            return self._sourceIndex

    @sourceIndex.setter
    def sourceIndex(self, value):
        self._sourceIndex = value
        # self._cfg.logger.info('Assignment to %s/sourceIndex in %s', self.location, self.filename)

    @sourceIndex.deleter
    def sourceIndex(self):
        self._sourceIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceIndex from %s', self.location, self.filename)

    @property
    def detectorIndex(self):
        if type(self._detectorIndex) is type(AbsentDataset):
            return None
        if type(self._detectorIndex) is type(PresentDataset) and 'detectorIndex' in self._h:
            return _read_int(self._h['detectorIndex'])
            self._cfg.logger.info('Dynamically loaded %s/detectorIndex from %s', self.location, self.filename)
        else:
            return self._detectorIndex

    @detectorIndex.setter
    def detectorIndex(self, value):
        self._detectorIndex = value
        # self._cfg.logger.info('Assignment to %s/detectorIndex in %s', self.location, self.filename)

    @detectorIndex.deleter
    def detectorIndex(self):
        self._detectorIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorIndex from %s', self.location, self.filename)

    @property
    def wavelengthIndex(self):
        if type(self._wavelengthIndex) is type(AbsentDataset):
            return None
        if type(self._wavelengthIndex) is type(PresentDataset) and 'wavelengthIndex' in self._h:
            return _read_int(self._h['wavelengthIndex'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengthIndex from %s', self.location, self.filename)
        else:
            return self._wavelengthIndex

    @wavelengthIndex.setter
    def wavelengthIndex(self, value):
        self._wavelengthIndex = value
        # self._cfg.logger.info('Assignment to %s/wavelengthIndex in %s', self.location, self.filename)

    @wavelengthIndex.deleter
    def wavelengthIndex(self):
        self._wavelengthIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthIndex from %s', self.location, self.filename)

    @property
    def wavelengthActual(self):
        if type(self._wavelengthActual) is type(AbsentDataset):
            return None
        if type(self._wavelengthActual) is type(PresentDataset) and 'wavelengthActual' in self._h:
            return _read_float(self._h['wavelengthActual'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengthActual from %s', self.location, self.filename)
        else:
            return self._wavelengthActual

    @wavelengthActual.setter
    def wavelengthActual(self, value):
        self._wavelengthActual = value
        # self._cfg.logger.info('Assignment to %s/wavelengthActual in %s', self.location, self.filename)

    @wavelengthActual.deleter
    def wavelengthActual(self):
        self._wavelengthActual = AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthActual from %s', self.location, self.filename)

    @property
    def wavelengthEmissionActual(self):
        if type(self._wavelengthEmissionActual) is type(AbsentDataset):
            return None
        if type(self._wavelengthEmissionActual) is type(PresentDataset) and 'wavelengthEmissionActual' in self._h:
            return _read_float(self._h['wavelengthEmissionActual'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengthEmissionActual from %s', self.location, self.filename)
        else:
            return self._wavelengthEmissionActual

    @wavelengthEmissionActual.setter
    def wavelengthEmissionActual(self, value):
        self._wavelengthEmissionActual = value
        # self._cfg.logger.info('Assignment to %s/wavelengthEmissionActual in %s', self.location, self.filename)

    @wavelengthEmissionActual.deleter
    def wavelengthEmissionActual(self):
        self._wavelengthEmissionActual = AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthEmissionActual from %s', self.location, self.filename)

    @property
    def dataType(self):
        if type(self._dataType) is type(AbsentDataset):
            return None
        if type(self._dataType) is type(PresentDataset) and 'dataType' in self._h:
            return _read_int(self._h['dataType'])
            self._cfg.logger.info('Dynamically loaded %s/dataType from %s', self.location, self.filename)
        else:
            return self._dataType

    @dataType.setter
    def dataType(self, value):
        self._dataType = value
        # self._cfg.logger.info('Assignment to %s/dataType in %s', self.location, self.filename)

    @dataType.deleter
    def dataType(self):
        self._dataType = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataType from %s', self.location, self.filename)

    @property
    def dataUnit(self):
        if type(self._dataUnit) is type(AbsentDataset):
            return None
        if type(self._dataUnit) is type(PresentDataset) and 'dataUnit' in self._h:
            return _read_string(self._h['dataUnit'])
            self._cfg.logger.info('Dynamically loaded %s/dataUnit from %s', self.location, self.filename)
        else:
            return self._dataUnit

    @dataUnit.setter
    def dataUnit(self, value):
        self._dataUnit = value
        # self._cfg.logger.info('Assignment to %s/dataUnit in %s', self.location, self.filename)

    @dataUnit.deleter
    def dataUnit(self):
        self._dataUnit = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataUnit from %s', self.location, self.filename)

    @property
    def dataTypeLabel(self):
        if type(self._dataTypeLabel) is type(AbsentDataset):
            return None
        if type(self._dataTypeLabel) is type(PresentDataset) and 'dataTypeLabel' in self._h:
            return _read_string(self._h['dataTypeLabel'])
            self._cfg.logger.info('Dynamically loaded %s/dataTypeLabel from %s', self.location, self.filename)
        else:
            return self._dataTypeLabel

    @dataTypeLabel.setter
    def dataTypeLabel(self, value):
        self._dataTypeLabel = value
        # self._cfg.logger.info('Assignment to %s/dataTypeLabel in %s', self.location, self.filename)

    @dataTypeLabel.deleter
    def dataTypeLabel(self):
        self._dataTypeLabel = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTypeLabel from %s', self.location, self.filename)

    @property
    def dataTypeIndex(self):
        if type(self._dataTypeIndex) is type(AbsentDataset):
            return None
        if type(self._dataTypeIndex) is type(PresentDataset) and 'dataTypeIndex' in self._h:
            return _read_int(self._h['dataTypeIndex'])
            self._cfg.logger.info('Dynamically loaded %s/dataTypeIndex from %s', self.location, self.filename)
        else:
            return self._dataTypeIndex

    @dataTypeIndex.setter
    def dataTypeIndex(self, value):
        self._dataTypeIndex = value
        # self._cfg.logger.info('Assignment to %s/dataTypeIndex in %s', self.location, self.filename)

    @dataTypeIndex.deleter
    def dataTypeIndex(self):
        self._dataTypeIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTypeIndex from %s', self.location, self.filename)

    @property
    def sourcePower(self):
        if type(self._sourcePower) is type(AbsentDataset):
            return None
        if type(self._sourcePower) is type(PresentDataset) and 'sourcePower' in self._h:
            return _read_float(self._h['sourcePower'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePower from %s', self.location, self.filename)
        else:
            return self._sourcePower

    @sourcePower.setter
    def sourcePower(self, value):
        self._sourcePower = value
        # self._cfg.logger.info('Assignment to %s/sourcePower in %s', self.location, self.filename)

    @sourcePower.deleter
    def sourcePower(self):
        self._sourcePower = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePower from %s', self.location, self.filename)

    @property
    def detectorGain(self):
        if type(self._detectorGain) is type(AbsentDataset):
            return None
        if type(self._detectorGain) is type(PresentDataset) and 'detectorGain' in self._h:
            return _read_float(self._h['detectorGain'])
            self._cfg.logger.info('Dynamically loaded %s/detectorGain from %s', self.location, self.filename)
        else:
            return self._detectorGain

    @detectorGain.setter
    def detectorGain(self, value):
        self._detectorGain = value
        # self._cfg.logger.info('Assignment to %s/detectorGain in %s', self.location, self.filename)

    @detectorGain.deleter
    def detectorGain(self):
        self._detectorGain = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorGain from %s', self.location, self.filename)

    @property
    def moduleIndex(self):
        if type(self._moduleIndex) is type(AbsentDataset):
            return None
        if type(self._moduleIndex) is type(PresentDataset) and 'moduleIndex' in self._h:
            return _read_int(self._h['moduleIndex'])
            self._cfg.logger.info('Dynamically loaded %s/moduleIndex from %s', self.location, self.filename)
        else:
            return self._moduleIndex

    @moduleIndex.setter
    def moduleIndex(self, value):
        self._moduleIndex = value
        # self._cfg.logger.info('Assignment to %s/moduleIndex in %s', self.location, self.filename)

    @moduleIndex.deleter
    def moduleIndex(self):
        self._moduleIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/moduleIndex from %s', self.location, self.filename)

    @property
    def sourceModuleIndex(self):
        if type(self._sourceModuleIndex) is type(AbsentDataset):
            return None
        if type(self._sourceModuleIndex) is type(PresentDataset) and 'sourceModuleIndex' in self._h:
            return _read_int(self._h['sourceModuleIndex'])
            self._cfg.logger.info('Dynamically loaded %s/sourceModuleIndex from %s', self.location, self.filename)
        else:
            return self._sourceModuleIndex

    @sourceModuleIndex.setter
    def sourceModuleIndex(self, value):
        self._sourceModuleIndex = value
        # self._cfg.logger.info('Assignment to %s/sourceModuleIndex in %s', self.location, self.filename)

    @sourceModuleIndex.deleter
    def sourceModuleIndex(self):
        self._sourceModuleIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceModuleIndex from %s', self.location, self.filename)

    @property
    def detectorModuleIndex(self):
        if type(self._detectorModuleIndex) is type(AbsentDataset):
            return None
        if type(self._detectorModuleIndex) is type(PresentDataset) and 'detectorModuleIndex' in self._h:
            return _read_int(self._h['detectorModuleIndex'])
            self._cfg.logger.info('Dynamically loaded %s/detectorModuleIndex from %s', self.location, self.filename)
        else:
            return self._detectorModuleIndex

    @detectorModuleIndex.setter
    def detectorModuleIndex(self, value):
        self._detectorModuleIndex = value
        # self._cfg.logger.info('Assignment to %s/detectorModuleIndex in %s', self.location, self.filename)

    @detectorModuleIndex.deleter
    def detectorModuleIndex(self):
        self._detectorModuleIndex = AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorModuleIndex from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/sourceIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourceIndex
        if type(self._sourceIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorIndex
        if type(self._detectorIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/wavelengthIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.wavelengthIndex
        if type(self._wavelengthIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/wavelengthActual'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.wavelengthActual
        if type(self._wavelengthActual) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/wavelengthEmissionActual'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.wavelengthEmissionActual
        if type(self._wavelengthEmissionActual) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataType'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataType
        if type(self._dataType) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataUnit'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataUnit
        if type(self._dataUnit) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataTypeLabel'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataTypeLabel
        if type(self._dataTypeLabel) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataTypeIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataTypeIndex
        if type(self._dataTypeIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/sourcePower'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourcePower
        if type(self._sourcePower) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorGain'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorGain
        if type(self._detectorGain) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/moduleIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.moduleIndex
        if type(self._moduleIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/sourceModuleIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.sourceModuleIndex
        if type(self._sourceModuleIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/detectorModuleIndex'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.detectorModuleIndex
        if type(self._detectorModuleIndex) not in [type(AbsentDataset), type(None)]:
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)

    def _validate(self, result: ValidationResult):
        name = self.location + '/sourceIndex'
        if type(self._sourceIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourceIndex) is type(PresentDataset):
                    dataset = self._h['sourceIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'sourceIndex', self._sourceIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorIndex'
        if type(self._detectorIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorIndex) is type(PresentDataset):
                    dataset = self._h['detectorIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'detectorIndex', self._detectorIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/wavelengthIndex'
        if type(self._wavelengthIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._wavelengthIndex) is type(PresentDataset):
                    dataset = self._h['wavelengthIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'wavelengthIndex', self._wavelengthIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/wavelengthActual'
        if type(self._wavelengthActual) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._wavelengthActual) is type(PresentDataset):
                    dataset = self._h['wavelengthActual']
                else:
                    dataset = _create_dataset_float(tmp, 'wavelengthActual', self._wavelengthActual)
                result._add(name, _validate_float(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/wavelengthEmissionActual'
        if type(self._wavelengthEmissionActual) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._wavelengthEmissionActual) is type(PresentDataset):
                    dataset = self._h['wavelengthEmissionActual']
                else:
                    dataset = _create_dataset_float(tmp, 'wavelengthEmissionActual', self._wavelengthEmissionActual)
                result._add(name, _validate_float(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataType'
        if type(self._dataType) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataType) is type(PresentDataset):
                    dataset = self._h['dataType']
                else:
                    dataset = _create_dataset_int(tmp, 'dataType', self._dataType)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataUnit'
        if type(self._dataUnit) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataUnit) is type(PresentDataset):
                    dataset = self._h['dataUnit']
                else:
                    dataset = _create_dataset_string(tmp, 'dataUnit', self._dataUnit)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataTypeLabel'
        if type(self._dataTypeLabel) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataTypeLabel) is type(PresentDataset):
                    dataset = self._h['dataTypeLabel']
                else:
                    dataset = _create_dataset_string(tmp, 'dataTypeLabel', self._dataTypeLabel)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataTypeIndex'
        if type(self._dataTypeIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataTypeIndex) is type(PresentDataset):
                    dataset = self._h['dataTypeIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'dataTypeIndex', self._dataTypeIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/sourcePower'
        if type(self._sourcePower) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourcePower) is type(PresentDataset):
                    dataset = self._h['sourcePower']
                else:
                    dataset = _create_dataset_float(tmp, 'sourcePower', self._sourcePower)
                result._add(name, _validate_float(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorGain'
        if type(self._detectorGain) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorGain) is type(PresentDataset):
                    dataset = self._h['detectorGain']
                else:
                    dataset = _create_dataset_float(tmp, 'detectorGain', self._detectorGain)
                result._add(name, _validate_float(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/moduleIndex'
        if type(self._moduleIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._moduleIndex) is type(PresentDataset):
                    dataset = self._h['moduleIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'moduleIndex', self._moduleIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/sourceModuleIndex'
        if type(self._sourceModuleIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._sourceModuleIndex) is type(PresentDataset):
                    dataset = self._h['sourceModuleIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'sourceModuleIndex', self._sourceModuleIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/detectorModuleIndex'
        if type(self._detectorModuleIndex) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._detectorModuleIndex) is type(PresentDataset):
                    dataset = self._h['detectorModuleIndex']
                else:
                    dataset = _create_dataset_int(tmp, 'detectorModuleIndex', self._detectorModuleIndex)
                result._add(name, _validate_int(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


class MeasurementList(IndexedGroup):

    _name: str = 'measurementList'
    _element: Group = MeasurementListElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class StimElement(Group):

    _name = AbsentDataset  # "s"+
    _data = AbsentDataset  # [[<f>,...]]+
    _dataLabels = AbsentDataset  # ["s",...]
    _snirf_names = ['name', 'data', 'dataLabels', ]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._indexed_groups = []
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._name = PresentDataset
        else:  # if the dataset is not found on disk
            self._name = AbsentDataset
        if 'data' in self._h:
            if not self._cfg.dynamic_loading:
                self._data = _read_float_array(self._h['data'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._data = PresentDataset
        else:  # if the dataset is not found on disk
            self._data = AbsentDataset
        if 'dataLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataLabels = _read_string_array(self._h['dataLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataLabels = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataLabels = AbsentDataset
                

    @property
    def name(self):
        if type(self._name) is type(AbsentDataset):
            return None
        if type(self._name) is type(PresentDataset) and 'name' in self._h:
            return _read_string(self._h['name'])
            self._cfg.logger.info('Dynamically loaded %s/name from %s', self.location, self.filename)
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value
        # self._cfg.logger.info('Assignment to %s/name in %s', self.location, self.filename)

    @name.deleter
    def name(self):
        self._name = AbsentDataset
        self._cfg.logger.info('Deleted %s/name from %s', self.location, self.filename)

    @property
    def data(self):
        if type(self._data) is type(AbsentDataset):
            return None
        if type(self._data) is type(PresentDataset) and 'data' in self._h:
            return _read_float_array(self._h['data'])
            self._cfg.logger.info('Dynamically loaded %s/data from %s', self.location, self.filename)
        else:
            return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # self._cfg.logger.info('Assignment to %s/data in %s', self.location, self.filename)

    @data.deleter
    def data(self):
        self._data = AbsentDataset
        self._cfg.logger.info('Deleted %s/data from %s', self.location, self.filename)

    @property
    def dataLabels(self):
        if type(self._dataLabels) is type(AbsentDataset):
            return None
        if type(self._dataLabels) is type(PresentDataset) and 'dataLabels' in self._h:
            return _read_string_array(self._h['dataLabels'])
            self._cfg.logger.info('Dynamically loaded %s/dataLabels from %s', self.location, self.filename)
        else:
            return self._dataLabels

    @dataLabels.setter
    def dataLabels(self, value):
        self._dataLabels = value
        # self._cfg.logger.info('Assignment to %s/dataLabels in %s', self.location, self.filename)

    @dataLabels.deleter
    def dataLabels(self):
        self._dataLabels = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataLabels from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/name'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.name
        if type(self._name) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/data'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.data
        if type(self._data) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataLabels'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataLabels
        if type(self._dataLabels) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)

    def _validate(self, result: ValidationResult):
        name = self.location + '/name'
        if type(self._name) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._name) is type(PresentDataset):
                    dataset = self._h['name']
                else:
                    dataset = _create_dataset_string(tmp, 'name', self._name)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/data'
        if type(self._data) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._data) is type(PresentDataset):
                    dataset = self._h['data']
                else:
                    dataset = _create_dataset_float_array(tmp, 'data', self._data)
                result._add(name, _validate_float_array(dataset, ndims=[2]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataLabels'
        if type(self._dataLabels) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataLabels) is type(PresentDataset):
                    dataset = self._h['dataLabels']
                else:
                    dataset = _create_dataset_string_array(tmp, 'dataLabels', self._dataLabels)
                result._add(name, _validate_string_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


class Stim(IndexedGroup):

    _name: str = 'stim'
    _element: Group = StimElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class AuxElement(Group):

    _name = AbsentDataset  # "s"+
    _dataTimeSeries = AbsentDataset  # [<f>,...]+
    _dataUnit = AbsentDataset  # "s"
    _time = AbsentDataset  # [<f>,...]+
    _timeOffset = AbsentDataset  # [<f>,...]
    _snirf_names = ['name', 'dataTimeSeries', 'dataUnit', 'time', 'timeOffset', ]


    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._indexed_groups = []
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._name = PresentDataset
        else:  # if the dataset is not found on disk
            self._name = AbsentDataset
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(self._h['dataTimeSeries'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTimeSeries = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTimeSeries = AbsentDataset
        if 'dataUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataUnit = _read_string(self._h['dataUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataUnit = PresentDataset
        else:  # if the dataset is not found on disk
            self._dataUnit = AbsentDataset
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._time = PresentDataset
        else:  # if the dataset is not found on disk
            self._time = AbsentDataset
        if 'timeOffset' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeOffset = _read_float_array(self._h['timeOffset'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeOffset = PresentDataset
        else:  # if the dataset is not found on disk
            self._timeOffset = AbsentDataset
                

    @property
    def name(self):
        if type(self._name) is type(AbsentDataset):
            return None
        if type(self._name) is type(PresentDataset) and 'name' in self._h:
            return _read_string(self._h['name'])
            self._cfg.logger.info('Dynamically loaded %s/name from %s', self.location, self.filename)
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value
        # self._cfg.logger.info('Assignment to %s/name in %s', self.location, self.filename)

    @name.deleter
    def name(self):
        self._name = AbsentDataset
        self._cfg.logger.info('Deleted %s/name from %s', self.location, self.filename)

    @property
    def dataTimeSeries(self):
        if type(self._dataTimeSeries) is type(AbsentDataset):
            return None
        if type(self._dataTimeSeries) is type(PresentDataset) and 'dataTimeSeries' in self._h:
            return _read_float_array(self._h['dataTimeSeries'])
            self._cfg.logger.info('Dynamically loaded %s/dataTimeSeries from %s', self.location, self.filename)
        else:
            return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value
        # self._cfg.logger.info('Assignment to %s/dataTimeSeries in %s', self.location, self.filename)

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTimeSeries from %s', self.location, self.filename)

    @property
    def dataUnit(self):
        if type(self._dataUnit) is type(AbsentDataset):
            return None
        if type(self._dataUnit) is type(PresentDataset) and 'dataUnit' in self._h:
            return _read_string(self._h['dataUnit'])
            self._cfg.logger.info('Dynamically loaded %s/dataUnit from %s', self.location, self.filename)
        else:
            return self._dataUnit

    @dataUnit.setter
    def dataUnit(self, value):
        self._dataUnit = value
        # self._cfg.logger.info('Assignment to %s/dataUnit in %s', self.location, self.filename)

    @dataUnit.deleter
    def dataUnit(self):
        self._dataUnit = AbsentDataset
        self._cfg.logger.info('Deleted %s/dataUnit from %s', self.location, self.filename)

    @property
    def time(self):
        if type(self._time) is type(AbsentDataset):
            return None
        if type(self._time) is type(PresentDataset) and 'time' in self._h:
            return _read_float_array(self._h['time'])
            self._cfg.logger.info('Dynamically loaded %s/time from %s', self.location, self.filename)
        else:
            return self._time

    @time.setter
    def time(self, value):
        self._time = value
        # self._cfg.logger.info('Assignment to %s/time in %s', self.location, self.filename)

    @time.deleter
    def time(self):
        self._time = AbsentDataset
        self._cfg.logger.info('Deleted %s/time from %s', self.location, self.filename)

    @property
    def timeOffset(self):
        if type(self._timeOffset) is type(AbsentDataset):
            return None
        if type(self._timeOffset) is type(PresentDataset) and 'timeOffset' in self._h:
            return _read_float_array(self._h['timeOffset'])
            self._cfg.logger.info('Dynamically loaded %s/timeOffset from %s', self.location, self.filename)
        else:
            return self._timeOffset

    @timeOffset.setter
    def timeOffset(self, value):
        self._timeOffset = value
        # self._cfg.logger.info('Assignment to %s/timeOffset in %s', self.location, self.filename)

    @timeOffset.deleter
    def timeOffset(self):
        self._timeOffset = AbsentDataset
        self._cfg.logger.info('Deleted %s/timeOffset from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/name'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.name
        if type(self._name) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataTimeSeries'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataTimeSeries
        if type(self._dataTimeSeries) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/dataUnit'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.dataUnit
        if type(self._dataUnit) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/time'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.time
        if type(self._time) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        name = self.location + '/timeOffset'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.timeOffset
        if type(self._timeOffset) not in [type(AbsentDataset), type(None)]:
            _create_dataset_float_array(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)

    def _validate(self, result: ValidationResult):
        name = self.location + '/name'
        if type(self._name) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._name) is type(PresentDataset):
                    dataset = self._h['name']
                else:
                    dataset = _create_dataset_string(tmp, 'name', self._name)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataTimeSeries'
        if type(self._dataTimeSeries) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataTimeSeries) is type(PresentDataset):
                    dataset = self._h['dataTimeSeries']
                else:
                    dataset = _create_dataset_float_array(tmp, 'dataTimeSeries', self._dataTimeSeries)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/dataUnit'
        if type(self._dataUnit) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._dataUnit) is type(PresentDataset):
                    dataset = self._h['dataUnit']
                else:
                    dataset = _create_dataset_string(tmp, 'dataUnit', self._dataUnit)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/time'
        if type(self._time) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._time) is type(PresentDataset):
                    dataset = self._h['time']
                else:
                    dataset = _create_dataset_float_array(tmp, 'time', self._time)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/timeOffset'
        if type(self._timeOffset) in [type(AbsentDataset), type(None)]:
            result._add(name, 'OPTIONAL_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._timeOffset) is type(PresentDataset):
                    dataset = self._h['timeOffset']
                else:
                    dataset = _create_dataset_float_array(tmp, 'timeOffset', self._timeOffset)
                result._add(name, _validate_float_array(dataset, ndims=[1]))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


class Aux(IndexedGroup):

    _name: str = 'aux'
    _element: Group = AuxElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class Snirf(Group):
    
    _name = '/'
    _formatVersion = AbsentDataset  # "s"*
    _nirs = AbsentDataset  # {i}*
    _snirf_names = ['formatVersion', 'nirs', ]

    
    # overload
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
                    self._cfg.logger.info('Loading from file %s', path)
                    self._h = h5py.File(path, 'r+')
                else:
                    self._cfg.logger.info('Creating new file at %s', path)
                    self._h = h5py.File(path, 'w')
            else:
                raise TypeError(str(path) + ' is not a valid filename')
        else:
            self._cfg.logger = _logger
            self._cfg.logger.info('Created Snirf object based on tempfile')
            path = None
            self._h = h5py.File(TemporaryFile(), 'w')
        self._indexed_groups = []
        if 'formatVersion' in self._h:
            if not self._cfg.dynamic_loading:
                self._formatVersion = _read_string(self._h['formatVersion'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._formatVersion = PresentDataset
        else:  # if the dataset is not found on disk
            self._formatVersion = AbsentDataset
        self.nirs = Nirs(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.nirs)
                

    @property
    def formatVersion(self):
        if type(self._formatVersion) is type(AbsentDataset):
            return None
        if type(self._formatVersion) is type(PresentDataset) and 'formatVersion' in self._h:
            return _read_string(self._h['formatVersion'])
            self._cfg.logger.info('Dynamically loaded %s/formatVersion from %s', self.location, self.filename)
        else:
            return self._formatVersion

    @formatVersion.setter
    def formatVersion(self, value):
        self._formatVersion = value
        # self._cfg.logger.info('Assignment to %s/formatVersion in %s', self.location, self.filename)

    @formatVersion.deleter
    def formatVersion(self):
        self._formatVersion = AbsentDataset
        self._cfg.logger.info('Deleted %s/formatVersion from %s', self.location, self.filename)

    @property
    def nirs(self):
        return self._nirs

    @nirs.setter
    def nirs(self, value):
        self._nirs = value
        # self._cfg.logger.info('Assignment to %s/nirs in %s', self.location, self.filename)

    @nirs.deleter
    def nirs(self):
        raise AttributeError('IndexedGroup ' + str(type(self._nirs)) + ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/nirs from %s', self.location, self.filename)


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
            if self.location not in file:
                file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
        else:
            if self.location not in file:
                # Assign the wrapper to the new HDF5 Group on disk
                self._h = file.create_group(self.location)
                # self._cfg.logger.info('Created Group at %s in %s', self.location, file)
            if self._h != {}:
                file = self._h.file
            else:
                raise ValueError('Cannot save an anonymous ' + self.__class__.__name__ + ' instance without a filename')
        name = self.location + '/formatVersion'
        if name in file:
            del file[name]
            self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        data = self.formatVersion
        if type(self._formatVersion) not in [type(AbsentDataset), type(None)]:
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        self.nirs._save(*args)

    def _validate(self, result: ValidationResult):
        name = self.location + '/formatVersion'
        if type(self._formatVersion) in [type(AbsentDataset), type(None)]:
            result._add(name, 'REQUIRED_DATASET_MISSING')
            print(name, 'not found')
        else:
            # Validate unwritten datasets after writing them to this tempfile
            tmp = h5py.File(TemporaryFile(), 'w')
            try:
                if type(self._formatVersion) is type(PresentDataset):
                    dataset = self._h['formatVersion']
                else:
                    dataset = _create_dataset_string(tmp, 'formatVersion', self._formatVersion)
                result._add(name, _validate_string(dataset))
            except ValueError:  # If the _create_dataset function can't convert the data
                result._add(name, 'INVALID_DATASET_TYPE')
            print(dataset)
        name = self.location + '/nirs'
        self.nirs._validate(result)
        for key in self._h.keys():  # TODO attributes too
            if not any([key.startswith(name) for name in self._snirf_names]):
                if type(self._h[key]) is h5py.Group:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_GROUP')
                elif type(self._h[key]) is h5py.Dataset:
                    result._add(self.location + '/' + key, 'UNRECOGNIZED_DATASET')


    # overload
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
            self._cfg.logger.info('Saved Snirf file at %s to copy at %s', self.filename, path)
            new_file.close()
        else:
            self._save(self._h.file)

    def validate(self):
        '''
        Returns True, ValidationResult
        '''
        result = ValidationResult()
        self._validate(result)
        return (result.is_valid(), result)

    def close(self):
        self._cfg.logger.info('Closing Snirf file %s', self.filename)
        self._h.close()

    def __enter__(self):
        return self

#    def __del__(self):
#        self.close()

    def __getitem__(self, key):
        if self._h != {}:
            if key in self._h:
                return self._h[key]
        else:
            return None

# Extend metaDataTags to support addition of new unspecified datasets

class MetaDataTags(MetaDataTags):
    
    def add(self, name, value):
        """
        Add a new tag to the list.
        """
        if type(name) is not str:
            raise ValueError('name must be str, not ' + str(type(name)))
        try:
            setattr(self, name, value)
        except AttributeError as e:
            raise AttributeError("can't set attribute. You cannot set the required metaDataTags fields using add() or use protected attributes of MetaDataTags such as 'location' or 'filename'")
        if name not in self._unspecified_names:
            self._unspecified_names.append(name)
