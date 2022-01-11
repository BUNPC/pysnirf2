# -*- coding: utf-8 -*-
"""Module for reading, writing and validating SNIRF files.

SNIRF files are HDF5 files designed to facilitate the sharing of near-infrared
spectrocopy data. Their specification is defined at https://github.com/fNIRS/snirf.

This library wraps each HDF5 Group and offers a Pythonic interface on lists
of like-Groups which the SNIRF speicification calls "indexed Groups".

Example:
    Load a file::

        >>> from pysnirf2 import Snirf
        >>> s = Snirf(<filename>)

Maintained by the Boston University Neurophotonics Center
"""

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
import colorama
from typing import Tuple

try:
    from pysnirf2.__version__ import __version__ as __version__
except Exception:
    warn('Failed to load pysnirf2 library version')
    __version__ = '0.0.0'


if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')


class SnirfFormatError(Exception):
    """Raised when SNIRF-specific error prevents file from loading properly."""
    pass


# Colored prints for validation output to console
if os.name == 'nt':
    colorama.init()

_printr = lambda x: termcolor.cprint(x, 'red')
_printg = lambda x: termcolor.cprint(x, 'green')
_printb = lambda x: termcolor.cprint(x, 'blue')
_printm = lambda x: termcolor.cprint(x, 'magenta')


_loggers = {}
def _create_logger(name, log_file, level=logging.INFO):
    if name in _loggers.keys():
        return _loggers[name]
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s v%(version)s | %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, {'version': __version__})
    _loggers[name] = logger
    return logger

# Package-wide logger
_logger = _create_logger('pysnirf2', 'pysnirf2.log')


# -- methods to cast data prior to writing to and after reading from h5py interfaces------

_varlen_str_type = h5py.string_dtype(encoding='ascii', length=None)  # Length=None creates HDF5 variable length string
_DTYPE_FLOAT32 = 'f4'
_DTYPE_FLOAT64 = 'f8'
_DTYPE_INT32 = 'i4'
_DTYPE_INT64 = 'i8'
_DTYPE_FIXED_LEN_STR = 'S'  # Not sure how robust this is, but fixed length strings will always at least contain S
_DTYPE_VAR_LEN_STR = 'O'  # Variable length string

_INT_DTYPES = [int, np.int32, np.int64]
_FLOAT_DTYPES = [float, np.float, np.float64]
_STR_DTYPES = [str, np.string_]

# -- Dataset creators  ---------------------------------------


def _create_dataset(file: h5py.File, name: str, data):
    """Saves a variable to an h5py.File on disk as a new Dataset.

    Discerns the type of a given variable and adds it to an h5py File as a
    new dataset with SNIRF compliant formatting.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The variable to save to the Dataset.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

    Raises:
        TypeError: The data could not be mapped to a SNIRF compliant h5py format.
    """
    data = np.array(data)  # Cast to numpy type to identify
    if data.size > 1:
        dtype = data[0].dtype
        print(dtype)
        if any([dtype == t for t in _INT_DTYPES]):  # int
            return _create_dataset_int_array(file, name, data)
        elif any([dtype == t for t in _FLOAT_DTYPES]):  # float
            return _create_dataset_float_array(file, name, data)
        elif any([dtype == t for t in _STR_DTYPES]) or any([t in dtype.str for t in ['U', 'S']]):  # string
            return _create_dataset_string_array(file, name, data)
    dtype = data.dtype
    if any([dtype == t for t in _INT_DTYPES]):  # int
        return _create_dataset_int(file, name, data)
    elif any([dtype == t for t in _FLOAT_DTYPES]):  # float
        return _create_dataset_float(file, name, data)
    elif any([dtype == t for t in _STR_DTYPES]) or any([t in dtype.str for t in ['U', 'S']]):  # string
        return _create_dataset_string(file, name, data)
    raise TypeError("Unrecognized data type '" + str(dtype)
                    + "'. Please provide an int, float, or str, or an iterable of these.")


def _create_dataset_string(file: h5py.File, name: str, data: str):
    """Saves a variable to an h5py.File on disk as a new SNIRF compliant variable length string Dataset.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The string to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    return file.create_dataset(name, dtype=_varlen_str_type, data=str(data))


def _create_dataset_int(file: h5py.File, name: str, data: int):
    """Saves a variable to an h5py.File on disk as a new SNIRF compliant integer Dataset.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The integer to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    return file.create_dataset(name, dtype=_DTYPE_INT32, data=int(data))


def _create_dataset_float(file: h5py.File, name: str, data: float):
    """Saves a variable to an h5py.File on disk as a new SNIRF compliant float Dataset.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The float to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    return file.create_dataset(name, dtype=_DTYPE_FLOAT64, data=float(data))


def _create_dataset_string_array(file: h5py.File, name: str, data: np.ndarray):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of variable length strings.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype('O')
    return file.create_dataset(name, dtype=_varlen_str_type, data=array)


def _create_dataset_int_array(file: h5py.File, name: str, data: np.ndarray):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of 32-bit integers.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype(int)
    return file.create_dataset(name, dtype=_DTYPE_INT32, data=array)


def _create_dataset_float_array(file: h5py.File, name: str, data: np.ndarray):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of 64-bit floats.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype(float)
    return file.create_dataset(name, dtype=_DTYPE_FLOAT64, data=array)


# -- Dataset readers  ---------------------------------------


def _read_dataset(dataset: h5py.Dataset):
    """Converts the contents of an h5py Dataset into a NumPy object.

    Converts the contents of an `h5py.Dataset` into a NumPy object after
    attempting to determine the appropriate SNIRF compliant type.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

    Raises:
        TypeError: The Dataset could not be mapped to a SNIRF compliant type.
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1:
        if _DTYPE_FIXED_LEN_STR in dataset.dtype or _DTYPE_VAR_LEN_STR in dataset.dtype.str:
            return _read_string_array(dataset)
        elif _DTYPE_INT32 in dataset.dtype.str or _DTYPE_INT64 in dataset.dtype.str:
            return _read_int_array(dataset)
        elif _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
            return _read_float_array(dataset)
    else:
        if _DTYPE_FIXED_LEN_STR in dataset.dtype.str or _DTYPE_VAR_LEN_STR in dataset.dtype.str:
            return _read_string(dataset)
        elif _DTYPE_INT32 in dataset.dtype.str or _DTYPE_INT64 in dataset.dtype.str:
            return _read_int(dataset)
        elif _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
            return _read_float(dataset)
    raise TypeError("Dataset dtype='" + str(dataset.dtype)
                    + "' not recognized. Expecting dtype to contain one of these: "
                    + str([_DTYPE_FIXED_LEN_STR, _DTYPE_VAR_LEN_STR, _DTYPE_INT32, _DTYPE_INT64, _DTYPE_FLOAT32, _DTYPE_FLOAT64]))


def _read_string(dataset: h5py.Dataset) -> str:
    """Reads the contents of an `h5py.Dataset` to a `str`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        A `str`
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    # Because many SNIRF files are saved with string values in length 1 arrays
    if dataset.ndim > 0:
        return str(dataset[0].decode('ascii'))
    else:
        return str(dataset[()].decode('ascii'))


def _read_int(dataset: h5py.Dataset) -> int:
    """Reads the contents of an `h5py.Dataset` to an `int`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        An `int`
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return int(dataset[0])
    else:
        return int(dataset[()])


def _read_float(dataset: h5py.Dataset) -> float:
    """Reads the contents of an `h5py.Dataset` to a `float`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        A `float`
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim > 0:
        return float(dataset[0])
    else:
        return float(dataset[()])


def _read_string_array(dataset: h5py.Dataset) -> np.ndarray:
    """Reads the contents of an `h5py.Dataset` to an array of `dtype=str`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        A numpy array `astype(str)`
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(str)


def _read_int_array(dataset: h5py.Dataset) -> np.ndarray:
    """Reads the contents of an `h5py.Dataset` to an array of `dtype=int`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        A numpy array `astype(int)`
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(int)


def _read_float_array(dataset: h5py.Dataset) -> np.ndarray:
    """Reads the contents of an `h5py.Dataset` to an array of `dtype=float`.
    
    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance
    Returns:
        A numpy array astype(float)
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    return np.array(dataset).astype(float)


# -- Validation types ---------------------------------------

_SEVERITY_LEVELS = {
                    0: 'OK     ',
                    1: 'INFO   ',
                    2: 'WARNING',
                    3: 'FATAL  ',
                    }

_SEVERITY_COLORS = {
                    0: 'green',
                    1: 'blue',
                    2: 'magenta',
                    3: 'red',
                    }
_CODES = {
        # Errors (Severity 1)
        'INVALID_FILE_NAME': (1, 3, 'Valid SNIRF files must end with .snirf'),
        'INVALID_FILE': (2, 3, 'The file could not be opened, or the validator crashed'),
        'REQUIRED_DATASET_MISSING': (3, 3, 'A required dataset is missing from the file'),
        'REQUIRED_GROUP_MISSING': (4, 3, 'A required Group is missing from the file'),
        'REQUIRED_INDEXED_GROUP_EMPTY': (5, 3, 'At least one member of the indexed group must be present in the file'),
        'INVALID_DATASET_TYPE': (6, 3, 'An HDF5 Dataset is not stored in the specified format'),
        'INVALID_DATASET_SHAPE': (7, 3, 'An HDF5 Dataset is not stored in the specified shape. Strings and scalars should never be stored as arrays of length 1.'),
        'INVALID_MEASUREMENTLIST': (8, 3, 'The number of measurementList elements does not match the second dimension of dataTimeSeries'),
        'INVALID_TIME': (9, 3, 'The length of the data/time vector does not match the first dimension of data/dataTimeSeries'),
        'INVALID_STIM_DATALABELS': (10, 3, 'The length of stim/dataLabels exceeds the second dimension of stim/data'),
        'INVALID_SOURCE_INDEX': (11, 3, 'measurementList/sourceIndex exceeds length of probe/sourceLabels'),
        'INVALID_DETECTOR_INDEX': (12, 3, 'measurementList/detectorIndex exceeds length of probe/detectorLabels'),
        'INVALID_WAVELENGTH_INDEX': (13, 3, 'measurementList/waveLengthIndex exceeds length of probe/wavelengths'),
        'NEGATIVE_INDEX': (14, 3, 'An index is negative'),
        # Warnings (Severity 2)
        'INDEX_OF_ZERO': (15, 2, 'An index of zero is usually undefined'),
        'UNRECOGNIZED_GROUP': (16, 2, 'An unspecified Group is a part of the file'),
        'UNRECOGNIZED_DATASET': (17, 2, 'An unspecified Dataset is a part of the file in an unexpected place'),
        'UNRECOGNIZED_DATATYPELABEL': (18, 2, 'measurementList/dataTypeLabel is not one of the recognized values listed in the Appendix'),
        'UNRECOGNIZED_DATATYPE': (19, 2, 'measurementList/dataType is not one of the recognized values listed in the Appendix'),
        'INT_64': (25, 2, 'The SNIRF specification limits users to the use of 32 bit native integer types'),
        'FIXED_LENGTH_STRING': (20, 2, 'The use of fixed-length strings is discouraged and may be banned by a future spec version. Rewrite this file with pysnirf2 to use variable length strings'),
        # Info (Severity 1)
        'OPTIONAL_GROUP_MISSING': (21, 1, 'Missing an optional Group in this location'),
        'OPTIONAL_DATASET_MISSING': (22, 1, 'Missing optional Dataset in this location'),
        'OPTIONAL_INDEXED_GROUP_EMPTY': (23, 1, 'The optional indexed group has no elements'),
        # OK (Severity 0)
        'OK': (24, 0, 'No issues detected'),
        }


class ValidationIssue:
    """Information about the validity of a given SNIRF file location.

    Properties:
        location: A relative HDF5 name corresponding to the location of the issue
        name: A string describing the issue. Must be predefined in `_CODES`
        id: An integer corresponding to the predefined error type
        severity: An integer ranking the serverity level of the issue. 
            0 OK, Nothing remarkable
            1 Potentially useful `INFO`
            2 `WARNING`, the file is valid but exhibits undefined behavior or features marked deprecation
            3 `FATAL`, The file is invalid.
        message: A string containing a more verbose description of the issue
    """
    
    def __init__(self, name: str, location: str):
        self.location = location  # A location in the Snirf file matching an HDF5 name
        self.name = name  # The name of the issue, a key in _CODES above
        self.id = _CODES[name][0]  # The ID of the issue
        self.severity = _CODES[name][1]  # The severity level of the issue
        self.message = _CODES[name][2]  # A string describing the issue

    def __repr__(self):
        s = super().__repr__()
        s += '\nlocation: ' + self.location + '\nseverity: '
        s += str(self.severity).ljust(4) + _SEVERITY_LEVELS[self.severity]
        s += '\nname:     ' + str(self.id).ljust(4) + self.name +  '\nmessage:  ' + self.message
        return s


class ValidationResult:
    """The result of Snirf file validation routines.
    
    Validation results in a list of issues. Each issue records information about
    the validity of each location (each named Dataset and Group) in a SNIRF file.
    ValidationResult organizes the issues catalogued during validation and affords interfaces
    to retrieve and display them.

    ```
    (<ValidationResult>.is_valid(), <ValidationResult>) = <Snirf instance>.validate()
    (<ValidationResult>.is_valid(), <ValidationResult>) = validateSnirf(<path>)
    ```
    """

    def __init__(self):
        """`ValidationResult` should only be created by a `Snirf` instance's `validate` method."""
        self._issues = []
        self._locations = []

    def __bool__(self):
        return self.is_valid()

    def is_valid(self) -> bool:
        """Returns True if no `FATAL` issues were catalogued during validation."""
        for issue in self._issues:
            if issue.severity > 2:
                return False
        return True
    
    @property
    def issues(self):
        """A comprehensive list of all `ValidationIssue` instances for the result."""
        return self._issues
    
    @property
    def locations(self):
        """A list of the HDF5 location associated with each issue."""
        return self._locations
    
    @property
    def codes(self):
        """A list of each unique code name associated with all catalogued issues."""
        return list(set([issue.name for issue in self._issues]))
    
    @property
    def errors(self):
        """A list of the `FATAL` issues catalogued during validation."""
        errors = []
        for issue in self._issues:
            if issue.severity == 3:
                errors.append(issue)
        return errors
    
    @property
    def warnings(self):
        """A list of the `WARNING` issues catalogued during validation."""
        warnings = []
        for issue in self._issues:
            if issue.severity == 2:
                warnings.append(issue)
        return warnings
    
    @property
    def info(self):
        """A list of the `INFO` issues catalogued during validation."""
        info = []
        for issue in self._issues:
            if issue.severity == 1:
                info.append(issue)
        return info
        
    def display(self, severity=2):
        """Reads the contents of an `h5py.Dataset` to an array of `dtype=str`.
        
        Args:
            severity: An `int` which sets the minimum severity message to
            display. Default is 2.
                severity=0 All messages will be shown, including `OK`
                severity=1 Prints `INFO`, `WARNING`, and `FATAL` messages
                severity=2 Prints `WARNING` and `FATAL` messages
                severity=3 Prints only `FATAL` error messages
        """
        try:
            longest_key = max([len(key) for key in self.locations])
            longest_code = max([len(code) for code in self.codes])
        except ValueError:
            print('Empty ValidationResult: nothing to display')
        s = object.__repr__(self) + '\n'
        printed = [0, 0, 0, 0]
        for issue in self._issues:
            sev = issue.severity
            printed[sev] += 1
            if sev >= severity:
                s += issue.location.ljust(longest_key) + ' ' + _SEVERITY_LEVELS[sev] + ' ' + issue.name.ljust(longest_code) + '\n'
        print(s)
        for i in range(0, severity):
            [_printg, _printb, _printm, _printr][i]('Found ' + str(printed[i]) + ' ' + termcolor.colored(_SEVERITY_LEVELS[i], _SEVERITY_COLORS[i]) + ' (hidden)')            
        for i in range(severity, 4):
            [_printg, _printb, _printm, _printr][i]('Found ' + str(printed[i]) + ' ' + termcolor.colored(_SEVERITY_LEVELS[i], _SEVERITY_COLORS[i]))
        i = int(self.is_valid())
        [_printr, _printg][i]('\nFile is ' +['INVALID', 'VALID'][i])

    def _add(self, location, key):
        if key not in _CODES.keys():
            raise KeyError("Invalid code '" + key + "'")
        if location not in self:  # only one issue per HDF5 name
            issue = ValidationIssue(key, location)
            self._locations.append(location)
            self._issues.append(issue) 
        
    def __contains__(self, key):
        for issue in self._issues:
            if issue.location == key:
                return True
        return False
        
    def __getitem__(self, key):
        for issue in self._issues:
            if issue.location == key:
                return issue
        raise KeyError("'" + key + "' not in issues list")
            
        
    def __repr__(self):
        return object.__repr__(self) + ' is_valid ' + str(self.is_valid()) 
        
    
# -- Validation functions ---------------------------------------


def _validate_string(dataset: h5py.Dataset) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance..

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
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


def _validate_int(dataset: h5py.Dataset) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1 or dataset.ndim > 0:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_INT32 in dataset.dtype.str:
        return 'OK'
    if _DTYPE_INT64 in dataset.dtype.str:
        return 'INT_64'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_float(dataset: h5py.Dataset) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.size > 1 or dataset.ndim > 0:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'


def _validate_string_array(dataset: h5py.Dataset, ndims=[1]) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
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


def _validate_int_array(dataset: h5py.Dataset, ndims=[1]) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim not in ndims:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_INT32 in dataset.dtype.str:
        return 'OK'
    if _DTYPE_INT64 in dataset.dtype.str:
        return 'INT_64'
    else:
        return 'INVALID_DATASET_TYPE'

def _validate_float_array(dataset: h5py.Dataset, ndims=[1]) -> str:
    """Determines an issue code (as predefined in `_CODES`) based on the contents an `h5py.Dataset` instance.

    Args:
        dataset (h5py.Dataset): An open` h5py.Dataset` instance 

    Returns:
        An issue code describing the validity of the dataset based on its format and shape
    """
    if type(dataset) is not h5py.Dataset:
        raise TypeError("'dataset' must be type h5py.Dataset")
    if dataset.ndim not in ndims:
        return 'INVALID_DATASET_SHAPE'
    if _DTYPE_FLOAT32 in dataset.dtype.str or _DTYPE_FLOAT64 in dataset.dtype.str:
        return 'OK'
    else:
        return 'INVALID_DATASET_TYPE'


# -----------------------------------------


class SnirfConfig:
    """Structure containing Snirf-wide data and settings.
    
    Properties:
        logger (logging.Logger): The logger that the Snirf instance writes to
        dynamic_loading (bool): If True, data is loaded from the HDF5 file only on access via property
    """
    def __init__(self):
        self.logger: logging.Logger = _logger  # The logger that the interface will write to
        self.dynamic_loading: bool = False  # If False, data is loaded in the constructor, if True, data is loaded on access


# Placeholder for a Dataset that is not on disk or in memory
class _AbsentDatasetType():
    pass


# Placeholder for a Group that is not on disk or in memory
class _AbsentGroupType():
    pass


# Placeholder for a Dataset that is available only on disk in a dynamic_loading=True wrapper
class _PresentDatasetType():
    pass


# Instantiate faux singletons
_AbsentDataset = _AbsentDatasetType()
_AbsentGroup = _AbsentGroupType()
_PresentDataset = _PresentDatasetType()


class Group(ABC):

    def __init__(self, varg, cfg: SnirfConfig):
        """Wrapper for an HDF5 Group element defined by SNIRF.

        Base class for an HDF5 Group element defined by SNIRF. Must be created with a
        Group ID or string specifying a complete path relative to file root--in
        the latter case, the wrapper will not correspond to a real HDF5 group on
        disk until `_save()` (with no arguments) is executed for the first time

        Args:
            varg (h5py.h5g.GroupID or str): Either a string which maps to a future Group location or an ID corresponding to a current one on disk
            cfg (SnirfConfig): Injected configuration of parent `Snirf` instance
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
        """Group level save to a SNIRF file on disk.
        
        Args:
            args (str or h5py.File): A path to a closed SNIRF file on disk or an open `h5py.File` instance
    
        Examples:
            save can be called on a Group already on disk to overwrite the current contents:
            >>> mysnirf.nirs[0].probe.save()
            
            or using a new filename to write the Group there:
            >>> mysnirf.nirs[0].probe.save(<new destination>)
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
        """The filename the Snirf object was loaded from and will save to.
        
        None if not associated with a Group on disk.        
        """
        if self._h != {}:
            return self._h.file.filename
        else:
            return None

    @property
    def location(self):
        """The HDF5 relative location indentifier.
        
        None if not associataed with a Group on disk.
        """
        if self._h != {}:
            return self._h.name
        else:
            return self._location

    def is_empty(self):
        """If the Group has no member Groups or Datasets.
        
        Returns:
            bool: True if empty, False if not
        """
        for name in self._snirf_names:
            attr = getattr(self, '_' + name)
            if isinstance(attr, Group) or isinstance(attr, IndexedGroup):
                if not attr.is_empty():
                    return False
            else:
                if not any([attr is a for a in [None, _AbsentGroup, _AbsentDataset]]):
                    return False
        return True

    @abstractmethod
    def _save(self, *args):
        raise NotImplementedError('_save is an abstract method')

    @abstractmethod
    def _validate(self, result: ValidationResult):
        raise NotImplementedError('_validate is an abstract method')

    def __repr__(self):
        props = [p for p in dir(self) if (not p.startswith('_') and not callable(getattr(self, p)))]
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

    def __contains__(self, key):
        return key in self._h


class IndexedGroup(MutableSequence, ABC):

    _name: str = ''  # The specified prefix to this indexed group's members, i.e. nirs, data, stim, aux, measurementList
    _element: Group = None  # The type of Group which belongs to this IndexedGroup

    def __init__(self, parent: Group, cfg: SnirfConfig):
        """Represents several Groups which share a name, an "indexed group".
        
        Represents the "indexed group" which is defined by v1.0 of the SNIRF
        specification as:
            If a data element is an HDF5 group and contains multiple sub-groups,
            it is referred to as an indexed group. Each element of the sub-group
            is uniquely identified by appending a string-formatted index (starting
            from 1, with no preceding zeros) in the name, for example, /.../name1
            denotes the first sub-group of data element name, and /.../name2
            denotes the 2nd element, and so on.
        
        Because the indexed group is not a true HDF5 group but rather an
        iterable list of HDF5 groups, it takes a base group or file and
        searches its keys, appending the appropriate elements to itself
        in order.
        
        The appropriate elements are identified using the `_name` attribute: if
        a key begins with `_name` and ends with a number, or is equal to `_name`.

        Args:
            parent (h5py.h5g.Group): An HDF5 group which is the parent of the indexed groups
            cfg (SnirfConfig): Injected configuration of parent `Snirf` instance

        """

        self._parent = parent
        self._cfg = cfg
        self._populate_list()
        self._cfg.logger.info('IndexedGroup %s at %s in %s initalized with %i instances of %s', self.__class__.__name__,
                              self._parent.location, self.filename, len(self._list), self._element)

    @property
    def filename(self):
        """The filename the Snirf object was loaded from and will save to."""
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
        """Returns True if the Indexed Group has no member Groups with contents.
        
        Returns:
            bool: True if empty, False if not
        """
        if len(self._list) > 0:
            for e in self._list:
                if not e.is_empty():
                    return False
        return True

    def insert(self, i, item):
        """Insert a new Group into the IndexedGroup.
        
        Args:
            i (int): an index
            item: must be of type _element
        """
        self._check_type(item)
        self._list.insert(i, item)
        self._cfg.logger.info('%i th element inserted into IndexedGroup %s at %s in %s at %i', len(self._list),
                              self.__class__.__name__, self._parent.location, self.filename, i)

    def append(self, item):
        """Append a new Group to the IndexedGroup.
        
        Args:
            item: must be of type _element
        """
        self._check_type(item)
        self._list.append(item)
        self._cfg.logger.info('%i th element appended to IndexedGroup %s at %s in %s', len(self._list),
                              self.__class__.__name__, self._parent.location, self.filename)

    def save(self, *args):
        """Save the groups to a SNIRF file on disk.
        
        When saving, the naming convention defined by the SNIRF spec is enforced:
        groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on.
        
        Args:
            args (str or h5py.File): A path to a closed SNIRF file on disk or an open `h5py.File` instance
    
        Examples:
            save can be called on an Indexed Group already on disk to overwrite the current contents:
            >>> mysnirf.nirs[0].stim.save()
            
            or using a new filename to write the Indexed Group there:
            >>> mysnirf.nirs[0].stim.save(<new destination>)
        """
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
        """Insert a new Group at the end of the Indexed Group.
        
        Creates an empty Group with the appropriate name at the end of the 
        list of Groups managed by the IndexedGroup.
        """
        location = self._parent.location + '/' + self._name + str(len(self._list) + 1)
        self._list.append(self._element(location, self._cfg))
        self._cfg.logger.info('%i th %s appended to IndexedGroup %s at %s in %s', len(self._list),
                          self._element, self.__class__.__name__, self._parent.location, self.filename)

    def insertGroup(self, i):
        """Insert a new Group following the index given.
        
        Creates an empty Group with a placeholder name within the list of Groups
        managed by the IndexedGroup. The placeholder name will be replaced with a
        name with the correct order once `save` is called.
        
        Args:
            i (int): the position at which to insert the new Group
        """
        location = self._parent.location + '/' + self._name + '0' + str(i) + 1
        self._list.append(self._element(location, self._cfg))
        self._cfg.logger.info('%i th %s appended to IndexedGroup %s at %s in %s', len(self._list),
                          self._element, self.__class__.__name__, self._parent.location, self.filename)

    def _populate_list(self):
        """Add all the appropriate groups found in parent's HDF5 keys to the list."""
        self._list = list()
        names = self._get_matching_keys()
        for name in names:
            if name in self._parent._h:
                self._list.append(self._element(self._parent._h[name].id, self._cfg))

    def _check_type(self, item):
        """Raise TypeError if an item does not match `_element`."""
        if type(item) is not self._element:
            raise TypeError('elements of ' + str(self.__class__.__name__) +
                            ' must be ' + str(self._element) + ', not ' +
                            str(type(item))
                            )

    def _order_names(self, h=None):
        """Renumber (rename) the HDF5 Groups in the wrapper and on disk such that they ascend in order.
        
        Enforce the format of the names of HDF5 groups within a group or file on disk. i.e. `IndexedGroup` `stim`'s elements
        will be renamed, in order, /stim1, /stim2, /stim3. This is expensive but can be avoided by `save()`ing individual groups
        within the IndexedGroup
        
        Args:
            h (`h5py.File` or `h5py.Group`): if supplied, the rename will be carried out on the given h5py wrapper. For copying.
        """
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
        """Return sorted list of a group or file's keys which match this `IndexedList`'s _name format."""
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
            elif key.endswith(self._name):  # Case of single Group with no index
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
