# -*- coding: utf-8 -*-
"""Module for reading, writing and validating SNIRF files.

SNIRF files are HDF5 files designed to facilitate the sharing of near-infrared
spectrocopy data. Their specification is defined at https://github.com/fNIRS/snirf.

This library wraps each HDF5 Group and offers a Pythonic interface on lists
of like-Groups which the SNIRF specification calls "indexed Groups".

Example:
    Load a file:

        >>> from snirf import Snirf
        >>> with Snirf(<filename>) as s:
            ...

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
from typing import Tuple
import time
import io
import json
import copy

try:
    from snirf.__version__ import __version__ as __version__
except Exception:
    warn('Failed to load pysnirf2 library version')
    __version__ = '0.0.0'

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')


class SnirfFormatError(Warning):
    """Raised when SNIRF-specific error prevents file from loading properly."""
    pass


# Colored prints for validation output to console
try:
    import termcolor
    import colorama

    if os.name == 'nt':
        colorama.init()

    _printr = lambda x: termcolor.cprint(x, 'red')
    _printg = lambda x: termcolor.cprint(x, 'green')
    _printb = lambda x: termcolor.cprint(x, 'blue')
    _printm = lambda x: termcolor.cprint(x, 'magenta')
    _colored = termcolor.colored

except ImportError:
    _printr = lambda x: print(x)
    _printg = lambda x: print(x)
    _printb = lambda x: print(x)
    _printm = lambda x: print(x)
    _colored = lambda x, c: x


def _isfilelike(o: object) -> bool:
    """Returns True if object is an instance of a file-like object like `io.IOBase` or `io.BufferedIOBase`."""
    return any([
        isinstance(o, io.TextIOBase),
        isinstance(o, io.BufferedIOBase),
        isinstance(o, io.RawIOBase),
        isinstance(o, io.IOBase)
    ])


_loggers = {}


def _create_logger(name, log_file, level=logging.INFO):
    if name in _loggers.keys():
        return _loggers[name]
    if log_file == '' or log_file is None:
        handler = logging.NullHandler()
    else:
        handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter('%(asctime)s | %(name)s v%(version)s | %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, {'version': __version__})
    _loggers[name] = logger
    return logger


def _close_logger(logger: logging.LoggerAdapter):
    if type(logger) is logging.LoggerAdapter:
        handlers = logger.logger.handlers[:]
    elif type(logger) is logging.Logger:
        handlers = logger.handlers[:]
    else:
        raise TypeError(
            'logger must be logging.LoggerAdapter or logging.Logger')
    for handler in handlers:
        handler.close()


# Package-wide logger
_logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pysnirf2.log')

if os.path.exists(_logfile):
    try:
        if (time.time() - os.path.getctime(_logfile)
            ) / 86400 > 10:  # Keep logs for only 10 days
            os.remove(_logfile)
        _logger = _create_logger('pysnirf2', _logfile)
    except (FileNotFoundError, PermissionError):
        _logger = _create_logger('pysnirf2', None)  # Null logger
else:
    _logger = _create_logger('pysnirf2',
                             os.path.join(os.getcwd(), 'pysnirf2.log'))
_logger.info('Library loaded by process {}'.format(os.getpid()))

# -- methods to cast data prior to writing to and after reading from h5py interfaces------

_varlen_str_type = h5py.string_dtype(
    encoding='ascii',
    length=None)  # Length=None creates HDF5 variable length string
_DTYPE_FLOAT32 = 'f4'
_DTYPE_FLOAT64 = 'f8'
_DTYPE_INT32 = 'i4'
_DTYPE_INT64 = 'i8'
_DTYPE_FIXED_LEN_STR = 'S'  # Not sure how robust this is, but fixed length strings will always at least contain S
_DTYPE_VAR_LEN_STR = 'O'  # Variable length string

_INT_DTYPES = [int, np.int32, np.int64]
_FLOAT_DTYPES = [float, np.float64]
_STR_DTYPES = [str, np.string_]

# -- Dataset creators  ---------------------------------------


def _get_padded_shape(name: str, data: np.ndarray,
                      desired_ndim: int) -> np.ndarray:
    """Utility function which pads data shape to ndim."""
    if desired_ndim is None:
        return data.shape
    elif desired_ndim > data.ndim:
        return np.concatenate(
            [data.shape,
             np.ones(int(desired_ndim) - int(data.ndim))])
    elif data.ndim == desired_ndim:
        return np.shape(data)
    else:
        raise ValueError(
            "Could not create dataset {}: ndim={} is incompatible with data which has shape {}."
            .format(name, desired_ndim, data.shape))


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
        elif any([dtype == t for t in _STR_DTYPES]) or any(
            [t in dtype.str for t in ['U', 'S']]):  # string
            return _create_dataset_string_array(file, name, data)
    dtype = data.dtype
    if any([dtype == t for t in _INT_DTYPES]):  # int
        return _create_dataset_int(file, name, data)
    elif any([dtype == t for t in _FLOAT_DTYPES]):  # float
        return _create_dataset_float(file, name, data)
    elif any([dtype == t for t in _STR_DTYPES]) or any(
        [t in dtype.str for t in ['U', 'S']]):  # string
        return _create_dataset_string(file, name, data)
    raise TypeError(
        "Unrecognized data type '" + str(dtype) +
        "'. Please provide an int, float, or str, or an iterable of these.")


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


def _create_dataset_string_array(file: h5py.File,
                                 name: str,
                                 data: np.ndarray,
                                 ndim=None):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of variable length strings.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype('O')
    shape = _get_padded_shape(name, array, ndim)
    return file.create_dataset(name, dtype=_varlen_str_type, data=array)


def _create_dataset_int_array(file: h5py.File,
                              name: str,
                              data: np.ndarray,
                              ndim=None):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of 32-bit integers.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype(int)
    shape = _get_padded_shape(name, array, ndim)
    return file.create_dataset(name, dtype=_DTYPE_INT32, data=array)


def _create_dataset_float_array(file: h5py.File,
                                name: str,
                                data: np.ndarray,
                                ndim=None):
    """Saves a NumPy array to an h5py.File on disk as a new SNIRF compliant array of 64-bit floats.

    Args:
        file: An open `h5py.File` or `h5py.Group` instance to which the Dataset will be added
        name (str): The name of the new dataset. Can be a relative HDF5 name.
        data: The array to save to the Dataset.

    Returns:
        An h5py.Dataset instance created
    """
    array = np.array(data).astype(float)
    shape = _get_padded_shape(name, array, ndim)
    return file.create_dataset(name,
                               dtype=_DTYPE_FLOAT64,
                               shape=shape,
                               data=array)


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
        if _DTYPE_FIXED_LEN_STR in dataset.dtype.str or _DTYPE_VAR_LEN_STR in dataset.dtype.str:
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
    raise TypeError(
        "Dataset dtype='" + str(dataset.dtype) +
        "' not recognized. Expecting dtype to contain one of these: " + str([
            _DTYPE_FIXED_LEN_STR, _DTYPE_VAR_LEN_STR, _DTYPE_INT32,
            _DTYPE_INT64, _DTYPE_FLOAT32, _DTYPE_FLOAT64
        ]))


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
    try:
        if dataset.ndim > 0:
            return str(dataset[0].decode('ascii'))
        else:
            return str(dataset[()].decode('ascii'))
    except AttributeError:  # If we expected a string and got something else, `decode` isn't there
        warn(
            'Expected dataset {} to be stringlike, is {} conversion may be incorrect'
            .format(dataset.name, dataset.dtype), SnirfFormatError)
        return str(dataset[0])


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
    'INVALID_FILE': (2, 3,
                     'The file could not be opened, or the validator crashed'),
    'REQUIRED_DATASET_MISSING':
    (3, 3, 'A required dataset is missing from the file'),
    'REQUIRED_GROUP_MISSING': (4, 3,
                               'A required Group is missing from the file'),
    'REQUIRED_INDEXED_GROUP_EMPTY':
    (5, 3,
     'At least one member of the indexed group must be present in the file'),
    'INVALID_DATASET_TYPE':
    (6, 3, 'An HDF5 Dataset is not stored in the specified format'),
    'INVALID_DATASET_SHAPE':
    (7, 3,
     'An HDF5 Dataset is not stored in the specified shape. Strings and scalars should never be stored as arrays of length 1.'
     ),
    'INVALID_MEASUREMENTLIST':
    (8, 3,
     'The number of measurementList elements does not match the second dimension of dataTimeSeries'
     ),
    'INVALID_TIME':
    (9, 3,
     'The length of the data/time vector does not match the first dimension of data/dataTimeSeries'
     ),
    'INVALID_STIM_DATALABELS':
    (10, 3,
     'The length of stim/dataLabels exceeds the second dimension of stim/data'
     ),
    'INVALID_SOURCE_INDEX':
    (11, 3,
     'measurementList/sourceIndex exceeds length of probe/sourceLabels'),
    'INVALID_DETECTOR_INDEX':
    (12, 3,
     'measurementList/detectorIndex exceeds length of probe/detectorLabels'),
    'INVALID_WAVELENGTH_INDEX':
    (13, 3,
     'measurementList/waveLengthIndex exceeds length of probe/wavelengths'),
    'NEGATIVE_INDEX': (14, 3, 'An index is negative'),
    # Warnings (Severity 2)
    'INDEX_OF_ZERO': (15, 2, 'An index of zero is usually undefined'),
    'UNRECOGNIZED_GROUP': (16, 2,
                           'An unspecified Group is a part of the file'),
    'UNRECOGNIZED_DATASET':
    (17, 2,
     'An unspecified Dataset is a part of the file in an unexpected place'),
    'UNRECOGNIZED_DATATYPELABEL':
    (18, 2,
     'measurementList/dataTypeLabel is not one of the recognized values listed in the Appendix'
     ),
    'UNRECOGNIZED_DATATYPE':
    (19, 2,
     'measurementList/dataType is not one of the recognized values listed in the Appendix'
     ),
    'INT_64':
    (25, 2,
     'The SNIRF specification limits users to the use of 32 bit native integer types'
     ),
    'UNRECOGNIZED_COORDINATE_SYSTEM':
    (26, 2,
     'The identifying string of the coordinate system was not recognized.'),
    'NO_COORDINATE_SYSTEM_DESCRIPTION':
    (27, 2,
     "The coordinate system was unrecognized or 'Other' but lacks a probe/coordinateSystemDescription"
     ),
    'FIXED_LENGTH_STRING':
    (20, 2,
     'The use of fixed-length strings is discouraged and may be banned by a future spec version. Rewrite this file with pysnirf2 to use variable length strings'
     ),
    # Info (Severity 1)
    'OPTIONAL_GROUP_MISSING': (21, 1,
                               'Missing an optional Group in this location'),
    'OPTIONAL_DATASET_MISSING': (22, 1,
                                 'Missing optional Dataset in this location'),
    'OPTIONAL_INDEXED_GROUP_EMPTY':
    (23, 1, 'The optional indexed group has no elements'),
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
        s += '\nname:     ' + str(
            self.id).ljust(4) + self.name + '\nmessage:  ' + self.message
        return s

    def dictize(self):
        """Return dictionary representation of Issue."""
        return {
            'location': self.location,
            'name': self.name,
            'id': self.id,
            'severity': self.severity,
            'message': self.message
        }


class ValidationResult:
    """The result of Snirf file validation routines.

    Validation results in a list of issues. Each issue records information about
    the validity of each location (each named Dataset and Group) in a SNIRF file.
    ValidationResult organizes the issues catalogued during validation and affords interfaces
    to retrieve and display them.

    ```
    <ValidationResult> = <Snirf instance>.validate()
     <ValidationResult> = validateSnirf(<path>)
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

    def serialize(self, indent=4):
        """Render serialized JSON ValidationResult."""
        d = {}
        for issue in self._issues:
            d[issue.location] = issue.dictize()
        return json.dumps(d, indent=indent)

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
                s += issue.location.ljust(
                    longest_key) + ' ' + _SEVERITY_LEVELS[
                        sev] + ' ' + issue.name.ljust(longest_code) + '\n'
        print(s)
        for i in range(0, severity):
            [_printg, _printb, _printm,
             _printr][i]('Found ' + str(printed[i]) + ' ' +
                         _colored(_SEVERITY_LEVELS[i], _SEVERITY_COLORS[i]) +
                         ' (hidden)')
        for i in range(severity, 4):
            [_printg, _printb, _printm,
             _printr][i]('Found ' + str(printed[i]) + ' ' +
                         _colored(_SEVERITY_LEVELS[i], _SEVERITY_COLORS[i]))
        i = int(self.is_valid())
        [_printr, _printg][i]('\nFile is ' + ['INVALID', 'VALID'][i])

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
        self.fmode: str = 'w'  # 'w' or 'r', mode to open HDF5 file with


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
        if type(
                varg
        ) is str:  # If a Group wrapper is created prior to a save to HDF Group object
            self._h = {}
            self._location = varg
        elif isinstance(varg, h5py.h5g.GroupID
                        ):  # If Group is created based on an HDF Group object
            self._h = h5py.Group(varg)
            self._location = self._h.name
        else:
            raise TypeError('must initialize ' + self.__class__.__name__ +
                            ' with a Group ID or string, not ' +
                            str(type(varg)))

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
                self._cfg.logger.info('Group-level save of %s in %s',
                                      self.location, self.filename)
                self._save(args[0])
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError(
                        "No such SNIRF file '" + path +
                        "'. Create a SNIRF file before attempting to save a Group to it."
                    )
                self._cfg.logger.info(
                    'Group-level save of %s in %s to new file %s',
                    self.location, self.filename, file)
                self._save(file)
                file.close()
            elif _isfilelike(args[0]):
                self._cfg.logger.info(
                    'Group-level write of %s in %s to filelike object',
                    self.location, self.filename)
                file = h5py.File(args[0], 'w')
                self._save(file)
        else:
            if self._h != {}:
                file = self._h.file
                self._save(file)
                self._cfg.logger.info(
                    'IndexedGroup-level save of %s at %s in %s',
                    self.__class__.__name__, self._parent.location,
                    self.filename)
            else:
                raise ValueError(
                    'File not saved. No file linked to {} instance. Call save with arguments to write to a file.'
                    .format(self.__class__.__name__))

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
            val = getattr(self, '_' + name)
            if isinstance(val, Group) or isinstance(val, IndexedGroup):
                if not val.is_empty():
                    return False
            else:
                if not any(
                    [val is a for a in [None, _AbsentGroup, _AbsentDataset]]):
                    return False
        return True

    @abstractmethod
    def _save(self, *args):
        raise NotImplementedError('_save is an abstract method')

    @abstractmethod
    def _validate(self, result: ValidationResult):
        raise NotImplementedError('_validate is an abstract method')

    def __repr__(self):
        props = [
            p for p in dir(self)
            if (not p.startswith('_') and not callable(getattr(self, p)))
        ]
        out = str(self.__class__.__name__) + ' at ' + str(self.location) + '\n'
        for prop in props:
            val = getattr(self, prop)
            out += prop + ': '
            if type(val) is np.ndarray or type(val) is list:
                if np.size(val) > 32:
                    out += '<' + str(np.shape(val)) + ' array of ' + str(
                        val.dtype) + '>'
                else:
                    out += str(val)
            else:
                prepr = str(val)
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
        self._cfg.logger.info(
            'IndexedGroup %s at %s in %s initalized with %i instances of %s',
            self.__class__.__name__, self._parent.location, self.filename,
            len(self._list), self._element)

    @property
    def filename(self):
        """The filename the Snirf object was loaded from and will save to."""
        return self._parent.filename

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __delitem__(self, i):
        del self._list[i]

    def __setitem__(self, i, item):
        self._check_type(item)
        self._list[i] = _recursive_hdf5_copy(self._list[i], item)

    def __getattr__(self, name):
        # If user tries to access an element's properties, raise informative exception
        if name in [
                p for p in dir(self._element)
                if ('_' not in p and not callable(getattr(self._element, p)))
        ]:
            raise AttributeError(self.__class__.__name__ +
                                 ' is an interable list of ' + str(len(self)) +
                                 ' ' + str(self._element) +
                                 ', access these with an index i.e. ' +
                                 str(self._name) + '[0].' + name)

    def __repr__(self):
        return str('<' + 'iterable of ' + str(len(self._list)) + ' ' +
                   str(self._element) + '>')

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
        self.insertGroup(i)
        self._list[i] = _recursive_hdf5_copy(self._list[i], item)
        self._cfg.logger.info(
            '%i th element inserted into IndexedGroup %s at %s in %s at %i',
            len(self._list), self.__class__.__name__, self._parent.location,
            self.filename, i)

    def append(self, item):
        """Append a new Group to the IndexedGroup.

        Args:
            item: must be of type _element
        """
        self._check_type(item)
        self.appendGroup()
        self._list[-1] = _recursive_hdf5_copy(self._list[-1], item)
        self._cfg.logger.info(
            '%i th element appended to IndexedGroup %s at %s in %s',
            len(self._list), self.__class__.__name__, self._parent.location,
            self.filename)

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
                self._cfg.logger.info(
                    'IndexedGroup-level save of %s at %s in %s',
                    self.__class__.__name__, self._parent.location,
                    self.filename)
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path.replace('.', '')
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'w')
                else:
                    raise FileNotFoundError(
                        "No such SNIRF file '" + path +
                        "'. Create a SNIRF file before attempting to save an IndexedGroup to it."
                    )
                self._cfg.logger.info(
                    'IndexedGroup-level save of %s at %s in %s to %s',
                    self.__class__.__name__, self._parent.location,
                    self.filename, file)
                self._save(file)
                file.close()
        else:
            if self._parent._h != {}:
                file = self._parent._h.file
                self._save(file)
                self._cfg.logger.info(
                    'IndexedGroup-level save of %s at %s in %s',
                    self.__class__.__name__, self._parent.location,
                    self.filename)
            else:
                raise ValueError(
                    'File not saved. No file linked to {} instance. Call save with arguments to write to a file.'
                    .format(self.__class__.__name__))

    def appendGroup(self):
        """Insert a new Group at the end of the Indexed Group.

        Creates an empty Group with the appropriate name at the end of the
        list of Groups managed by the IndexedGroup.
        """
        location = self._parent.location + '/' + self._name + str(
            len(self._list) + 1)
        self._list.append(self._element(location, self._cfg))
        self._cfg.logger.info(
            '%i th %s appended to IndexedGroup %s at %s in %s',
            len(self._list), self._element, self.__class__.__name__,
            self._parent.location, self.filename)

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
        self._cfg.logger.info(
            '%i th %s appended to IndexedGroup %s at %s in %s',
            len(self._list), self._element, self.__class__.__name__,
            self._parent.location, self.filename)

    def _populate_list(self):
        """Add all the appropriate groups found in parent's HDF5 keys to the list."""
        self._list = list()
        names = self._get_matching_keys()
        for name in names:
            if name in self._parent._h:
                self._list.append(
                    self._element(self._parent._h[name].id, self._cfg))

    def _check_type(self, item):
        """Raise TypeError if an item does not match `_element`."""
        if type(item) is not self._element:
            raise TypeError('elements of ' + str(self.__class__.__name__) +
                            ' must be ' + str(self._element) + ', not ' +
                            str(type(item)))

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
        if len(self._list) == 1 and self._name == 'nirs':
            e = self._list[0]
            indexstr = e.location.split('/' + self._name)[-1]
            if len(indexstr) > 0:  # Rename the element
                h.move(e.location,
                       '/'.join(e.location.split('/')[:-1]) + '/' + self._name)
                self._cfg.logger.info(
                    e.location, '--->',
                    '/'.join(e.location.split('/')[:-1]) + '/' + self._name)
        elif all([
                len(e.location.split('/' + self._name)[-1]) > 0
                for e in self._list
        ]):
            if not [
                    int(e.location.split('/' + self._name)[-1])
                    for e in self._list
            ] == list(range(1,
                            len(self._list) + 1)):
                self._cfg.logger.info('renaming elements of IndexedGroup ' +
                                      self.__class__.__name__ + ' at ' +
                                      self._parent.location + ' in ' +
                                      self.filename +
                                      ' to agree with naming format')
                # if list is not already ordered propertly
                for i, e in enumerate(self._list):
                    # To avoid assignment to an existing name, move all
                    h.move(
                        e.location, '/'.join(e.location.split('/')[:-1]) +
                        '/' + self._name + str(i + 1) + '_tmp')
                    self._cfg.logger.info(
                        e.location, '--->',
                        '/'.join(e.location.split('/')[:-1]) + '/' +
                        self._name + str(i + 1) + '_tmp')
                for i, e in enumerate(self._list):
                    h.move(
                        '/'.join(e.location.split('/')[:-1]) + '/' +
                        self._name + str(i + 1) + '_tmp',
                        '/'.join(e.location.split('/')[:-1]) + '/' +
                        self._name + str(i + 1))
                    self._cfg.logger.info(
                        '/'.join(e.location.split('/')[:-1]) + '/' +
                        self._name + str(i + 1) + '_tmp', '--->',
                        '/'.join(e.location.split('/')[:-1]) + '/' +
                        self._name + str(i + 1))

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
            elif key.endswith(
                    self._name):  # Case of single Group with no index
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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ + ' instance')
        names_in_file = self._get_matching_keys(
            h=h)  # List of all names in the file on disk
        names_to_save = [e.location.split('/')[-1]
                         for e in self._list]  # List of names in the wrapper
        # Remove groups which remain on disk after being removed from the wrapper
        for name in names_in_file:
            if name not in names_to_save:
                del h[self._parent.name + '/' +
                      name]  # Remove the actual data from the hdf5 file.
        for e in self._list:
            e._save(*args)  # Group save functions handle the write to disk
        self._order_names(h=h)  # Enforce order in the group names


def _recursive_hdf5_copy(g_dst: Group, g_src: Group):
    """Copy a Group to a new Group, modifying the h5py interfaces accordingly."""
    for sub_src, name in [(getattr(g_src, name), name)
                          for name in g_src._snirf_names]:
        if isinstance(getattr(g_src, name), Group):
            setattr(g_dst, name,
                    sub_src)  # Recursion is continued in the setter
        elif isinstance(getattr(g_src, name), IndexedGroup):
            getattr(g_dst, name)._list.clear(
            )  # Delete entire list and replace it. IndexedGroup methods continue the recursion
            for e in sub_src:
                getattr(g_dst, name).append(e)
        else:  # Other datasets
            setattr(g_dst, name, sub_src)
        if hasattr(g_src, '_unspecified_names'):
            for sub_src_unspec, name in [(getattr(g_src, name), name)
                                         for name in g_src._unspecified_names]:
                g_dst.add(
                    name, sub_src_unspec
                )  # If a Group hasattr _unspecified names, it should have add
    return g_dst


# generated by sstucker on 2022-08-04
# version v1.1 SNIRF specification parsed from https://raw.githubusercontent.com/fNIRS/snirf/v1.1/snirf_specification.md


class MetaDataTags(Group):
    """Wrapper for Group of type `metaDataTags`.

    The `metaDataTags` group contains the metadata associated with the measurements.
    Each metadata record is represented as a dataset under this group - with the name of
    the record, i.e. the key, as the dataset's name, and the value of the record as the 
    actual data stored in the dataset. Each metadata record can potentially have different 
    data types. Sub-groups should not be used to organize metadata records: a member of the `metaDataTags` Group must be a Dataset.

    The below five metadata records are minimally required in a SNIRF file

    """
    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self._SubjectID = _AbsentDataset  # "s"*
        self._MeasurementDate = _AbsentDataset  # "s"*
        self._MeasurementTime = _AbsentDataset  # "s"*
        self._LengthUnit = _AbsentDataset  # "s"*
        self._TimeUnit = _AbsentDataset  # "s"*
        self._FrequencyUnit = _AbsentDataset  # "s"*
        self._snirf_names = [
            'SubjectID',
            'MeasurementDate',
            'MeasurementTime',
            'LengthUnit',
            'TimeUnit',
            'FrequencyUnit',
        ]

        self._indexed_groups = []
        if 'SubjectID' in self._h:
            if not self._cfg.dynamic_loading:
                self._SubjectID = _read_string(self._h['SubjectID'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._SubjectID = _PresentDataset
        else:  # if the dataset is not found on disk
            self._SubjectID = _AbsentDataset
        if 'MeasurementDate' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementDate = _read_string(
                    self._h['MeasurementDate'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._MeasurementDate = _PresentDataset
        else:  # if the dataset is not found on disk
            self._MeasurementDate = _AbsentDataset
        if 'MeasurementTime' in self._h:
            if not self._cfg.dynamic_loading:
                self._MeasurementTime = _read_string(
                    self._h['MeasurementTime'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._MeasurementTime = _PresentDataset
        else:  # if the dataset is not found on disk
            self._MeasurementTime = _AbsentDataset
        if 'LengthUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._LengthUnit = _read_string(self._h['LengthUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._LengthUnit = _PresentDataset
        else:  # if the dataset is not found on disk
            self._LengthUnit = _AbsentDataset
        if 'TimeUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._TimeUnit = _read_string(self._h['TimeUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._TimeUnit = _PresentDataset
        else:  # if the dataset is not found on disk
            self._TimeUnit = _AbsentDataset
        if 'FrequencyUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._FrequencyUnit = _read_string(self._h['FrequencyUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._FrequencyUnit = _PresentDataset
        else:  # if the dataset is not found on disk
            self._FrequencyUnit = _AbsentDataset
        self._unspecified_names = []
        # Unspecified datasets are not properties and unaffected by dynamic_loading
        for key in self._h.keys():
            # If the name isn't specified
            if key not in self._snirf_names and not any([
                    key in indexed_group
                    for indexed_group in self._indexed_groups
            ]):
                self.__dict__[key] = _read_dataset(self._h[key])
                self._unspecified_names.append(key)

    @property
    def SubjectID(self):
        """SNIRF field `SubjectID`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the string-valued ID of the study subject or experiment.

        """
        if type(self._SubjectID) is type(_AbsentDataset):
            return None
        if type(self._SubjectID) is type(_PresentDataset):
            return _read_string(self._h['SubjectID'])
            self._cfg.logger.info('Dynamically loaded %s/SubjectID from %s',
                                  self.location, self.filename)
        return self._SubjectID

    @SubjectID.setter
    def SubjectID(self, value):
        self._SubjectID = value
        # self._cfg.logger.info('Assignment to %s/SubjectID in %s', self.location, self.filename)

    @SubjectID.deleter
    def SubjectID(self):
        self._SubjectID = _AbsentDataset
        self._cfg.logger.info('Deleted %s/SubjectID from %s', self.location,
                              self.filename)

    @property
    def MeasurementDate(self):
        """SNIRF field `MeasurementDate`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the date of the measurement as a string. The format of the date
        string must either be `"unknown"`, or follow the ISO 8601 date string format `YYYY-MM-DD`, where
        - `YYYY` is the 4-digit year
        - `MM` is the 2-digit month (padding zero if a single digit)
        - `DD` is the 2-digit date (padding zero if a single digit)

        """
        if type(self._MeasurementDate) is type(_AbsentDataset):
            return None
        if type(self._MeasurementDate) is type(_PresentDataset):
            return _read_string(self._h['MeasurementDate'])
            self._cfg.logger.info(
                'Dynamically loaded %s/MeasurementDate from %s', self.location,
                self.filename)
        return self._MeasurementDate

    @MeasurementDate.setter
    def MeasurementDate(self, value):
        self._MeasurementDate = value
        # self._cfg.logger.info('Assignment to %s/MeasurementDate in %s', self.location, self.filename)

    @MeasurementDate.deleter
    def MeasurementDate(self):
        self._MeasurementDate = _AbsentDataset
        self._cfg.logger.info('Deleted %s/MeasurementDate from %s',
                              self.location, self.filename)

    @property
    def MeasurementTime(self):
        """SNIRF field `MeasurementTime`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the time of the measurement as a string. The format of the time
        string must either be `"unknown"` or follow the ISO 8601 time string format `hh:mm:ss.sTZD`, where
        - `hh` is the 2-digit hour
        - `mm` is the 2-digit minute
        - `ss` is the 2-digit second
        - `.s` is 1 or more digit representing a decimal fraction of a second (optional)
        - `TZD` is the time zone designator (`Z` or `+hh:mm` or `-hh:mm`)

        """
        if type(self._MeasurementTime) is type(_AbsentDataset):
            return None
        if type(self._MeasurementTime) is type(_PresentDataset):
            return _read_string(self._h['MeasurementTime'])
            self._cfg.logger.info(
                'Dynamically loaded %s/MeasurementTime from %s', self.location,
                self.filename)
        return self._MeasurementTime

    @MeasurementTime.setter
    def MeasurementTime(self, value):
        self._MeasurementTime = value
        # self._cfg.logger.info('Assignment to %s/MeasurementTime in %s', self.location, self.filename)

    @MeasurementTime.deleter
    def MeasurementTime(self):
        self._MeasurementTime = _AbsentDataset
        self._cfg.logger.info('Deleted %s/MeasurementTime from %s',
                              self.location, self.filename)

    @property
    def LengthUnit(self):
        """SNIRF field `LengthUnit`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the **case-sensitive** SI length unit used in this 
        measurement. Sample length units include "mm", "cm", and "m". A value of 
        "um" is the same as "mm", i.e. micrometer.

        """
        if type(self._LengthUnit) is type(_AbsentDataset):
            return None
        if type(self._LengthUnit) is type(_PresentDataset):
            return _read_string(self._h['LengthUnit'])
            self._cfg.logger.info('Dynamically loaded %s/LengthUnit from %s',
                                  self.location, self.filename)
        return self._LengthUnit

    @LengthUnit.setter
    def LengthUnit(self, value):
        self._LengthUnit = value
        # self._cfg.logger.info('Assignment to %s/LengthUnit in %s', self.location, self.filename)

    @LengthUnit.deleter
    def LengthUnit(self):
        self._LengthUnit = _AbsentDataset
        self._cfg.logger.info('Deleted %s/LengthUnit from %s', self.location,
                              self.filename)

    @property
    def TimeUnit(self):
        """SNIRF field `TimeUnit`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the **case-sensitive** SI time unit used in this 
        measurement. Sample time units include "s", and "ms". A value of "us" 
        is the same as "ms", i.e. microsecond.

        """
        if type(self._TimeUnit) is type(_AbsentDataset):
            return None
        if type(self._TimeUnit) is type(_PresentDataset):
            return _read_string(self._h['TimeUnit'])
            self._cfg.logger.info('Dynamically loaded %s/TimeUnit from %s',
                                  self.location, self.filename)
        return self._TimeUnit

    @TimeUnit.setter
    def TimeUnit(self, value):
        self._TimeUnit = value
        # self._cfg.logger.info('Assignment to %s/TimeUnit in %s', self.location, self.filename)

    @TimeUnit.deleter
    def TimeUnit(self):
        self._TimeUnit = _AbsentDataset
        self._cfg.logger.info('Deleted %s/TimeUnit from %s', self.location,
                              self.filename)

    @property
    def FrequencyUnit(self):
        """SNIRF field `FrequencyUnit`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This record stores the **case-sensitive** SI frequency unit used in 
        this measurement. Sample frequency units "Hz", "MHz" and "GHz". Please
        note that "mHz" is milli-Hz while "MHz" denotes "mega-Hz" according to
        SI unit system.

        We do not limit the total number of metadata records in the `metaDataTags`. Users
        can add additional customized metadata records; no duplicated metadata record names
        are allowed.

        Additional metadata record samples can be found in the below table.

        | Metadata Key Name | Metadata value |
        |-------------------|----------------|
        |ManufacturerName | "Company Name" |
        |Model | "Model Name" |
        |SubjectName | "LastName, FirstName" |
        |DateOfBirth | "YYYY-MM-DD" |
        |AcquisitionStartTime | "1569465620" |
        |StudyID | "Infant Brain Development" |
        |StudyDescription | "In this study, we measure ...." |
        |AccessionNumber | "##########################" |
        |InstanceNumber  | 2 |
        |CalibrationFileName | "phantomcal_121015.snirf" |
        |UnixTime | "1569465667" |

        The metadata records `"StudyID"` and `"AccessionNumber"` are unique strings that 
        can be used to link the current dataset to a particular study and a particular 
        procedure, respectively. The `"StudyID"` tag is similar to the DICOM tag "Study 
        ID" (0020,0010) and `"AccessionNumber"` is similar to the DICOM tag "Accession 
        Number"(0008,0050), as defined in the DICOM standard (ISO 12052).

        The metadata record `"InstanceNumber"` is defined similarly to the DICOM tag 
        "Instance Number" (0020,0013), and can be used as the sequence number to group 
        multiple datasets into a larger dataset - for example, concatenating streamed 
        data segments during a long measurement session.

        The metadata record `"UnixTime"` defines the Unix Epoch Time, i.e. the total elapse
        time in seconds since 1970-01-01T00:00:00Z (UTC) minus the leap seconds.

        """
        if type(self._FrequencyUnit) is type(_AbsentDataset):
            return None
        if type(self._FrequencyUnit) is type(_PresentDataset):
            return _read_string(self._h['FrequencyUnit'])
            self._cfg.logger.info(
                'Dynamically loaded %s/FrequencyUnit from %s', self.location,
                self.filename)
        return self._FrequencyUnit

    @FrequencyUnit.setter
    def FrequencyUnit(self, value):
        self._FrequencyUnit = value
        # self._cfg.logger.info('Assignment to %s/FrequencyUnit in %s', self.location, self.filename)

    @FrequencyUnit.deleter
    def FrequencyUnit(self):
        self._FrequencyUnit = _AbsentDataset
        self._cfg.logger.info('Deleted %s/FrequencyUnit from %s',
                              self.location, self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/SubjectID'
        if type(self._SubjectID) not in [type(_AbsentDataset), type(None)]:
            data = self.SubjectID  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/MeasurementDate'
        if type(self._MeasurementDate) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.MeasurementDate  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/MeasurementTime'
        if type(self._MeasurementTime) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.MeasurementTime  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/LengthUnit'
        if type(self._LengthUnit) not in [type(_AbsentDataset), type(None)]:
            data = self.LengthUnit  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/TimeUnit'
        if type(self._TimeUnit) not in [type(_AbsentDataset), type(None)]:
            data = self.TimeUnit  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/FrequencyUnit'
        if type(self._FrequencyUnit) not in [type(_AbsentDataset), type(None)]:
            data = self.FrequencyUnit  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        for unspecified_name in self._unspecified_names:
            name = self.location + '/' + unspecified_name
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
            try:
                data = getattr(self, unspecified_name)
            except AttributeError:  # Dataset was deleted
                continue
            _create_dataset(file, name, data)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/SubjectID'
            if type(self._SubjectID) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._SubjectID) is type(
                            _PresentDataset) or 'SubjectID' in self._h:
                        dataset = self._h['SubjectID']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'SubjectID', self._SubjectID)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/MeasurementDate'
            if type(self._MeasurementDate) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._MeasurementDate) is type(
                            _PresentDataset) or 'MeasurementDate' in self._h:
                        dataset = self._h['MeasurementDate']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'MeasurementDate', self._MeasurementDate)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/MeasurementTime'
            if type(self._MeasurementTime) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._MeasurementTime) is type(
                            _PresentDataset) or 'MeasurementTime' in self._h:
                        dataset = self._h['MeasurementTime']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'MeasurementTime', self._MeasurementTime)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/LengthUnit'
            if type(self._LengthUnit) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._LengthUnit) is type(
                            _PresentDataset) or 'LengthUnit' in self._h:
                        dataset = self._h['LengthUnit']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'LengthUnit', self._LengthUnit)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/TimeUnit'
            if type(self._TimeUnit) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._TimeUnit) is type(
                            _PresentDataset) or 'TimeUnit' in self._h:
                        dataset = self._h['TimeUnit']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'TimeUnit', self._TimeUnit)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/FrequencyUnit'
            if type(self._FrequencyUnit) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._FrequencyUnit) is type(
                            _PresentDataset) or 'FrequencyUnit' in self._h:
                        dataset = self._h['FrequencyUnit']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'FrequencyUnit', self._FrequencyUnit)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')


class Probe(Group):
    """Wrapper for Group of type `probe`.

    This is a structured variable that describes the probe (source-detector) 
    geometry.  This variable has a number of required fields.

    """
    def __init__(self, var, cfg: SnirfConfig):
        super().__init__(var, cfg)
        self._wavelengths = _AbsentDataset  # [<f>,...]*
        self._wavelengthsEmission = _AbsentDataset  # [<f>,...]
        self._sourcePos2D = _AbsentDataset  # [[<f>,...]]*1
        self._sourcePos3D = _AbsentDataset  # [[<f>,...]]*1
        self._detectorPos2D = _AbsentDataset  # [[<f>,...]]*2
        self._detectorPos3D = _AbsentDataset  # [[<f>,...]]*2
        self._frequencies = _AbsentDataset  # [<f>,...]
        self._timeDelays = _AbsentDataset  # [<f>,...]
        self._timeDelayWidths = _AbsentDataset  # [<f>,...]
        self._momentOrders = _AbsentDataset  # [<f>,...]
        self._correlationTimeDelays = _AbsentDataset  # [<f>,...]
        self._correlationTimeDelayWidths = _AbsentDataset  # [<f>,...]
        self._sourceLabels = _AbsentDataset  # [["s",...]]
        self._detectorLabels = _AbsentDataset  # ["s",...]
        self._landmarkPos2D = _AbsentDataset  # [[<f>,...]]
        self._landmarkPos3D = _AbsentDataset  # [[<f>,...]]
        self._landmarkLabels = _AbsentDataset  # ["s",...]
        self._coordinateSystem = _AbsentDataset  # "s"
        self._coordinateSystemDescription = _AbsentDataset  # "s"
        self._useLocalIndex = _AbsentDataset  # <i>
        self._snirf_names = [
            'wavelengths',
            'wavelengthsEmission',
            'sourcePos2D',
            'sourcePos3D',
            'detectorPos2D',
            'detectorPos3D',
            'frequencies',
            'timeDelays',
            'timeDelayWidths',
            'momentOrders',
            'correlationTimeDelays',
            'correlationTimeDelayWidths',
            'sourceLabels',
            'detectorLabels',
            'landmarkPos2D',
            'landmarkPos3D',
            'landmarkLabels',
            'coordinateSystem',
            'coordinateSystemDescription',
            'useLocalIndex',
        ]

        self._indexed_groups = []
        if 'wavelengths' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengths = _read_float_array(self._h['wavelengths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengths = _PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengths = _AbsentDataset
        if 'wavelengthsEmission' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthsEmission = _read_float_array(
                    self._h['wavelengthsEmission'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthsEmission = _PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthsEmission = _AbsentDataset
        if 'sourcePos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos2D = _read_float_array(self._h['sourcePos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePos2D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePos2D = _AbsentDataset
        if 'sourcePos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePos3D = _read_float_array(self._h['sourcePos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePos3D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePos3D = _AbsentDataset
        if 'detectorPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos2D = _read_float_array(
                    self._h['detectorPos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorPos2D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorPos2D = _AbsentDataset
        if 'detectorPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorPos3D = _read_float_array(
                    self._h['detectorPos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorPos3D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorPos3D = _AbsentDataset
        if 'frequencies' in self._h:
            if not self._cfg.dynamic_loading:
                self._frequencies = _read_float_array(self._h['frequencies'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._frequencies = _PresentDataset
        else:  # if the dataset is not found on disk
            self._frequencies = _AbsentDataset
        if 'timeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelays = _read_float_array(self._h['timeDelays'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeDelays = _PresentDataset
        else:  # if the dataset is not found on disk
            self._timeDelays = _AbsentDataset
        if 'timeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeDelayWidths = _read_float_array(
                    self._h['timeDelayWidths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeDelayWidths = _PresentDataset
        else:  # if the dataset is not found on disk
            self._timeDelayWidths = _AbsentDataset
        if 'momentOrders' in self._h:
            if not self._cfg.dynamic_loading:
                self._momentOrders = _read_float_array(self._h['momentOrders'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._momentOrders = _PresentDataset
        else:  # if the dataset is not found on disk
            self._momentOrders = _AbsentDataset
        if 'correlationTimeDelays' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelays = _read_float_array(
                    self._h['correlationTimeDelays'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._correlationTimeDelays = _PresentDataset
        else:  # if the dataset is not found on disk
            self._correlationTimeDelays = _AbsentDataset
        if 'correlationTimeDelayWidths' in self._h:
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelayWidths = _read_float_array(
                    self._h['correlationTimeDelayWidths'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._correlationTimeDelayWidths = _PresentDataset
        else:  # if the dataset is not found on disk
            self._correlationTimeDelayWidths = _AbsentDataset
        if 'sourceLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceLabels = _read_string_array(
                    self._h['sourceLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceLabels = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceLabels = _AbsentDataset
        if 'detectorLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorLabels = _read_string_array(
                    self._h['detectorLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorLabels = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorLabels = _AbsentDataset
        if 'landmarkPos2D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos2D = _read_float_array(
                    self._h['landmarkPos2D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkPos2D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkPos2D = _AbsentDataset
        if 'landmarkPos3D' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkPos3D = _read_float_array(
                    self._h['landmarkPos3D'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkPos3D = _PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkPos3D = _AbsentDataset
        if 'landmarkLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._landmarkLabels = _read_string_array(
                    self._h['landmarkLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._landmarkLabels = _PresentDataset
        else:  # if the dataset is not found on disk
            self._landmarkLabels = _AbsentDataset
        if 'coordinateSystem' in self._h:
            if not self._cfg.dynamic_loading:
                self._coordinateSystem = _read_string(
                    self._h['coordinateSystem'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._coordinateSystem = _PresentDataset
        else:  # if the dataset is not found on disk
            self._coordinateSystem = _AbsentDataset
        if 'coordinateSystemDescription' in self._h:
            if not self._cfg.dynamic_loading:
                self._coordinateSystemDescription = _read_string(
                    self._h['coordinateSystemDescription'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._coordinateSystemDescription = _PresentDataset
        else:  # if the dataset is not found on disk
            self._coordinateSystemDescription = _AbsentDataset
        if 'useLocalIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._useLocalIndex = _read_int(self._h['useLocalIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._useLocalIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._useLocalIndex = _AbsentDataset

    @property
    def wavelengths(self):
        """SNIRF field `wavelengths`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the "nominal" wavelengths used (in `nm` unit).  This is indexed by the 
        `wavelengthIndex` of the measurementList variable. For example, `probe.wavelengths` = [690, 
        780, 830]; implies that the measurements were taken at three wavelengths (690 nm, 
        780 nm, and 830 nm).  The wavelength index of 
        `measurementList(k).wavelengthIndex` variable refers to this field.
        `measurementList(k).wavelengthIndex` = 2 means the k<sup>th</sup> measurement 
        was at 780 nm.

        Please note that this field stores the "nominal" wavelengths. If the precise 
        (measured) wavelengths differ from the nominal wavelengths, one can store those
        in the `measurementList.wavelengthActual` field in a per-channel fashion.

        The number of wavelengths is not limited (except that at least two are needed 
        to calculate the two forms of hemoglobin).  Each source-detector pair would 
        generally have measurements at all wavelengths.

        This field must present, but can be empty, for example, in the case that the stored
        data are processed data (`dataType=99999`, see Appendix).


        """
        if type(self._wavelengths) is type(_AbsentDataset):
            return None
        if type(self._wavelengths) is type(_PresentDataset):
            return _read_float_array(self._h['wavelengths'])
            self._cfg.logger.info('Dynamically loaded %s/wavelengths from %s',
                                  self.location, self.filename)
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        # self._cfg.logger.info('Assignment to %s/wavelengths in %s', self.location, self.filename)

    @wavelengths.deleter
    def wavelengths(self):
        self._wavelengths = _AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengths from %s', self.location,
                              self.filename)

    @property
    def wavelengthsEmission(self):
        """SNIRF field `wavelengthsEmission`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field is required only for fluorescence data types, and describes the 
        "nominal" emission wavelengths used (in `nm` unit).  The indexing of this variable is the same 
        wavelength index in measurementList used for `probe.wavelengths` such that the 
        excitation wavelength is paired with this emission wavelength for a given measurement.

        Please note that this field stores the "nominal" emission wavelengths. If the precise 
        (measured) emission wavelengths differ from the nominal ones, one can store those
        in the `measurementList.wavelengthEmissionActual` field in a per-channel fashion.


        """
        if type(self._wavelengthsEmission) is type(_AbsentDataset):
            return None
        if type(self._wavelengthsEmission) is type(_PresentDataset):
            return _read_float_array(self._h['wavelengthsEmission'])
            self._cfg.logger.info(
                'Dynamically loaded %s/wavelengthsEmission from %s',
                self.location, self.filename)
        return self._wavelengthsEmission

    @wavelengthsEmission.setter
    def wavelengthsEmission(self, value):
        self._wavelengthsEmission = value
        # self._cfg.logger.info('Assignment to %s/wavelengthsEmission in %s', self.location, self.filename)

    @wavelengthsEmission.deleter
    def wavelengthsEmission(self):
        self._wavelengthsEmission = _AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthsEmission from %s',
                              self.location, self.filename)

    @property
    def sourcePos2D(self):
        """SNIRF field `sourcePos2D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the position (in `LengthUnit` units) of each source 
        optode. The positions are coordinates in a flattened 2D probe layout. 
        This field has size `<number of sources> x 2`. For example, 
        `probe.sourcePos2D(1,:) = [1.4 1]`, and `LengthUnit='cm'` places source 
        number 1 at x=1.4 cm and y=1 cm.


        """
        if type(self._sourcePos2D) is type(_AbsentDataset):
            return None
        if type(self._sourcePos2D) is type(_PresentDataset):
            return _read_float_array(self._h['sourcePos2D'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePos2D from %s',
                                  self.location, self.filename)
        return self._sourcePos2D

    @sourcePos2D.setter
    def sourcePos2D(self, value):
        self._sourcePos2D = value
        # self._cfg.logger.info('Assignment to %s/sourcePos2D in %s', self.location, self.filename)

    @sourcePos2D.deleter
    def sourcePos2D(self):
        self._sourcePos2D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePos2D from %s', self.location,
                              self.filename)

    @property
    def sourcePos3D(self):
        """SNIRF field `sourcePos3D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the position (in `LengthUnit` units) of each source 
        optode in 3D. This field has size `<number of sources> x 3`.


        """
        if type(self._sourcePos3D) is type(_AbsentDataset):
            return None
        if type(self._sourcePos3D) is type(_PresentDataset):
            return _read_float_array(self._h['sourcePos3D'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePos3D from %s',
                                  self.location, self.filename)
        return self._sourcePos3D

    @sourcePos3D.setter
    def sourcePos3D(self, value):
        self._sourcePos3D = value
        # self._cfg.logger.info('Assignment to %s/sourcePos3D in %s', self.location, self.filename)

    @sourcePos3D.deleter
    def sourcePos3D(self):
        self._sourcePos3D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePos3D from %s', self.location,
                              self.filename)

    @property
    def detectorPos2D(self):
        """SNIRF field `detectorPos2D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Same as `probe.sourcePos2D`, but describing the detector positions in a 
        flattened 2D probe layout.


        """
        if type(self._detectorPos2D) is type(_AbsentDataset):
            return None
        if type(self._detectorPos2D) is type(_PresentDataset):
            return _read_float_array(self._h['detectorPos2D'])
            self._cfg.logger.info(
                'Dynamically loaded %s/detectorPos2D from %s', self.location,
                self.filename)
        return self._detectorPos2D

    @detectorPos2D.setter
    def detectorPos2D(self, value):
        self._detectorPos2D = value
        # self._cfg.logger.info('Assignment to %s/detectorPos2D in %s', self.location, self.filename)

    @detectorPos2D.deleter
    def detectorPos2D(self):
        self._detectorPos2D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorPos2D from %s',
                              self.location, self.filename)

    @property
    def detectorPos3D(self):
        """SNIRF field `detectorPos3D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the position (in `LengthUnit` units) of each detector 
        optode in 3D, defined similarly to `sourcePos3D`.


        """
        if type(self._detectorPos3D) is type(_AbsentDataset):
            return None
        if type(self._detectorPos3D) is type(_PresentDataset):
            return _read_float_array(self._h['detectorPos3D'])
            self._cfg.logger.info(
                'Dynamically loaded %s/detectorPos3D from %s', self.location,
                self.filename)
        return self._detectorPos3D

    @detectorPos3D.setter
    def detectorPos3D(self, value):
        self._detectorPos3D = value
        # self._cfg.logger.info('Assignment to %s/detectorPos3D in %s', self.location, self.filename)

    @detectorPos3D.deleter
    def detectorPos3D(self):
        self._detectorPos3D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorPos3D from %s',
                              self.location, self.filename)

    @property
    def frequencies(self):
        """SNIRF field `frequencies`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the frequencies used (in `FrequencyUnit` units)  for 
        frequency domain measurements. This field is only required for frequency 
        domain data types, and is indexed by `measurementList(k).dataTypeIndex`.


        """
        if type(self._frequencies) is type(_AbsentDataset):
            return None
        if type(self._frequencies) is type(_PresentDataset):
            return _read_float_array(self._h['frequencies'])
            self._cfg.logger.info('Dynamically loaded %s/frequencies from %s',
                                  self.location, self.filename)
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value):
        self._frequencies = value
        # self._cfg.logger.info('Assignment to %s/frequencies in %s', self.location, self.filename)

    @frequencies.deleter
    def frequencies(self):
        self._frequencies = _AbsentDataset
        self._cfg.logger.info('Deleted %s/frequencies from %s', self.location,
                              self.filename)

    @property
    def timeDelays(self):
        """SNIRF field `timeDelays`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the time delays (in `TimeUnit` units) used for gated time domain measurements. 
        This field is only required for gated time domain data types, and is indexed by 
        `measurementList(k).dataTypeIndex`. The indexing of this field is paired with 
        the indexing of `probe.timeDelayWidths`. 


        """
        if type(self._timeDelays) is type(_AbsentDataset):
            return None
        if type(self._timeDelays) is type(_PresentDataset):
            return _read_float_array(self._h['timeDelays'])
            self._cfg.logger.info('Dynamically loaded %s/timeDelays from %s',
                                  self.location, self.filename)
        return self._timeDelays

    @timeDelays.setter
    def timeDelays(self, value):
        self._timeDelays = value
        # self._cfg.logger.info('Assignment to %s/timeDelays in %s', self.location, self.filename)

    @timeDelays.deleter
    def timeDelays(self):
        self._timeDelays = _AbsentDataset
        self._cfg.logger.info('Deleted %s/timeDelays from %s', self.location,
                              self.filename)

    @property
    def timeDelayWidths(self):
        """SNIRF field `timeDelayWidths`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the time delay widths (in `TimeUnit` units) used for gated time domain 
        measurements. This field is only required for gated time domain data types, and 
        is indexed by `measurementList(k).dataTypeIndex`.  The indexing of this field 
        is paired with the indexing of `probe.timeDelays`.


        """
        if type(self._timeDelayWidths) is type(_AbsentDataset):
            return None
        if type(self._timeDelayWidths) is type(_PresentDataset):
            return _read_float_array(self._h['timeDelayWidths'])
            self._cfg.logger.info(
                'Dynamically loaded %s/timeDelayWidths from %s', self.location,
                self.filename)
        return self._timeDelayWidths

    @timeDelayWidths.setter
    def timeDelayWidths(self, value):
        self._timeDelayWidths = value
        # self._cfg.logger.info('Assignment to %s/timeDelayWidths in %s', self.location, self.filename)

    @timeDelayWidths.deleter
    def timeDelayWidths(self):
        self._timeDelayWidths = _AbsentDataset
        self._cfg.logger.info('Deleted %s/timeDelayWidths from %s',
                              self.location, self.filename)

    @property
    def momentOrders(self):
        """SNIRF field `momentOrders`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the moment orders of the temporal point spread function (TPSF) or the distribution of time-of-flight (DTOF)
        for moment time domain measurements. This field is only required for moment time domain data types, and is indexed by `measurementList(k).dataTypeIndex`.  
        Note that the numeric value in this array is the exponent in the integral used for calculating the moments. For detailed/specific definitions of moments, see [Wabnitz et al, 2020](https://doi.org/10.1364/BOE.396585); for general definitions of moments see [here](https://en.wikipedia.org/wiki/Moment_(mathematics) ).

        In brief, given a TPSF or DTOF N(t) (photon counts vs. photon arrival time at the detector): /
        momentOrder = 0: total counts: `N_total = /intergral N(t)dt` /
        momentOrder = 1: mean time of flight: `m = <t> = (1/N_total) /integral t N(t) dt` /
        momentOrder = 2: variance/second central moment: `V = (1/N_total) /integral (t - <t>)^2 N(t) dt` /
        Please note that all moments (for orders >=1) are expected to be normalized by the total counts (i.e. n=0); Additionally all moments (for orders >= 2) are expected to be centralized.


        """
        if type(self._momentOrders) is type(_AbsentDataset):
            return None
        if type(self._momentOrders) is type(_PresentDataset):
            return _read_float_array(self._h['momentOrders'])
            self._cfg.logger.info('Dynamically loaded %s/momentOrders from %s',
                                  self.location, self.filename)
        return self._momentOrders

    @momentOrders.setter
    def momentOrders(self, value):
        self._momentOrders = value
        # self._cfg.logger.info('Assignment to %s/momentOrders in %s', self.location, self.filename)

    @momentOrders.deleter
    def momentOrders(self):
        self._momentOrders = _AbsentDataset
        self._cfg.logger.info('Deleted %s/momentOrders from %s', self.location,
                              self.filename)

    @property
    def correlationTimeDelays(self):
        """SNIRF field `correlationTimeDelays`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the time delays (in `TimeUnit` units) used for diffuse correlation spectroscopy 
        measurements. This field is only required for diffuse correlation spectroscopy 
        data types, and is indexed by `measurementList(k).dataTypeIndex`.  The indexing 
        of this field is paired with the indexing of `probe.correlationTimeDelayWidths`.


        """
        if type(self._correlationTimeDelays) is type(_AbsentDataset):
            return None
        if type(self._correlationTimeDelays) is type(_PresentDataset):
            return _read_float_array(self._h['correlationTimeDelays'])
            self._cfg.logger.info(
                'Dynamically loaded %s/correlationTimeDelays from %s',
                self.location, self.filename)
        return self._correlationTimeDelays

    @correlationTimeDelays.setter
    def correlationTimeDelays(self, value):
        self._correlationTimeDelays = value
        # self._cfg.logger.info('Assignment to %s/correlationTimeDelays in %s', self.location, self.filename)

    @correlationTimeDelays.deleter
    def correlationTimeDelays(self):
        self._correlationTimeDelays = _AbsentDataset
        self._cfg.logger.info('Deleted %s/correlationTimeDelays from %s',
                              self.location, self.filename)

    @property
    def correlationTimeDelayWidths(self):
        """SNIRF field `correlationTimeDelayWidths`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This field describes the time delay widths (in `TimeUnit` units) used for diffuse correlation 
        spectroscopy measurements. This field is only required for gated time domain 
        data types, and is indexed by `measurementList(k).dataTypeIndex`. The indexing 
        of this field is paired with the indexing of `probe.correlationTimeDelays`.  


        """
        if type(self._correlationTimeDelayWidths) is type(_AbsentDataset):
            return None
        if type(self._correlationTimeDelayWidths) is type(_PresentDataset):
            return _read_float_array(self._h['correlationTimeDelayWidths'])
            self._cfg.logger.info(
                'Dynamically loaded %s/correlationTimeDelayWidths from %s',
                self.location, self.filename)
        return self._correlationTimeDelayWidths

    @correlationTimeDelayWidths.setter
    def correlationTimeDelayWidths(self, value):
        self._correlationTimeDelayWidths = value
        # self._cfg.logger.info('Assignment to %s/correlationTimeDelayWidths in %s', self.location, self.filename)

    @correlationTimeDelayWidths.deleter
    def correlationTimeDelayWidths(self):
        self._correlationTimeDelayWidths = _AbsentDataset
        self._cfg.logger.info('Deleted %s/correlationTimeDelayWidths from %s',
                              self.location, self.filename)

    @property
    def sourceLabels(self):
        """SNIRF field `sourceLabels`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a string array providing user friendly or instrument specific labels 
        for each source. Each element of the array must be a unique string among both 
        `probe.sourceLabels` and `probe.detectorLabels`.This can be of size `<number 
        of sources>x 1` or `<number of sources> x <number of 
        wavelengths>`. This is indexed by `measurementList(k).sourceIndex` and 
        `measurementList(k).wavelengthIndex`.


        """
        if type(self._sourceLabels) is type(_AbsentDataset):
            return None
        if type(self._sourceLabels) is type(_PresentDataset):
            return _read_string_array(self._h['sourceLabels'])
            self._cfg.logger.info('Dynamically loaded %s/sourceLabels from %s',
                                  self.location, self.filename)
        return self._sourceLabels

    @sourceLabels.setter
    def sourceLabels(self, value):
        self._sourceLabels = value
        # self._cfg.logger.info('Assignment to %s/sourceLabels in %s', self.location, self.filename)

    @sourceLabels.deleter
    def sourceLabels(self):
        self._sourceLabels = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceLabels from %s', self.location,
                              self.filename)

    @property
    def detectorLabels(self):
        """SNIRF field `detectorLabels`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a string array providing user friendly or instrument specific labels 
        for each detector. Each element of the array must be a unique string among both 
        `probe.sourceLabels` and `probe.detectorLabels`. This is indexed by 
        `measurementList(k).detectorIndex`.


        """
        if type(self._detectorLabels) is type(_AbsentDataset):
            return None
        if type(self._detectorLabels) is type(_PresentDataset):
            return _read_string_array(self._h['detectorLabels'])
            self._cfg.logger.info(
                'Dynamically loaded %s/detectorLabels from %s', self.location,
                self.filename)
        return self._detectorLabels

    @detectorLabels.setter
    def detectorLabels(self, value):
        self._detectorLabels = value
        # self._cfg.logger.info('Assignment to %s/detectorLabels in %s', self.location, self.filename)

    @detectorLabels.deleter
    def detectorLabels(self):
        self._detectorLabels = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorLabels from %s',
                              self.location, self.filename)

    @property
    def landmarkPos2D(self):
        """SNIRF field `landmarkPos2D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a 2-D array storing the neurological landmark positions projected
        along the 2-D (flattened) probe plane in order to map optical data from the
        flattened optode positions to brain anatomy. This array should contain a minimum 
        of 2 columns, representing the x and y coordinates (in `LengthUnit` units)
        of the 2-D projected landmark positions. If a 3rd column presents, it stores 
        the index to the labels of the given landmark. Label names are stored in the 
        `probe.landmarkLabels` subfield. An label index of 0 refers to an undefined landmark. 


        """
        if type(self._landmarkPos2D) is type(_AbsentDataset):
            return None
        if type(self._landmarkPos2D) is type(_PresentDataset):
            return _read_float_array(self._h['landmarkPos2D'])
            self._cfg.logger.info(
                'Dynamically loaded %s/landmarkPos2D from %s', self.location,
                self.filename)
        return self._landmarkPos2D

    @landmarkPos2D.setter
    def landmarkPos2D(self, value):
        self._landmarkPos2D = value
        # self._cfg.logger.info('Assignment to %s/landmarkPos2D in %s', self.location, self.filename)

    @landmarkPos2D.deleter
    def landmarkPos2D(self):
        self._landmarkPos2D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkPos2D from %s',
                              self.location, self.filename)

    @property
    def landmarkPos3D(self):
        """SNIRF field `landmarkPos3D`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a 2-D array storing the neurological landmark positions measurement 
        from 3-D digitization and tracking systems to facilitate the registration and 
        mapping of optical data to brain anatomy. This array should contain a minimum 
        of 3 columns, representing the x, y and z coordinates (in `LengthUnit` units) 
        of the digitized landmark positions. If a 4th column presents, it stores the 
        index to the labels of the given landmark. Label names are stored in the 
        `probe.landmarkLabels` subfield. An label index of 0 refers to an undefined landmark. 


        """
        if type(self._landmarkPos3D) is type(_AbsentDataset):
            return None
        if type(self._landmarkPos3D) is type(_PresentDataset):
            return _read_float_array(self._h['landmarkPos3D'])
            self._cfg.logger.info(
                'Dynamically loaded %s/landmarkPos3D from %s', self.location,
                self.filename)
        return self._landmarkPos3D

    @landmarkPos3D.setter
    def landmarkPos3D(self, value):
        self._landmarkPos3D = value
        # self._cfg.logger.info('Assignment to %s/landmarkPos3D in %s', self.location, self.filename)

    @landmarkPos3D.deleter
    def landmarkPos3D(self):
        self._landmarkPos3D = _AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkPos3D from %s',
                              self.location, self.filename)

    @property
    def landmarkLabels(self):
        """SNIRF field `landmarkLabels`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This string array stores the names of the landmarks. The first string denotes 
        the name of the landmarks with an index of 1 in the 4th column of 
        `probe.landmark`, and so on. One can adopt the commonly used 10-20 landmark 
        names, such as "Nasion", "Inion", "Cz" etc, or use user-defined landmark 
        labels. The landmark label can also use the unique source and detector labels 
        defined in `probe.sourceLabels` and `probe.detectorLabels`, respectively, to 
        associate the given landmark to a specific source or detector. All strings are 
        ASCII encoded char arrays.


        """
        if type(self._landmarkLabels) is type(_AbsentDataset):
            return None
        if type(self._landmarkLabels) is type(_PresentDataset):
            return _read_string_array(self._h['landmarkLabels'])
            self._cfg.logger.info(
                'Dynamically loaded %s/landmarkLabels from %s', self.location,
                self.filename)
        return self._landmarkLabels

    @landmarkLabels.setter
    def landmarkLabels(self, value):
        self._landmarkLabels = value
        # self._cfg.logger.info('Assignment to %s/landmarkLabels in %s', self.location, self.filename)

    @landmarkLabels.deleter
    def landmarkLabels(self):
        self._landmarkLabels = _AbsentDataset
        self._cfg.logger.info('Deleted %s/landmarkLabels from %s',
                              self.location, self.filename)

    @property
    def coordinateSystem(self):
        """SNIRF field `coordinateSystem`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Defines the coordinate system for sensor positions.
        The string must be one of the coordinate systems listed in the
        [BIDS specification (Appendix VII)](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html#standard-template-identifiers)
        such as "MNI152NLin2009bAsym", "CapTrak" or "Other".
        If the value "Other" is specified, then a defition of the coordinate
        system must be provided in `/nirs(i)/probe/coordinateSystemDescription`.
        See the [FieldTrip toolbox web page](https://www.fieldtriptoolbox.org/faq/coordsys/)
        for detailed descriptions of different coordinate systems.


        """
        if type(self._coordinateSystem) is type(_AbsentDataset):
            return None
        if type(self._coordinateSystem) is type(_PresentDataset):
            return _read_string(self._h['coordinateSystem'])
            self._cfg.logger.info(
                'Dynamically loaded %s/coordinateSystem from %s',
                self.location, self.filename)
        return self._coordinateSystem

    @coordinateSystem.setter
    def coordinateSystem(self, value):
        self._coordinateSystem = value
        # self._cfg.logger.info('Assignment to %s/coordinateSystem in %s', self.location, self.filename)

    @coordinateSystem.deleter
    def coordinateSystem(self):
        self._coordinateSystem = _AbsentDataset
        self._cfg.logger.info('Deleted %s/coordinateSystem from %s',
                              self.location, self.filename)

    @property
    def coordinateSystemDescription(self):
        """SNIRF field `coordinateSystemDescription`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Free-form text description of the coordinate system.
        May also include a link to a documentation page or
        paper describing the system in greater detail.
        This field is required if the `coordinateSystem` field is set to "Other".


        """
        if type(self._coordinateSystemDescription) is type(_AbsentDataset):
            return None
        if type(self._coordinateSystemDescription) is type(_PresentDataset):
            return _read_string(self._h['coordinateSystemDescription'])
            self._cfg.logger.info(
                'Dynamically loaded %s/coordinateSystemDescription from %s',
                self.location, self.filename)
        return self._coordinateSystemDescription

    @coordinateSystemDescription.setter
    def coordinateSystemDescription(self, value):
        self._coordinateSystemDescription = value
        # self._cfg.logger.info('Assignment to %s/coordinateSystemDescription in %s', self.location, self.filename)

    @coordinateSystemDescription.deleter
    def coordinateSystemDescription(self):
        self._coordinateSystemDescription = _AbsentDataset
        self._cfg.logger.info('Deleted %s/coordinateSystemDescription from %s',
                              self.location, self.filename)

    @property
    def useLocalIndex(self):
        """SNIRF field `useLocalIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        For modular NIRS systems, setting this flag to a non-zero integer indicates 
        that `measurementList(k).sourceIndex` and `measurementList(k).detectorIndex` 
        are module-specific local-indices. One must also include 
        `measurementList(k).moduleIndex`, or when cross-module channels present, both 
        `measurementList(k).sourceModuleIndex` and `measurementList(k).detectorModuleIndex` 
        in the `measurementList` structure in order to restore the global indices 
        of the sources/detectors.


        """
        if type(self._useLocalIndex) is type(_AbsentDataset):
            return None
        if type(self._useLocalIndex) is type(_PresentDataset):
            return _read_int(self._h['useLocalIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/useLocalIndex from %s', self.location,
                self.filename)
        return self._useLocalIndex

    @useLocalIndex.setter
    def useLocalIndex(self, value):
        self._useLocalIndex = value
        # self._cfg.logger.info('Assignment to %s/useLocalIndex in %s', self.location, self.filename)

    @useLocalIndex.deleter
    def useLocalIndex(self):
        self._useLocalIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/useLocalIndex from %s',
                              self.location, self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/wavelengths'
        if type(self._wavelengths) not in [type(_AbsentDataset), type(None)]:
            data = self.wavelengths  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/wavelengthsEmission'
        if type(self._wavelengthsEmission) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.wavelengthsEmission  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/sourcePos2D'
        if type(self._sourcePos2D) not in [type(_AbsentDataset), type(None)]:
            data = self.sourcePos2D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/sourcePos3D'
        if type(self._sourcePos3D) not in [type(_AbsentDataset), type(None)]:
            data = self.sourcePos3D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorPos2D'
        if type(self._detectorPos2D) not in [type(_AbsentDataset), type(None)]:
            data = self.detectorPos2D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorPos3D'
        if type(self._detectorPos3D) not in [type(_AbsentDataset), type(None)]:
            data = self.detectorPos3D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/frequencies'
        if type(self._frequencies) not in [type(_AbsentDataset), type(None)]:
            data = self.frequencies  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/timeDelays'
        if type(self._timeDelays) not in [type(_AbsentDataset), type(None)]:
            data = self.timeDelays  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/timeDelayWidths'
        if type(self._timeDelayWidths) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.timeDelayWidths  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/momentOrders'
        if type(self._momentOrders) not in [type(_AbsentDataset), type(None)]:
            data = self.momentOrders  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/correlationTimeDelays'
        if type(self._correlationTimeDelays) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.correlationTimeDelays  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/correlationTimeDelayWidths'
        if type(self._correlationTimeDelayWidths) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.correlationTimeDelayWidths  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/sourceLabels'
        if type(self._sourceLabels) not in [type(_AbsentDataset), type(None)]:
            data = self.sourceLabels  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorLabels'
        if type(self._detectorLabels) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.detectorLabels  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/landmarkPos2D'
        if type(self._landmarkPos2D) not in [type(_AbsentDataset), type(None)]:
            data = self.landmarkPos2D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/landmarkPos3D'
        if type(self._landmarkPos3D) not in [type(_AbsentDataset), type(None)]:
            data = self.landmarkPos3D  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/landmarkLabels'
        if type(self._landmarkLabels) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.landmarkLabels  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/coordinateSystem'
        if type(self._coordinateSystem) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.coordinateSystem  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/coordinateSystemDescription'
        if type(self._coordinateSystemDescription) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.coordinateSystemDescription  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/useLocalIndex'
        if type(self._useLocalIndex) not in [type(_AbsentDataset), type(None)]:
            data = self.useLocalIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/wavelengths'
            if type(self._wavelengths) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._wavelengths) is type(
                            _PresentDataset) or 'wavelengths' in self._h:
                        dataset = self._h['wavelengths']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'wavelengths', self._wavelengths)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/wavelengthsEmission'
            if type(self._wavelengthsEmission) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._wavelengthsEmission) is type(
                            _PresentDataset
                    ) or 'wavelengthsEmission' in self._h:
                        dataset = self._h['wavelengthsEmission']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'wavelengthsEmission',
                            self._wavelengthsEmission)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/sourcePos2D'
            if type(self._sourcePos2D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._sourcePos2D) is type(
                            _PresentDataset) or 'sourcePos2D' in self._h:
                        dataset = self._h['sourcePos2D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'sourcePos2D', self._sourcePos2D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/sourcePos3D'
            if type(self._sourcePos3D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._sourcePos3D) is type(
                            _PresentDataset) or 'sourcePos3D' in self._h:
                        dataset = self._h['sourcePos3D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'sourcePos3D', self._sourcePos3D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorPos2D'
            if type(self._detectorPos2D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorPos2D) is type(
                            _PresentDataset) or 'detectorPos2D' in self._h:
                        dataset = self._h['detectorPos2D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'detectorPos2D', self._detectorPos2D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorPos3D'
            if type(self._detectorPos3D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorPos3D) is type(
                            _PresentDataset) or 'detectorPos3D' in self._h:
                        dataset = self._h['detectorPos3D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'detectorPos3D', self._detectorPos3D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/frequencies'
            if type(self._frequencies) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._frequencies) is type(
                            _PresentDataset) or 'frequencies' in self._h:
                        dataset = self._h['frequencies']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'frequencies', self._frequencies)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/timeDelays'
            if type(self._timeDelays) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._timeDelays) is type(
                            _PresentDataset) or 'timeDelays' in self._h:
                        dataset = self._h['timeDelays']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'timeDelays', self._timeDelays)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/timeDelayWidths'
            if type(self._timeDelayWidths) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._timeDelayWidths) is type(
                            _PresentDataset) or 'timeDelayWidths' in self._h:
                        dataset = self._h['timeDelayWidths']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'timeDelayWidths', self._timeDelayWidths)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/momentOrders'
            if type(self._momentOrders) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._momentOrders) is type(
                            _PresentDataset) or 'momentOrders' in self._h:
                        dataset = self._h['momentOrders']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'momentOrders', self._momentOrders)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/correlationTimeDelays'
            if type(self._correlationTimeDelays) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._correlationTimeDelays) is type(
                            _PresentDataset
                    ) or 'correlationTimeDelays' in self._h:
                        dataset = self._h['correlationTimeDelays']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'correlationTimeDelays',
                            self._correlationTimeDelays)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/correlationTimeDelayWidths'
            if type(self._correlationTimeDelayWidths) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._correlationTimeDelayWidths) is type(
                            _PresentDataset
                    ) or 'correlationTimeDelayWidths' in self._h:
                        dataset = self._h['correlationTimeDelayWidths']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'correlationTimeDelayWidths',
                            self._correlationTimeDelayWidths)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/sourceLabels'
            if type(self._sourceLabels) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._sourceLabels) is type(
                            _PresentDataset) or 'sourceLabels' in self._h:
                        dataset = self._h['sourceLabels']
                    else:
                        dataset = _create_dataset_string_array(
                            tmp, 'sourceLabels', self._sourceLabels)
                    result._add(name, _validate_string_array(dataset,
                                                             ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorLabels'
            if type(self._detectorLabels) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorLabels) is type(
                            _PresentDataset) or 'detectorLabels' in self._h:
                        dataset = self._h['detectorLabels']
                    else:
                        dataset = _create_dataset_string_array(
                            tmp, 'detectorLabels', self._detectorLabels)
                    result._add(name, _validate_string_array(dataset,
                                                             ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/landmarkPos2D'
            if type(self._landmarkPos2D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._landmarkPos2D) is type(
                            _PresentDataset) or 'landmarkPos2D' in self._h:
                        dataset = self._h['landmarkPos2D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'landmarkPos2D', self._landmarkPos2D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/landmarkPos3D'
            if type(self._landmarkPos3D) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._landmarkPos3D) is type(
                            _PresentDataset) or 'landmarkPos3D' in self._h:
                        dataset = self._h['landmarkPos3D']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'landmarkPos3D', self._landmarkPos3D)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/landmarkLabels'
            if type(self._landmarkLabels) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._landmarkLabels) is type(
                            _PresentDataset) or 'landmarkLabels' in self._h:
                        dataset = self._h['landmarkLabels']
                    else:
                        dataset = _create_dataset_string_array(
                            tmp, 'landmarkLabels', self._landmarkLabels)
                    result._add(name, _validate_string_array(dataset,
                                                             ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/coordinateSystem'
            if type(self._coordinateSystem) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._coordinateSystem) is type(
                            _PresentDataset) or 'coordinateSystem' in self._h:
                        dataset = self._h['coordinateSystem']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'coordinateSystem', self._coordinateSystem)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/coordinateSystemDescription'
            if type(self._coordinateSystemDescription) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._coordinateSystemDescription) is type(
                            _PresentDataset
                    ) or 'coordinateSystemDescription' in self._h:
                        dataset = self._h['coordinateSystemDescription']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'coordinateSystemDescription',
                            self._coordinateSystemDescription)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/useLocalIndex'
            if type(self._useLocalIndex) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._useLocalIndex) is type(
                            _PresentDataset) or 'useLocalIndex' in self._h:
                        dataset = self._h['useLocalIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'useLocalIndex',
                                                      self._useLocalIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class NirsElement(Group):
    """Wrapper for an element of indexed group `Nirs`."""
    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._metaDataTags = _AbsentGroup  # {.}*
        self._data = _AbsentDataset  # {i}*
        self._stim = _AbsentDataset  # {i}
        self._probe = _AbsentGroup  # {.}*
        self._aux = _AbsentDataset  # {i}
        self._snirf_names = [
            'metaDataTags',
            'data',
            'stim',
            'probe',
            'aux',
        ]

        self._indexed_groups = []
        if 'metaDataTags' in self._h:
            self._metaDataTags = MetaDataTags(self._h['metaDataTags'].id,
                                              self._cfg)  # Group
        else:
            self._metaDataTags = MetaDataTags(
                self.location + '/' + 'metaDataTags',
                self._cfg)  # Anonymous group (wrapper only)
        self.data = Data(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.data)
        self.stim = Stim(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.stim)
        if 'probe' in self._h:
            self._probe = Probe(self._h['probe'].id, self._cfg)  # Group
        else:
            self._probe = Probe(self.location + '/' + 'probe',
                                self._cfg)  # Anonymous group (wrapper only)
        self.aux = Aux(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.aux)

    @property
    def metaDataTags(self):
        """SNIRF field `metaDataTags`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        The `metaDataTags` group contains the metadata associated with the measurements.
        Each metadata record is represented as a dataset under this group - with the name of
        the record, i.e. the key, as the dataset's name, and the value of the record as the 
        actual data stored in the dataset. Each metadata record can potentially have different 
        data types. Sub-groups should not be used to organize metadata records: a member of the `metaDataTags` Group must be a Dataset.

        The below five metadata records are minimally required in a SNIRF file

        """
        if type(self._metaDataTags) is type(_AbsentGroup):
            return None
        return self._metaDataTags

    @metaDataTags.setter
    def metaDataTags(self, value):
        if isinstance(value, MetaDataTags):
            self._metaDataTags = _recursive_hdf5_copy(self._metaDataTags,
                                                      value)
        else:
            raise ValueError(
                "Only a Group of type MetaDataTags can be assigned to metaDataTags."
            )
        # self._cfg.logger.info('Assignment to %s/metaDataTags in %s', self.location, self.filename)

    @metaDataTags.deleter
    def metaDataTags(self):
        self._metaDataTags = _AbsentGroup
        self._cfg.logger.info('Deleted %s/metaDataTags from %s', self.location,
                              self.filename)

    @property
    def data(self):
        """SNIRF field `data`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This group stores one block of NIRS data.  This can be extended adding the 
        count number (e.g. `data1`, `data2`,...) to the group name.  This is intended to 
        allow the storage of 1 or more blocks of NIRS data from within the same `/nirs` 
        entry
        * `/nirs/data1` =  data block 1
        * `/nirs/data2` =  data block 2 

         
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # self._cfg.logger.info('Assignment to %s/data in %s', self.location, self.filename)

    @data.deleter
    def data(self):
        raise AttributeError('IndexedGroup ' + str(type(self._data)) +
                             ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/data from %s', self.location,
                              self.filename)

    @property
    def stim(self):
        """SNIRF field `stim`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is an array describing any stimulus conditions. Each element of the array 
        has the following required fields.


        """
        return self._stim

    @stim.setter
    def stim(self, value):
        self._stim = value
        # self._cfg.logger.info('Assignment to %s/stim in %s', self.location, self.filename)

    @stim.deleter
    def stim(self):
        raise AttributeError('IndexedGroup ' + str(type(self._stim)) +
                             ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/stim from %s', self.location,
                              self.filename)

    @property
    def probe(self):
        """SNIRF field `probe`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a structured variable that describes the probe (source-detector) 
        geometry.  This variable has a number of required fields.

        """
        if type(self._probe) is type(_AbsentGroup):
            return None
        return self._probe

    @probe.setter
    def probe(self, value):
        if isinstance(value, Probe):
            self._probe = _recursive_hdf5_copy(self._probe, value)
        else:
            raise ValueError(
                "Only a Group of type Probe can be assigned to probe.")
        # self._cfg.logger.info('Assignment to %s/probe in %s', self.location, self.filename)

    @probe.deleter
    def probe(self):
        self._probe = _AbsentGroup
        self._cfg.logger.info('Deleted %s/probe from %s', self.location,
                              self.filename)

    @property
    def aux(self):
        """SNIRF field `aux`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This optional array specifies any recorded auxiliary data. Each element of 
        `aux` has the following required fields:

        """
        return self._aux

    @aux.setter
    def aux(self, value):
        self._aux = value
        # self._cfg.logger.info('Assignment to %s/aux in %s', self.location, self.filename)

    @aux.deleter
    def aux(self):
        raise AttributeError('IndexedGroup ' + str(type(self._aux)) +
                             ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/aux from %s', self.location,
                              self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        if type(self._metaDataTags) is type(
                _AbsentGroup) or self._metaDataTags.is_empty():
            if 'metaDataTags' in file:
                del file['metaDataTags']
                self._cfg.logger.info('Deleted Group %s/metaDataTags from %s',
                                      self.location, file)
        else:
            self.metaDataTags._save(*args)
        self.data._save(*args)
        self.stim._save(*args)
        if type(self._probe) is type(_AbsentGroup) or self._probe.is_empty():
            if 'probe' in file:
                del file['probe']
                self._cfg.logger.info('Deleted Group %s/probe from %s',
                                      self.location, file)
        else:
            self.probe._save(*args)
        self.aux._save(*args)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/metaDataTags'
            # If Group is not present in file and empty in the wrapper, it is missing
            if type(self._metaDataTags) in [
                    type(_AbsentGroup), type(None)
            ] or ('metaDataTags' not in self._h
                  and self._metaDataTags.is_empty()):
                result._add(name, 'REQUIRED_GROUP_MISSING')
            else:
                self._metaDataTags._validate(result)
            name = self.location + '/data'
            if len(self._data) == 0:
                result._add(name, 'REQUIRED_INDEXED_GROUP_EMPTY')
            else:
                self.data._validate(result)
            name = self.location + '/stim'
            if len(self._stim) == 0:
                result._add(name, 'OPTIONAL_INDEXED_GROUP_EMPTY')
            else:
                self.stim._validate(result)
            name = self.location + '/probe'
            # If Group is not present in file and empty in the wrapper, it is missing
            if type(self._probe) in [
                    type(_AbsentGroup), type(None)
            ] or ('probe' not in self._h and self._probe.is_empty()):
                result._add(name, 'REQUIRED_GROUP_MISSING')
            else:
                self._probe._validate(result)
            name = self.location + '/aux'
            if len(self._aux) == 0:
                result._add(name, 'OPTIONAL_INDEXED_GROUP_EMPTY')
            else:
                self.aux._validate(result)
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class Nirs(IndexedGroup):
    """Interface for indexed group `Nirs`.

    Can be indexed like a list to retrieve `Nirs` elements.

    To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively.

    This group stores one set of NIRS data.  This can be extended by adding the count 
    number (e.g. `/nirs1`, `/nirs2`,...) to the group name. This is intended to 
    allow the storage of 1 or more complete NIRS datasets inside a single SNIRF 
    document.  For example, a two-subject hyperscanning can be stored using the notation
    * `/nirs1` =  first subject's data
    * `/nirs2` =  second subject's data
    The use of a non-indexed (e.g. `/nirs`) entry is allowed when only one entry 
    is present and is assumed to be entry 1.


    """
    _name: str = 'nirs'
    _element: Group = NirsElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class DataElement(Group):
    """Wrapper for an element of indexed group `Data`."""
    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._dataTimeSeries = _AbsentDataset  # [[<f>,...]]*
        self._time = _AbsentDataset  # [<f>,...]*
        self._measurementList = _AbsentDataset  # {i}*
        self._snirf_names = [
            'dataTimeSeries',
            'time',
            'measurementList',
        ]

        self._indexed_groups = []
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(
                    self._h['dataTimeSeries'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTimeSeries = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTimeSeries = _AbsentDataset
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._time = _PresentDataset
        else:  # if the dataset is not found on disk
            self._time = _AbsentDataset
        self.measurementList = MeasurementList(self,
                                               self._cfg)  # Indexed group
        self._indexed_groups.append(self.measurementList)

    @property
    def dataTimeSeries(self):
        """SNIRF field `dataTimeSeries`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is the actual raw or processed data variable. This variable has dimensions 
        of `<number of time points> x <number of channels>`. Columns in 
        `dataTimeSeries` are mapped to the measurement list (`measurementList` variable 
        described below).

        `dataTimeSeries` can be compressed using the HDF5 filter (using the built-in 
        [`deflate`](https://portal.hdfgroup.org/display/HDF5/H5P_SET_DEFLATE)
        filter or [3rd party filters such as `305-LZO` or `307-bzip2`](https://portal.hdfgroup.org/display/support/Registered+Filter+Plugins)

        Chunked data is allowed to support real-time streaming of data in this array. 

        """
        if type(self._dataTimeSeries) is type(_AbsentDataset):
            return None
        if type(self._dataTimeSeries) is type(_PresentDataset):
            return _read_float_array(self._h['dataTimeSeries'])
            self._cfg.logger.info(
                'Dynamically loaded %s/dataTimeSeries from %s', self.location,
                self.filename)
        return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value
        # self._cfg.logger.info('Assignment to %s/dataTimeSeries in %s', self.location, self.filename)

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTimeSeries from %s',
                              self.location, self.filename)

    @property
    def time(self):
        """SNIRF field `time`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        The `time` variable. This provides the acquisition time of the measurement 
        relative to the time origin.  This will usually be a straight line with slope 
        equal to the acquisition frequency, but does not need to be equal spacing.  For 
        the special case of equal sample spacing an array of length `<2>` is allowed 
        where the first entry is the start time and the 
        second entry is the sample time spacing in `TimeUnit` specified in the 
        `metaDataTags`. The default time unit is in second ("s"). For example, 
        a time spacing of 0.2 (s) indicates a sampling rate of 5 Hz.
          
        * **Option 1** - The size of this variable is `<number of time points>` and 
                     corresponds to the sample time of every data point
        * **Option 2**-  The size of this variable is `<2>` and corresponds to the start
              time and sample spacing.

        Chunked data is allowed to support real-time streaming of data in this array.

        """
        if type(self._time) is type(_AbsentDataset):
            return None
        if type(self._time) is type(_PresentDataset):
            return _read_float_array(self._h['time'])
            self._cfg.logger.info('Dynamically loaded %s/time from %s',
                                  self.location, self.filename)
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        # self._cfg.logger.info('Assignment to %s/time in %s', self.location, self.filename)

    @time.deleter
    def time(self):
        self._time = _AbsentDataset
        self._cfg.logger.info('Deleted %s/time from %s', self.location,
                              self.filename)

    @property
    def measurementList(self):
        """SNIRF field `measurementList`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        The measurement list. This variable serves to map the data array onto the probe 
        geometry (sources and detectors), data type, and wavelength. This variable is 
        an array structure that has the size `<number of channels>` that 
        describes the corresponding column in the data matrix. For example, the 
        `measurementList3` describes the third column of the data matrix (i.e. 
        `dataTimeSeries(:,3)`).

        Each element of the array is a structure which describes the measurement 
        conditions for this data with the following fields:


        """
        return self._measurementList

    @measurementList.setter
    def measurementList(self, value):
        self._measurementList = value
        # self._cfg.logger.info('Assignment to %s/measurementList in %s', self.location, self.filename)

    @measurementList.deleter
    def measurementList(self):
        raise AttributeError('IndexedGroup ' +
                             str(type(self._measurementList)) +
                             ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/measurementList from %s',
                              self.location, self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/dataTimeSeries'
        if type(self._dataTimeSeries) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.dataTimeSeries  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/time'
        if type(self._time) not in [type(_AbsentDataset), type(None)]:
            data = self.time  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        self.measurementList._save(*args)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/dataTimeSeries'
            if type(self._dataTimeSeries) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._dataTimeSeries) is type(
                            _PresentDataset) or 'dataTimeSeries' in self._h:
                        dataset = self._h['dataTimeSeries']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'dataTimeSeries', self._dataTimeSeries)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/time'
            if type(self._time) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._time) is type(
                            _PresentDataset) or 'time' in self._h:
                        dataset = self._h['time']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'time', self._time)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/measurementList'
            if len(self._measurementList) == 0:
                result._add(name, 'REQUIRED_INDEXED_GROUP_EMPTY')
            else:
                self.measurementList._validate(result)
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class Data(IndexedGroup):
    """Interface for indexed group `Data`.

    Can be indexed like a list to retrieve `Data` elements.

    To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively.

    This group stores one block of NIRS data.  This can be extended adding the 
    count number (e.g. `data1`, `data2`,...) to the group name.  This is intended to 
    allow the storage of 1 or more blocks of NIRS data from within the same `/nirs` 
    entry
    * `/nirs/data1` =  data block 1
    * `/nirs/data2` =  data block 2 

     
    """
    _name: str = 'data'
    _element: Group = DataElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class MeasurementListElement(Group):
    """Wrapper for an element of indexed group `MeasurementList`."""
    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._sourceIndex = _AbsentDataset  # <i>*
        self._detectorIndex = _AbsentDataset  # <i>*
        self._wavelengthIndex = _AbsentDataset  # <i>*
        self._wavelengthActual = _AbsentDataset  # <f>
        self._wavelengthEmissionActual = _AbsentDataset  # <f>
        self._dataType = _AbsentDataset  # <i>*
        self._dataUnit = _AbsentDataset  # "s"
        self._dataTypeLabel = _AbsentDataset  # "s"
        self._dataTypeIndex = _AbsentDataset  # <i>*
        self._sourcePower = _AbsentDataset  # <f>
        self._detectorGain = _AbsentDataset  # <f>
        self._moduleIndex = _AbsentDataset  # <i>
        self._sourceModuleIndex = _AbsentDataset  # <i>
        self._detectorModuleIndex = _AbsentDataset  # <i>
        self._snirf_names = [
            'sourceIndex',
            'detectorIndex',
            'wavelengthIndex',
            'wavelengthActual',
            'wavelengthEmissionActual',
            'dataType',
            'dataUnit',
            'dataTypeLabel',
            'dataTypeIndex',
            'sourcePower',
            'detectorGain',
            'moduleIndex',
            'sourceModuleIndex',
            'detectorModuleIndex',
        ]

        self._indexed_groups = []
        if 'sourceIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceIndex = _read_int(self._h['sourceIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceIndex = _AbsentDataset
        if 'detectorIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorIndex = _read_int(self._h['detectorIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorIndex = _AbsentDataset
        if 'wavelengthIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthIndex = _read_int(self._h['wavelengthIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthIndex = _AbsentDataset
        if 'wavelengthActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthActual = _read_float(
                    self._h['wavelengthActual'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthActual = _PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthActual = _AbsentDataset
        if 'wavelengthEmissionActual' in self._h:
            if not self._cfg.dynamic_loading:
                self._wavelengthEmissionActual = _read_float(
                    self._h['wavelengthEmissionActual'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._wavelengthEmissionActual = _PresentDataset
        else:  # if the dataset is not found on disk
            self._wavelengthEmissionActual = _AbsentDataset
        if 'dataType' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataType = _read_int(self._h['dataType'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataType = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataType = _AbsentDataset
        if 'dataUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataUnit = _read_string(self._h['dataUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataUnit = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataUnit = _AbsentDataset
        if 'dataTypeLabel' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeLabel = _read_string(self._h['dataTypeLabel'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTypeLabel = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTypeLabel = _AbsentDataset
        if 'dataTypeIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTypeIndex = _read_int(self._h['dataTypeIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTypeIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTypeIndex = _AbsentDataset
        if 'sourcePower' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourcePower = _read_float(self._h['sourcePower'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourcePower = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourcePower = _AbsentDataset
        if 'detectorGain' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorGain = _read_float(self._h['detectorGain'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorGain = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorGain = _AbsentDataset
        if 'moduleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._moduleIndex = _read_int(self._h['moduleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._moduleIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._moduleIndex = _AbsentDataset
        if 'sourceModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._sourceModuleIndex = _read_int(
                    self._h['sourceModuleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._sourceModuleIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._sourceModuleIndex = _AbsentDataset
        if 'detectorModuleIndex' in self._h:
            if not self._cfg.dynamic_loading:
                self._detectorModuleIndex = _read_int(
                    self._h['detectorModuleIndex'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._detectorModuleIndex = _PresentDataset
        else:  # if the dataset is not found on disk
            self._detectorModuleIndex = _AbsentDataset

    @property
    def sourceIndex(self):
        """SNIRF field `sourceIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of the source.
         
        """
        if type(self._sourceIndex) is type(_AbsentDataset):
            return None
        if type(self._sourceIndex) is type(_PresentDataset):
            return _read_int(self._h['sourceIndex'])
            self._cfg.logger.info('Dynamically loaded %s/sourceIndex from %s',
                                  self.location, self.filename)
        return self._sourceIndex

    @sourceIndex.setter
    def sourceIndex(self, value):
        self._sourceIndex = value
        # self._cfg.logger.info('Assignment to %s/sourceIndex in %s', self.location, self.filename)

    @sourceIndex.deleter
    def sourceIndex(self):
        self._sourceIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceIndex from %s', self.location,
                              self.filename)

    @property
    def detectorIndex(self):
        """SNIRF field `detectorIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of the detector.

        """
        if type(self._detectorIndex) is type(_AbsentDataset):
            return None
        if type(self._detectorIndex) is type(_PresentDataset):
            return _read_int(self._h['detectorIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/detectorIndex from %s', self.location,
                self.filename)
        return self._detectorIndex

    @detectorIndex.setter
    def detectorIndex(self, value):
        self._detectorIndex = value
        # self._cfg.logger.info('Assignment to %s/detectorIndex in %s', self.location, self.filename)

    @detectorIndex.deleter
    def detectorIndex(self):
        self._detectorIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorIndex from %s',
                              self.location, self.filename)

    @property
    def wavelengthIndex(self):
        """SNIRF field `wavelengthIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of the "nominal" wavelength (in `probe.wavelengths`).

        """
        if type(self._wavelengthIndex) is type(_AbsentDataset):
            return None
        if type(self._wavelengthIndex) is type(_PresentDataset):
            return _read_int(self._h['wavelengthIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/wavelengthIndex from %s', self.location,
                self.filename)
        return self._wavelengthIndex

    @wavelengthIndex.setter
    def wavelengthIndex(self, value):
        self._wavelengthIndex = value
        # self._cfg.logger.info('Assignment to %s/wavelengthIndex in %s', self.location, self.filename)

    @wavelengthIndex.deleter
    def wavelengthIndex(self):
        self._wavelengthIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthIndex from %s',
                              self.location, self.filename)

    @property
    def wavelengthActual(self):
        """SNIRF field `wavelengthActual`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Actual (measured) wavelength in nm, if available, for the source in a given channel.

        """
        if type(self._wavelengthActual) is type(_AbsentDataset):
            return None
        if type(self._wavelengthActual) is type(_PresentDataset):
            return _read_float(self._h['wavelengthActual'])
            self._cfg.logger.info(
                'Dynamically loaded %s/wavelengthActual from %s',
                self.location, self.filename)
        return self._wavelengthActual

    @wavelengthActual.setter
    def wavelengthActual(self, value):
        self._wavelengthActual = value
        # self._cfg.logger.info('Assignment to %s/wavelengthActual in %s', self.location, self.filename)

    @wavelengthActual.deleter
    def wavelengthActual(self):
        self._wavelengthActual = _AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthActual from %s',
                              self.location, self.filename)

    @property
    def wavelengthEmissionActual(self):
        """SNIRF field `wavelengthEmissionActual`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Actual (measured) emission wavelength in nm, if available, for the source in a given channel.
         
        """
        if type(self._wavelengthEmissionActual) is type(_AbsentDataset):
            return None
        if type(self._wavelengthEmissionActual) is type(_PresentDataset):
            return _read_float(self._h['wavelengthEmissionActual'])
            self._cfg.logger.info(
                'Dynamically loaded %s/wavelengthEmissionActual from %s',
                self.location, self.filename)
        return self._wavelengthEmissionActual

    @wavelengthEmissionActual.setter
    def wavelengthEmissionActual(self, value):
        self._wavelengthEmissionActual = value
        # self._cfg.logger.info('Assignment to %s/wavelengthEmissionActual in %s', self.location, self.filename)

    @wavelengthEmissionActual.deleter
    def wavelengthEmissionActual(self):
        self._wavelengthEmissionActual = _AbsentDataset
        self._cfg.logger.info('Deleted %s/wavelengthEmissionActual from %s',
                              self.location, self.filename)

    @property
    def dataType(self):
        """SNIRF field `dataType`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Data-type identifier. See Appendix for list possible values.

        """
        if type(self._dataType) is type(_AbsentDataset):
            return None
        if type(self._dataType) is type(_PresentDataset):
            return _read_int(self._h['dataType'])
            self._cfg.logger.info('Dynamically loaded %s/dataType from %s',
                                  self.location, self.filename)
        return self._dataType

    @dataType.setter
    def dataType(self, value):
        self._dataType = value
        # self._cfg.logger.info('Assignment to %s/dataType in %s', self.location, self.filename)

    @dataType.deleter
    def dataType(self):
        self._dataType = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataType from %s', self.location,
                              self.filename)

    @property
    def dataUnit(self):
        """SNIRF field `dataUnit`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        International System of Units (SI units) identifier for the given channel. Encoding should follow the [CMIXF-12 standard](https://people.csail.mit.edu/jaffer/MIXF/CMIXF-12), avoiding special unicode symbols like U+03BC (m) or U+00B5 (u) and using '/' rather than 'per' for units such as `V/us`. The recommended export format is in unscaled units such as V, s, Mole.

        """
        if type(self._dataUnit) is type(_AbsentDataset):
            return None
        if type(self._dataUnit) is type(_PresentDataset):
            return _read_string(self._h['dataUnit'])
            self._cfg.logger.info('Dynamically loaded %s/dataUnit from %s',
                                  self.location, self.filename)
        return self._dataUnit

    @dataUnit.setter
    def dataUnit(self, value):
        self._dataUnit = value
        # self._cfg.logger.info('Assignment to %s/dataUnit in %s', self.location, self.filename)

    @dataUnit.deleter
    def dataUnit(self):
        self._dataUnit = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataUnit from %s', self.location,
                              self.filename)

    @property
    def dataTypeLabel(self):
        """SNIRF field `dataTypeLabel`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Data-type label. Only required if dataType is "processed" (`99999`). See Appendix 
        for list of possible values.

        """
        if type(self._dataTypeLabel) is type(_AbsentDataset):
            return None
        if type(self._dataTypeLabel) is type(_PresentDataset):
            return _read_string(self._h['dataTypeLabel'])
            self._cfg.logger.info(
                'Dynamically loaded %s/dataTypeLabel from %s', self.location,
                self.filename)
        return self._dataTypeLabel

    @dataTypeLabel.setter
    def dataTypeLabel(self, value):
        self._dataTypeLabel = value
        # self._cfg.logger.info('Assignment to %s/dataTypeLabel in %s', self.location, self.filename)

    @dataTypeLabel.deleter
    def dataTypeLabel(self):
        self._dataTypeLabel = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTypeLabel from %s',
                              self.location, self.filename)

    @property
    def dataTypeIndex(self):
        """SNIRF field `dataTypeIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Data-type specific parameter indices. The data type index specifies additional data type specific parameters that are further elaborated by other fields in the probe structure, as detailed below. Note that the Time Domain and Diffuse Correlation Spectroscopy data types have two additional parameters and so the data type index must be a vector with 2 elements that index the additional parameters. One use of this parameter is as a 
        stimulus condition index when `measurementList(k).dataType = 99999` (i.e, `processed` and 
        `measurementList(k).dataTypeLabel = 'HRF ...'` .

        """
        if type(self._dataTypeIndex) is type(_AbsentDataset):
            return None
        if type(self._dataTypeIndex) is type(_PresentDataset):
            return _read_int(self._h['dataTypeIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/dataTypeIndex from %s', self.location,
                self.filename)
        return self._dataTypeIndex

    @dataTypeIndex.setter
    def dataTypeIndex(self, value):
        self._dataTypeIndex = value
        # self._cfg.logger.info('Assignment to %s/dataTypeIndex in %s', self.location, self.filename)

    @dataTypeIndex.deleter
    def dataTypeIndex(self):
        self._dataTypeIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTypeIndex from %s',
                              self.location, self.filename)

    @property
    def sourcePower(self):
        """SNIRF field `sourcePower`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        The units are not defined, unless the user takes the option of using a `metaDataTag` as described below.

        """
        if type(self._sourcePower) is type(_AbsentDataset):
            return None
        if type(self._sourcePower) is type(_PresentDataset):
            return _read_float(self._h['sourcePower'])
            self._cfg.logger.info('Dynamically loaded %s/sourcePower from %s',
                                  self.location, self.filename)
        return self._sourcePower

    @sourcePower.setter
    def sourcePower(self, value):
        self._sourcePower = value
        # self._cfg.logger.info('Assignment to %s/sourcePower in %s', self.location, self.filename)

    @sourcePower.deleter
    def sourcePower(self):
        self._sourcePower = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourcePower from %s', self.location,
                              self.filename)

    @property
    def detectorGain(self):
        """SNIRF field `detectorGain`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Detector gain

        """
        if type(self._detectorGain) is type(_AbsentDataset):
            return None
        if type(self._detectorGain) is type(_PresentDataset):
            return _read_float(self._h['detectorGain'])
            self._cfg.logger.info('Dynamically loaded %s/detectorGain from %s',
                                  self.location, self.filename)
        return self._detectorGain

    @detectorGain.setter
    def detectorGain(self, value):
        self._detectorGain = value
        # self._cfg.logger.info('Assignment to %s/detectorGain in %s', self.location, self.filename)

    @detectorGain.deleter
    def detectorGain(self):
        self._detectorGain = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorGain from %s', self.location,
                              self.filename)

    @property
    def moduleIndex(self):
        """SNIRF field `moduleIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of a repeating module. If `moduleIndex` is provided while `useLocalIndex`
        is set to `true`, then, both `measurementList(k).sourceIndex` and 
        `measurementList(k).detectorIndex` are assumed to be the local indices
        of the same module specified by `moduleIndex`. If the source and
        detector are located on different modules, one must use `sourceModuleIndex`
        and `detectorModuleIndex` instead to specify separate parent module 
        indices. See below.


        """
        if type(self._moduleIndex) is type(_AbsentDataset):
            return None
        if type(self._moduleIndex) is type(_PresentDataset):
            return _read_int(self._h['moduleIndex'])
            self._cfg.logger.info('Dynamically loaded %s/moduleIndex from %s',
                                  self.location, self.filename)
        return self._moduleIndex

    @moduleIndex.setter
    def moduleIndex(self, value):
        self._moduleIndex = value
        # self._cfg.logger.info('Assignment to %s/moduleIndex in %s', self.location, self.filename)

    @moduleIndex.deleter
    def moduleIndex(self):
        self._moduleIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/moduleIndex from %s', self.location,
                              self.filename)

    @property
    def sourceModuleIndex(self):
        """SNIRF field `sourceModuleIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of the module that contains the source of the channel. 
        This index must be used together with `detectorModuleIndex`, and 
        can not be used when `moduleIndex` presents.

        """
        if type(self._sourceModuleIndex) is type(_AbsentDataset):
            return None
        if type(self._sourceModuleIndex) is type(_PresentDataset):
            return _read_int(self._h['sourceModuleIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/sourceModuleIndex from %s',
                self.location, self.filename)
        return self._sourceModuleIndex

    @sourceModuleIndex.setter
    def sourceModuleIndex(self, value):
        self._sourceModuleIndex = value
        # self._cfg.logger.info('Assignment to %s/sourceModuleIndex in %s', self.location, self.filename)

    @sourceModuleIndex.deleter
    def sourceModuleIndex(self):
        self._sourceModuleIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/sourceModuleIndex from %s',
                              self.location, self.filename)

    @property
    def detectorModuleIndex(self):
        """SNIRF field `detectorModuleIndex`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        Index of the module that contains the detector of the channel. 
        This index must be used together with `sourceModuleIndex`, and 
        can not be used when `moduleIndex` presents.


        For example, if `measurementList5` is a structure with `sourceIndex=2`, 
        `detectorIndex=3`, `wavelengthIndex=1`, `dataType=1`, `dataTypeIndex=1` would 
        imply that the data in the 5th column of the `dataTimeSeries` variable was 
        measured with source #2 and detector #3 at wavelength #1.  Wavelengths (in 
        nanometers) are described in the `probe.wavelengths` variable  (described 
        later). The data type in this case is 1, implying that it was a continuous wave 
        measurement.  The complete list of currently supported data types is found in 
        the Appendix. The data type index specifies additional data type specific 
        parameters that are further elaborated by other fields in the `probe` 
        structure, as detailed below. Note that the Time Domain and Diffuse Correlation 
        Spectroscopy data types have two additional parameters and so the data type 
        index must be a vector with 2 elements that index the additional parameters.

        `sourcePower` provides the option for information about the source power for 
        that channel to be saved along with the data. The units are not defined, unless 
        the user takes the option of using a `metaDataTag` described below to define, 
        for instance, `sourcePowerUnit`. `detectorGain` provides the option for 
        information about the detector gain for that channel to be saved along with the 
        data.

        Note:  The source indices generally refer to the optode naming (probe 
        positions) and not necessarily the physical laser numbers on the instrument. 
        The same is true for the detector indices.  Each source optode would generally, 
        but not necessarily, have 2 or more wavelengths (hence lasers) plugged into it 
        in order to calculate deoxy- and oxy-hemoglobin concentrations. The data from 
        these two wavelengths will be indexed by the same source, detector, and data 
        type values, but have different wavelength indices. Using the same source index 
        for lasers at the same location but with different wavelengths simplifies the 
        bookkeeping for converting intensity measurements into concentration changes. 
        As described below, optional variables `probe.sourceLabels` and 
        `probe.detectorLabels` are provided for indicating the instrument specific 
        label for sources and detectors.

        """
        if type(self._detectorModuleIndex) is type(_AbsentDataset):
            return None
        if type(self._detectorModuleIndex) is type(_PresentDataset):
            return _read_int(self._h['detectorModuleIndex'])
            self._cfg.logger.info(
                'Dynamically loaded %s/detectorModuleIndex from %s',
                self.location, self.filename)
        return self._detectorModuleIndex

    @detectorModuleIndex.setter
    def detectorModuleIndex(self, value):
        self._detectorModuleIndex = value
        # self._cfg.logger.info('Assignment to %s/detectorModuleIndex in %s', self.location, self.filename)

    @detectorModuleIndex.deleter
    def detectorModuleIndex(self):
        self._detectorModuleIndex = _AbsentDataset
        self._cfg.logger.info('Deleted %s/detectorModuleIndex from %s',
                              self.location, self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/sourceIndex'
        if type(self._sourceIndex) not in [type(_AbsentDataset), type(None)]:
            data = self.sourceIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorIndex'
        if type(self._detectorIndex) not in [type(_AbsentDataset), type(None)]:
            data = self.detectorIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/wavelengthIndex'
        if type(self._wavelengthIndex) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.wavelengthIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/wavelengthActual'
        if type(self._wavelengthActual) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.wavelengthActual  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/wavelengthEmissionActual'
        if type(self._wavelengthEmissionActual) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.wavelengthEmissionActual  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataType'
        if type(self._dataType) not in [type(_AbsentDataset), type(None)]:
            data = self.dataType  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataUnit'
        if type(self._dataUnit) not in [type(_AbsentDataset), type(None)]:
            data = self.dataUnit  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataTypeLabel'
        if type(self._dataTypeLabel) not in [type(_AbsentDataset), type(None)]:
            data = self.dataTypeLabel  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataTypeIndex'
        if type(self._dataTypeIndex) not in [type(_AbsentDataset), type(None)]:
            data = self.dataTypeIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/sourcePower'
        if type(self._sourcePower) not in [type(_AbsentDataset), type(None)]:
            data = self.sourcePower  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorGain'
        if type(self._detectorGain) not in [type(_AbsentDataset), type(None)]:
            data = self.detectorGain  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/moduleIndex'
        if type(self._moduleIndex) not in [type(_AbsentDataset), type(None)]:
            data = self.moduleIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/sourceModuleIndex'
        if type(self._sourceModuleIndex) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.sourceModuleIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/detectorModuleIndex'
        if type(self._detectorModuleIndex) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.detectorModuleIndex  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_int(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/sourceIndex'
            if type(self._sourceIndex) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._sourceIndex) is type(
                            _PresentDataset) or 'sourceIndex' in self._h:
                        dataset = self._h['sourceIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'sourceIndex',
                                                      self._sourceIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorIndex'
            if type(self._detectorIndex) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorIndex) is type(
                            _PresentDataset) or 'detectorIndex' in self._h:
                        dataset = self._h['detectorIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'detectorIndex',
                                                      self._detectorIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/wavelengthIndex'
            if type(self._wavelengthIndex) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._wavelengthIndex) is type(
                            _PresentDataset) or 'wavelengthIndex' in self._h:
                        dataset = self._h['wavelengthIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'wavelengthIndex',
                                                      self._wavelengthIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/wavelengthActual'
            if type(self._wavelengthActual) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._wavelengthActual) is type(
                            _PresentDataset) or 'wavelengthActual' in self._h:
                        dataset = self._h['wavelengthActual']
                    else:
                        dataset = _create_dataset_float(
                            tmp, 'wavelengthActual', self._wavelengthActual)
                    result._add(name, _validate_float(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/wavelengthEmissionActual'
            if type(self._wavelengthEmissionActual) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._wavelengthEmissionActual) is type(
                            _PresentDataset
                    ) or 'wavelengthEmissionActual' in self._h:
                        dataset = self._h['wavelengthEmissionActual']
                    else:
                        dataset = _create_dataset_float(
                            tmp, 'wavelengthEmissionActual',
                            self._wavelengthEmissionActual)
                    result._add(name, _validate_float(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataType'
            if type(self._dataType) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._dataType) is type(
                            _PresentDataset) or 'dataType' in self._h:
                        dataset = self._h['dataType']
                    else:
                        dataset = _create_dataset_int(tmp, 'dataType',
                                                      self._dataType)
                    result._add(name, _validate_int(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataUnit'
            if type(self._dataUnit) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._dataUnit) is type(
                            _PresentDataset) or 'dataUnit' in self._h:
                        dataset = self._h['dataUnit']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'dataUnit', self._dataUnit)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataTypeLabel'
            if type(self._dataTypeLabel) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._dataTypeLabel) is type(
                            _PresentDataset) or 'dataTypeLabel' in self._h:
                        dataset = self._h['dataTypeLabel']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'dataTypeLabel', self._dataTypeLabel)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataTypeIndex'
            if type(self._dataTypeIndex) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._dataTypeIndex) is type(
                            _PresentDataset) or 'dataTypeIndex' in self._h:
                        dataset = self._h['dataTypeIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'dataTypeIndex',
                                                      self._dataTypeIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/sourcePower'
            if type(self._sourcePower) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._sourcePower) is type(
                            _PresentDataset) or 'sourcePower' in self._h:
                        dataset = self._h['sourcePower']
                    else:
                        dataset = _create_dataset_float(
                            tmp, 'sourcePower', self._sourcePower)
                    result._add(name, _validate_float(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorGain'
            if type(self._detectorGain) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorGain) is type(
                            _PresentDataset) or 'detectorGain' in self._h:
                        dataset = self._h['detectorGain']
                    else:
                        dataset = _create_dataset_float(
                            tmp, 'detectorGain', self._detectorGain)
                    result._add(name, _validate_float(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/moduleIndex'
            if type(self._moduleIndex) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._moduleIndex) is type(
                            _PresentDataset) or 'moduleIndex' in self._h:
                        dataset = self._h['moduleIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'moduleIndex',
                                                      self._moduleIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/sourceModuleIndex'
            if type(self._sourceModuleIndex) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._sourceModuleIndex) is type(
                            _PresentDataset) or 'sourceModuleIndex' in self._h:
                        dataset = self._h['sourceModuleIndex']
                    else:
                        dataset = _create_dataset_int(tmp, 'sourceModuleIndex',
                                                      self._sourceModuleIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/detectorModuleIndex'
            if type(self._detectorModuleIndex) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._detectorModuleIndex) is type(
                            _PresentDataset
                    ) or 'detectorModuleIndex' in self._h:
                        dataset = self._h['detectorModuleIndex']
                    else:
                        dataset = _create_dataset_int(
                            tmp, 'detectorModuleIndex',
                            self._detectorModuleIndex)
                    err_code = _validate_int(dataset)
                    if _read_int(dataset) < 0 and err_code == 'OK':
                        result._add(name, 'NEGATIVE_INDEX')
                    elif _read_int(dataset) == 0 and err_code == 'OK':
                        result._add(name, 'INDEX_OF_ZERO')
                    else:
                        result._add(name, err_code)
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class MeasurementList(IndexedGroup):
    """Interface for indexed group `MeasurementList`.

    Can be indexed like a list to retrieve `MeasurementList` elements.

    To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively.

    The measurement list. This variable serves to map the data array onto the probe 
    geometry (sources and detectors), data type, and wavelength. This variable is 
    an array structure that has the size `<number of channels>` that 
    describes the corresponding column in the data matrix. For example, the 
    `measurementList3` describes the third column of the data matrix (i.e. 
    `dataTimeSeries(:,3)`).

    Each element of the array is a structure which describes the measurement 
    conditions for this data with the following fields:


    """
    _name: str = 'measurementList'
    _element: Group = MeasurementListElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class StimElement(Group):
    """Wrapper for an element of indexed group `Stim`."""
    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._name = _AbsentDataset  # "s"+
        self._data = _AbsentDataset  # [[<f>,...]]+
        self._dataLabels = _AbsentDataset  # ["s",...]
        self._snirf_names = [
            'name',
            'data',
            'dataLabels',
        ]

        self._indexed_groups = []
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._name = _PresentDataset
        else:  # if the dataset is not found on disk
            self._name = _AbsentDataset
        if 'data' in self._h:
            if not self._cfg.dynamic_loading:
                self._data = _read_float_array(self._h['data'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._data = _PresentDataset
        else:  # if the dataset is not found on disk
            self._data = _AbsentDataset
        if 'dataLabels' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataLabels = _read_string_array(self._h['dataLabels'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataLabels = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataLabels = _AbsentDataset

    @property
    def name(self):
        """SNIRF field `name`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a string describing the j<sup>th</sup> stimulus condition.


        """
        if type(self._name) is type(_AbsentDataset):
            return None
        if type(self._name) is type(_PresentDataset):
            return _read_string(self._h['name'])
            self._cfg.logger.info('Dynamically loaded %s/name from %s',
                                  self.location, self.filename)
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        # self._cfg.logger.info('Assignment to %s/name in %s', self.location, self.filename)

    @name.deleter
    def name(self):
        self._name = _AbsentDataset
        self._cfg.logger.info('Deleted %s/name from %s', self.location,
                              self.filename)

    @property
    def data(self):
        """SNIRF field `data`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        * **Allowed attribute**: `names`

        This is a numeric 2-D array with at least 3 columns, specifying the stimulus 
        time course for the j<sup>th</sup> condition. Each row corresponds to a 
        specific stimulus trial. The first three columns indicate `[starttime duration value]`.  
        The starttime, in seconds, is the time relative to the time origin when the 
        stimulus takes on a value; the duration is the time in seconds that the stimulus 
        value continues, and value is the stimulus amplitude.  The number of rows is 
        not constrained. (see examples in the appendix).

        Additional columns can be used to store user-specified data associated with 
        each stimulus trial. An optional record `/nirs(i)/stim(j)/dataLabels` can be 
        used to annotate the meanings of each data column. 

        """
        if type(self._data) is type(_AbsentDataset):
            return None
        if type(self._data) is type(_PresentDataset):
            return _read_float_array(self._h['data'])
            self._cfg.logger.info('Dynamically loaded %s/data from %s',
                                  self.location, self.filename)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        # self._cfg.logger.info('Assignment to %s/data in %s', self.location, self.filename)

    @data.deleter
    def data(self):
        self._data = _AbsentDataset
        self._cfg.logger.info('Deleted %s/data from %s', self.location,
                              self.filename)

    @property
    def dataLabels(self):
        """SNIRF field `dataLabels`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a string array providing annotations for each data column in 
        `/nirs(i)/stim(j)/data`. Each element of the array must be a string;
        the total length of this array must be the same as the column number
        of `/nirs(i)/stim(j)/data`, including the first 3 required columns.

        """
        if type(self._dataLabels) is type(_AbsentDataset):
            return None
        if type(self._dataLabels) is type(_PresentDataset):
            return _read_string_array(self._h['dataLabels'])
            self._cfg.logger.info('Dynamically loaded %s/dataLabels from %s',
                                  self.location, self.filename)
        return self._dataLabels

    @dataLabels.setter
    def dataLabels(self, value):
        self._dataLabels = value
        # self._cfg.logger.info('Assignment to %s/dataLabels in %s', self.location, self.filename)

    @dataLabels.deleter
    def dataLabels(self):
        self._dataLabels = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataLabels from %s', self.location,
                              self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/name'
        if type(self._name) not in [type(_AbsentDataset), type(None)]:
            data = self.name  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/data'
        if type(self._data) not in [type(_AbsentDataset), type(None)]:
            data = self.data  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataLabels'
        if type(self._dataLabels) not in [type(_AbsentDataset), type(None)]:
            data = self.dataLabels  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/name'
            if type(self._name) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._name) is type(
                            _PresentDataset) or 'name' in self._h:
                        dataset = self._h['name']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'name', self._name)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/data'
            if type(self._data) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._data) is type(
                            _PresentDataset) or 'data' in self._h:
                        dataset = self._h['data']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'data', self._data)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataLabels'
            if type(self._dataLabels) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._dataLabels) is type(
                            _PresentDataset) or 'dataLabels' in self._h:
                        dataset = self._h['dataLabels']
                    else:
                        dataset = _create_dataset_string_array(
                            tmp, 'dataLabels', self._dataLabels)
                    result._add(name, _validate_string_array(dataset,
                                                             ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class Stim(IndexedGroup):
    """Interface for indexed group `Stim`.

    Can be indexed like a list to retrieve `Stim` elements.

    To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively.

    This is an array describing any stimulus conditions. Each element of the array 
    has the following required fields.


    """
    _name: str = 'stim'
    _element: Group = StimElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class AuxElement(Group):
    """Wrapper for an element of indexed group `Aux`."""
    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        self._name = _AbsentDataset  # "s"+
        self._dataTimeSeries = _AbsentDataset  # [[<f>,...]]+
        self._dataUnit = _AbsentDataset  # "s"
        self._time = _AbsentDataset  # [<f>,...]+
        self._timeOffset = _AbsentDataset  # [<f>,...]
        self._snirf_names = [
            'name',
            'dataTimeSeries',
            'dataUnit',
            'time',
            'timeOffset',
        ]

        self._indexed_groups = []
        if 'name' in self._h:
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._name = _PresentDataset
        else:  # if the dataset is not found on disk
            self._name = _AbsentDataset
        if 'dataTimeSeries' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = _read_float_array(
                    self._h['dataTimeSeries'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataTimeSeries = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataTimeSeries = _AbsentDataset
        if 'dataUnit' in self._h:
            if not self._cfg.dynamic_loading:
                self._dataUnit = _read_string(self._h['dataUnit'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._dataUnit = _PresentDataset
        else:  # if the dataset is not found on disk
            self._dataUnit = _AbsentDataset
        if 'time' in self._h:
            if not self._cfg.dynamic_loading:
                self._time = _read_float_array(self._h['time'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._time = _PresentDataset
        else:  # if the dataset is not found on disk
            self._time = _AbsentDataset
        if 'timeOffset' in self._h:
            if not self._cfg.dynamic_loading:
                self._timeOffset = _read_float_array(self._h['timeOffset'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._timeOffset = _PresentDataset
        else:  # if the dataset is not found on disk
            self._timeOffset = _AbsentDataset

    @property
    def name(self):
        """SNIRF field `name`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is string describing the j<sup>th</sup> auxiliary data timecourse. While auxiliary data can be given any title, standard names for commonly used auxiliary channels (i.e. accelerometer data) are specified in the appendix.

        """
        if type(self._name) is type(_AbsentDataset):
            return None
        if type(self._name) is type(_PresentDataset):
            return _read_string(self._h['name'])
            self._cfg.logger.info('Dynamically loaded %s/name from %s',
                                  self.location, self.filename)
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        # self._cfg.logger.info('Assignment to %s/name in %s', self.location, self.filename)

    @name.deleter
    def name(self):
        self._name = _AbsentDataset
        self._cfg.logger.info('Deleted %s/name from %s', self.location,
                              self.filename)

    @property
    def dataTimeSeries(self):
        """SNIRF field `dataTimeSeries`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is the aux data variable. This variable has dimensions of `<number of 
        time points> x <number of channels>`. If multiple channels of related data are generated by a system, they may be encoded in the multiple columns of the time series (i.e. complex numbers). For example, a system containing more than one accelerometer may output this data as a set of `ACCEL_X`/`ACCEL_Y`/`ACCEL_Z` auxiliary time series, where each has the dimension of `<number of time points> x <number of accelerometers>`. Note that it is NOT recommended to encode the various accelerometer dimensions as multiple channels of the same `aux` Group: instead follow the `"ACCEL_X"`, `"ACCEL_Y"`, `"ACCEL_Z"` naming conventions described in the appendix. Chunked data is allowed to support real-time data streaming.

        """
        if type(self._dataTimeSeries) is type(_AbsentDataset):
            return None
        if type(self._dataTimeSeries) is type(_PresentDataset):
            return _read_float_array(self._h['dataTimeSeries'])
            self._cfg.logger.info(
                'Dynamically loaded %s/dataTimeSeries from %s', self.location,
                self.filename)
        return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value
        # self._cfg.logger.info('Assignment to %s/dataTimeSeries in %s', self.location, self.filename)

    @dataTimeSeries.deleter
    def dataTimeSeries(self):
        self._dataTimeSeries = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataTimeSeries from %s',
                              self.location, self.filename)

    @property
    def dataUnit(self):
        """SNIRF field `dataUnit`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        International System of Units (SI units) identifier for the given channel. Encoding should follow the [CMIXF-12 standard](https://people.csail.mit.edu/jaffer/MIXF/CMIXF-12), avoiding special unicode symbols like U+03BC (m) or U+00B5 (u) and using '/' rather than 'per' for units such as `V/us`. The recommended export format is in unscaled units such as V, s, Mole.

        """
        if type(self._dataUnit) is type(_AbsentDataset):
            return None
        if type(self._dataUnit) is type(_PresentDataset):
            return _read_string(self._h['dataUnit'])
            self._cfg.logger.info('Dynamically loaded %s/dataUnit from %s',
                                  self.location, self.filename)
        return self._dataUnit

    @dataUnit.setter
    def dataUnit(self, value):
        self._dataUnit = value
        # self._cfg.logger.info('Assignment to %s/dataUnit in %s', self.location, self.filename)

    @dataUnit.deleter
    def dataUnit(self):
        self._dataUnit = _AbsentDataset
        self._cfg.logger.info('Deleted %s/dataUnit from %s', self.location,
                              self.filename)

    @property
    def time(self):
        """SNIRF field `time`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        The time variable. This provides the acquisition time (in `TimeUnit` units) 
        of the aux measurement relative to the time origin.  This will usually be 
        a straight line with slope equal to the acquisition frequency, but does 
        not need to be equal spacing. The size of this variable is 
        `<number of time points>` or `<2>` similar  to definition of the 
        `/nirs(i)/data(j)/time` field.

        Chunked data is allowed to support real-time data streaming

        """
        if type(self._time) is type(_AbsentDataset):
            return None
        if type(self._time) is type(_PresentDataset):
            return _read_float_array(self._h['time'])
            self._cfg.logger.info('Dynamically loaded %s/time from %s',
                                  self.location, self.filename)
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        # self._cfg.logger.info('Assignment to %s/time in %s', self.location, self.filename)

    @time.deleter
    def time(self):
        self._time = _AbsentDataset
        self._cfg.logger.info('Deleted %s/time from %s', self.location,
                              self.filename)

    @property
    def timeOffset(self):
        """SNIRF field `timeOffset`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This variable specifies the offset of the file time origin relative to absolute
        (clock) time in `TimeUnit` units.



        """
        if type(self._timeOffset) is type(_AbsentDataset):
            return None
        if type(self._timeOffset) is type(_PresentDataset):
            return _read_float_array(self._h['timeOffset'])
            self._cfg.logger.info('Dynamically loaded %s/timeOffset from %s',
                                  self.location, self.filename)
        return self._timeOffset

    @timeOffset.setter
    def timeOffset(self, value):
        self._timeOffset = value
        # self._cfg.logger.info('Assignment to %s/timeOffset in %s', self.location, self.filename)

    @timeOffset.deleter
    def timeOffset(self):
        self._timeOffset = _AbsentDataset
        self._cfg.logger.info('Deleted %s/timeOffset from %s', self.location,
                              self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/name'
        if type(self._name) not in [type(_AbsentDataset), type(None)]:
            data = self.name  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataTimeSeries'
        if type(self._dataTimeSeries) not in [
                type(_AbsentDataset), type(None)
        ]:
            data = self.dataTimeSeries  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=2)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/dataUnit'
        if type(self._dataUnit) not in [type(_AbsentDataset), type(None)]:
            data = self.dataUnit  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/time'
        if type(self._time) not in [type(_AbsentDataset), type(None)]:
            data = self.time  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        name = self.location + '/timeOffset'
        if type(self._timeOffset) not in [type(_AbsentDataset), type(None)]:
            data = self.timeOffset  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_float_array(file, name, data, ndim=1)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/name'
            if type(self._name) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._name) is type(
                            _PresentDataset) or 'name' in self._h:
                        dataset = self._h['name']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'name', self._name)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataTimeSeries'
            if type(self._dataTimeSeries) in [
                    type(_AbsentDataset), type(None)
            ]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._dataTimeSeries) is type(
                            _PresentDataset) or 'dataTimeSeries' in self._h:
                        dataset = self._h['dataTimeSeries']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'dataTimeSeries', self._dataTimeSeries)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/dataUnit'
            if type(self._dataUnit) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._dataUnit) is type(
                            _PresentDataset) or 'dataUnit' in self._h:
                        dataset = self._h['dataUnit']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'dataUnit', self._dataUnit)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/time'
            if type(self._time) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._time) is type(
                            _PresentDataset) or 'time' in self._h:
                        dataset = self._h['time']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'time', self._time)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/timeOffset'
            if type(self._timeOffset) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._timeOffset) is type(
                            _PresentDataset) or 'timeOffset' in self._h:
                        dataset = self._h['timeOffset']
                    else:
                        dataset = _create_dataset_float_array(
                            tmp, 'timeOffset', self._timeOffset)
                    result._add(name, _validate_float_array(dataset,
                                                            ndims=[1]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


class Aux(IndexedGroup):
    """Interface for indexed group `Aux`.

    Can be indexed like a list to retrieve `Aux` elements.

    To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively.

    This optional array specifies any recorded auxiliary data. Each element of 
    `aux` has the following required fields:

    """
    _name: str = 'aux'
    _element: Group = AuxElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class Snirf(Group):

    _name = '/'

    # overload
    def __init__(self,
                 *args,
                 dynamic_loading: bool = False,
                 enable_logging: bool = False):
        self._cfg = SnirfConfig()
        self._cfg.dynamic_loading = dynamic_loading
        self._cfg.fmode = ''
        if len(args) > 0:
            path = args[0]
            if enable_logging:
                self._cfg.logger = _create_logger(path,
                                                  path.split('.')[0] + '.log')
            else:
                self._cfg.logger = _create_logger('',
                                                  None)  # Do not log to file
            if len(args) > 1:
                assert type(
                    args[1]
                ) is str, 'Positional argument 2 must be "r"/"w" mode'
                if args[1] == 'r':
                    self._cfg.fmode = 'r'
                elif args[1] == 'r+':
                    self._cfg.fmode = 'r+'
                elif args[1] == 'w':
                    self._cfg.fmode = 'w'
                else:
                    raise ValueError(
                        "Invalid mode: '{}'. Only 'r', 'r+' and 'w' are supported."
                        .format(args[1]))
            else:
                warn(
                    'Use `Snirf(<path>, <mode>)` to open SNIRF file from path. Path-only construction is deprecated.',
                    DeprecationWarning)
                # fmode is ''
            if type(path) is str:
                if not path.endswith('.snirf'):
                    path.replace('.', '')
                    path = path + '.snirf'
                if os.path.exists(path):
                    self._cfg.logger.info('Loading from file %s', path)
                    if self._cfg.fmode == '':
                        self._cfg.fmode = 'r+'
                    self._h = h5py.File(path, self._cfg.fmode)
                else:
                    self._cfg.logger.info('Creating new file at %s', path)
                    if self._cfg.fmode == '':
                        self._cfg.fmode = 'w'
                    self._h = h5py.File(path, self._cfg.fmode)
            elif _isfilelike(args[0]):
                self._cfg.logger.info('Loading from filelike object')
                if self._cfg.fmode == '':
                    self._cfg.fmode = 'r'
                self._h = h5py.File(path, 'r')
            else:
                raise TypeError(str(path) + ' is not a valid filename')
        else:
            path = None
            if enable_logging:
                self._cfg.logger = _logger
                self._cfg.logger.info('Created Snirf object based on tempfile')
            else:
                self._cfg.logger = _create_logger('',
                                                  None)  # Do not log to file
            self._cfg.fmode = 'w'
            self._h = h5py.File(TemporaryFile(), 'w')
        self._formatVersion = _AbsentDataset  # "s"*
        self._nirs = _AbsentDataset  # {i}*
        self._snirf_names = [
            'formatVersion',
            'nirs',
        ]

        self._indexed_groups = []
        if 'formatVersion' in self._h:
            if not self._cfg.dynamic_loading:
                self._formatVersion = _read_string(self._h['formatVersion'])
            else:  # if the dataset is found on disk but dynamic_loading=True
                self._formatVersion = _PresentDataset
        else:  # if the dataset is not found on disk
            self._formatVersion = _AbsentDataset
        self.nirs = Nirs(self, self._cfg)  # Indexed group
        self._indexed_groups.append(self.nirs)

    @property
    def formatVersion(self):
        """SNIRF field `formatVersion`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This is a string that specifies the version of the file format.  This document 
        describes format version "1.0"
         
        """
        if type(self._formatVersion) is type(_AbsentDataset):
            return None
        if type(self._formatVersion) is type(_PresentDataset):
            return _read_string(self._h['formatVersion'])
            self._cfg.logger.info(
                'Dynamically loaded %s/formatVersion from %s', self.location,
                self.filename)
        return self._formatVersion

    @formatVersion.setter
    def formatVersion(self, value):
        self._formatVersion = value
        # self._cfg.logger.info('Assignment to %s/formatVersion in %s', self.location, self.filename)

    @formatVersion.deleter
    def formatVersion(self):
        self._formatVersion = _AbsentDataset
        self._cfg.logger.info('Deleted %s/formatVersion from %s',
                              self.location, self.filename)

    @property
    def nirs(self):
        """SNIRF field `nirs`.

        If dynamic_loading=True, the data is loaded from the SNIRF file only
        when accessed through the getter

        This group stores one set of NIRS data.  This can be extended by adding the count 
        number (e.g. `/nirs1`, `/nirs2`,...) to the group name. This is intended to 
        allow the storage of 1 or more complete NIRS datasets inside a single SNIRF 
        document.  For example, a two-subject hyperscanning can be stored using the notation
        * `/nirs1` =  first subject's data
        * `/nirs2` =  second subject's data
        The use of a non-indexed (e.g. `/nirs`) entry is allowed when only one entry 
        is present and is assumed to be entry 1.


        """
        return self._nirs

    @nirs.setter
    def nirs(self, value):
        self._nirs = value
        # self._cfg.logger.info('Assignment to %s/nirs in %s', self.location, self.filename)

    @nirs.deleter
    def nirs(self):
        raise AttributeError('IndexedGroup ' + str(type(self._nirs)) +
                             ' cannot be deleted')
        self._cfg.logger.info('Deleted %s/nirs from %s', self.location,
                              self.filename)

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
                raise ValueError('Cannot save an anonymous ' +
                                 self.__class__.__name__ +
                                 ' instance without a filename')
        name = self.location + '/formatVersion'
        if type(self._formatVersion) not in [type(_AbsentDataset), type(None)]:
            data = self.formatVersion  # Use loader function via getter
            if name in file:
                del file[name]
            _create_dataset_string(file, name, data)
            # self._cfg.logger.info('Creating Dataset %s in %s', name, file)
        else:
            if name in file:
                del file[name]
                self._cfg.logger.info('Deleted Dataset %s from %s', name, file)
        self.nirs._save(*args)

    def _validate(self, result: ValidationResult):
        # Validate unwritten datasets after writing them to this tempfile
        with h5py.File(TemporaryFile(), 'w') as tmp:
            name = self.location + '/formatVersion'
            if type(self._formatVersion) in [type(_AbsentDataset), type(None)]:
                result._add(name, 'REQUIRED_DATASET_MISSING')
            else:
                try:
                    if type(self._formatVersion) is type(
                            _PresentDataset) or 'formatVersion' in self._h:
                        dataset = self._h['formatVersion']
                    else:
                        dataset = _create_dataset_string(
                            tmp, 'formatVersion', self._formatVersion)
                    result._add(name, _validate_string(dataset))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(name, 'INVALID_DATASET_TYPE')
            name = self.location + '/nirs'
            if len(self._nirs) == 0:
                result._add(name, 'REQUIRED_INDEXED_GROUP_EMPTY')
            else:
                self.nirs._validate(result)
            for key in self._h.keys():
                if not any(
                    [key.startswith(name) for name in self._snirf_names]):
                    if type(self._h[key]) is h5py.Group:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_GROUP')
                    elif type(self._h[key]) is h5py.Dataset:
                        result._add(self.location + '/' + key,
                                    'UNRECOGNIZED_DATASET')


_RECOGNIZED_COORDINATE_SYSTEM_NAMES = [
    'ICBM452AirSpace',
    'ICBM452Warp5Space',
    'IXI549Space',
    'fsaverage',
    'fsaverageSym',
    'fsLR',
    'MNIColin27',
    'MNI152Lin',
    'MNI152NLin2009[a-c][Sym|Asym]',
    'MNI152NLin6Sym',
    'MNI152NLin6ASym',
    'MNI305',
    'NIHPD',
    'OASIS30AntsOASISAnts',
    'OASIS30Atropos',
    'Talairach',
    'UNCInfant',
]

# -- Extend metaDataTags to support addition of new unspecified datasets ------


class MetaDataTags(MetaDataTags):
    def add(self, name, value):
        """Add a new tag to the list.
        
        Args:
            name (str): The name of the tag to add (will be added as an attribute of this `MetaDataTags` instance)
            value: The value of the new tag
        """
        if type(name) is not str:
            raise ValueError('name must be str, not ' + str(type(name)))
        try:
            self.__dict__[name] = value
        except AttributeError as e:
            raise AttributeError(
                "can't set tag. You cannot set the required metaDataTags fields using add() or use protected attributes of MetaDataTags such as 'location' or 'filename'"
            )
        if name not in self._unspecified_names:
            self._unspecified_names.append(name)

    def remove(self, name):
        """Remove a tag from the list. You cannot remove a required tag.
        
        Args:
            name (str): The name of the tag to remove.
        """
        if type(name) is not str:
            raise ValueError('name must be str, not ' + str(type(name)))
        if name not in self._unspecified_names:
            raise AttributeError("no unspecified tag '" + name + "'")
        del self.__dict__[name]


# -- Manually extend _validate to provide detailed error codes ----------------


class StimElement(StimElement):
    def _validate(self, result: ValidationResult):
        super()._validate(result)

        if all(attr is not None for attr in [self.data, self.dataLabels]):
            try:
                if np.shape(self.data)[1] != self.dataLabels.size:
                    result._add(self.location + '/dataLabels',
                                'INVALID_STIM_DATALABELS')
            except IndexError:  # If data doesn't have columns
                result._add(self.location + '/data', 'INVALID_DATASET_SHAPE')


class Stim(Stim):
    _element = StimElement


class AuxElement(AuxElement):
    def _validate(self, result: ValidationResult):
        super()._validate(result)

        if all(attr is not None for attr in [self.time, self.dataTimeSeries]):
            if self.time.size != self.dataTimeSeries.size:
                result._add(self.location + '/time', 'INVALID_TIME')


class Aux(Aux):
    _element = AuxElement


class DataElement(DataElement):
    def _validate(self, result: ValidationResult):
        super()._validate(result)

        if all(attr is not None for attr in [self.time, self.dataTimeSeries]):
            if self.time.size != np.shape(self.dataTimeSeries)[0]:
                result._add(self.location + '/time', 'INVALID_TIME')

            if len(self.measurementList) != np.shape(self.dataTimeSeries)[1]:
                result._add(self.location, 'INVALID_MEASUREMENTLIST')


class Data(Data):
    _element = DataElement


class Probe(Probe):
    def _validate(self, result: ValidationResult):

        # Override sourceLabels validation, can be 1D or 2D
        with h5py.File(TemporaryFile(), 'w') as tmp:
            if type(self._sourceLabels) in [type(_AbsentDataset), type(None)]:
                result._add(self.location + '/sourceLabels',
                            'OPTIONAL_DATASET_MISSING')
            else:
                try:
                    if type(self._sourceLabels) is type(
                            _PresentDataset) or 'sourceLabels' in self._h:
                        dataset = self._h['sourceLabels']
                    else:
                        dataset = _create_dataset_string_array(
                            tmp, 'sourceLabels', self._sourceLabels)
                    result._add(self.location + '/sourceLabels',
                                _validate_string_array(dataset, ndims=[1, 2]))
                except ValueError:  # If the _create_dataset function can't convert the data
                    result._add(self.location + '/sourceLabels',
                                'INVALID_DATASET_TYPE')

        s2 = self.sourcePos2D is not None
        d2 = self.detectorPos2D is not None
        s3 = self.sourcePos3D is not None
        d3 = self.detectorPos3D is not None
        if (s2 and d2):
            result._add(self.location + '/sourcePos2D', 'OK')
            result._add(self.location + '/detectorPos2D', 'OK')
            result._add(self.location + '/sourcePos3D',
                        'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/detectorPos3D',
                        'OPTIONAL_DATASET_MISSING')
        elif (s3 and d3):
            result._add(self.location + '/sourcePos2D',
                        'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/detectorPos2D',
                        'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/sourcePos3D', 'OK')
            result._add(self.location + '/detectorPos3D', 'OK')
        else:
            result._add(self.location + '/sourcePos2D',
                        ['REQUIRED_DATASET_MISSING', 'OK'][int(s2)])
            result._add(self.location + '/detectorPos2D',
                        ['REQUIRED_DATASET_MISSING', 'OK'][int(d2)])
            result._add(self.location + '/sourcePos3D',
                        ['REQUIRED_DATASET_MISSING', 'OK'][int(s3)])
            result._add(self.location + '/detectorPos3D',
                        ['REQUIRED_DATASET_MISSING', 'OK'][int(d3)])

        if self.coordinateSystem is not None:
            if not self.coordinateSystem in _RECOGNIZED_COORDINATE_SYSTEM_NAMES:
                result._add(self.location + '/coordinateSystem',
                            'UNRECOGNIZED_COORDINATE_SYSTEM')
                if self.coordinateSystemDescription is None:
                    result._add(self.location + '/coordinateSystemDescription',
                                'NO_COORDINATE_SYSTEM_DESCRIPTION')

        # The above will supersede the errors from the template code because
        # duplicate names cannot be added to the issues list
        super()._validate(result)


class Snirf(Snirf):

    # overload
    def save(self, *args):
        """Save a SNIRF file to disk.

        Args:
            args (str or h5py.File or file-like): A path to a closed or nonexistant SNIRF file on disk or an open `h5py.File` instance

        Examples:
            save can overwrite the current contents of a Snirf file:
            >>> mysnirf.save()

            or take a new filename to write the file there:
            >>> mysnirf.save(<new destination>)
            
            or write to an IO stream:
            >>> mysnirf.save(<io.BytesIO stream>)
        """
        if len(args) > 0 and type(args[0]) is str:
            path = args[0]
            if not path.endswith('.snirf'):
                path.replace('.', '')
                path += '.snirf'
            if self.filename == path:
                self._save(self._h.file)
                return
            with h5py.File(path, 'w') as new_file:
                self._save(new_file)
                self._cfg.logger.info('Saved Snirf file at %s to copy at %s',
                                      self.filename, path)
        elif len(args) > 0 and _isfilelike(args[0]):
            with h5py.File(args[0], 'w') as stream:
                self._save(stream)
                self._cfg.logger.info('Saved Snirf file to filelike object')
        else:
            self._save(self._h.file)

    def copy(self) -> Snirf:
        """Return a copy of the Snirf instance.
            
        A copy of a Snirf instance is a brand new HDF5 file in memory. This can 
        be expensive to create. Note that in lieu of copying you can make assignments
        between Snirf instances. 
        """
        s = Snirf('r+')
        s = _recursive_hdf5_copy(s, self)
        return s

    def validate(self) -> ValidationResult:
        """Validate a `Snirf` instance.

        Returns the validity of the current state of a `Snirf` object, including
        modifications made in memory to a loaded SNIRF file.

        Returns:
            ValidationResult: truthy structure containing detailed validation results
        """
        result = ValidationResult()
        self._validate(result)
        return result

    # overload
    @property
    def filename(self):
        """The filename the Snirf object was loaded from and will save to."""
        if self._h != {}:
            return self._h.filename
        else:
            return None

    def close(self):
        """Close the file underlying a `Snirf` instance.

        After closing, the underlying SNIRF file cannot be accessed from this interface again.
        Use `close` if you need to open a new interface on the same HDF5 file.

        `close` is called automatically by the destructor.
        """
        self._cfg.logger.info('Closing Snirf file %s', self.filename)
        _close_logger(self._cfg.logger)
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return True if exc_type is None else False

    def __getitem__(self, key):
        if self._h != {}:
            if key in self._h:
                return self._h[key]
        else:
            return None

    def _validate(self, result: ValidationResult):
        super()._validate(result)

        # TODO INVALID_FILENAME, INVALID_FILE detection

        for nirs in self.nirs:
            if type(nirs.probe) not in [type(None), type(_AbsentGroup)]:
                if nirs.probe.sourceLabels is not None:
                    lenSourceLabels = nirs.probe.sourceLabels.shape[0]
                else:
                    lenSourceLabels = 0
                if nirs.probe.detectorLabels is not None:
                    lenDetectorLabels = nirs.probe.detectorLabels.size
                else:
                    lenDetectorLabels = 0
                if nirs.probe.wavelengths is not None:
                    lenWavelengths = nirs.probe.wavelengths.size
                else:
                    lenWavelengths = 0
                for data in nirs.data:
                    for ml in data.measurementList:
                        if ml.sourceIndex is not None:
                            if ml.sourceIndex > lenSourceLabels:
                                result._add(ml.location + '/sourceIndex',
                                            'INVALID_SOURCE_INDEX')
                        if ml.detectorIndex is not None:
                            if ml.detectorIndex > lenDetectorLabels:
                                result._add(ml.location + '/detectorIndex',
                                            'INVALID_DETECTOR_INDEX')
                        if ml.wavelengthIndex is not None:
                            if ml.wavelengthIndex > lenWavelengths:
                                result._add(ml.location + '/wavelengthIndex',
                                            'INVALID_WAVELENGTH_INDEX')


# -- Interface functions ----------------------------------------------------


def loadSnirf(path: str,
              dynamic_loading: bool = False,
              enable_logging: bool = False) -> Snirf:
    """Load a SNIRF file from disk.
    
    Returns a `Snirf` object loaded from path if a SNIRF file exists there. Takes
    the same kwargs as the Snirf object constructor
    
    Args:
        path (str): Path to a SNIRF file on disk.
        dynamic_loading (bool): If True, Datasets will not be read from the SNIRF file
            unless accessed with a property, conserving memory and loading time with larger datasets. Default False.
        enable_logging (bool): If True, the `Snirf` instance will write to a log file which shares its name. Default False.
    
    Returns:
        `Snirf`: a `Snirf` instance loaded from the SNIRF file.   
    
    Raises:
        FileNotFoundError: `path` was not found on disk.
    """
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        return Snirf(path,
                     'r+',
                     dynamic_loading=dynamic_loading,
                     enable_logging=enable_logging)
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)


def saveSnirf(path: str, snirf: Snirf):
    """Saves a SNIRF file to disk.
    
    Args:
        path (str): Path to save the file.
        snirf (Snirf): `Snirf` instance to write to disk.
    """
    if type(path) is not str:
        raise TypeError('path must be str, not ' + str(type(path)))
    if not isinstance(snirf, Snirf):
        raise TypeError('snirf must be Snirf, not ' + str(type(snirf)))
    snirf.save(path)


def validateSnirf(path: str) -> ValidationResult:
    """Validate a SNIRF file on disk.
    
    Returns truthy ValidationResult instance which holds detailed results of validation
    """
    if type(path) is not str:
        raise TypeError('path must be str, not ' + str(type(path)))
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        with Snirf(path, 'r') as snirf:
            return snirf.validate()
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)
