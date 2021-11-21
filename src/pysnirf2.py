from abc import ABC, abstractmethod
import h5py
import os
import sys
import numpy as np
from warnings import warn
from collections import MutableSequence
from tempfile import TemporaryFile

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')

# -- methods for writing and reading ------

    
def _read_string(dataset):
    # Because many SNIRF files are saved with string values in length 1 arrays
    if dataset.ndim > 0:
        return dataset[0].decode('ascii')
    else:
        return dataset[()].decode('ascii')


# -----------------------------------------


class SnirfFormatError(Exception):
    pass


class SnirfConfig:
    dynamic_loading: bool = False  # If False, data is loaded in the constructor, if True, data is loaded on access


class Group(ABC):

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        self._id = gid
        if not isinstance(gid, h5py.h5g.GroupID):
            raise TypeError('must initialize with a Group ID, not ' + str(type(gid)))
        self._h = h5py.Group(self._id)
        self._cfg = cfg

    def save(self, *args):
        if len(args) > 0:
            if type(args[0]) is h5py.File:
                self._save(args[0])
            elif type(args[0]) is str:
                path = args[0]
                if not path.endswith('.snirf'):
                    path += '.snirf'
                if os.path.exists(path):
                    file = h5py.File(path, 'r+')
                else:
                    file = h5py.File(path, 'w')
                self._save(file)
                file.close()
        else:
            self._save()

    @property
    def filename(self):
        return self._h.file.filename
    
    @abstractmethod
    def _save(self, *args):
        raise NotImplementedError('_save is an abstract method')
        
    def __repr__(self):
        props = [p for p in dir(self) if ('_' not in p and not callable(getattr(self, p)))]
        out = str(self.__class__.__name__) + ' at ' + str(self._h.name) + '\n'
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
    
    _name: str = ''
    _element: Group = None

    def __init__(self, parent: (h5py.Group, h5py.File), cfg: SnirfConfig):
        if isinstance(parent, (h5py.Group, h5py.File)):
            # Because the indexed group is not a true HDF5 group but rather an
            # iterable list of HDF5 groups, it takes a base group or file and
            # searches its keys, appending the appropriate elements to itself
            # in order
            self._parent = parent
            self._cfg = cfg
            self._list = list()
            unordered = []
            indices = []
            for key in self._parent.keys():
                numsplit = key.split(self._name)
                if len(numsplit) > 1 and len(numsplit[1]) > 0:
                    if len(numsplit[1]) == len(str(int(numsplit[1]))):
                        unordered.append(self._element(self._parent[key].id, self._cfg))
                        indices.append(int(numsplit[1]))
                elif key.endswith(self._name):
                    unordered.append(self._element(self._parent[key].id, self._cfg))
                    indices.append(0)
            ordered = np.argsort(indices)
            for i, j in enumerate(ordered):
                self._list.append(unordered[j])
        else:
            raise TypeError('must initialize IndexedGroup with a Group or File')
    
    
    def __len__(self): return len(self._list)

    def __getitem__(self, i): return self._list[i]

    def __delitem__(self, i): del self._list[i]
    
    def __setitem__(self, i, item):
        self._check_type(item)
        self._list[i] = item
        self._order_names

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
        self._save(*args)
    
    def appendGroup(self):
        'Adds a group to the end of the list'
        g = self._parent.createGroup(self._name + str(len(self._list) + 1))
        gid = g.id
        self._list.append(self._element(gid, self._cfg))
    
    def _check_type(self, item):
        if type(item) is not self._element:
            raise TypeError('elements of ' + str(self.__class__.__name__) +
                            ' must be ' + str(self._element) + ', not ' +
                            str(type(item))
                            )
        
    def _order_names(self):
        for i, element in enumerate(self._list):
            self._parent.move(element._h.name.split('/')[-1], self._name + str(i + 1))
#            print(element._h.name.split('/')[-1], '--->', self._name + str(i + 1))
        
    def _save(self, *args):
        self._order_names()
        [element._save(*args) for element in self._list]


# generated by sstucker on 2021-11-21
# version 1.0 SNIRF specification parsed from https://raw.githubusercontent.com/fNIRS/snirf/v1.0/snirf_specification.md


class MetaDataTags(Group):

    _SubjectID = None  # "s"*
    _MeasurementDate = None  # "s"*
    _MeasurementTime = None  # "s"*
    _LengthUnit = None  # "s"*
    _TimeUnit = None  # "s"*
    _FrequencyUnit = None  # "s"*

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'SubjectID' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._SubjectID = _read_string(self._h['SubjectID'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"SubjectID"')

        if 'MeasurementDate' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._MeasurementDate = _read_string(self._h['MeasurementDate'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"MeasurementDate"')

        if 'MeasurementTime' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._MeasurementTime = _read_string(self._h['MeasurementTime'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"MeasurementTime"')

        if 'LengthUnit' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._LengthUnit = _read_string(self._h['LengthUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"LengthUnit"')

        if 'TimeUnit' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._TimeUnit = _read_string(self._h['TimeUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"TimeUnit"')

        if 'FrequencyUnit' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._FrequencyUnit = _read_string(self._h['FrequencyUnit'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"FrequencyUnit"')


    @property
    def SubjectID(self):
        if self._cfg.dynamic_loading and self._SubjectID is None:
            if 'SubjectID' in self._h.keys():
                return _read_string(self._h['SubjectID'])
        return self._SubjectID

    @SubjectID.setter
    def SubjectID(self, value):
        self._SubjectID = value

    @property
    def MeasurementDate(self):
        if self._cfg.dynamic_loading and self._MeasurementDate is None:
            if 'MeasurementDate' in self._h.keys():
                return _read_string(self._h['MeasurementDate'])
        return self._MeasurementDate

    @MeasurementDate.setter
    def MeasurementDate(self, value):
        self._MeasurementDate = value

    @property
    def MeasurementTime(self):
        if self._cfg.dynamic_loading and self._MeasurementTime is None:
            if 'MeasurementTime' in self._h.keys():
                return _read_string(self._h['MeasurementTime'])
        return self._MeasurementTime

    @MeasurementTime.setter
    def MeasurementTime(self, value):
        self._MeasurementTime = value

    @property
    def LengthUnit(self):
        if self._cfg.dynamic_loading and self._LengthUnit is None:
            if 'LengthUnit' in self._h.keys():
                return _read_string(self._h['LengthUnit'])
        return self._LengthUnit

    @LengthUnit.setter
    def LengthUnit(self, value):
        self._LengthUnit = value

    @property
    def TimeUnit(self):
        if self._cfg.dynamic_loading and self._TimeUnit is None:
            if 'TimeUnit' in self._h.keys():
                return _read_string(self._h['TimeUnit'])
        return self._TimeUnit

    @TimeUnit.setter
    def TimeUnit(self, value):
        self._TimeUnit = value

    @property
    def FrequencyUnit(self):
        if self._cfg.dynamic_loading and self._FrequencyUnit is None:
            if 'FrequencyUnit' in self._h.keys():
                return _read_string(self._h['FrequencyUnit'])
        return self._FrequencyUnit

    @FrequencyUnit.setter
    def FrequencyUnit(self, value):
        self._FrequencyUnit = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'SubjectID' in self._h:
            name = self._h['SubjectID'].name
            data = self.SubjectID
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'MeasurementDate' in self._h:
            name = self._h['MeasurementDate'].name
            data = self.MeasurementDate
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'MeasurementTime' in self._h:
            name = self._h['MeasurementTime'].name
            data = self.MeasurementTime
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'LengthUnit' in self._h:
            name = self._h['LengthUnit'].name
            data = self.LengthUnit
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'TimeUnit' in self._h:
            name = self._h['TimeUnit'].name
            data = self.TimeUnit
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'FrequencyUnit' in self._h:
            name = self._h['FrequencyUnit'].name
            data = self.FrequencyUnit
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)



class Probe(Group):

    _wavelengths = None  # [<f>,...]*
    _wavelengthsEmission = None  # [<f>,...]
    _sourcePos2D = None  # [[<f>,...]]*1
    _sourcePos3D = None  # [[<f>,...]]*1
    _detectorPos2D = None  # [[<f>,...]]*2
    _detectorPos3D = None  # [[<f>,...]]*2
    _frequencies = None  # [<f>,...]
    _timeDelays = None  # [<f>,...]
    _timeDelayWidths = None  # [<f>,...]
    _momentOrders = None  # [<f>,...]
    _correlationTimeDelays = None  # [<f>,...]
    _correlationTimeDelayWidths = None  # [<f>,...]
    _sourceLabels = None  # ["s",...]
    _detectorLabels = None  # ["s",...]
    _landmarkPos2D = None  # [[<f>,...]]
    _landmarkPos3D = None  # [[<f>,...]]
    _landmarkLabels = None  # ["s",...]
    _useLocalIndex = None  # <i>

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'wavelengths' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._wavelengths = np.array(self._h['wavelengths']).astype(float)  # Array
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"wavelengths"')

        if 'wavelengthsEmission' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._wavelengthsEmission = np.array(self._h['wavelengthsEmission']).astype(float)  # Array

        if 'sourcePos2D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourcePos2D = np.array(self._h['sourcePos2D']).astype(float)  # Array

        if 'sourcePos3D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourcePos3D = np.array(self._h['sourcePos3D']).astype(float)  # Array

        if 'detectorPos2D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorPos2D = np.array(self._h['detectorPos2D']).astype(float)  # Array

        if 'detectorPos3D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorPos3D = np.array(self._h['detectorPos3D']).astype(float)  # Array

        if 'frequencies' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._frequencies = np.array(self._h['frequencies']).astype(float)  # Array

        if 'timeDelays' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._timeDelays = np.array(self._h['timeDelays']).astype(float)  # Array

        if 'timeDelayWidths' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._timeDelayWidths = np.array(self._h['timeDelayWidths']).astype(float)  # Array

        if 'momentOrders' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._momentOrders = np.array(self._h['momentOrders']).astype(float)  # Array

        if 'correlationTimeDelays' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelays = np.array(self._h['correlationTimeDelays']).astype(float)  # Array

        if 'correlationTimeDelayWidths' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._correlationTimeDelayWidths = np.array(self._h['correlationTimeDelayWidths']).astype(float)  # Array

        if 'sourceLabels' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourceLabels = np.array(self._h['sourceLabels']).astype(str)  # Array

        if 'detectorLabels' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorLabels = np.array(self._h['detectorLabels']).astype(str)  # Array

        if 'landmarkPos2D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._landmarkPos2D = np.array(self._h['landmarkPos2D']).astype(float)  # Array

        if 'landmarkPos3D' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._landmarkPos3D = np.array(self._h['landmarkPos3D']).astype(float)  # Array

        if 'landmarkLabels' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._landmarkLabels = np.array(self._h['landmarkLabels']).astype(str)  # Array

        if 'useLocalIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._useLocalIndex = int(self._h['useLocalIndex'][()])


    @property
    def wavelengths(self):
        if self._cfg.dynamic_loading and self._wavelengths is None:
            if 'wavelengths' in self._h.keys():
                return np.array(self._h['wavelengths']).astype(float)  # Array
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    @property
    def wavelengthsEmission(self):
        if self._cfg.dynamic_loading and self._wavelengthsEmission is None:
            if 'wavelengthsEmission' in self._h.keys():
                return np.array(self._h['wavelengthsEmission']).astype(float)  # Array
        return self._wavelengthsEmission

    @wavelengthsEmission.setter
    def wavelengthsEmission(self, value):
        self._wavelengthsEmission = value

    @property
    def sourcePos2D(self):
        if self._cfg.dynamic_loading and self._sourcePos2D is None:
            if 'sourcePos2D' in self._h.keys():
                return np.array(self._h['sourcePos2D']).astype(float)  # Array
        return self._sourcePos2D

    @sourcePos2D.setter
    def sourcePos2D(self, value):
        self._sourcePos2D = value

    @property
    def sourcePos3D(self):
        if self._cfg.dynamic_loading and self._sourcePos3D is None:
            if 'sourcePos3D' in self._h.keys():
                return np.array(self._h['sourcePos3D']).astype(float)  # Array
        return self._sourcePos3D

    @sourcePos3D.setter
    def sourcePos3D(self, value):
        self._sourcePos3D = value

    @property
    def detectorPos2D(self):
        if self._cfg.dynamic_loading and self._detectorPos2D is None:
            if 'detectorPos2D' in self._h.keys():
                return np.array(self._h['detectorPos2D']).astype(float)  # Array
        return self._detectorPos2D

    @detectorPos2D.setter
    def detectorPos2D(self, value):
        self._detectorPos2D = value

    @property
    def detectorPos3D(self):
        if self._cfg.dynamic_loading and self._detectorPos3D is None:
            if 'detectorPos3D' in self._h.keys():
                return np.array(self._h['detectorPos3D']).astype(float)  # Array
        return self._detectorPos3D

    @detectorPos3D.setter
    def detectorPos3D(self, value):
        self._detectorPos3D = value

    @property
    def frequencies(self):
        if self._cfg.dynamic_loading and self._frequencies is None:
            if 'frequencies' in self._h.keys():
                return np.array(self._h['frequencies']).astype(float)  # Array
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value):
        self._frequencies = value

    @property
    def timeDelays(self):
        if self._cfg.dynamic_loading and self._timeDelays is None:
            if 'timeDelays' in self._h.keys():
                return np.array(self._h['timeDelays']).astype(float)  # Array
        return self._timeDelays

    @timeDelays.setter
    def timeDelays(self, value):
        self._timeDelays = value

    @property
    def timeDelayWidths(self):
        if self._cfg.dynamic_loading and self._timeDelayWidths is None:
            if 'timeDelayWidths' in self._h.keys():
                return np.array(self._h['timeDelayWidths']).astype(float)  # Array
        return self._timeDelayWidths

    @timeDelayWidths.setter
    def timeDelayWidths(self, value):
        self._timeDelayWidths = value

    @property
    def momentOrders(self):
        if self._cfg.dynamic_loading and self._momentOrders is None:
            if 'momentOrders' in self._h.keys():
                return np.array(self._h['momentOrders']).astype(float)  # Array
        return self._momentOrders

    @momentOrders.setter
    def momentOrders(self, value):
        self._momentOrders = value

    @property
    def correlationTimeDelays(self):
        if self._cfg.dynamic_loading and self._correlationTimeDelays is None:
            if 'correlationTimeDelays' in self._h.keys():
                return np.array(self._h['correlationTimeDelays']).astype(float)  # Array
        return self._correlationTimeDelays

    @correlationTimeDelays.setter
    def correlationTimeDelays(self, value):
        self._correlationTimeDelays = value

    @property
    def correlationTimeDelayWidths(self):
        if self._cfg.dynamic_loading and self._correlationTimeDelayWidths is None:
            if 'correlationTimeDelayWidths' in self._h.keys():
                return np.array(self._h['correlationTimeDelayWidths']).astype(float)  # Array
        return self._correlationTimeDelayWidths

    @correlationTimeDelayWidths.setter
    def correlationTimeDelayWidths(self, value):
        self._correlationTimeDelayWidths = value

    @property
    def sourceLabels(self):
        if self._cfg.dynamic_loading and self._sourceLabels is None:
            if 'sourceLabels' in self._h.keys():
                return np.array(self._h['sourceLabels']).astype(str)  # Array
        return self._sourceLabels

    @sourceLabels.setter
    def sourceLabels(self, value):
        self._sourceLabels = value

    @property
    def detectorLabels(self):
        if self._cfg.dynamic_loading and self._detectorLabels is None:
            if 'detectorLabels' in self._h.keys():
                return np.array(self._h['detectorLabels']).astype(str)  # Array
        return self._detectorLabels

    @detectorLabels.setter
    def detectorLabels(self, value):
        self._detectorLabels = value

    @property
    def landmarkPos2D(self):
        if self._cfg.dynamic_loading and self._landmarkPos2D is None:
            if 'landmarkPos2D' in self._h.keys():
                return np.array(self._h['landmarkPos2D']).astype(float)  # Array
        return self._landmarkPos2D

    @landmarkPos2D.setter
    def landmarkPos2D(self, value):
        self._landmarkPos2D = value

    @property
    def landmarkPos3D(self):
        if self._cfg.dynamic_loading and self._landmarkPos3D is None:
            if 'landmarkPos3D' in self._h.keys():
                return np.array(self._h['landmarkPos3D']).astype(float)  # Array
        return self._landmarkPos3D

    @landmarkPos3D.setter
    def landmarkPos3D(self, value):
        self._landmarkPos3D = value

    @property
    def landmarkLabels(self):
        if self._cfg.dynamic_loading and self._landmarkLabels is None:
            if 'landmarkLabels' in self._h.keys():
                return np.array(self._h['landmarkLabels']).astype(str)  # Array
        return self._landmarkLabels

    @landmarkLabels.setter
    def landmarkLabels(self, value):
        self._landmarkLabels = value

    @property
    def useLocalIndex(self):
        if self._cfg.dynamic_loading and self._useLocalIndex is None:
            if 'useLocalIndex' in self._h.keys():
                return int(self._h['useLocalIndex'][()])
        return self._useLocalIndex

    @useLocalIndex.setter
    def useLocalIndex(self, value):
        self._useLocalIndex = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'wavelengths' in self._h:
            name = self._h['wavelengths'].name
            data = np.array(self.wavelengths)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'wavelengthsEmission' in self._h:
            name = self._h['wavelengthsEmission'].name
            data = np.array(self.wavelengthsEmission)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'sourcePos2D' in self._h:
            name = self._h['sourcePos2D'].name
            data = np.array(self.sourcePos2D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'sourcePos3D' in self._h:
            name = self._h['sourcePos3D'].name
            data = np.array(self.sourcePos3D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'detectorPos2D' in self._h:
            name = self._h['detectorPos2D'].name
            data = np.array(self.detectorPos2D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'detectorPos3D' in self._h:
            name = self._h['detectorPos3D'].name
            data = np.array(self.detectorPos3D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'frequencies' in self._h:
            name = self._h['frequencies'].name
            data = np.array(self.frequencies)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'timeDelays' in self._h:
            name = self._h['timeDelays'].name
            data = np.array(self.timeDelays)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'timeDelayWidths' in self._h:
            name = self._h['timeDelayWidths'].name
            data = np.array(self.timeDelayWidths)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'momentOrders' in self._h:
            name = self._h['momentOrders'].name
            data = np.array(self.momentOrders)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'correlationTimeDelays' in self._h:
            name = self._h['correlationTimeDelays'].name
            data = np.array(self.correlationTimeDelays)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'correlationTimeDelayWidths' in self._h:
            name = self._h['correlationTimeDelayWidths'].name
            data = np.array(self.correlationTimeDelayWidths)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'sourceLabels' in self._h:
            name = self._h['sourceLabels'].name
            data = np.array(self.sourceLabels).astype('O')
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'detectorLabels' in self._h:
            name = self._h['detectorLabels'].name
            data = np.array(self.detectorLabels).astype('O')
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'landmarkPos2D' in self._h:
            name = self._h['landmarkPos2D'].name
            data = np.array(self.landmarkPos2D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'landmarkPos3D' in self._h:
            name = self._h['landmarkPos3D'].name
            data = np.array(self.landmarkPos3D)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'landmarkLabels' in self._h:
            name = self._h['landmarkLabels'].name
            data = np.array(self.landmarkLabels).astype('O')
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'useLocalIndex' in self._h:
            name = self._h['useLocalIndex'].name
            data = self.useLocalIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)



class NirsElement(Group):

    _metaDataTags = None  # {.}*
    _data = None  # {i}*
    _stim = None  # {i}
    _probe = None  # {.}*
    _aux = None  # {i}

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'metaDataTags' in self._h.keys():
            self._metaDataTags = MetaDataTags(self._h['metaDataTags'].id, self._cfg)  # Group
        else:  # Create an empty group
            self._h.create_group('metaDataTags')
            self._metaDataTags = MetaDataTags(self._h['metaDataTags'].id, self._cfg)  # Group

        self.data = Data(self._h, self._cfg)  # Indexed group

        self.stim = Stim(self._h, self._cfg)  # Indexed group

        if 'probe' in self._h.keys():
            self._probe = Probe(self._h['probe'].id, self._cfg)  # Group
        else:  # Create an empty group
            self._h.create_group('probe')
            self._probe = Probe(self._h['probe'].id, self._cfg)  # Group

        self.aux = Aux(self._h, self._cfg)  # Indexed group


    @property
    def metaDataTags(self):
        if 'metaDataTags' in self._h.keys():
            return self._metaDataTags

    @metaDataTags.setter
    def metaDataTags(self, value):
        self._metaDataTags = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def stim(self):
        return self._stim

    @stim.setter
    def stim(self, value):
        self._stim = value

    @property
    def probe(self):
        if 'probe' in self._h.keys():
            return self._probe

    @probe.setter
    def probe(self, value):
        self._probe = value

    @property
    def aux(self):
        return self._aux

    @aux.setter
    def aux(self, value):
        self._aux = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        self.metaDataTags._save(*args)
        self.data._save(*args)
        self.stim._save(*args)
        self.probe._save(*args)
        self.aux._save(*args)


class Nirs(IndexedGroup):

    _name: str = 'nirs'
    _element: Group = NirsElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class DataElement(Group):

    _dataTimeSeries = None  # [[<f>,...]]*
    _time = None  # [<f>,...]*
    _measurementList = None  # {i}*

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'dataTimeSeries' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = np.array(self._h['dataTimeSeries']).astype(float)  # Array
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataTimeSeries"')

        if 'time' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._time = np.array(self._h['time']).astype(float)  # Array
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"time"')

        self.measurementList = MeasurementList(self._h, self._cfg)  # Indexed group


    @property
    def dataTimeSeries(self):
        if self._cfg.dynamic_loading and self._dataTimeSeries is None:
            if 'dataTimeSeries' in self._h.keys():
                return np.array(self._h['dataTimeSeries']).astype(float)  # Array
        return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value

    @property
    def time(self):
        if self._cfg.dynamic_loading and self._time is None:
            if 'time' in self._h.keys():
                return np.array(self._h['time']).astype(float)  # Array
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def measurementList(self):
        return self._measurementList

    @measurementList.setter
    def measurementList(self, value):
        self._measurementList = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'dataTimeSeries' in self._h:
            name = self._h['dataTimeSeries'].name
            data = np.array(self.dataTimeSeries)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'time' in self._h:
            name = self._h['time'].name
            data = np.array(self.time)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        self.measurementList._save(*args)


class Data(IndexedGroup):

    _name: str = 'data'
    _element: Group = DataElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class MeasurementListElement(Group):

    _sourceIndex = None  # <i>*
    _detectorIndex = None  # <i>*
    _wavelengthIndex = None  # <i>*
    _wavelengthActual = None  # <f>
    _wavelengthEmissionActual = None  # <f>
    _dataType = None  # <i>*
    _dataTypeLabel = None  # "s"
    _dataTypeIndex = None  # <i>*
    _sourcePower = None  # <f>
    _detectorGain = None  # <f>
    _moduleIndex = None  # <i>
    _sourceModuleIndex = None  # <i>
    _detectorModuleIndex = None  # <i>

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'sourceIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourceIndex = int(self._h['sourceIndex'][()])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"sourceIndex"')

        if 'detectorIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorIndex = int(self._h['detectorIndex'][()])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"detectorIndex"')

        if 'wavelengthIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._wavelengthIndex = int(self._h['wavelengthIndex'][()])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"wavelengthIndex"')

        if 'wavelengthActual' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._wavelengthActual = float(self._h['wavelengthActual'][()])

        if 'wavelengthEmissionActual' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._wavelengthEmissionActual = float(self._h['wavelengthEmissionActual'][()])

        if 'dataType' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataType = int(self._h['dataType'][()])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataType"')

        if 'dataTypeLabel' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataTypeLabel = _read_string(self._h['dataTypeLabel'])

        if 'dataTypeIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataTypeIndex = int(self._h['dataTypeIndex'][()])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"dataTypeIndex"')

        if 'sourcePower' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourcePower = float(self._h['sourcePower'][()])

        if 'detectorGain' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorGain = float(self._h['detectorGain'][()])

        if 'moduleIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._moduleIndex = int(self._h['moduleIndex'][()])

        if 'sourceModuleIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._sourceModuleIndex = int(self._h['sourceModuleIndex'][()])

        if 'detectorModuleIndex' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._detectorModuleIndex = int(self._h['detectorModuleIndex'][()])


    @property
    def sourceIndex(self):
        if self._cfg.dynamic_loading and self._sourceIndex is None:
            if 'sourceIndex' in self._h.keys():
                return int(self._h['sourceIndex'][()])
        return self._sourceIndex

    @sourceIndex.setter
    def sourceIndex(self, value):
        self._sourceIndex = value

    @property
    def detectorIndex(self):
        if self._cfg.dynamic_loading and self._detectorIndex is None:
            if 'detectorIndex' in self._h.keys():
                return int(self._h['detectorIndex'][()])
        return self._detectorIndex

    @detectorIndex.setter
    def detectorIndex(self, value):
        self._detectorIndex = value

    @property
    def wavelengthIndex(self):
        if self._cfg.dynamic_loading and self._wavelengthIndex is None:
            if 'wavelengthIndex' in self._h.keys():
                return int(self._h['wavelengthIndex'][()])
        return self._wavelengthIndex

    @wavelengthIndex.setter
    def wavelengthIndex(self, value):
        self._wavelengthIndex = value

    @property
    def wavelengthActual(self):
        if self._cfg.dynamic_loading and self._wavelengthActual is None:
            if 'wavelengthActual' in self._h.keys():
                return float(self._h['wavelengthActual'][()])
        return self._wavelengthActual

    @wavelengthActual.setter
    def wavelengthActual(self, value):
        self._wavelengthActual = value

    @property
    def wavelengthEmissionActual(self):
        if self._cfg.dynamic_loading and self._wavelengthEmissionActual is None:
            if 'wavelengthEmissionActual' in self._h.keys():
                return float(self._h['wavelengthEmissionActual'][()])
        return self._wavelengthEmissionActual

    @wavelengthEmissionActual.setter
    def wavelengthEmissionActual(self, value):
        self._wavelengthEmissionActual = value

    @property
    def dataType(self):
        if self._cfg.dynamic_loading and self._dataType is None:
            if 'dataType' in self._h.keys():
                return int(self._h['dataType'][()])
        return self._dataType

    @dataType.setter
    def dataType(self, value):
        self._dataType = value

    @property
    def dataTypeLabel(self):
        if self._cfg.dynamic_loading and self._dataTypeLabel is None:
            if 'dataTypeLabel' in self._h.keys():
                return _read_string(self._h['dataTypeLabel'])
        return self._dataTypeLabel

    @dataTypeLabel.setter
    def dataTypeLabel(self, value):
        self._dataTypeLabel = value

    @property
    def dataTypeIndex(self):
        if self._cfg.dynamic_loading and self._dataTypeIndex is None:
            if 'dataTypeIndex' in self._h.keys():
                return int(self._h['dataTypeIndex'][()])
        return self._dataTypeIndex

    @dataTypeIndex.setter
    def dataTypeIndex(self, value):
        self._dataTypeIndex = value

    @property
    def sourcePower(self):
        if self._cfg.dynamic_loading and self._sourcePower is None:
            if 'sourcePower' in self._h.keys():
                return float(self._h['sourcePower'][()])
        return self._sourcePower

    @sourcePower.setter
    def sourcePower(self, value):
        self._sourcePower = value

    @property
    def detectorGain(self):
        if self._cfg.dynamic_loading and self._detectorGain is None:
            if 'detectorGain' in self._h.keys():
                return float(self._h['detectorGain'][()])
        return self._detectorGain

    @detectorGain.setter
    def detectorGain(self, value):
        self._detectorGain = value

    @property
    def moduleIndex(self):
        if self._cfg.dynamic_loading and self._moduleIndex is None:
            if 'moduleIndex' in self._h.keys():
                return int(self._h['moduleIndex'][()])
        return self._moduleIndex

    @moduleIndex.setter
    def moduleIndex(self, value):
        self._moduleIndex = value

    @property
    def sourceModuleIndex(self):
        if self._cfg.dynamic_loading and self._sourceModuleIndex is None:
            if 'sourceModuleIndex' in self._h.keys():
                return int(self._h['sourceModuleIndex'][()])
        return self._sourceModuleIndex

    @sourceModuleIndex.setter
    def sourceModuleIndex(self, value):
        self._sourceModuleIndex = value

    @property
    def detectorModuleIndex(self):
        if self._cfg.dynamic_loading and self._detectorModuleIndex is None:
            if 'detectorModuleIndex' in self._h.keys():
                return int(self._h['detectorModuleIndex'][()])
        return self._detectorModuleIndex

    @detectorModuleIndex.setter
    def detectorModuleIndex(self, value):
        self._detectorModuleIndex = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'sourceIndex' in self._h:
            name = self._h['sourceIndex'].name
            data = self.sourceIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'detectorIndex' in self._h:
            name = self._h['detectorIndex'].name
            data = self.detectorIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'wavelengthIndex' in self._h:
            name = self._h['wavelengthIndex'].name
            data = self.wavelengthIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'wavelengthActual' in self._h:
            name = self._h['wavelengthActual'].name
            data = self.wavelengthActual
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'wavelengthEmissionActual' in self._h:
            name = self._h['wavelengthEmissionActual'].name
            data = self.wavelengthEmissionActual
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'dataType' in self._h:
            name = self._h['dataType'].name
            data = self.dataType
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'dataTypeLabel' in self._h:
            name = self._h['dataTypeLabel'].name
            data = self.dataTypeLabel
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'dataTypeIndex' in self._h:
            name = self._h['dataTypeIndex'].name
            data = self.dataTypeIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'sourcePower' in self._h:
            name = self._h['sourcePower'].name
            data = self.sourcePower
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'detectorGain' in self._h:
            name = self._h['detectorGain'].name
            data = self.detectorGain
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'moduleIndex' in self._h:
            name = self._h['moduleIndex'].name
            data = self.moduleIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'sourceModuleIndex' in self._h:
            name = self._h['sourceModuleIndex'].name
            data = self.sourceModuleIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)
        if 'detectorModuleIndex' in self._h:
            name = self._h['detectorModuleIndex'].name
            data = self.detectorModuleIndex
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='i4', data=data)


class MeasurementList(IndexedGroup):

    _name: str = 'measurementList'
    _element: Group = MeasurementListElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class StimElement(Group):

    _name = None  # "s"+
    _data = None  # [<f>,...]+
    _dataLabels = None  # ["s",...]

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'name' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])

        if 'data' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._data = np.array(self._h['data']).astype(float)  # Array

        if 'dataLabels' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataLabels = np.array(self._h['dataLabels']).astype(str)  # Array


    @property
    def name(self):
        if self._cfg.dynamic_loading and self._name is None:
            if 'name' in self._h.keys():
                return _read_string(self._h['name'])
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def data(self):
        if self._cfg.dynamic_loading and self._data is None:
            if 'data' in self._h.keys():
                return np.array(self._h['data']).astype(float)  # Array
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def dataLabels(self):
        if self._cfg.dynamic_loading and self._dataLabels is None:
            if 'dataLabels' in self._h.keys():
                return np.array(self._h['dataLabels']).astype(str)  # Array
        return self._dataLabels

    @dataLabels.setter
    def dataLabels(self, value):
        self._dataLabels = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'name' in self._h:
            name = self._h['name'].name
            data = self.name
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'data' in self._h:
            name = self._h['data'].name
            data = np.array(self.data)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'dataLabels' in self._h:
            name = self._h['dataLabels'].name
            data = np.array(self.dataLabels).astype('O')
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)


class Stim(IndexedGroup):

    _name: str = 'stim'
    _element: Group = StimElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class AuxElement(Group):

    _name = None  # "s"+
    _dataTimeSeries = None  # [[<f>,...]]+
    _time = None  # [<f>,...]+
    _timeOffset = None  # [<f>,...]

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        super().__init__(gid, cfg)
        if 'name' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._name = _read_string(self._h['name'])

        if 'dataTimeSeries' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._dataTimeSeries = np.array(self._h['dataTimeSeries']).astype(float)  # Array

        if 'time' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._time = np.array(self._h['time']).astype(float)  # Array

        if 'timeOffset' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._timeOffset = np.array(self._h['timeOffset']).astype(float)  # Array


    @property
    def name(self):
        if self._cfg.dynamic_loading and self._name is None:
            if 'name' in self._h.keys():
                return _read_string(self._h['name'])
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def dataTimeSeries(self):
        if self._cfg.dynamic_loading and self._dataTimeSeries is None:
            if 'dataTimeSeries' in self._h.keys():
                return np.array(self._h['dataTimeSeries']).astype(float)  # Array
        return self._dataTimeSeries

    @dataTimeSeries.setter
    def dataTimeSeries(self, value):
        self._dataTimeSeries = value

    @property
    def time(self):
        if self._cfg.dynamic_loading and self._time is None:
            if 'time' in self._h.keys():
                return np.array(self._h['time']).astype(float)  # Array
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def timeOffset(self):
        if self._cfg.dynamic_loading and self._timeOffset is None:
            if 'timeOffset' in self._h.keys():
                return np.array(self._h['timeOffset']).astype(float)  # Array
        return self._timeOffset

    @timeOffset.setter
    def timeOffset(self, value):
        self._timeOffset = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'name' in self._h:
            name = self._h['name'].name
            data = self.name
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        if 'dataTimeSeries' in self._h:
            name = self._h['dataTimeSeries'].name
            data = np.array(self.dataTimeSeries)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'time' in self._h:
            name = self._h['time'].name
            data = np.array(self.time)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)
        if 'timeOffset' in self._h:
            name = self._h['timeOffset'].name
            data = np.array(self.timeOffset)
            if name in file:
                del file[name]
            file.create_dataset(name, dtype='f8', data=data)


class Aux(IndexedGroup):

    _name: str = 'aux'
    _element: Group = AuxElement

    def __init__(self, h: h5py.File, cfg: SnirfConfig):
        super().__init__(h, cfg)


class Snirf():
    
    _name = '/'
    _formatVersion = None  # "s"*
    _nirs = None  # {i}*

    def __init__(self, *args, dynamic_loading: bool = False):
        if len(args) > 0:
            path = args[0]
            if type(path) is str:
                if not path.endswith('.snirf'):
                    path = path.join('.snirf')
                if os.path.exists(path):
                    self._h = h5py.File(path, 'r+')
                else:
                    self._h = h5py.File(path, 'w')
            else:
                raise TypeError(str(path) + ' is not a valid filename')
        else:
            self._h = h5py.File(TemporaryFile(), 'w')
        self._cfg = SnirfConfig()
        self._cfg.dynamic_loading = dynamic_loading
        if 'formatVersion' in self._h.keys():
            if not self._cfg.dynamic_loading:
                self._formatVersion = _read_string(self._h['formatVersion'])
        else:
            warn(str(self.__class__.__name__) + ' missing required key ' + '"formatVersion"')

        self.nirs = Nirs(self._h, self._cfg)  # Indexed group


    @property
    def formatVersion(self):
        if self._cfg.dynamic_loading and self._formatVersion is None:
            if 'formatVersion' in self._h.keys():
                return _read_string(self._h['formatVersion'])
        return self._formatVersion

    @formatVersion.setter
    def formatVersion(self, value):
        self._formatVersion = value

    @property
    def nirs(self):
        return self._nirs

    @nirs.setter
    def nirs(self, value):
        self._nirs = value


    def _save(self, *args):
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        if 'formatVersion' in self._h:
            name = self._h['formatVersion'].name
            data = self.formatVersion
            if name in file:
                del file[name]
            file.create_dataset(name, dtype=h5py.string_dtype(encoding='ascii', length=None), data=data)
        self.nirs._save(*args)

    # overload
    def save(self, path: str):
        ...

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
            self._save()

    @property
    def filename(self):
        return self._h.filename

    def close(self):
        self._h.close()

    def __del__(self):
        self._h.close()

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