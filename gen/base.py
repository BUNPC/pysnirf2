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
        g = self._parent.create_group(self._name + str(len(self._list) + 1))
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