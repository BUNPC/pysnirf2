from abc import ABC, abstractmethod
import h5py
import os
import sys
import numpy as np
from warnings import warn

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')


class SnirfFormatError(Exception):
    pass


class SnirfConfig:
    filepath: str = ''
    dynamic_loading: bool = True  # If False, data is loaded in the constructor, if True, data is loaded on access


class _Group():

    def __init__(self, gid: h5py.h5g.GroupID, cfg: SnirfConfig):
        self._id = gid
        if not isinstance(gid, h5py.h5g.GroupID):
            raise TypeError('must initialize with a Group ID, not ' + str(type(gid)))
        self._h = h5py.Group(self._id)
        self._cfg = cfg
        
    def __repr__(self):
        props = [p for p in dir(self) if '_' not in p]
        out = ''
        for prop in props:
            out += prop + ': '
            prepr = str(getattr(self, prop))
            if len(prepr) < 64:
                out += prepr
            else:
                out += '\n' + prepr
            out += '\n'
        
        return str(out[:-1])


class _IndexedGroup(list, ABC):
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
    _element: _Group = None

    def __init__(self, parent: (h5py.Group, h5py.File), cfg: SnirfConfig):
        if isinstance(parent, (h5py.Group, h5py.File)):
            # Because the indexed group is not a true HDF5 group but rather an
            # iterable list of HDF5 groups, it takes a base group or file and
            # searches its keys, appending the appropriate elements to itself
            # in order
            self._parent = parent
            self._cfg = cfg
            i = 1
            for key in self._parent.keys():
                if key.startswith(self._name):
#                    print('adding numbered member of indexed group', str(key))
                    self.append(self._element(self._parent[key].id, self._cfg))
                    i += 1
                elif i == 1 and key.endswith(self._name):
#                    print('adding member of indexed group', str(key))
                    self.append(self._element(self._parent[key].id, self._cfg))
        else:
            raise TypeError('must initialize _IndexedGroup with a Group or File')
         
    def __new__(cls, *args, **kwargs):
        if cls is _IndexedGroup:
            raise NotImplementedError('_IndexedGroup is an abstract class')
        return super().__new__(cls, *args, **kwargs)

    @abstractmethod
    def _append_group(self, gid: h5py.h5g.GroupID):
        raise NotImplementedError('_append_group is an abstract method')
    
    def __getattr__(self, name):
        raise AttributeError(self.__class__.__name__ + ' is an interable list of '
                             + str(len(self)) + ' ' + str(self._element)
                             + ', access these with an index')
        
    def __repr__(self):
        return str('<' + 'iterable of ' + str(len(self)) + ' ' + str(self._element) + '>')
