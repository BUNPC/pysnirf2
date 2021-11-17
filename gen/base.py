from abc import ABC, abstractmethod
import h5py
import os
import sys

if sys.version_info[0] < 3:
    raise ImportError('pysnirf2 requires Python > 3')


class SnirfFormatError(Exception):
    pass


class _Group():
    
    def __init__(self, gid: h5py.h5g.GroupID):
        self._id = gid
        if not isinstance(gid, h5py.h5g.GroupID):
            raise TypeError('must initialize with a Group ID')
        self._group = h5py.Group(self._id)

    def __repr__(self):
        return str(list(filter(lambda x: not x.startswith('_'), vars(self).keys())))



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
    
    def __init__(self, parent):
        if isinstance(parent, (h5py.Group, h5py.File)):
            # Because the indexed group is not a true HDF5 group but rather an
            # iterable list of HDF5 groups, it takes a base group or file and
            # searches its keys, appending the appropriate elements to itself
            # in order
            self._parent = parent
            print('class name', self.__class__.__name__, 'signature', self._name)
            i = 1
            for key in self._parent.keys():
                name = str(key).split('/')[-1]
                print('Looking for keys starting with', self._name)
                if key.startswith(self._name):
                    if key.endswith(str(i)):
                        print('adding numbered key', i, name)
                        self.append(self._element(self._parent[key].id))
                        i += 1
                    elif i == 1 and key.endswith(self._name):
                        print('adding non-numbered key', name)
                        self.append(self._element(self._parent[key].id))
           
        else:
            raise TypeError('must initialize _IndexedGroup with a Group or File')
         
    def __new__(cls, *args, **kwargs):
        if cls is _IndexedGroup:
            raise NotImplementedError('_IndexedGroup is an abstract class')
        return super().__new__(cls, *args, **kwargs)
            
    @abstractmethod
    def _append_group(self, gid: h5py.h5g.GroupID):
        raise NotImplementedError('_append_group is an abstract method')
    
    def __repr__(self):
        prettylist = ''
        for i in range(len(self)):
            prettylist += (str(self[i]) + '\n')
        return prettylist
