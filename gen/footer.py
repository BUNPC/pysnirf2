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
            raise AttributeError("can't set tag. You cannot set the required metaDataTags fields using add() or use protected attributes of MetaDataTags such as 'location' or 'filename'")
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
                if np.shape(self.data)[1] != len(self.dataLabels):
                    result._add(self.location + '/dataLabels', 'INVALID_STIM_DATALABELS')        
            except IndexError:  # If data doesn't have columns
                result._add(self.location + '/data', 'INVALID_DATASET_SHAPE')    


class Stim(Stim):
    _element = StimElement


class AuxElement(AuxElement):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)
        
        if all(attr is not None for attr in [self.time, self.dataTimeSeries]):
            if len(self.time) != len(self.dataTimeSeries):
                result._add(self.location + '/time', 'INVALID_TIME')


class Aux(Aux):
    _element = AuxElement


class DataElement(DataElement):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)  
        
        if all(attr is not None for attr in [self.time, self.dataTimeSeries]):
            if len(self.time) != np.shape(self.dataTimeSeries)[0]:
                result._add(self.location + '/time', 'INVALID_TIME')
            
            if len(self.measurementList) != np.shape(self.dataTimeSeries)[1]:
                result._add(self.location, 'INVALID_MEASUREMENTLIST')


class Data(Data):
    _element = DataElement


class Probe(Probe):
    
    def _validate(self, result: ValidationResult):
        
        s2 = self.sourcePos2D is not None
        d2 = self.detectorPos2D is not None
        s3 = self.sourcePos3D is not None
        d3 = self.detectorPos3D is not None
        if (s2 and d2):
            result._add(self.location + '/sourcePos2D', 'OK')
            result._add(self.location + '/detectorPos2D', 'OK')
            result._add(self.location + '/sourcePos3D', 'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/detectorPos3D', 'OPTIONAL_DATASET_MISSING')
        elif (s3 and d3):
            result._add(self.location + '/sourcePos2D', 'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/detectorPos2D', 'OPTIONAL_DATASET_MISSING')
            result._add(self.location + '/sourcePos3D', 'OK')
            result._add(self.location + '/detectorPos3D', 'OK')
        else:
            result._add(self.location + '/sourcePos2D', ['REQUIRED_DATASET_MISSING', 'OK'][int(s2)])
            result._add(self.location + '/detectorPos2D', ['REQUIRED_DATASET_MISSING', 'OK'][int(d2)])
            result._add(self.location + '/sourcePos3D', ['REQUIRED_DATASET_MISSING', 'OK'][int(s3)])
            result._add(self.location + '/detectorPos3D', ['REQUIRED_DATASET_MISSING', 'OK'][int(d3)])
        
        if self.coordinateSystem is not None:
            if not self.coordinateSystem in _RECOGNIZED_COORDINATE_SYSTEM_NAMES:
                result._add(self.location + '/coordinateSystem', 'UNRECOGNIZED_COORDINATE_SYSTEM')            
                if self.coordinateSystemDescription is None:
                    result._add(self.location + '/coordinateSystemDescription', 'NO_COORDINATE_SYSTEM_DESCRIPTION')
        
        # The above will supersede the errors from the template code because
        # duplicate names cannot be added to the issues list
        super()._validate(result)
    

class Snirf(Snirf):
    
    # overload
    def save(self, *args):
        """Save a SNIRF file to disk.

        Args:
            args (str or h5py.File): A path to a closed or nonexistant SNIRF file on disk or an open `h5py.File` instance

        Examples:
            save can overwrite the current contents of a Snirf file:
            >>> mysnirf.save()

            or take a new filename to write the file there:
            >>> mysnirf.save(<new destination>)
        """
        if len(args) > 0 and type(args[0]) is str:
            path = args[0]
            if not path.endswith('.snirf'):
                path += '.snirf'
            with h5py.File(path, 'w') as new_file:
                self._save(new_file)
                self._cfg.logger.info('Saved Snirf file at %s to copy at %s', self.filename, path)
        else:
            self._save(self._h.file)

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
                    lenSourceLabels = len(nirs.probe.sourceLabels)
                else:
                    lenSourceLabels = 0
                if nirs.probe.detectorLabels is not None:
                    lenDetectorLabels = len(nirs.probe.detectorLabels)
                else:
                    lenDetectorLabels = 0
                if nirs.probe.wavelengths is not None:
                    lenWavelengths = len(nirs.probe.wavelengths)
                else:
                    lenWavelengths = 0
                for data in nirs.data:
                    for ml in data.measurementList:
                        if ml.sourceIndex is not None:
                            if ml.sourceIndex > lenSourceLabels:
                                result._add(ml.location + '/sourceIndex', 'INVALID_SOURCE_INDEX')
                        if ml.detectorIndex is not None:
                            if ml.detectorIndex > lenDetectorLabels:
                                result._add(ml.location + '/detectorIndex', 'INVALID_DETECTOR_INDEX')
                        if ml.wavelengthIndex is not None:
                            if ml.wavelengthIndex > lenWavelengths:
                                result._add(ml.location + '/wavelengthIndex', 'INVALID_WAVELENGTH_INDEX')


# -- Interface functions ----------------------------------------------------
            
        
def loadSnirf(path: str, dynamic_loading: bool=False, logfile: bool=False) -> Snirf:
    """Load a SNIRF file from disk.
    
    Returns a `Snirf` object loaded from path if a SNIRF file exists there. Takes
    the same kwargs as the Snirf object constructor
    
    Args:
        path (str): Path to a SNIRF file on disk.
        dynamic_loading (bool): If True, Datasets will not be read from the SNIRF file
            unless accessed with a property, conserving memory and loading time with larger datasets. Default False.
        logfile (bool): If True, the `Snirf` instance will write to a log file which shares its name. Default False.
    
    Returns:
        `Snirf`: a `Snirf` instance loaded from the SNIRF file.   
    
    Raises:
        FileNotFoundError: `path` was not found on disk.
    """
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        return Snirf(path, dynamic_loading=dynamic_loading, logfile=logfile)
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)
                    
        
def saveSnirf(path: str, snirf: Snirf):
    """Saves a SNIRF file to disk.
    
    Args:
        path (str): Path to save the file.
        snirf (Snirf): `Snirf` instance to write to disk.
    """
    if type(path) is not str:
        raise TypeError('path must be str, not '+ type(path))
    if not isinstance(snirfobj, Snirf):
        raise TypeError('snirfobj must be Snirf, not ' + type(snirfobj))
    snirfobj.save(path)


def validateSnirf(path: str) -> ValidationResult:
    """Validate a SNIRF file on disk.
    
    Returns truthy ValidationResult instance which holds detailed results of validation
    """
    if type(path) is not str:
        raise TypeError('path must be str, not '+ type(path))
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        with Snirf(path) as snirf:
            return snirf.validate()
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)
