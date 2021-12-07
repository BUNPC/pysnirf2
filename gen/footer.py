# -- Extend metaDataTags to support addition of new unspecified datasets ------

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


# -- Manually extend _validate to provide detailed error codes ----------------


class StimElement(StimElement):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)
        
        if np.shape(self.data)[1] != len(self.dataLabels):
            result._add(self.location + '/dataLabels', 'INVALID_STIM_DATALABELS')        


class Stim(Stim):
    _element = StimElement


class AuxElement(AuxElement):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)
        
        if len(self.time) != len(self.dataTimeSeries):
            result._add(self.location + '/time', 'INVALID_TIME')


class Aux(Aux):
    _element = AuxElement


class DataElement(DataElement):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)
        
        if len(self.time) != np.shape(self.dataTimeSeries)[0]:
            result._add(self.location + '/time', 'INVALID_TIME')
        
        print(len(self.measurementList), np.shape(self.dataTimeSeries))
        if len(self.measurementList) != np.shape(self.dataTimeSeries)[1]:
            result._add(self.location, 'INVALID_MEASUREMENTLIST')


class Data(Data):
    _element = DataElement
    
    
class Snirf(Snirf):
    
    def _validate(self, result: ValidationResult):
        super()._validate(result)
        
        # TODO invalid filename, file
            
        for nirs in self.nirs:
            if type(nirs.probe) not in [type(None), type(AbsentGroup)]:
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


# -- convenience functions ----------------------------------------------------
            
        
def loadSnirf(path: str, dynamic_loading: bool=False, logfile: bool=False) -> Snirf:
    """
    Returns a Snirf object loaded from path if a Snirf file exists there. Takes
    the same kwargs as the Snirf object constructor
    """
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        return Snirf(path, dynamic_loading=dynamic_loading, logfile=logfile)
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)
                    
        
def saveSnirf(path: str, snirfobj: Snirf):
    """
    Saves a SNIRF file snirfobj to disk at path
    """
    if type(path) is not str:
        raise TypeError('path must be str, not '+ type(path))
    if not isinstance(snirfobj, Snirf):
        raise TypeError('snirfobj must be Snirf, not ' + type(snirfobj))
    snirfobj.save(path)


def validateSnirf(path: str) -> Tuple[bool, ValidationResult]:
    """
    Returns a bool representing the validity of the Snirf object on disk at
    path along with the detailed output structure ValidationResult instance
    """
    if type(path) is not str:
        raise TypeError('path must be str, not '+ type(path))
    if not path.endswith('.snirf'):
        path += '.snirf'
    if os.path.exists(path):
        s = Snirf(path)
        valid, result = s.validate()
        s.close()
        return (valid, result)
    else:
        raise FileNotFoundError('No SNIRF file at ' + path)
