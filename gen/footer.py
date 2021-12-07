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


# Manually extend _validate to provide detailed error codes


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
        
        if not self._h.filename.endswith('.snirf'):
            result._add('/', 'INVALID_FILE_NAME')
            
        for nirs in self.nirs:
            lenSourceLabels = len(nirs.probe.sourceLabels)
            lenDetectorLabels = len(nirs.probe.sourceLabels)
            lenWavelengths = len(nirs.probe.sourceLabels)
            for data in nirs.data:
                for ml in data:
                    if ml.sourceIndex > lenSourceLabels:
                        result._add(ml.location + '/sourceIndex', 'INVALID_SOURCE_INDEX')
                    if ml.detectorIndex > lenDetectorLabels:
                        result._add(ml.location + '/detectorIndex', 'INVALID_DETECTOR_INDEX')
                    if ml.wavelengthIndex > lenWavelengths:
                        result._add(ml.location + '/wavelengthIndex', 'INVALID_WAVELENGTH_INDEX')
                    
                    
            
        
        
        
        
        