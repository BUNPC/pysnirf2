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
