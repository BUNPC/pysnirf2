# Extend metaDataTags to support loading and saving of unspecified datasets

class MetaDataTags(MetaDataTags):
    
    def __init__(self, varg, cfg):
        super().__init__(varg, cfg)
        self.__other = []
        for key in self._h:
            if key not in self.__snirfnames:
                data = np.array(self._h[key])
                self.__other.append(key)
                setattr(self, key, data)
                
    def _save(self, *args):
        super()._save(*args)
        if len(args) > 0 and type(args[0]) is h5py.File:
            file = args[0]
        else:
            file = self._h.file
        print(self.__class__.__name__ + ' saving misc datasets: ', self.__other)
        for key in self.__other:
            print('Writing misc dataset to', self.location + '/' + key, 'in', file)
            file[self.location + '/' + key]
                