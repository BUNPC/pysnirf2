<!-- markdownlint-disable -->

# API Overview

## Modules

- [`pysnirf2`](./pysnirf2.md#module-pysnirf2)

## Classes

- [`pysnirf2.AbsentDatasetType`](./pysnirf2.md#class-absentdatasettype)
- [`pysnirf2.AbsentGroupType`](./pysnirf2.md#class-absentgrouptype)
- [`pysnirf2.Aux`](./pysnirf2.md#class-aux)
- [`pysnirf2.AuxElement`](./pysnirf2.md#class-auxelement)
- [`pysnirf2.Data`](./pysnirf2.md#class-data)
- [`pysnirf2.DataElement`](./pysnirf2.md#class-dataelement)
- [`pysnirf2.Group`](./pysnirf2.md#class-group)
- [`pysnirf2.IndexedGroup`](./pysnirf2.md#class-indexedgroup): Represents the "indexed group" which is defined by v1.0 of the SNIRF
- [`pysnirf2.MeasurementList`](./pysnirf2.md#class-measurementlist)
- [`pysnirf2.MeasurementListElement`](./pysnirf2.md#class-measurementlistelement)
- [`pysnirf2.MetaDataTags`](./pysnirf2.md#class-metadatatags)
- [`pysnirf2.Nirs`](./pysnirf2.md#class-nirs)
- [`pysnirf2.NirsElement`](./pysnirf2.md#class-nirselement)
- [`pysnirf2.PresentDatasetType`](./pysnirf2.md#class-presentdatasettype)
- [`pysnirf2.Probe`](./pysnirf2.md#class-probe)
- [`pysnirf2.Snirf`](./pysnirf2.md#class-snirf)
- [`pysnirf2.SnirfConfig`](./pysnirf2.md#class-snirfconfig): Structure for containing Snirf-wide data and settings
- [`pysnirf2.SnirfFormatError`](./pysnirf2.md#class-snirfformaterror)
- [`pysnirf2.Stim`](./pysnirf2.md#class-stim)
- [`pysnirf2.StimElement`](./pysnirf2.md#class-stimelement)
- [`pysnirf2.ValidationIssue`](./pysnirf2.md#class-validationissue): Pretty-printable structure for the result of validation on an HDF5 name
- [`pysnirf2.ValidationResult`](./pysnirf2.md#class-validationresult): ORganzies the result of the pysnirf2 validation routine like so:

## Functions

- [`pysnirf2.loadSnirf`](./pysnirf2.md#function-loadsnirf): Returns a Snirf object loaded from path if a Snirf file exists there. Takes
- [`pysnirf2.saveSnirf`](./pysnirf2.md#function-savesnirf): Saves a SNIRF file snirfobj to disk at path
- [`pysnirf2.validateSnirf`](./pysnirf2.md#function-validatesnirf): Returns a bool representing the validity of the Snirf object on disk at


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
