<!-- markdownlint-disable -->

# API Overview

## Modules

- [`pysnirf2`](./pysnirf2.md#module-pysnirf2): pysnirf2

## Classes

- [`pysnirf2.Aux`](./pysnirf2.md#class-aux)
- [`pysnirf2.AuxElement`](./pysnirf2.md#class-auxelement)
- [`pysnirf2.Data`](./pysnirf2.md#class-data)
- [`pysnirf2.DataElement`](./pysnirf2.md#class-dataelement)
- [`pysnirf2.Group`](./pysnirf2.md#class-group)
- [`pysnirf2.IndexedGroup`](./pysnirf2.md#class-indexedgroup): Represents the "indexed group" which is defined by v1.0 of the SNIRF
- [`pysnirf2.MeasurementList`](./pysnirf2.md#class-measurementlist): The measurement list. This variable serves to map the data array onto the probe 
- [`pysnirf2.MeasurementListElement`](./pysnirf2.md#class-measurementlistelement): An element of indexed group `MeasurementList`.
- [`pysnirf2.MetaDataTags`](./pysnirf2.md#class-metadatatags)
- [`pysnirf2.Nirs`](./pysnirf2.md#class-nirs): This group stores one set of NIRS data.  This can be extended by adding the count 
- [`pysnirf2.NirsElement`](./pysnirf2.md#class-nirselement): An element of indexed group `Nirs`.
- [`pysnirf2.Probe`](./pysnirf2.md#class-probe)
- [`pysnirf2.Snirf`](./pysnirf2.md#class-snirf)
- [`pysnirf2.SnirfConfig`](./pysnirf2.md#class-snirfconfig): Structure containing Snirf-wide data and settings
- [`pysnirf2.SnirfFormatError`](./pysnirf2.md#class-snirfformaterror)
- [`pysnirf2.Stim`](./pysnirf2.md#class-stim)
- [`pysnirf2.StimElement`](./pysnirf2.md#class-stimelement)
- [`pysnirf2.ValidationIssue`](./pysnirf2.md#class-validationissue): Information about the validity of a given SNIRF file location
- [`pysnirf2.ValidationResult`](./pysnirf2.md#class-validationresult): The result of Snirf file validation routines.

## Functions

- [`pysnirf2.loadSnirf`](./pysnirf2.md#function-loadsnirf): Returns a Snirf object loaded from path if a Snirf file exists there. Takes
- [`pysnirf2.saveSnirf`](./pysnirf2.md#function-savesnirf): Saves a SNIRF file snirfobj to disk at path
- [`pysnirf2.validateSnirf`](./pysnirf2.md#function-validatesnirf): Returns a bool representing the validity of the Snirf object on disk at


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
