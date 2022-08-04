<h1 style="text-align:center">

pysnirf2

`pip install snirf`

</h1>


![testing](https://github.com/BUNPC/pysnirf2/actions/workflows/test.yml/badge.svg)
![lazydocs](https://github.com/BUNPC/pysnirf2/actions/workflows/lazydocs.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/snirf.svg)](https://badge.fury.io/py/snirf)
[![DOI](https://zenodo.org/badge/426339949.svg)](https://zenodo.org/badge/latestdoi/426339949)

Dynamically generated Python library for reading, writing, and validating [Shared Near Infrared Spectroscopy Format (SNIRF) files](https://github.com/fNIRS/snirf).

Developed and maintained by the [Boston University Neurophotonics Center](https://www.bu.edu/neurophotonics/).

## Documentation

[Documentation](https://github.com/BUNPC/pysnirf2/tree/main/docs) is generated from source using [lazydocs](https://github.com/ml-tooling/lazydocs)

## Installation
`pip install snirf`

pysnirf2 requires Python > 3.6.

# Features

The library generated via metaprogramming, but the resulting classes explicitly implement each and every specified SNIRF field so as to provide an extensible object-oriented foundation for SNIRF applications.

## Open a SNIRF file
`Snirf(<path>, <mode>)` opens a SNIRF file at `<path>` _or creates a new one if it doesn't exist._ Use mode 'w' to create a new file, 'r' to read a file, and 'r+' to edit an existing file.
```python
from snirf import Snirf
snirf = Snirf(r'some\path\subj1_run01.snirf', 'r+')
```
## Create a SNIRF file object
`Snirf()` with no arguments creates a temporary file which can be written later using `save()`.
```python
snirf = Snirf()
```

## Closing a SNIRF file
A `Snirf` instance wraps a file on disk. It should be closed when you're done reading from it or saving.
```python
snirf.close()
```
Use a `with` statement to ensure that the file is closed when you're done with it:
```python
with Snirf(r'some\path\subj1_run01.snirf', 'r+') as snirf:
     # Read/write
     snirf.save()
```

## Copy a SNIRF file object
Any `Snirf` object can be copied to a new instance in memory, after which the original can be closed.
```python
snirf2 = snirf.copy()
snirf.close()
# snirf2 is free for manipulation
```

## View or retrieve a file's contents
```python
>>> snirf

Snirf at /
filename: 
C:\Users\you\some\path\subj1_run01.snirf
formatVersion: v1.0
nirs: <iterable of 2 <class 'pysnirf2.NirsElement'>>
```
```python
>>> snirf.nirs[0].probe

Probe at /nirs1/probe
correlationTimeDelayWidths: [0.]
correlationTimeDelays: [0.]
detectorLabels: ['D1' 'D2']
detectorPos2D: [[30.  0.]
 [ 0. 30.]]
detectorPos3D: [[30.  0.  0.]
 [ 0. 30.  0.]]
filename: 
C:\Users\you\some\path\subj1_run01.snirf
frequencies: [1.]
landmarkLabels: None
landmarkPos2D: None
landmarkPos3D: None
location: /nirs/probe
momentOrders: None
sourceLabels: ['S1']
sourcePos2D: [[0. 0.]]
sourcePos3D: [[0.]
 [0.]
 [0.]]
timeDelayWidths: [0.]
timeDelays: [0.]
useLocalIndex: None
wavelengths: [690. 830.]
wavelengthsEmission: None
```
## Edit a SNIRF file
Assign a new value to a field
```python
>>> snirf.nirs[0].metaDataTags.SubjectID = 'subj1'
>>> snirf.nirs[0].metaDataTags.SubjectID

'subj1'
```
```python
>>> snirf.nirs[0].probe.detectorPos3D[0, :] = [90, 90, 90]
>>> snirf.nirs[0].probe.detectorPos3D

array([[90.,  90.,  90.],
      [  0.,  30.,   0.]])
```
> Note: assignment via slicing is not possible in `dynamic_loading` mode. 
## Indexed groups
Indexed groups are defined by the SNIRF file format as groups of the same type which are indexed via their name + a 1-based index, i.e.  `data1`, `data2`, ... or `stim1`, `stim2`, `stim3`, ...

pysnirf2 provides an iterable interface for these groups using Pythonic 0-based indexing, i.e. `data[0]`, `data[1]`, ... or `stim[0]`, `stim[1]]`, `stim[2]`, ...

```python
>>> snirf.nirs[0].stim


<iterable of 0 <class 'pysnirf2.StimElement'>>

>>> len(nirs[0].stim)

0
```
To add an indexed group, use the `appendGroup()` method of any `IndexedGroup` class. Indexed groups are created automatically. `nirs` is an indexed group.
```python
>>> snirf.nirs[0].stim.appendGroup()
>>> len(nirs[0].stim)

1

>>> snirf.nirs[0].stim[0]

StimElement at /nirs/stim2
data: None
dataLabels: None
filename: 
C:\Users\you\some\path\subj1_run01.snirf
name: None
```
To remove an indexed group
```python
del snirf.nirs[0].stim[0]
```
## Save a SNIRF file
Overwrite the open file
```python
snirf.save()
```
Save As in a new location
```python
snirf.save(r'some\new\path\subj1_run01_edited.snirf')
```
The `save()` function can be called for any group or indexed group:
```python
snirf.nirs[0].metaDataTags.save('subj1_run01_edited_metadata_only.snirf')
```
## Dynamic loading mode
For larger files, it may be useful to load data dynamically: data will only be loaded on access, and only changed datasets will be written on `save()`. When creating a new `Snirf` instance, set `dynamic_loading` to `True` (Default `False`).
```python
snirf = Snirf(r'some\path\subj1_run01.snirf', 'r+', dynamic_loading=True)
```
> Note: in dynamic loading mode, array data cannot be modified with indices like in the example above:
> ```python
> >>> snirf = Snirf(TESTPATH, 'r+', dynamic_loading=True)
> >>> snirf.nirs[0].probe.detectorPos3D
> 
> array([[30.,  0.,  0.],
>       [ 0., 30.,  0.]])
> 
> >>> snirf.nirs[0].probe.detectorPos3D[0, :] = [90, 90, 90]
> >>> snirf.nirs[0].probe.detectorPos3D
> 
> array([[30.,  0.,  0.],
>        [ 0., 30.,  0.]])
> ```
> To modify an array in `dynamic_loading` mode, assign it, modify it, and assign it back to the Snirf object.
> ```python
> >>> detectorPos3D = snirf.nirs[0].probe.detectorPos3D
> >>> detectorPos3D[0, :] = [90, 90, 90]
> >>> snirf.nirs[0].probe.detectorPos3D = detectorPos3D
> 
> array([[90.,  90.,  90.],
>        [ 0.,   30.,  0.]])

# Validating a SNIRF file
pysnirf2 features functions for validating SNIRF files against the specification and generating detailed error reports.
## Validate a Snirf object you have created
```python
result = snirf.validate()
```
## Validate a SNIRF file on disk
To validate a SNIRF file on disk
```python
from snirf import validateSnirf
result = validateSnirf(r'some\path\subj1_run01.snirf')
assert result, 'Invalid SNIRF file!\n' + result.display()  # Crash and display issues if the file is invalid.
```
## Validation results
The validation functions return a [`ValidationResult`](https://github.com/BUNPC/pysnirf2/blob/main/docs/pysnirf2.md#class-validationresult) instance which contains details about the SNIRF file.
To view the validation result:
```python
>>> result.display(severity=3)  # Display all fatal errors

<pysnirf2.pysnirf2.ValidationResult object at 0x000001C0CCF05A00>
/nirs1/data1/measurementList103/dataType                 FATAL   REQUIRED_DATASET_MISSING
/nirs1/data1/measurementList103/dataTypeIndex            FATAL   REQUIRED_DATASET_MISSING
/nirs1/data1                                             FATAL   INVALID_MEASUREMENTLIST 

Found 668 OK      (hidden)
Found 635 INFO    (hidden)
Found 204 WARNING (hidden)
Found 3 FATAL  

File is INVALID
```
To look at a particular result:
```python
>>> result.errors[2]

<pysnirf2.pysnirf2.ValidationIssue object at 0x000001C0CB502F70>
location: /nirs1/data1
severity: 3   FATAL  
name:     8   INVALID_MEASUREMENTLIST
message:  The number of measurementList elements does not match the second dimension of dataTimeSeries
```
The full list of validation results `result.issues` can be explored programatically.
# Code generation

The interface and validator are generated via metacode that downloads and parses [the latest SNIRF specification](https://raw.githubusercontent.com/fNIRS/snirf/master/snirf_specification.md). 

See [\gen](https://github.com/BUNPC/pysnirf2/tree/main/gen) for details.


  

