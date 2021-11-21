# pysnirf2

Dynamically generated Python library for loading, saving, and validating Snirf files.

Developed and maintained by the [Boston University Neurophotonics Center]().

pysnirf2 requires Python 3.

# Features

## Load a SNIRF file
```python
from pysnirf2 import Snirf
>>> snirf = Snirf(r'some\path\subj1_run01.snirf')
```

## View or retrive its contents
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
detectorLabels: ['D1' 'D2' 'D3' 'D4' 'D5' 'D6' 'D7' 'D8' 'D9' 'D10' 'D11' 'D12' 'D13'
 'D14' 'D15' 'D16' 'D17' 'D18' 'D19' 'D20' 'D21' 'D22' 'D23' 'D24' 'D25'
 'D26' 'D27' 'D28' 'D29' 'D30' 'D31']
detectorPos2D: <(31, 2) array of float64>
detectorPos3D: <(31, 3) array of float64>
filename: 
C:\Users\you\some\path\subj1_run01.snirf
frequencies: [1.]
landmarkLabels: None
landmarkPos2D: None
landmarkPos3D: None
momentOrders: None
sourceLabels: ['S1' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'S9' 'S10' 'S11' 'S12' 'S13'
 'S14' 'S15']
sourcePos2D: [[-125.    42.8]
 [-125.     0. ]
 [ -83.    42.8]
 [ -83.     0. ]
 [ -41.    42.8]
 [ -41.     0. ]
 [  41.    42.8]
 [  41.     0. ]
 [  83.    42.8]
 [  83.     0. ]
 [ 125.    42.8]
 [ 125.     0. ]
 [ -60.    90. ]
 [   0.    90. ]
 [  60.    90. ]]
sourcePos3D: <(15, 3) array of float64>
timeDelayWidths: [0.]
timeDelays: [0.]
useLocalIndex: None
wavelengths: [690. 830.]
wavelengthsEmission: None
```
## Edit a SNIRF file
```python
>>> snirf.nirs[0].metaDataTags.SubjectID = 'subj1'
>>> snirf.nirs[0].metaDataTags.SubjectID
'subj1'
```
```python
>>> snirf.nirs[0].probe.sourcePos2D[0, :] = [0, 0]
>>> snirf.nirs[0].probe.sourcePos2D
array([[   0. ,    0. ],
       [-125. ,    0. ],
       [ -83. ,   42.8],
       [ -83. ,    0. ],
       [ -41. ,   42.8],
       [ -41. ,    0. ],
       [  41. ,   42.8],
       [  41. ,    0. ],
       [  83. ,   42.8],
       [  83. ,    0. ],
       [ 125. ,   42.8],
       [ 125. ,    0. ],
       [ -60. ,   90. ],
       [   0. ,   90. ],
       [  60. ,   90. ]])
```


## Save a SNIRF file
Overwrite the open file
```python
>>> snirf.save()
```
Save As in a new location
```python
>>> snirf.save(r'some\new\path\subj1_run01_edited.snirf')
```
The `save()` function can be called for any group or indexed group:
```python
>>> snirf.nirs[0].metaDataTags.save('subj1_run01_edited_metadata_only.snirf')
```
## Dynamic loading mode
For larger files, it may be useful to load data dynamically: data will only be loaded on access, and only changed datasets will be written on `save()`. When creating a new `Snirf` instance, set `dynamic_loading` to `True` (Default `False`).
```python
>>> snirf = Snirf(r'some\path\subj1_run01.snirf', dynamic_loading=True)
```
> Warning: in dynamic loading mode, array data cannot be modified with indices like in the example above:
> ```python
> >>> snirf = Snirf(TESTPATH, dynamic_loading=True)
> >>> snirf.nirs[0].probe.sourcePos2D
> array([[-125. ,   42.8],
>        [-125. ,    0. ],
>        [ -83. ,   42.8],
>        [ -83. ,    0. ],
>        [ -41. ,   42.8],
>        [ -41. ,    0. ],
>        [  41. ,   42.8],
>        [  41. ,    0. ],
>        [  83. ,   42.8],
>        [  83. ,    0. ],
>        [ 125. ,   42.8],
>        [ 125. ,    0. ],
>        [ -60. ,   90. ],
>        [   0. ,   90. ],
>        [  60. ,   90. ]])
> >>> snirf.nirs[0].probe.sourcePos2D[0, :] = [0, 0]
> >>> snirf.nirs[0].probe.sourcePos2D
> array([[-125. ,   42.8],
>        [-125. ,    0. ],
>        [ -83. ,   42.8],
>        [ -83. ,    0. ],
>        [ -41. ,   42.8],
>        [ -41. ,    0. ],
>        [  41. ,   42.8],
>        [  41. ,    0. ],
>        [  83. ,   42.8],
>        [  83. ,    0. ],
>        [ 125. ,   42.8],
>        [ 125. ,    0. ],
>        [ -60. ,   90. ],
>        [   0. ,   90. ],
>        [  60. ,   90. ]])
> ```
> To modify an array in `dynamic_loading` mode, assign it, modify it, and assign it back to the Snirf object.
> ```python
> >>> sourcePos2D = snirf.nirs[0].probe.sourcePos2D
> >>> sourcePos2D[0, :] = [0, 0]
> >>> snirf.nirs[0].probe.sourcePos2D = sourcePos2D
> array([[   0. ,    0. ],
>        [-125. ,    0. ],
>        [ -83. ,   42.8],
>        [ -83. ,    0. ],
>        [ -41. ,   42.8],
>        [ -41. ,    0. ],
>        [  41. ,   42.8],
>        [  41. ,    0. ],
>        [  83. ,   42.8],
>        [  83. ,    0. ],
>        [ 125. ,   42.8],
>        [ 125. ,    0. ],
>        [ -60. ,   90. ],
>        [   0. ,   90. ],
>        [  60. ,   90. ]])

# Code generation

The interface and validator are generated via metacode that downloads and parses the latest [SNIRF specification](https://github.com/fNIRS/snirf). 

See [\gen]() for details.


  

