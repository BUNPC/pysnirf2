<!-- markdownlint-disable -->

<a href="../snirf/pysnirf2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pysnirf2`
Module for reading, writing and validating SNIRF files. 

SNIRF files are HDF5 files designed to facilitate the sharing of near-infrared spectrocopy data. Their specification is defined at https://github.com/fNIRS/snirf. 

This library wraps each HDF5 Group and offers a Pythonic interface on lists of like-Groups which the SNIRF specification calls "indexed Groups". 



**Example:**
  Load a file: 

``` from snirf import Snirf```
         >>> with Snirf(<filename>) as s:
             ...

Maintained by the Boston University Neurophotonics Center



---

<a href="../snirf/pysnirf2.py#L6222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loadSnirf`

```python
loadSnirf(
    path: str,
    dynamic_loading: bool = False,
    enable_logging: bool = False
) → Snirf
```

Load a SNIRF file from disk. 

Returns a `Snirf` object loaded from path if a SNIRF file exists there. Takes the same kwargs as the Snirf object constructor 



**Args:**
 
 - <b>`path`</b> (str):  Path to a SNIRF file on disk. 
 - <b>`dynamic_loading`</b> (bool):  If True, Datasets will not be read from the SNIRF file  unless accessed with a property, conserving memory and loading time with larger datasets. Default False. 
 - <b>`enable_logging`</b> (bool):  If True, the `Snirf` instance will write to a log file which shares its name. Default False. 



**Returns:**
 
 - <b>``Snirf``</b>:  a `Snirf` instance loaded from the SNIRF file.    



**Raises:**
 
 - <b>`FileNotFoundError`</b>:  `path` was not found on disk. 


---

<a href="../snirf/pysnirf2.py#L6253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `saveSnirf`

```python
saveSnirf(path: str, snirf: Snirf)
```

Saves a SNIRF file to disk. 



**Args:**
 
 - <b>`path`</b> (str):  Path to save the file. 
 - <b>`snirf`</b> (Snirf):  `Snirf` instance to write to disk. 


---

<a href="../snirf/pysnirf2.py#L6267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validateSnirf`

```python
validateSnirf(path: str) → ValidationResult
```

Validate a SNIRF file on disk. 

Returns truthy ValidationResult instance which holds detailed results of validation 


---

<a href="../snirf/pysnirf2.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SnirfFormatError`
Raised when SNIRF-specific error prevents file from loading properly. 





---

<a href="../snirf/pysnirf2.py#L549"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValidationIssue`
Information about the validity of a given SNIRF file location. 

Properties:  location: A relative HDF5 name corresponding to the location of the issue  name: A string describing the issue. Must be predefined in `_CODES`  id: An integer corresponding to the predefined error type  severity: An integer ranking the serverity level of the issue.  0 OK, Nothing remarkable  1 Potentially useful `INFO`  2 `WARNING`, the file is valid but exhibits undefined behavior or features marked deprecation  3 `FATAL`, The file is invalid.  message: A string containing a more verbose description of the issue 

<a href="../snirf/pysnirf2.py#L563"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name: str, location: str)
```








---

<a href="../snirf/pysnirf2.py#L578"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dictize`

```python
dictize()
```

Return dictionary representation of Issue. 


---

<a href="../snirf/pysnirf2.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValidationResult`
The result of Snirf file validation routines. 

Validation results in a list of issues. Each issue records information about the validity of each location (each named Dataset and Group) in a SNIRF file. ValidationResult organizes the issues catalogued during validation and affords interfaces to retrieve and display them. 

```
<ValidationResult> = <Snirf instance>.validate()
 <ValidationResult> = validateSnirf(<path>)
``` 

<a href="../snirf/pysnirf2.py#L602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

`ValidationResult` should only be created by a `Snirf` instance's `validate` method. 


---

#### <kbd>property</kbd> codes

A list of each unique code name associated with all catalogued issues. 

---

#### <kbd>property</kbd> errors

A list of the `FATAL` issues catalogued during validation. 

---

#### <kbd>property</kbd> info

A list of the `INFO` issues catalogued during validation. 

---

#### <kbd>property</kbd> issues

A comprehensive list of all `ValidationIssue` instances for the result. 

---

#### <kbd>property</kbd> locations

A list of the HDF5 location associated with each issue. 

---

#### <kbd>property</kbd> warnings

A list of the `WARNING` issues catalogued during validation. 



---

<a href="../snirf/pysnirf2.py#L666"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `display`

```python
display(severity=2)
```

Reads the contents of an `h5py.Dataset` to an array of `dtype=str`. 



**Args:**
 
 - <b>`severity`</b>:  An `int` which sets the minimum severity message to display. Default is 2.  severity=0 All messages will be shown, including `OK`  severity=1 Prints `INFO`, `WARNING`, and `FATAL` messages  severity=2 Prints `WARNING` and `FATAL` messages  severity=3 Prints only `FATAL` error messages 

---

<a href="../snirf/pysnirf2.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_valid`

```python
is_valid() → bool
```

Returns True if no `FATAL` issues were catalogued during validation. 

---

<a href="../snirf/pysnirf2.py#L659"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize(indent=4)
```

Render serialized JSON ValidationResult. 


---

<a href="../snirf/pysnirf2.py#L860"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SnirfConfig`
Structure containing Snirf-wide data and settings. 

Properties:  logger (logging.Logger): The logger that the Snirf instance writes to  dynamic_loading (bool): If True, data is loaded from the HDF5 file only on access via property 

<a href="../snirf/pysnirf2.py#L867"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```









---

<a href="../snirf/pysnirf2.py#L894"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Group`




<a href="../snirf/pysnirf2.py#L895"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(varg, cfg: SnirfConfig)
```

Wrapper for an HDF5 Group element defined by SNIRF. 

Base class for an HDF5 Group element defined by SNIRF. Must be created with a Group ID or string specifying a complete path relative to file root--in the latter case, the wrapper will not correspond to a real HDF5 group on disk until `_save()` (with no arguments) is executed for the first time 



**Args:**
 
 - <b>`varg`</b> (h5py.h5g.GroupID or str):  Either a string which maps to a future Group location or an ID corresponding to a current one on disk 
 - <b>`cfg`</b> (SnirfConfig):  Injected configuration of parent `Snirf` instance 


---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L1050"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IndexedGroup`




<a href="../snirf/pysnirf2.py#L1055"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(parent: Group, cfg: SnirfConfig)
```

Represents several Groups which share a name, an "indexed group". 

Represents the "indexed group" which is defined by v1.0 of the SNIRF specification as:  If a data element is an HDF5 group and contains multiple sub-groups,  it is referred to as an indexed group. Each element of the sub-group  is uniquely identified by appending a string-formatted index (starting  from 1, with no preceding zeros) in the name, for example, /.../name1  denotes the first sub-group of data element name, and /.../name2  denotes the 2nd element, and so on. 

Because the indexed group is not a true HDF5 group but rather an iterable list of HDF5 groups, it takes a base group or file and searches its keys, appending the appropriate elements to itself in order. 

The appropriate elements are identified using the `_name` attribute: if a key begins with `_name` and ends with a number, or is equal to `_name`. 



**Args:**
 
 - <b>`parent`</b> (h5py.h5g.Group):  An HDF5 group which is the parent of the indexed groups 
 - <b>`cfg`</b> (SnirfConfig):  Injected configuration of parent `Snirf` instance 


---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L1397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MetaDataTags`




<a href="../snirf/pysnirf2.py#L1409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(var, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> FrequencyUnit

SNIRF field `FrequencyUnit`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the **case-sensitive** SI frequency unit used in  this measurement. Sample frequency units "Hz", "MHz" and "GHz". Please note that "mHz" is milli-Hz while "MHz" denotes "mega-Hz" according to SI unit system. 

We do not limit the total number of metadata records in the `metaDataTags`. Users can add additional customized metadata records; no duplicated metadata record names are allowed. 

Additional metadata record samples can be found in the below table. 

| Metadata Key Name | Metadata value | |-------------------|----------------| |ManufacturerName | "Company Name" | |Model | "Model Name" | |SubjectName | "LastName, FirstName" | |DateOfBirth | "YYYY-MM-DD" | |AcquisitionStartTime | "1569465620" | |StudyID | "Infant Brain Development" | |StudyDescription | "In this study, we measure ...." | |AccessionNumber | "##########################" | |InstanceNumber  | 2 | |CalibrationFileName | "phantomcal_121015.snirf" | |UnixTime | "1569465667" | 

The metadata records `"StudyID"` and `"AccessionNumber"` are unique strings that  can be used to link the current dataset to a particular study and a particular  procedure, respectively. The `"StudyID"` tag is similar to the DICOM tag "Study  ID" (0020,0010) and `"AccessionNumber"` is similar to the DICOM tag "Accession  Number"(0008,0050), as defined in the DICOM standard (ISO 12052). 

The metadata record `"InstanceNumber"` is defined similarly to the DICOM tag  "Instance Number" (0020,0013), and can be used as the sequence number to group  multiple datasets into a larger dataset - for example, concatenating streamed  data segments during a long measurement session. 

The metadata record `"UnixTime"` defines the Unix Epoch Time, i.e. the total elapse time in seconds since 1970-01-01T00:00:00Z (UTC) minus the leap seconds. 

---

#### <kbd>property</kbd> LengthUnit

SNIRF field `LengthUnit`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the **case-sensitive** SI length unit used in this  measurement. Sample length units include "mm", "cm", and "m". A value of  "um" is the same as "mm", i.e. micrometer. 

---

#### <kbd>property</kbd> MeasurementDate

SNIRF field `MeasurementDate`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the date of the measurement as a string. The format of the date string must either be `"unknown"`, or follow the ISO 8601 date string format `YYYY-MM-DD`, where 
- `YYYY` is the 4-digit year 
- `MM` is the 2-digit month (padding zero if a single digit) 
- `DD` is the 2-digit date (padding zero if a single digit) 

---

#### <kbd>property</kbd> MeasurementTime

SNIRF field `MeasurementTime`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the time of the measurement as a string. The format of the time string must either be `"unknown"` or follow the ISO 8601 time string format `hh:mm:ss.sTZD`, where 
- `hh` is the 2-digit hour 
- `mm` is the 2-digit minute 
- `ss` is the 2-digit second 
- `.s` is 1 or more digit representing a decimal fraction of a second (optional) 
- `TZD` is the time zone designator (`Z` or `+hh:mm` or `-hh:mm`) 

---

#### <kbd>property</kbd> SubjectID

SNIRF field `SubjectID`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the string-valued ID of the study subject or experiment. 

---

#### <kbd>property</kbd> TimeUnit

SNIRF field `TimeUnit`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This record stores the **case-sensitive** SI time unit used in this  measurement. Sample time units include "s", and "ms". A value of "us"  is the same as "ms", i.e. microsecond. 

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 



---

<a href="../snirf/pysnirf2.py#L5944"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(name, value)
```

Add a new tag to the list. 



**Args:**
 
 - <b>`name`</b> (str):  The name of the tag to add (will be added as an attribute of this `MetaDataTags` instance) 
 - <b>`value`</b>:  The value of the new tag 

---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L5962"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove`

```python
remove(name)
```

Remove a tag from the list. You cannot remove a required tag. 



**Args:**
 
 - <b>`name`</b> (str):  The name of the tag to remove. 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L1901"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Probe`




<a href="../snirf/pysnirf2.py#L1908"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(var, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> coordinateSystem

SNIRF field `coordinateSystem`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Defines the coordinate system for sensor positions. The string must be one of the coordinate systems listed in the [BIDS specification (Appendix VII)](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html#standard-template-identifiers) such as "MNI152NLin2009bAsym", "CapTrak" or "Other". If the value "Other" is specified, then a defition of the coordinate system must be provided in `/nirs(i)/probe/coordinateSystemDescription`. See the [FieldTrip toolbox web page](https://www.fieldtriptoolbox.org/faq/coordsys/) for detailed descriptions of different coordinate systems. 

---

#### <kbd>property</kbd> coordinateSystemDescription

SNIRF field `coordinateSystemDescription`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Free-form text description of the coordinate system. May also include a link to a documentation page or paper describing the system in greater detail. This field is required if the `coordinateSystem` field is set to "Other". 

---

#### <kbd>property</kbd> correlationTimeDelayWidths

SNIRF field `correlationTimeDelayWidths`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the time delay widths (in `TimeUnit` units) used for diffuse correlation  spectroscopy measurements. This field is only required for gated time domain  data types, and is indexed by `measurementList(k).dataTypeIndex`. The indexing  of this field is paired with the indexing of `probe.correlationTimeDelays`.   

---

#### <kbd>property</kbd> correlationTimeDelays

SNIRF field `correlationTimeDelays`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the time delays (in `TimeUnit` units) used for diffuse correlation spectroscopy  measurements. This field is only required for diffuse correlation spectroscopy  data types, and is indexed by `measurementList(k).dataTypeIndex`.  The indexing  of this field is paired with the indexing of `probe.correlationTimeDelayWidths`. 

---

#### <kbd>property</kbd> detectorLabels

SNIRF field `detectorLabels`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a string array providing user friendly or instrument specific labels  for each detector. Each element of the array must be a unique string among both  `probe.sourceLabels` and `probe.detectorLabels`. This is indexed by  `measurementList(k).detectorIndex`. 

---

#### <kbd>property</kbd> detectorPos2D

SNIRF field `detectorPos2D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Same as `probe.sourcePos2D`, but describing the detector positions in a  flattened 2D probe layout. 

---

#### <kbd>property</kbd> detectorPos3D

SNIRF field `detectorPos3D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the position (in `LengthUnit` units) of each detector  optode in 3D, defined similarly to `sourcePos3D`. 

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> frequencies

SNIRF field `frequencies`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the frequencies used (in `FrequencyUnit` units)  for  frequency domain measurements. This field is only required for frequency  domain data types, and is indexed by `measurementList(k).dataTypeIndex`. 

---

#### <kbd>property</kbd> landmarkLabels

SNIRF field `landmarkLabels`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This string array stores the names of the landmarks. The first string denotes  the name of the landmarks with an index of 1 in the 4th column of  `probe.landmark`, and so on. One can adopt the commonly used 10-20 landmark  names, such as "Nasion", "Inion", "Cz" etc, or use user-defined landmark  labels. The landmark label can also use the unique source and detector labels  defined in `probe.sourceLabels` and `probe.detectorLabels`, respectively, to  associate the given landmark to a specific source or detector. All strings are  ASCII encoded char arrays. 

---

#### <kbd>property</kbd> landmarkPos2D

SNIRF field `landmarkPos2D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a 2-D array storing the neurological landmark positions projected along the 2-D (flattened) probe plane in order to map optical data from the flattened optode positions to brain anatomy. This array should contain a minimum  of 2 columns, representing the x and y coordinates (in `LengthUnit` units) of the 2-D projected landmark positions. If a 3rd column presents, it stores  the index to the labels of the given landmark. Label names are stored in the  `probe.landmarkLabels` subfield. An label index of 0 refers to an undefined landmark.  

---

#### <kbd>property</kbd> landmarkPos3D

SNIRF field `landmarkPos3D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a 2-D array storing the neurological landmark positions measurement  from 3-D digitization and tracking systems to facilitate the registration and  mapping of optical data to brain anatomy. This array should contain a minimum  of 3 columns, representing the x, y and z coordinates (in `LengthUnit` units)  of the digitized landmark positions. If a 4th column presents, it stores the  index to the labels of the given landmark. Label names are stored in the  `probe.landmarkLabels` subfield. An label index of 0 refers to an undefined landmark.  

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> momentOrders

SNIRF field `momentOrders`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the moment orders of the temporal point spread function (TPSF) or the distribution of time-of-flight (DTOF) for moment time domain measurements. This field is only required for moment time domain data types, and is indexed by `measurementList(k).dataTypeIndex`.   Note that the numeric value in this array is the exponent in the integral used for calculating the moments. For detailed/specific definitions of moments, see [Wabnitz et al, 2020](https://doi.org/10.1364/BOE.396585); for general definitions of moments see [here](https://en.wikipedia.org/wiki/Moment_(mathematics) ). 

In brief, given a TPSF or DTOF N(t) (photon counts vs. photon arrival time at the detector): / momentOrder = 0: total counts: `N_total = /intergral N(t)dt` / momentOrder = 1: mean time of flight: `m = <t> = (1/N_total) /integral t N(t) dt` / momentOrder = 2: variance/second central moment: `V = (1/N_total) /integral (t - <t>)^2 N(t) dt` / Please note that all moments (for orders >=1) are expected to be normalized by the total counts (i.e. n=0); Additionally all moments (for orders >= 2) are expected to be centralized. 

---

#### <kbd>property</kbd> sourceLabels

SNIRF field `sourceLabels`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a string array providing user friendly or instrument specific labels  for each source. Each element of the array must be a unique string among both  `probe.sourceLabels` and `probe.detectorLabels`.This can be of size `<number  of sources>x 1` or `<number of sources> x <number of  wavelengths>`. This is indexed by `measurementList(k).sourceIndex` and  `measurementList(k).wavelengthIndex`. 

---

#### <kbd>property</kbd> sourcePos2D

SNIRF field `sourcePos2D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the position (in `LengthUnit` units) of each source  optode. The positions are coordinates in a flattened 2D probe layout.  This field has size `<number of sources> x 2`. For example,  `probe.sourcePos2D(1,:) = [1.4 1]`, and `LengthUnit='cm'` places source  number 1 at x=1.4 cm and y=1 cm. 

---

#### <kbd>property</kbd> sourcePos3D

SNIRF field `sourcePos3D`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the position (in `LengthUnit` units) of each source  optode in 3D. This field has size `<number of sources> x 3`. 

---

#### <kbd>property</kbd> timeDelayWidths

SNIRF field `timeDelayWidths`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the time delay widths (in `TimeUnit` units) used for gated time domain  measurements. This field is only required for gated time domain data types, and  is indexed by `measurementList(k).dataTypeIndex`.  The indexing of this field  is paired with the indexing of `probe.timeDelays`. 

---

#### <kbd>property</kbd> timeDelays

SNIRF field `timeDelays`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the time delays (in `TimeUnit` units) used for gated time domain measurements.  This field is only required for gated time domain data types, and is indexed by  `measurementList(k).dataTypeIndex`. The indexing of this field is paired with  the indexing of `probe.timeDelayWidths`.  

---

#### <kbd>property</kbd> useLocalIndex

SNIRF field `useLocalIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

For modular NIRS systems, setting this flag to a non-zero integer indicates  that `measurementList(k).sourceIndex` and `measurementList(k).detectorIndex`  are module-specific local-indices. One must also include  `measurementList(k).moduleIndex`, or when cross-module channels present, both  `measurementList(k).sourceModuleIndex` and `measurementList(k).detectorModuleIndex`  in the `measurementList` structure in order to restore the global indices  of the sources/detectors. 

---

#### <kbd>property</kbd> wavelengths

SNIRF field `wavelengths`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field describes the "nominal" wavelengths used (in `nm` unit).  This is indexed by the  `wavelengthIndex` of the measurementList variable. For example, `probe.wavelengths` = [690,  780, 830]; implies that the measurements were taken at three wavelengths (690 nm,  780 nm, and 830 nm).  The wavelength index of  `measurementList(k).wavelengthIndex` variable refers to this field. `measurementList(k).wavelengthIndex` = 2 means the k<sup>th</sup> measurement  was at 780 nm. 

Please note that this field stores the "nominal" wavelengths. If the precise  (measured) wavelengths differ from the nominal wavelengths, one can store those in the `measurementList.wavelengthActual` field in a per-channel fashion. 

The number of wavelengths is not limited (except that at least two are needed  to calculate the two forms of hemoglobin).  Each source-detector pair would  generally have measurements at all wavelengths. 

This field must present, but can be empty, for example, in the case that the stored data are processed data (`dataType=99999`, see Appendix). 

---

#### <kbd>property</kbd> wavelengthsEmission

SNIRF field `wavelengthsEmission`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This field is required only for fluorescence data types, and describes the  "nominal" emission wavelengths used (in `nm` unit).  The indexing of this variable is the same  wavelength index in measurementList used for `probe.wavelengths` such that the  excitation wavelength is paired with this emission wavelength for a given measurement. 

Please note that this field stores the "nominal" emission wavelengths. If the precise  (measured) emission wavelengths differ from the nominal ones, one can store those in the `measurementList.wavelengthEmissionActual` field in a per-channel fashion. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L3412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NirsElement`
Wrapper for an element of indexed group `Nirs`. 

<a href="../snirf/pysnirf2.py#L3414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid: GroupID, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> aux

SNIRF field `aux`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This optional array specifies any recorded auxiliary data. Each element of  `aux` has the following required fields: 

---

#### <kbd>property</kbd> data

SNIRF field `data`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This group stores one block of NIRS data.  This can be extended adding the  count number (e.g. `data1`, `data2`,...) to the group name.  This is intended to  allow the storage of 1 or more blocks of NIRS data from within the same `/nirs`  entry * `/nirs/data1` =  data block 1 * `/nirs/data2` =  data block 2  





---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> metaDataTags

SNIRF field `metaDataTags`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

The `metaDataTags` group contains the metadata associated with the measurements. Each metadata record is represented as a dataset under this group - with the name of the record, i.e. the key, as the dataset's name, and the value of the record as the  actual data stored in the dataset. Each metadata record can potentially have different  data types. Sub-groups should not be used to organize metadata records: a member of the `metaDataTags` Group must be a Dataset. 

The below five metadata records are minimally required in a SNIRF file 

---

#### <kbd>property</kbd> probe

SNIRF field `probe`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a structured variable that describes the probe (source-detector)  geometry.  This variable has a number of required fields. 

---

#### <kbd>property</kbd> stim

SNIRF field `stim`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is an array describing any stimulus conditions. Each element of the array  has the following required fields. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L3679"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Nirs`
Interface for indexed group `Nirs`. 

Can be indexed like a list to retrieve `Nirs` elements. 

To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively. 

This group stores one set of NIRS data.  This can be extended by adding the count  number (e.g. `/nirs1`, `/nirs2`,...) to the group name. This is intended to  allow the storage of 1 or more complete NIRS datasets inside a single SNIRF  document.  For example, a two-subject hyperscanning can be stored using the notation * `/nirs1` =  first subject's data * `/nirs2` =  second subject's data The use of a non-indexed (e.g. `/nirs`) entry is allowed when only one entry  is present and is assumed to be entry 1. 

<a href="../snirf/pysnirf2.py#L3700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h: File, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L3704"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DataElement`




<a href="../snirf/pysnirf2.py#L3706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid: GroupID, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> dataTimeSeries

SNIRF field `dataTimeSeries`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is the actual raw or processed data variable. This variable has dimensions  of `<number of time points> x <number of channels>`. Columns in  `dataTimeSeries` are mapped to the measurement list (`measurementList` variable  described below). 

`dataTimeSeries` can be compressed using the HDF5 filter (using the built-in  [`deflate`](https://portal.hdfgroup.org/display/HDF5/H5P_SET_DEFLATE) filter or [3rd party filters such as `305-LZO` or `307-bzip2`](https://portal.hdfgroup.org/display/support/Registered+Filter+Plugins) 

Chunked data is allowed to support real-time streaming of data in this array.  

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> measurementList

SNIRF field `measurementList`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

The measurement list. This variable serves to map the data array onto the probe  geometry (sources and detectors), data type, and wavelength. This variable is  an array structure that has the size `<number of channels>` that  describes the corresponding column in the data matrix. For example, the  `measurementList3` describes the third column of the data matrix (i.e.  `dataTimeSeries(:,3)`). 

Each element of the array is a structure which describes the measurement  conditions for this data with the following fields: 

---

#### <kbd>property</kbd> time

SNIRF field `time`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

The `time` variable. This provides the acquisition time of the measurement  relative to the time origin.  This will usually be a straight line with slope  equal to the acquisition frequency, but does not need to be equal spacing.  For  the special case of equal sample spacing an array of length `<2>` is allowed  where the first entry is the start time and the  second entry is the sample time spacing in `TimeUnit` specified in the  `metaDataTags`. The default time unit is in second ("s"). For example,  a time spacing of 0.2 (s) indicates a sampling rate of 5 Hz.  

* **Option 1** - The size of this variable is `<number of time points>` and   corresponds to the sample time of every data point * **Option 2**-  The size of this variable is `<2>` and corresponds to the start  time and sample spacing. 

Chunked data is allowed to support real-time streaming of data in this array. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L3947"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Data`




<a href="../snirf/pysnirf2.py#L3966"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h: File, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L3970"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MeasurementListElement`
Wrapper for an element of indexed group `MeasurementList`. 

<a href="../snirf/pysnirf2.py#L3972"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid: GroupID, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> dataType

SNIRF field `dataType`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Data-type identifier. See Appendix for list possible values. 

---

#### <kbd>property</kbd> dataTypeIndex

SNIRF field `dataTypeIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Data-type specific parameter indices. The data type index specifies additional data type specific parameters that are further elaborated by other fields in the probe structure, as detailed below. Note that the Time Domain and Diffuse Correlation Spectroscopy data types have two additional parameters and so the data type index must be a vector with 2 elements that index the additional parameters. One use of this parameter is as a  stimulus condition index when `measurementList(k).dataType = 99999` (i.e, `processed` and  `measurementList(k).dataTypeLabel = 'HRF ...'` . 

---

#### <kbd>property</kbd> dataTypeLabel

SNIRF field `dataTypeLabel`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Data-type label. Only required if dataType is "processed" (`99999`). See Appendix  for list of possible values. 

---

#### <kbd>property</kbd> dataUnit

SNIRF field `dataUnit`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

International System of Units (SI units) identifier for the given channel. Encoding should follow the [CMIXF-12 standard](https://people.csail.mit.edu/jaffer/MIXF/CMIXF-12), avoiding special unicode symbols like U+03BC (m) or U+00B5 (u) and using '/' rather than 'per' for units such as `V/us`. The recommended export format is in unscaled units such as V, s, Mole. 

---

#### <kbd>property</kbd> detectorGain

SNIRF field `detectorGain`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Detector gain 

---

#### <kbd>property</kbd> detectorIndex

SNIRF field `detectorIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of the detector. 

---

#### <kbd>property</kbd> detectorModuleIndex

SNIRF field `detectorModuleIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of the module that contains the detector of the channel.  This index must be used together with `sourceModuleIndex`, and  can not be used when `moduleIndex` presents. 



For example, if `measurementList5` is a structure with `sourceIndex=2`,  `detectorIndex=3`, `wavelengthIndex=1`, `dataType=1`, `dataTypeIndex=1` would  imply that the data in the 5th column of the `dataTimeSeries` variable was  measured with source #2 and detector #3 at wavelength #1.  Wavelengths (in  nanometers) are described in the `probe.wavelengths` variable  (described  later). The data type in this case is 1, implying that it was a continuous wave  measurement.  The complete list of currently supported data types is found in  the Appendix. The data type index specifies additional data type specific  parameters that are further elaborated by other fields in the `probe`  structure, as detailed below. Note that the Time Domain and Diffuse Correlation  Spectroscopy data types have two additional parameters and so the data type  index must be a vector with 2 elements that index the additional parameters. 

`sourcePower` provides the option for information about the source power for  that channel to be saved along with the data. The units are not defined, unless  the user takes the option of using a `metaDataTag` described below to define,  for instance, `sourcePowerUnit`. `detectorGain` provides the option for  information about the detector gain for that channel to be saved along with the  data. 

Note:  The source indices generally refer to the optode naming (probe  positions) and not necessarily the physical laser numbers on the instrument.  The same is true for the detector indices.  Each source optode would generally,  but not necessarily, have 2 or more wavelengths (hence lasers) plugged into it  in order to calculate deoxy- and oxy-hemoglobin concentrations. The data from  these two wavelengths will be indexed by the same source, detector, and data  type values, but have different wavelength indices. Using the same source index  for lasers at the same location but with different wavelengths simplifies the  bookkeeping for converting intensity measurements into concentration changes.  As described below, optional variables `probe.sourceLabels` and  `probe.detectorLabels` are provided for indicating the instrument specific  label for sources and detectors. 

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> moduleIndex

SNIRF field `moduleIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of a repeating module. If `moduleIndex` is provided while `useLocalIndex` is set to `true`, then, both `measurementList(k).sourceIndex` and  `measurementList(k).detectorIndex` are assumed to be the local indices of the same module specified by `moduleIndex`. If the source and detector are located on different modules, one must use `sourceModuleIndex` and `detectorModuleIndex` instead to specify separate parent module  indices. See below. 

---

#### <kbd>property</kbd> sourceIndex

SNIRF field `sourceIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of the source. 



---

#### <kbd>property</kbd> sourceModuleIndex

SNIRF field `sourceModuleIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of the module that contains the source of the channel.  This index must be used together with `detectorModuleIndex`, and  can not be used when `moduleIndex` presents. 

---

#### <kbd>property</kbd> sourcePower

SNIRF field `sourcePower`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

The units are not defined, unless the user takes the option of using a `metaDataTag` as described below. 

---

#### <kbd>property</kbd> wavelengthActual

SNIRF field `wavelengthActual`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Actual (measured) wavelength in nm, if available, for the source in a given channel. 

---

#### <kbd>property</kbd> wavelengthEmissionActual

SNIRF field `wavelengthEmissionActual`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Actual (measured) emission wavelength in nm, if available, for the source in a given channel. 



---

#### <kbd>property</kbd> wavelengthIndex

SNIRF field `wavelengthIndex`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

Index of the "nominal" wavelength (in `probe.wavelengths`). 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5019"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MeasurementList`
Interface for indexed group `MeasurementList`. 

Can be indexed like a list to retrieve `MeasurementList` elements. 

To add or remove an element from the list, use the `appendGroup` method and the `del` operator, respectively. 

The measurement list. This variable serves to map the data array onto the probe  geometry (sources and detectors), data type, and wavelength. This variable is  an array structure that has the size `<number of channels>` that  describes the corresponding column in the data matrix. For example, the  `measurementList3` describes the third column of the data matrix (i.e.  `dataTimeSeries(:,3)`). 

Each element of the array is a structure which describes the measurement  conditions for this data with the following fields: 

<a href="../snirf/pysnirf2.py#L5041"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h: File, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5045"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StimElement`




<a href="../snirf/pysnirf2.py#L5047"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid: GroupID, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> data

SNIRF field `data`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

* **Allowed attribute**: `names` 

This is a numeric 2-D array with at least 3 columns, specifying the stimulus  time course for the j<sup>th</sup> condition. Each row corresponds to a  specific stimulus trial. The first three columns indicate `[starttime duration value]`.   The starttime, in seconds, is the time relative to the time origin when the  stimulus takes on a value; the duration is the time in seconds that the stimulus  value continues, and value is the stimulus amplitude.  The number of rows is  not constrained. (see examples in the appendix). 

Additional columns can be used to store user-specified data associated with  each stimulus trial. An optional record `/nirs(i)/stim(j)/dataLabels` can be  used to annotate the meanings of each data column.  

---

#### <kbd>property</kbd> dataLabels

SNIRF field `dataLabels`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a string array providing annotations for each data column in  `/nirs(i)/stim(j)/data`. Each element of the array must be a string; the total length of this array must be the same as the column number of `/nirs(i)/stim(j)/data`, including the first 3 required columns. 

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> name

SNIRF field `name`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a string describing the j<sup>th</sup> stimulus condition. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Stim`




<a href="../snirf/pysnirf2.py#L5308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h: File, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AuxElement`




<a href="../snirf/pysnirf2.py#L5314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid: GroupID, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> dataTimeSeries

SNIRF field `dataTimeSeries`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is the aux data variable. This variable has dimensions of `<number of  time points> x <number of channels>`. If multiple channels of related data are generated by a system, they may be encoded in the multiple columns of the time series (i.e. complex numbers). For example, a system containing more than one accelerometer may output this data as a set of `ACCEL_X`/`ACCEL_Y`/`ACCEL_Z` auxiliary time series, where each has the dimension of `<number of time points> x <number of accelerometers>`. Note that it is NOT recommended to encode the various accelerometer dimensions as multiple channels of the same `aux` Group: instead follow the `"ACCEL_X"`, `"ACCEL_Y"`, `"ACCEL_Z"` naming conventions described in the appendix. Chunked data is allowed to support real-time data streaming. 

---

#### <kbd>property</kbd> dataUnit

SNIRF field `dataUnit`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

International System of Units (SI units) identifier for the given channel. Encoding should follow the [CMIXF-12 standard](https://people.csail.mit.edu/jaffer/MIXF/CMIXF-12), avoiding special unicode symbols like U+03BC (m) or U+00B5 (u) and using '/' rather than 'per' for units such as `V/us`. The recommended export format is in unscaled units such as V, s, Mole. 

---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

None if not associated with a Group on disk. 

---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> name

SNIRF field `name`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is string describing the j<sup>th</sup> auxiliary data timecourse. While auxiliary data can be given any title, standard names for commonly used auxiliary channels (i.e. accelerometer data) are specified in the appendix. 

---

#### <kbd>property</kbd> time

SNIRF field `time`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

The time variable. This provides the acquisition time (in `TimeUnit` units)  of the aux measurement relative to the time origin.  This will usually be  a straight line with slope equal to the acquisition frequency, but does  not need to be equal spacing. The size of this variable is  `<number of time points>` or `<2>` similar  to definition of the  `/nirs(i)/data(j)/time` field. 

Chunked data is allowed to support real-time data streaming 

---

#### <kbd>property</kbd> timeOffset

SNIRF field `timeOffset`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This variable specifies the offset of the file time origin relative to absolute (clock) time in `TimeUnit` units. 



---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L922"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Group level save to a SNIRF file on disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on a Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].probe.save()```

    or using a new filename to write the Group there:
    >>> mysnirf.nirs[0].probe.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5688"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Aux`




<a href="../snirf/pysnirf2.py#L5702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h: File, cfg: SnirfConfig)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 



---

<a href="../snirf/pysnirf2.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```

Append a new Group to the IndexedGroup. 



**Args:**
 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Insert a new Group at the end of the Indexed Group. 

Creates an empty Group with the appropriate name at the end of the list of Groups managed by the IndexedGroup. 

---

<a href="../snirf/pysnirf2.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```

Insert a new Group into the IndexedGroup. 



**Args:**
 
 - <b>`i`</b> (int):  an index 
 - <b>`item`</b>:  must be of type _element 

---

<a href="../snirf/pysnirf2.py#L1235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insertGroup`

```python
insertGroup(i)
```

Insert a new Group following the index given. 

Creates an empty Group with a placeholder name within the list of Groups managed by the IndexedGroup. The placeholder name will be replaced with a name with the correct order once `save` is called. 



**Args:**
 
 - <b>`i`</b> (int):  the position at which to insert the new Group 

---

<a href="../snirf/pysnirf2.py#L1126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

Returns True if the Indexed Group has no member Groups with contents. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L1167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save the groups to a SNIRF file on disk. 

When saving, the naming convention defined by the SNIRF spec is enforced: groups are named `/<name>1`, `/<name>2`, `/<name>3`, and so on. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File):  A path to a closed SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can be called on an Indexed Group already on disk to overwrite the current contents: ``` mysnirf.nirs[0].stim.save()```

    or using a new filename to write the Indexed Group there:
    >>> mysnirf.nirs[0].stim.save(<new destination>)



---

<a href="../snirf/pysnirf2.py#L5706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Snirf`




<a href="../snirf/pysnirf2.py#L5711"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, dynamic_loading: bool = False, enable_logging: bool = False)
```






---

#### <kbd>property</kbd> filename

The filename the Snirf object was loaded from and will save to. 

---

#### <kbd>property</kbd> formatVersion

SNIRF field `formatVersion`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This is a string that specifies the version of the file format.  This document  describes format version "1.0" 



---

#### <kbd>property</kbd> location

The HDF5 relative location indentifier. 

None if not associataed with a Group on disk. 

---

#### <kbd>property</kbd> nirs

SNIRF field `nirs`. 

If dynamic_loading=True, the data is loaded from the SNIRF file only when accessed through the getter 

This group stores one set of NIRS data.  This can be extended by adding the count  number (e.g. `/nirs1`, `/nirs2`,...) to the group name. This is intended to  allow the storage of 1 or more complete NIRS datasets inside a single SNIRF  document.  For example, a two-subject hyperscanning can be stored using the notation * `/nirs1` =  first subject's data * `/nirs2` =  second subject's data The use of a non-indexed (e.g. `/nirs`) entry is allowed when only one entry  is present and is assumed to be entry 1. 



---

<a href="../snirf/pysnirf2.py#L6158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `close`

```python
close()
```

Close the file underlying a `Snirf` instance. 

After closing, the underlying SNIRF file cannot be accessed from this interface again. Use `close` if you need to open a new interface on the same HDF5 file. 

`close` is called automatically by the destructor. 

---

<a href="../snirf/pysnirf2.py#L6125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy`

```python
copy() → Snirf
```

Return a copy of the Snirf instance.  

A copy of a Snirf instance is a brand new HDF5 file in memory. This can  be expensive to create. Note that in lieu of copying you can make assignments between Snirf instances.  

---

<a href="../snirf/pysnirf2.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```

If the Group has no member Groups or Datasets. 



**Returns:**
 
 - <b>`bool`</b>:  True if empty, False if not 

---

<a href="../snirf/pysnirf2.py#L6090"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save a SNIRF file to disk. 



**Args:**
 
 - <b>`args`</b> (str or h5py.File or file-like):  A path to a closed or nonexistant SNIRF file on disk or an open `h5py.File` instance 



**Examples:**
 save can overwrite the current contents of a Snirf file: ``` mysnirf.save()```

    or take a new filename to write the file there:
    >>> mysnirf.save(<new destination>)
    
    or write to an IO stream:
    >>> mysnirf.save(<io.BytesIO stream>)


---

<a href="../snirf/pysnirf2.py#L6136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validate`

```python
validate() → ValidationResult
```

Validate a `Snirf` instance. 

Returns the validity of the current state of a `Snirf` object, including modifications made in memory to a loaded SNIRF file. 



**Returns:**
 
 - <b>`ValidationResult`</b>:  truthy structure containing detailed validation results 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
