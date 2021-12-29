<!-- markdownlint-disable -->

<a href="../pysnirf2/pysnirf2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pysnirf2`




**Global Variables**
---------------
- **AbsentDataset**
- **AbsentGroup**
- **PresentDataset**

---

<a href="../pysnirf2/pysnirf2.py#L4030"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loadSnirf`

```python
loadSnirf(path:str, dynamic_loading:bool=False, logfile:bool=False) → Snirf
```

Returns a Snirf object loaded from path if a Snirf file exists there. Takes the same kwargs as the Snirf object constructor 


---

<a href="../pysnirf2/pysnirf2.py#L4043"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `saveSnirf`

```python
saveSnirf(path:str, snirfobj:Snirf)
```

Saves a SNIRF file snirfobj to disk at path 


---

<a href="../pysnirf2/pysnirf2.py#L4054"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validateSnirf`

```python
validateSnirf(path:str) → Tuple[bool, ValidationResult]
```

Returns a bool representing the validity of the Snirf object on disk at path along with the detailed output structure ValidationResult instance 


---

<a href="../pysnirf2/pysnirf2.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValidationIssue`
Pretty-printable structure for the result of validation on an HDF5 name 

<a href="../pysnirf2/pysnirf2.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name:str, location:str)
```









---

<a href="../pysnirf2/pysnirf2.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValidationResult`
ORganzies the result of the pysnirf2 validation routine like so: <ValidationResult>.is_valid(), <ValidationResult> = <Snirf>.validate() 

<a href="../pysnirf2/pysnirf2.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```






---

#### <kbd>property</kbd> codes





---

#### <kbd>property</kbd> errors

Returns list of fatal errors 
------- errors : List of ValidationIssue 

---

#### <kbd>property</kbd> info

Returns list of info issues 
------- info : List of ValidationIssue 

---

#### <kbd>property</kbd> issues





---

#### <kbd>property</kbd> locations





---

#### <kbd>property</kbd> warnings

Returns list of warnings 
------- warnings : List of ValidationIssue 



---

<a href="../pysnirf2/pysnirf2.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `display`

```python
display(severity=2)
```

Prints the detailed results of the validation. kwarg severity filters the display by message severity (default 2) severity=0: All messages will be shown, including OK severity=1: Prints INFO, WARNING, and FATAL messages severity=2: Prints WARNING and FATAL messages severity=3: Prints only FATAL error messages 

---

<a href="../pysnirf2/pysnirf2.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_valid`

```python
is_valid()
```

Returns True if valid (if no FATAL errors found) 


---

<a href="../pysnirf2/pysnirf2.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SnirfFormatError`








---

<a href="../pysnirf2/pysnirf2.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SnirfConfig`
Structure for containing Snirf-wide data and settings 





---

<a href="../pysnirf2/pysnirf2.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AbsentDatasetType`








---

<a href="../pysnirf2/pysnirf2.py#L478"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AbsentGroupType`








---

<a href="../pysnirf2/pysnirf2.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PresentDatasetType`








---

<a href="../pysnirf2/pysnirf2.py#L493"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Group`




<a href="../pysnirf2/pysnirf2.py#L495"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(varg, cfg:SnirfConfig)
```

Wrapper for an HDF5 Group element defined by SNIRF. Must be created with a Group ID or string specifying a complete path relative to file root--in the latter case, the wrapper will not correspond to a real HDF5 group on disk until _save() (with no arguments) is executed for the first time 


---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L600"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IndexedGroup`
Represents the "indexed group" which is defined by v1.0 of the SNIRF specification as:  If a data element is an HDF5 group and contains multiple sub-groups,  it is referred to as an indexed group. Each element of the sub-group  is uniquely identified by appending a string-formatted index (starting  from 1, with no preceding zeros) in the name, for example, /.../name1  denotes the first sub-group of data element name, and /.../name2  denotes the 2nd element, and so on. 

<a href="../pysnirf2/pysnirf2.py#L615"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(parent:Group, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MetaDataTags`




<a href="../pysnirf2/pysnirf2.py#L799"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(var, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> FrequencyUnit





---

#### <kbd>property</kbd> LengthUnit





---

#### <kbd>property</kbd> MeasurementDate





---

#### <kbd>property</kbd> MeasurementTime





---

#### <kbd>property</kbd> SubjectID





---

#### <kbd>property</kbd> TimeUnit





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location







---

<a href="../pysnirf2/pysnirf2.py#L3902"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(name, value)
```

Add a new tag to the list. 

---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L1147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Probe`




<a href="../pysnirf2/pysnirf2.py#L1149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(var, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> correlationTimeDelayWidths





---

#### <kbd>property</kbd> correlationTimeDelays





---

#### <kbd>property</kbd> detectorLabels





---

#### <kbd>property</kbd> detectorPos2D





---

#### <kbd>property</kbd> detectorPos3D





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> frequencies





---

#### <kbd>property</kbd> landmarkLabels





---

#### <kbd>property</kbd> landmarkPos2D





---

#### <kbd>property</kbd> landmarkPos3D





---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> momentOrders





---

#### <kbd>property</kbd> sourceLabels





---

#### <kbd>property</kbd> sourcePos2D





---

#### <kbd>property</kbd> sourcePos3D





---

#### <kbd>property</kbd> timeDelayWidths





---

#### <kbd>property</kbd> timeDelays





---

#### <kbd>property</kbd> useLocalIndex





---

#### <kbd>property</kbd> wavelengths





---

#### <kbd>property</kbd> wavelengthsEmission







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L2091"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NirsElement`




<a href="../pysnirf2/pysnirf2.py#L2093"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid:GroupID, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> aux





---

#### <kbd>property</kbd> data





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> metaDataTags





---

#### <kbd>property</kbd> probe





---

#### <kbd>property</kbd> stim







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L2261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Nirs`




<a href="../pysnirf2/pysnirf2.py#L2266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h:File, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L2270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DataElement`




<a href="../pysnirf2/pysnirf2.py#L2272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid:GroupID, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> dataTimeSeries





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> measurementList





---

#### <kbd>property</kbd> time







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L2430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Data`




<a href="../pysnirf2/pysnirf2.py#L2435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h:File, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L2439"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MeasurementListElement`




<a href="../pysnirf2/pysnirf2.py#L2441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid:GroupID, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> dataType





---

#### <kbd>property</kbd> dataTypeIndex





---

#### <kbd>property</kbd> dataTypeLabel





---

#### <kbd>property</kbd> dataUnit





---

#### <kbd>property</kbd> detectorGain





---

#### <kbd>property</kbd> detectorIndex





---

#### <kbd>property</kbd> detectorModuleIndex





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> moduleIndex





---

#### <kbd>property</kbd> sourceIndex





---

#### <kbd>property</kbd> sourceModuleIndex





---

#### <kbd>property</kbd> sourcePower





---

#### <kbd>property</kbd> wavelengthActual





---

#### <kbd>property</kbd> wavelengthEmissionActual





---

#### <kbd>property</kbd> wavelengthIndex







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L3218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MeasurementList`




<a href="../pysnirf2/pysnirf2.py#L3223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h:File, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L3227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StimElement`




<a href="../pysnirf2/pysnirf2.py#L3229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid:GroupID, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> data





---

#### <kbd>property</kbd> dataLabels





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> name







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L3414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Stim`




<a href="../pysnirf2/pysnirf2.py#L3419"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h:File, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L3423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AuxElement`




<a href="../pysnirf2/pysnirf2.py#L3425"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gid:GroupID, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> dataTimeSeries





---

#### <kbd>property</kbd> dataUnit





---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> name





---

#### <kbd>property</kbd> time





---

#### <kbd>property</kbd> timeOffset







---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Entry to Group-level save 


---

<a href="../pysnirf2/pysnirf2.py#L3710"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Aux`




<a href="../pysnirf2/pysnirf2.py#L3715"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(h:File, cfg:SnirfConfig)
```






---

#### <kbd>property</kbd> filename







---

<a href="../pysnirf2/pysnirf2.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `append`

```python
append(item)
```





---

<a href="../pysnirf2/pysnirf2.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `appendGroup`

```python
appendGroup()
```

Adds a group to the end of the list 

---

<a href="../pysnirf2/pysnirf2.py#L665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `insert`

```python
insert(i, item)
```





---

<a href="../pysnirf2/pysnirf2.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```






---

<a href="../pysnirf2/pysnirf2.py#L3719"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Snirf`




<a href="../pysnirf2/pysnirf2.py#L3724"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, dynamic_loading:bool=False, logfile:bool=False)
```






---

#### <kbd>property</kbd> filename

Returns None if the wrapper is not associated with a Group on disk         

---

#### <kbd>property</kbd> formatVersion





---

#### <kbd>property</kbd> location





---

#### <kbd>property</kbd> nirs







---

<a href="../pysnirf2/pysnirf2.py#L3881"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `close`

```python
close()
```





---

<a href="../pysnirf2/pysnirf2.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_empty`

```python
is_empty()
```





---

<a href="../pysnirf2/pysnirf2.py#L3856"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(*args)
```

Save changes you have made to the Snirf object to disk. If a filepath is supplied, the changes will be 'saved as' in a new file. 

---

<a href="../pysnirf2/pysnirf2.py#L3872"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validate`

```python
validate() → Tuple[bool, ValidationResult]
```

Returns a bool representing the validity of the Snirf object and the detailed output structure ValidationResult instance 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
