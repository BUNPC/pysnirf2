# pysnirf2

Python library for loading, saving, and validating Snirf files

--

pysnirf2.py

```python
From pysnirf2 import *

<Snirf object> = loadSnirf(<source filename>)
saveSnirf(<Snirf object>, <destination filename>)

<error> = validate(<source filename>)
<error> = validate(<Snirf object>)
<error>, <verbose result> = validate

--

Snirf

Snirf(<source filename>)
Snirf.save(<destination filename>)
```

Things to consider:

Code style
https://www.python.org/dev/peps/pep-0008/

How to make objects (like Snirf or perhaps SnirfValidationResult, if you defined <verbose result> as a class) print nicely by implementing __repr__:
https://stackoverflow.com/questions/1535327/how-to-print-instances-of-a-class-using-print
  
How to package the project so that users can `pip install pysnirf2` your library:
https://packaging.python.org/tutorials/packaging-projects/
  
  

