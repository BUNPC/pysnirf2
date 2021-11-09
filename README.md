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

prettyprint(<verbose result>)

--

Snirf

Snirf(<source filename>)
Snirf.save(<destination filename>)
```
