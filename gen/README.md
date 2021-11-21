# Generating pysnirf2 from the specification text

pysnirf2 is generated dynamically from the SNIRF specification document maintained in the official repository. 

## Overview
 
gen.py parses the spec document at the location defined as `SPEC_SRC` in data.py.
 
gen.py then generates a dictionary of data which is combined with jinja2 template [pysnirf2.jinja]() which metaprogrammatically generates pysnirf2.py.

## How-to


1. Ensure that data.py contains correct data to parse the latest spec. Make sure `SPEC_SRC` and `VERSION` are up to date.
2. Using a Python 3 environment equipped with [gen/requirements.txt](), run gen.py
3. Test the output at src/pysnirf.py