# Generating pysnirf2 from the specification text

A significant portion of the pysnirf2 code is generated dynamically from the [SNIRF specification document](https://raw.githubusercontent.com/fNIRS/snirf/master/snirf_specification.md) maintained in the official repository.

This ensures easy maintenance of the project as the specification develops.

## Overview
 
[gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) parses the spec document at the location defined as `SPEC_SRC` in [data.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/data.py).
 
[gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) then generates a dictionary of data which is combined with jinja2 template [pysnirf2.jinja](https://github.com/BUNPC/pysnirf2/blob/main/gen/pysnirf2.jinja) which metaprogrammatically generates the library source [pysnirf2.py](https://github.com/BUNPC/pysnirf2/blob/main/pysnirf2/pysnirf2.py). Code is inserted between the start and end delimeters as specified. [data.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/data.py) should be updated and [gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) should be run when a new version of the SNIRF specification has been released.

## How-to

1. Ensure that [data.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/data.py) contains correct data to parse the latest spec. Make sure `SPEC_SRC` and `VERSION` are up to date.
2. IMPORTANT! Back up or commit local changes to the code via git. The generation process may delete your changes.
3. Using a Python > 3.9 environment equipped with [gen/requirements.txt](https://github.com/BUNPC/pysnirf2/blob/main/gen/requirements.txt), run [gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) from the project root
4. Test the resulting library
