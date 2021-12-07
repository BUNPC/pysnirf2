# Generating pysnirf2 from the specification text

pysnirf2 is generated dynamically from the [SNIRF specification document](https://raw.githubusercontent.com/fNIRS/snirf/master/snirf_specification.md) maintained in the official repository.

This ensures easy maintenance of the project as the specification develops.

## Overview
 
[gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) parses the spec document at the location defined as `SPEC_SRC` in [data.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/data.py).
 
[gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) then generates a dictionary of data which is combined with jinja2 template [pysnirf2.jinja](https://github.com/BUNPC/pysnirf2/blob/main/gen/pysnirf2.jinja) which metaprogrammatically generates the library source [pysnirf2.py](https://github.com/BUNPC/pysnirf2/blob/main/pysnirf2/pysnirf2.py). [header.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/header.py) and [footer.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/footer.py) sandwich the code generated from the template.

## How-to

1. Ensure that [data.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/data.py) contains correct data to parse the latest spec. Make sure `SPEC_SRC` and `VERSION` are up to date.
2. Using a Python > 3.6 environment equipped with [gen/requirements.txt](https://github.com/BUNPC/pysnirf2/blob/main/gen/requirements.txt), run [gen.py](https://github.com/BUNPC/pysnirf2/blob/main/gen/gen.py) from the project root
3. Test the resulting library
