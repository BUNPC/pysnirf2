SPEC_SRC = 'https://raw.githubusercontent.com/fNIRS/snirf/v1.1/snirf_specification.md'
SPEC_VERSION = 'v1.1'  # Version of the spec linked above

"""
These types are fragments of the string codes used to describe the types of
SNIRF datasets in the table of the specification document.

i.e. if TYPELUT[<type>] in <typecode>:
"""
TYPELUT = {
        'GROUP': '{.}',
        'INDEXED_GROUP': '{i}',
        'REQUIRED': '*',
        'ARRAY_2D': '...]]',  
        'ARRAY_1D': '...]',  # Parsed with if-else so these can be distinguished with a simple "in"
        'INT_VALUE': '<i>',
        'FLOAT_VALUE': '<f>',
        'VARLEN_STRING': '"s"'
        }

"""
Groups with these names will be exempt from warnings and attempt to load all Datasets
"""
UNSPECIFIED_DATASETS_OK = ['metaDataTags']

TEMPLATE = 'gen/pysnirf2.jinja'  # The name of the template file to use
HEADER = 'gen/header.py'  # The contents of this file are appended to the front of the template
FOOTER = 'gen/footer.py'  # The contents of the file and appended to the end of the file

"""
These strings are used to identify the beginning and end of the table
and definitions sections of the spec document
"""
TABLE_DELIM_START = '[//]: # (SCHEMA BEGIN)'
TABLE_DELIM_END = '[//]: # (SCHEMA END)'

DEFINITIONS_DELIM_START = '### SNIRF data container definitions'
DEFINITIONS_DELIM_END = '## Appendix'

# -- BIDS Probe name identifiers ---------------------------------------------

BIDS_PROBE_NAMES = ['ICBM452AirSpace',
                    'ICBM452Warp5Space',
                    'IXI549Space',
                    'fsaverage',
                    'fsaverageSym',
                    'fsLR',
                    'MNIColin27',
                    'MNI152Lin',
                    'MNI152NLin2009[a-c][Sym|Asym]',
                    'MNI152NLin6Sym',
                    'MNI152NLin6ASym',
                    'MNI305',
                    'NIHPD',
                    'OASIS30AntsOASISAnts',
                    'OASIS30Atropos',
                    'Talairach',
                    'UNCInfant']

