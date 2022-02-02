SPEC_SRC = 'https://raw.githubusercontent.com/fNIRS/snirf/master/snirf_specification.md'
VERSION = 'v1.0.1-development'  # Version of the spec linked above

"""
These types are fragments of the string codes used to describe the types of
SNIRF datasets in the table of the specification document.

i.e. if TYPELUT[<type>] in <typecode>:
"""
TYPELUT = {
        'GROUP': '{.}',
        'INDEXED_GROUP': '{i}',
        'REQUIRED': '*',
        'ARRAY_1D': '...]',
        'ARRAY_2D': '...]]',
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
TABLE_DELIM_START = '### SNIRF data format summary'
TABLE_DELIM_END = 'In the above table, the used notations are explained below'

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

