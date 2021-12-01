SPEC_SRC = 'https://raw.githubusercontent.com/fNIRS/snirf/v1.0/snirf_specification.md'
VERSION = '1.0'  # Version of the spec linked above
TYPELUT = {  # usage: if TYPELUT[<type>] in <typecode>:
        'GROUP': '{.}',
        'INDEXED_GROUP': '{i}',
        'REQUIRED': '*',
        'ARRAY_1D': '...]',
        'ARRAY_2D': '...]]',
        'INT_VALUE': '<i>',
        'FLOAT_VALUE': '<f>',
        'VARLEN_STRING': '"s"'
        }
TEMPLATE = 'pysnirf2.jinja'
HEADER = 'header.py'
FOOTER = 'footer.py'
TABLE_DELIM_START = '### SNIRF data format summary'
TABLE_DELIM_END = 'In the above table, the used notations are explained below'
DEFINITIONS_DELIM_START = '### SNIRF data container definitions'
DEFINITIONS_DELIM_END = '## Appendix'
