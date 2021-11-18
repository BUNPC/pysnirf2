from jinja2 import Environment, FileSystemLoader, select_autoescape
import requests
from unidecode import unidecode
from pathlib import Path
from datetime import date
import getpass
import os


from data import TYPELUT, SPEC_SRC, VERSION

"""
Generates SNIRF interface and validator from the summary table of the specification
hosted at SPEC_SRC.
"""

env = Environment(
    loader=FileSystemLoader(searchpath='./'),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,    
    )

template = env.get_template('pysnirf2.jinja')

# Retrieve the SNIRF specification from GitHub and parse it for the summary table
spec = requests.get(SPEC_SRC)
table = unidecode(spec.text).split('### SNIRF data format summary')[1].split('In the above table, the used notations are explained below')[0]
                 
rows = table.split('\n')
while '' in rows:
    rows.remove('')

type_codes = []

# Parse each row of the summary table for programmatic info
for row in rows[1:]:  # Skip header row
    delim = row.split('|')
    
    # Number of leading spaces determines indent level, hierarchy
    namews = delim[1].replace('`', '').replace('.','').replace('-', '')
    name = namews.replace(' ', '').replace('/', '').replace('{i}', '')
    
    # Get name: format pairs for each name
    if len(name) > 1:  # Skip the empty row
        type_code = delim[-2].replace(' ', '').replace('`', '')
        type_codes.append(type_code)

# Parse headings in spec for complete HDF style locations with which to build tree
definitions = unidecode(spec.text).split('### SNIRF data container definitions')[1].split('## Appendix')[0]
lines = definitions.split('\n')
while '' in lines:
    lines.remove('')
    
locations = []
for line in lines:
    line = line.replace(' ', '')
    if line.startswith('####'):
        locations.append(line.replace('`', '').replace('#', '').replace(',', ''))

# Create flat list of all nodes
flat = []

# Append root
flat.append({
            'name': '',
            'location': '/',
            'parent': None,
            'type': '/',
            'children': [],
            'required': False
            })

for i, location in enumerate(locations):
        type_code = type_codes[i]
        name = location.split('/')[-1].split('(')[0]  # Remove (i), (j)
        parent = location.split('/')[-2].split('(')[0]  # Remove (i), (j)
        print('Importing', location, 'with type', type_code)
        if type_code is not None:
            required = TYPELUT['REQUIRED'] in type_code
        else:
            required = None
#        if name in type_codes.keys():
#            type_code = type_codes[name]
#            required = TYPELUT['REQUIRED'] in type_code
#        else:
#            warn('Found a name with no corresponding type: ' + name)
#            type_code = None
#            required = None
        if not any([location == node['location'] for node in flat]):   
            flat.append({
                        'name': name,
                        'location': location,
                        'parent': parent,
                        'type': type_code,
                        'children': [],
                        'required': required
                        })

#  Generate data for template
SNIRF = {
        'VERSION': VERSION,
        'SPEC_SRC': SPEC_SRC,
        'BASE': '',
        'ROOT': flat[0],  # Snirf root '/'
        'TYPES': TYPELUT, 
        'USER': getpass.getuser(),
        'DATE': str(date.today()),
        'INDEXED_GROUPS': [], 
        'GROUPS': [], 
        }

#  Build list of groups and indexed groups
for node in flat:
    if node['type'] is not None:
        for other_node in flat:
            if other_node['parent'] == node['name']:
                node['children'].append(other_node)
        if TYPELUT['GROUP'] in node['type']:
            SNIRF['GROUPS'].append(node)
        elif TYPELUT['INDEXED_GROUP'] in node['type']:
            SNIRF['INDEXED_GROUPS'].append(node) 

# Generate the complete Snirf interface from base.py and the template + data
dst = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
output_path = dst + '/src/' + 'pysnirf2.py'
with open('base.py', 'r') as f_base:
    with open(output_path, "w") as f_out:
        SNIRF['BASE'] = f_base.read()
        f_out.write(template.render(SNIRF))
        
print('\nWrote script to ' + output_path + '.')
