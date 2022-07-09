from jinja2 import Environment, FileSystemLoader, select_autoescape
from yapf.yapflib.yapf_api import FormatFile 
import requests
from unidecode import unidecode
from pathlib import Path
from datetime import date, datetime
import getpass
import os
import sys
import warnings
from pylint import lint

"""
Generates SNIRF interface and validator from the summary table of the specification
hosted at SPEC_SRC.
"""

LIB_VERSION = '0.5.2'  # Version for this script

if __name__ == '__main__':
    
    cwd = os.path.abspath(os.getcwd())
    if not cwd.endswith('pysnirf2'):
        sys.exit('The gen script must be run from the pysnirf2 project root, not ' + cwd)    
    
    output_path = cwd + '/pysnirf2/' + 'pysnirf2.py'
    
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass
    open(output_path, 'w')
    
    try:
        from data import *
    except ModuleNotFoundError:
        from gen.data import *
    
    sys.path.append(cwd)
    
    print('-------------------------------------')
    print('pysnirf2 generation script v' + LIB_VERSION)
    print('-------------------------------------')
    
    local_spec = SPEC_SRC.split('/')[-1].split('.')[0] + '_retrieved_' + datetime.now().strftime('%d_%m_%y') + '.txt'
    
    if os.path.exists(local_spec):
        print('Loading specification from local document', local_spec, '...')
        with open(local_spec, 'r') as f:
            text = f.read()
    else:
        print('Attempting to retrieve spec from', SPEC_SRC, '...')
    
        # Retrieve the SNIRF specification from GitHub and parse it for the summary table
        
        try:
            spec = requests.get(SPEC_SRC)
        except (ConnectionError, requests.exceptions.ConnectionError):
            sys.exit('No internet connection and no downloaded specification document. pysnirf2 generation aborted.')
        
        if spec.text == '404: Not Found':
            print(spec.text)
            sys.exit('The Snirf specification could not be found at ' + SPEC_SRC)
        
        text = unidecode(spec.text)
        
        for file in os.listdir():
            if file.startswith(SPEC_SRC.split('/')[-1].split('.')[0] + '_retrieved_') and file.endswith('.txt'):
                os.remove(file)
        
        with open(local_spec, 'w') as f:
            f.write(text)
            
        
    table = text.split(TABLE_DELIM_START)[1].split(TABLE_DELIM_END)[0]
    
    rows = table.split('\n')
    while '' in rows:
        rows.remove('')
    
    type_codes = []
    
    # Parse each row of the summary table for programmatic info
    for row in rows[1:]:  # Skip header row
        delim = row.split('|')
        
        if len(delim) > 1:
        
            # Number of leading spaces determines indent level, hierarchy
            namews = delim[1].replace('`', '').replace('.','').replace('-', '')
            name = namews.replace(' ', '').replace('/', '').replace('{i}', '')
            
            # Get name: format pairs for each name
            if len(name) > 1:  # Skip the empty row
                type_code = delim[-2].replace(' ', '').replace('`', '')
                type_codes.append(type_code)
        
    print('Found', len(type_codes), 'types in the table...')
    
    # Parse headings in spec for complete HDF style locations with which to build tree
    definitions = unidecode(text).split(DEFINITIONS_DELIM_START)[1].split(DEFINITIONS_DELIM_END)[0]
    definitions_lines = definitions.split('\n')
        
    # Create list of hdf5 names ("locations") by assuming each follows a '####' header character in the definitions section
    locations = []
    descriptions = []
    for i, line in enumerate(definitions_lines):
        line = line.replace(' ', '')
        if line.startswith('####'):
            locations.append(line.replace('`', '').replace('#', '').replace(',', ''))
            description_lines = []
            j = i + 4
            while not definitions_lines[j].startswith('####'):
                description_lines.append(definitions_lines[j])
                j += 1
                if j >= len(definitions_lines):
                    break
            descriptions.append('\n'.join(description_lines))
                
    # Format descriptions
    descriptions = [description.replace('\\', '/').replace('\t', ' ').lstrip() for description in descriptions]
            
    # Create flat list of all nodes
    flat = []

    print('Found', len(descriptions), 'descriptions...')
    print('Found', len(locations), 'locations...')
    
    # Write locations to file
    if os.path.exists('locations.txt'):
        os.remove('locations.txt')
    with open('locations.txt', 'w') as f:
        for location in locations:
            f.write(location.replace('(i)', '').replace('(j)', '').replace('(k)', '') + '\n')
    print('Wrote to locations.txt')
    
    if len(locations) != len(type_codes) or len(locations) != len(descriptions):
        sys.exit('Parsed ' + str(len(type_codes)) + ' type codes from the summary table but '
                 + str(len(locations)) + ' names from the definitions and ' + str(len(descriptions))
                 + ' descriptions: the specification hosted at ' + SPEC_SRC +' was parsed incorrectly. Try adjusting the delimiters and then debug the parsing code (gen.py).')
    
    # Append root (flat will have 1 extra element compared to locations)
    flat.append({
                'name': '',
                'location': '/',
                'parent': None,
                'type': '/',
                'children': [],
                'required': False
                })
    
    for i, (location, description) in enumerate(zip(locations, descriptions)):
            type_code = type_codes[i]
            name = location.split('/')[-1].split('(')[0]  # Remove (i), (j)
            parent = location.split('/')[-2].split('(')[0]  # Remove (i), (j)
            print('Found', location, 'with type', type_code)
            if type_code is not None:
                required = TYPELUT['REQUIRED'] in type_code
            else:
                required = None
            if not any([location == node['location'] for node in flat]):   
                flat.append({
                            'name': name,
                            'location': location,
                            'description': description,
                            'parent': parent,
                            'type': type_code,
                            'children': [],
                            'required': required
                            })
    
    ans = input('Proceed? y/n\n')
    if ans not in ['y', 'Y']:
        sys.exit('pysnirf2 generation aborted.')
    
    print('Loading BIDS-specified Probe names from gen/data.py...')
    for name in BIDS_PROBE_NAMES:
        print('Found', name)
    
    ans = input('Proceed? y/n\n')
    if ans not in ['y', 'Y']:
        sys.exit('pysnirf2 generation aborted.')
    
    #  Generate data for template
    SNIRF = {
            'VERSION': SPEC_VERSION,
            'SPEC_SRC': SPEC_SRC,
            'HEADER': '',
            'FOOTER': '',
            'ROOT': flat[0],  # Snirf root '/'
            'TYPES': TYPELUT, 
            'USER': getpass.getuser(),
            'DATE': str(date.today()),
            'INDEXED_GROUPS': [], 
            'GROUPS': [], 
            'UNSPECIFIED_DATASETS_OK': UNSPECIFIED_DATASETS_OK,
            'BIDS_COORDINATE_SYSTEM_NAMES': BIDS_PROBE_NAMES
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
    
    # %%
    
    env = Environment(
        loader=FileSystemLoader(searchpath='./'),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,    
        )
    print('\nCreated jinja2 environment.')
    
    print('Loading template from', TEMPLATE)
    template = env.get_template(TEMPLATE)
    
    # Generate the complete Snirf interface from base.py and the template + data
    with open(HEADER, 'r') as f_header:
        with open(FOOTER, 'r') as f_footer:
            print('Loading base class definitions and file header from', HEADER)
            print('Loading file footer from', FOOTER)
            with open(output_path, "w") as f_out:
                SNIRF['HEADER'] = f_header.read()
                SNIRF['FOOTER'] = f_footer.read()
                f_out.write(template.render(SNIRF))
            
    print('\nWrote script to ' + output_path + '.')
    
    with open(output_path) as generated:
        lines = generated.read().split('\n')
        print('pysnirf2.py is', len(lines), 'lines long')
        errors = 0
        for i, line in enumerate(lines):
            if 'TEMPLATE_ERROR' in line:
                errors += 1
                print('ERROR on line', i, '\n', line)
        if errors == 0:
            print('pysnirf2.py generated with', errors, 'errors.')
    
    FormatFile(output_path, in_place=True)[:2]
    lint.Run(['--errors-only', output_path])
    
    