from rope.refactor import rename
import nbformat
from nbconvert import PythonExporter

# Load the notebook file as a notebook object
file_path = '2022_TP_Unsupervised_1_PCA-ICA.ipynb'
project_path = './'
with open(file_path, 'r') as file:
    nb = nbformat.read(file, as_version=nbformat.NO_CONVERT)

# Export the notebook as a Python script
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

# Extract variable names from the Python script
variables = []
for line in source.split('\n'):
    if '=' in line:
        variable = line.split('=')[0].strip()
        if variable not in ('', '#'):
            if  '#' not in variable:
                if '{' not in variable: 
                    if ' ' not in variable:
                        if '(' not in variable:
                            if '[' not in variable:
                                variables.append(variable)

# Print the list of variable names
print(variables)
'''
all_variables = dir()
  
# Iterate over the whole list where dir( )
# is stored.
for name in all_variables:
  # Print the item if it doesn't start with '__'
    if not name.startswith('_'):
        myvalue = eval(name)
        print(name, type(myvalue))
project_path = 'path/to/project/'
old_name = 'x'
new_name = 'y'

rename(project_path, old_name, new_name)
'''
import random
import string

def generate_random_string(n):
    """Generates a random string of length n."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(n))



for variable in variables:
    # Read the contents of the file
    with open(file_path, 'r') as f:
        file_contents = f.read()
    # Replace the old word with the new word
    new_variable = generate_random_string(len(variable))
    updated_contents = file_contents.replace(variable, new_variable)

    # Write the updated contents back to the file
    with open(file_path, 'w') as f:
        f.write(updated_contents)



