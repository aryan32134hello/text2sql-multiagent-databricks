import yaml
from models import SpaceConfig, ColumnDef, TableDef
import os

_spaces = {}

_SPACES_DIR = '/Workspace/Users/aryan32134@gmail.com/text2sql_prototype/config/spaces'


def _load_space_from_yaml(filepath):
    with open(filepath) as f:
        space = yaml.safe_load(f)
    
    tables = []
    for table in space['tables']:
        columns = [ColumnDef(**col) for col in table['columns']]
        table_def = TableDef(full_name=table['full_name'], description=table['description'], columns=columns)
        tables.append(table_def)

    space_config = SpaceConfig(name = space['name'], description = space['description'], tables = tables, keywords = space['keywords'])
    _spaces[space_config.name] = space_config


for filename in os.listdir(_SPACES_DIR):
    #print(filename)
    if filename.endswith(".yaml"):
        _load_space_from_yaml(os.path.join(_SPACES_DIR, filename))


def get_space(name):
    if name not in _spaces.keys():
        raise KeyError(f"Space {name} not found")
    return _spaces[name]


def list_spaces():
    return list(_spaces.values())

#print(_spaces)