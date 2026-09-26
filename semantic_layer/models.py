from dataclasses import dataclass
from typing import List

@dataclass
class ColumnDef:
    """Describes a single column so the LLM knows what it means and how to use it in SQL."""
    name:str
    data_type:str
    description:str

@dataclass
class TableDef:
    """Describes one physical table: where it lives, what it represents, and its columns."""
    full_name:str
    description:str
    columns:List[ColumnDef]

@dataclass
class SpaceConfig:
    """Describes one config-driven 'space' — a domain-scoped view of one or more related tables."""
    name:str
    description:str
    tables:List[TableDef]
    keywords:List[str]

