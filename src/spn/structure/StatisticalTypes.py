"""
Created on April 15, 2018

@author: Alejandro Molina
"""
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetaType(Enum):
    REAL = 1
    BINARY = 2
    DISCRETE = 3
    UTILITY = 4
    STATE = 5


class Type(Enum):
    REAL = (1, MetaType.REAL)
    INTERVAL = (2, MetaType.REAL)
    POSITIVE = (3, MetaType.REAL)
    CATEGORICAL = (4, MetaType.DISCRETE)
    ORDINAL = (5, MetaType.DISCRETE)
    COUNT = (6, MetaType.DISCRETE)
    BINARY = (7, MetaType.BINARY)
    UTILITY = (8, MetaType.UTILITY)
    STATE = (9, MetaType.STATE)

    def __init__(self, enum_val, meta_type):
        self._enum_val = enum_val
        self._meta_type = meta_type

    @property
    def meta_type(self):
        return self._meta_type


META_TYPE_MAP = {
    MetaType.REAL: [Type.REAL, Type.INTERVAL, Type.POSITIVE],
    MetaType.BINARY: [Type.BINARY],
    MetaType.DISCRETE: [Type.CATEGORICAL, Type.ORDINAL, Type.COUNT],
    MetaType.UTILITY: [Type.UTILITY],
    MetaType.STATE: [Type.STATE]
}
