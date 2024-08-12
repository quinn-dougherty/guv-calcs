from .room import Room
from .lamp import Lamp
from .calc_zone import CalcVol, CalcPlane
from .trigonometry import to_polar, to_cartesian, attitude
from ._calculate import get_disinfection_table
from ._version import __version__

__all__ = [
    "Room",
    "Lamp",
    "CalcVol",
    "CalcPlane",
    "to_polar",
    "to_cartesian",
    "attitude",
    "get_disinfection_table"
]

__version__ = __version__
