from .room import Room
from .lamp import Lamp
from .calc_zone import CalcVol, CalcPlane
from .spectrum import Spectrum, sum_spectrum
from .trigonometry import to_polar, to_cartesian, attitude
from ._calculate import get_disinfection_table, get_tlv, get_spectral_weightings
from ._version import __version__

__all__ = [
    "Room",
    "Lamp",
    "CalcVol",
    "CalcPlane",
    "Spectrum",
    "sum_spectrum",
    "to_polar",
    "to_cartesian",
    "attitude",
    "get_disinfection_table",
    "get_tlv",
    "get_spectral_weightings",
]

__version__ = __version__
