from .chair import ChairGenerator
from .table import TableGenerator
from .mug import MugGenerator
from .bottle import BottleGenerator
from .airplane import AirplaneGenerator
from .car import CarGenerator
from .lamp import LampGenerator
from .sofa import SofaGenerator
from .monitor import MonitorGenerator
from .bookshelf import BookshelfGenerator

ALL_GENERATORS = {
    "chair": ChairGenerator,
    "table": TableGenerator,
    "mug": MugGenerator,
    "bottle": BottleGenerator,
    "airplane": AirplaneGenerator,
    "car": CarGenerator,
    "lamp": LampGenerator,
    "sofa": SofaGenerator,
    "monitor": MonitorGenerator,
    "bookshelf": BookshelfGenerator,
}
