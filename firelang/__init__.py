import os

debug_on = {"on": True, "off": False}.get(os.environ.get("FIRE_DEBUG", "off"), False)

from .measure import *
from .function import *
from .models import *
from .map import *