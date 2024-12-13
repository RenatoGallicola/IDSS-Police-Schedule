from enum import Enum
import csv
import numpy as np
import os
import sys
module_dir = os.path.dirname(__file__)
module_path = os.path.join(module_dir, '../Models/scripts')
sys.path.append(module_path)
from utils import mapLosAngeles as map

class Location(Enum):
    MINLAT = 0.0
    MAXLAT = 70.0
    MINLONG = 0.0
    MAXLONG = 50.0
    LAT_CELLSIZE = 2.33333
    LON_CELLSIZE = 1.66666
    PRECISION = 5

def get_map():
    return np.flipud(map)

# filename = 'location_data.csv'
filename = 'ResourceAllocation/location_data.csv'

mult = 10 ** Location.PRECISION.value

data = []
for lat in range(int(Location.MINLAT.value * mult), int(Location.MAXLAT.value * mult), int(Location.LAT_CELLSIZE.value * mult)):
    for long in range(int(Location.MINLONG.value * mult), int(Location.MAXLONG.value * mult), int(Location.LON_CELLSIZE.value * mult)):
        data.append((lat / float(mult), long / float(mult)))

# Header: Location Latitude, Location Longitude
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)