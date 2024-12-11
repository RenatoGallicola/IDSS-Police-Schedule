from enum import Enum
import csv

class Location(Enum):
    MINLAT = 0.0
    MAXLAT = 70.0
    MINLONG = 0.0
    MAXLONG = 50.0
    CELLSIZE = 5.0
    PRECISION = 1

filename = 'location_data.csv'
# filename = 'ResourceAllocation/location_data.csv'

mult = 10 ** Location.PRECISION.value

data = []
for lat in range(int(Location.MINLAT.value * mult), int(Location.MAXLAT.value * mult) + int(Location.CELLSIZE.value * mult), int(Location.CELLSIZE.value * mult)):
    for long in range(int(Location.MINLONG.value * mult), int(Location.MAXLONG.value * mult) + int(Location.CELLSIZE.value * mult), int(Location.CELLSIZE.value * mult)):
        data.append((lat / float(mult), long / float(mult)))

# Header: Location Latitude, Location Longitude
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)