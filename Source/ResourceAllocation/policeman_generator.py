import csv
import random

import matplotlib.pyplot as plt
from faker import Faker
from location_generator import Location, get_map
from policeman_turn import Turn


class PolicemanGenerator:
    def __init__(
            self,
            file_name='Source/ResourceAllocation/policeman_data.csv',
            tot_pol=4000):
        self.__file_name = file_name
        self.__tot_pol = tot_pol
        self.__fake = Faker()

    def __generate_coordinates(self):
        found = False
        while not found:
            lat = round(random.uniform(Location.MINLAT.value, Location.MAXLAT.value), Location.PRECISION.value)
            long = round(random.uniform(Location.MINLONG.value, Location.MAXLONG.value), Location.PRECISION.value)
            y_cells = int((Location.MAXLAT.value - Location.MINLAT.value) / Location.LAT_CELLSIZE.value) + 1
            x_cells = int((Location.MAXLONG.value - Location.MINLONG.value) / Location.LON_CELLSIZE.value) + 1
            lat_idx = lat * y_cells / (Location.MAXLAT.value - Location.MINLAT.value)
            long_idx = long * x_cells / (Location.MAXLONG.value - Location.MINLONG.value)
            if get_map()[int(lat_idx)][int(long_idx)] == 1:
                found = True
                return long, lat


    def __generate_badge_numbers(self, total):
        return random.sample(range(10000, 99999), total)

    def show_policemen(self):
        latitudes = []
        longitudes = []
        
        with open(self.__file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                longitudes.append(float(row[1]))
                latitudes.append(float(row[2]))
        
        plt.scatter(longitudes, latitudes, c='blue', marker='o')
        plt.title('Policemen Locations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.show()

    def generate_policemen(self):
        badge_numbers = self.__generate_badge_numbers(self.__tot_pol)

        # Header: 'Full Name', 'Home Address Latitude', 'Home Address Longitude', 'Turn', 'Badge Number'
        with open(self.__file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            for i in range(self.__tot_pol):
                full_name = self.__fake.name()
                longitude, latitude = self.__generate_coordinates()
                turn = Turn(i % Turn.NUM_DAILY_TURNS.value).value
                badge = badge_numbers[i]
                writer.writerow([full_name, longitude, latitude, turn, badge])

    def update_shift_numbers(self):
            updated_pol_data = []
            with open(self.__file_name, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    curr = int(row[3])
                    if curr == 0:
                        curr = 3
                    else:
                        curr -= 1
                    row[3] = str(curr)
                    updated_pol_data.append(row)

            with open(self.__file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(updated_pol_data)
    
# pol_generator = PolicemanGenerator()
# pol_generator.generate_policemen()
# pol_generator.show_policemen()