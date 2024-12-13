import csv
import random
from faker import Faker
from location_generator import Location
from policeman_turn import Turn

class PolicemanGenerator:
    def __init__(
            self,
            file_name='ResourceAllocation/policeman_data.csv',
            tot_pol=4000):
        self.__file_name = file_name
        self.__tot_pol = tot_pol
        self.__fake = Faker()

    def __generate_coordinates(self):
        return round(random.uniform(Location.MINLAT.value, Location.MAXLAT.value), Location.PRECISION.value), \
        round(random.uniform(Location.MINLONG.value, Location.MAXLONG.value), Location.PRECISION.value)

    def __generate_badge_numbers(self, total):
        return random.sample(range(10000, 99999), total)

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
    
pol_generator = PolicemanGenerator()
pol_generator.generate_policemen()