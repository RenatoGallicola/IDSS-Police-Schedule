import csv
import os
import sys

import numpy as np
from location_generator import Location, get_map
from resource_allocation import allocate_policemen as ap

module_dir = os.path.dirname(__file__)
module_path = os.path.join(module_dir, '../Models/scripts')
sys.path.append(module_path)
import matplotlib.pyplot as plt
import pandas as pd
from policeman_generator import PolicemanGenerator
from policeman_turn import Turn
from PredictionClass import PredictionClass
from utils import get_area_number
from utils import lat as lat_array
from utils import lon as lon_array


class UIAllocation:
    def __init__(
            self,
            tolerance = 15,
            num_policemen = 1000,
            day = 1,
            month = 1,
            year = 2024,
            avg_vel = 40
        ):
        self.__tolerance = tolerance
        self.__num_policemen = num_policemen
        self.__location_data = self.__get_location_data()
        self.__policeman_data = None
        self.__day = day
        self.__month = month
        self.__year = year
        self.__turn = 1
        self.__p_class = PredictionClass()
        self.__csv_name = 'ResourceAllocation/ui_allocation.csv'
        self.__avg_vel = avg_vel
    
    def __get_current_location_data(self):
        return self.__location_data

    def __get_location_data(self):
        loc_data = []
        with open('ResourceAllocation/location_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                loc_data.append((float(row[0]), float(row[1])))
        return loc_data

    def __set_policeman_data(self):
        pol_data = []
        with open('ResourceAllocation/policeman_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if float(row[3]) == self.__turn % Turn.NUM_DAILY_TURNS.value:
                    pol_data.append((str(row[0]), float(row[1]), float(row[2]), int(row[3]), int(row[4])))
        self.__policeman_data = pol_data
        self.__num_policemen = len(pol_data)

    def __compute_location_distance(self):
        loc_data = self.__location_data
        pol_data = self.__policeman_data
        loc_dist = [[round(((loc_data[j][0] - pol_data[i][1]) ** 2 + (loc_data[j][1] - pol_data[i][2]) ** 2) ** 0.5, Location.PRECISION.value) \
             for j in range(len(loc_data))] for i in range(len(pol_data))]
        return loc_dist
    
    def __location_requirements(self):
        model_result = self.__p_class.predict(self.__turn, self.__day, self.__month, self.__year)
        model_result = np.flip(model_result, axis=0)

        total_probability = np.sum(model_result)
        normalized_probabilities = model_result / total_probability

        initial_allocation = np.ones_like(model_result, dtype=int)
        map_la = get_map()
        initial_allocation[map_la == 0] = 0

        remaining_policemen = self.__num_policemen - np.sum(initial_allocation)

        raw_allocation = normalized_probabilities * remaining_policemen

        additional_allocation = np.floor(raw_allocation).astype(int)

        loc_req = initial_allocation + additional_allocation

        remaining_policemen = self.__num_policemen - np.sum(loc_req)
        loc_req = loc_req.flatten().tolist()
        
        # self.__show_location_requirements(loc_req)

        return loc_req, remaining_policemen
    
    def __show_location_requirements(self, loc_req):
        matrix = np.zeros((31, 31), dtype=int)
        for idx, val in enumerate(loc_req):
            row = idx // 31
            col = idx % 31
            matrix[row, col] = val

        plt.imshow(matrix, interpolation='nearest')
        plt.colorbar()
        plt.title('Location Requirements Heatmap')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        for i in range(31):
            for j in range(31):
                plt.text(j, i, matrix[i, j], ha='center', va='center', color='black', fontsize=8)
        plt.show()

    def __compute_margin(self, remain):
        map_la = get_map()
        valid_loc = np.sum(map_la)
        margin = remain // valid_loc + 1
        return margin

    def show_allocation_map(self, dest):
        tot = []
        for i in range(len(self.__get_current_location_data())):
            tot_police = 0
            for j in range(len(dest)):
                tot_police += 1 if dest[j] == i else 0
            tot.append(tot_police)

        matrix = np.zeros((31, 31), dtype=int)
        for idx, val in enumerate(tot):
            row = idx // 31
            col = idx % 31
            matrix[row, col] = val

        plt.imshow(matrix, interpolation='nearest')
        plt.colorbar()
        plt.title('Police Allocation Heatmap')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        for i in range(31):
            for j in range(31):
                plt.text(j, i, matrix[i, j], ha='center', va='center', color='black', fontsize=8)
        plt.show()

    def __allocate(self):
        loc_dist = self.__compute_location_distance()
        loc_req, remain = self.__location_requirements()
        tol = self.__tolerance
        margin = self.__compute_margin(remain)
        dest, dist = ap(loc_dist, loc_req, tol, margin, self.__avg_vel)
        return dest, dist

    def __empty_csv(self):
        if os.path.exists(self.__csv_name):
            os.remove(self.__csv_name)

    def __update_ui_csv(self, dest, dist): 
        columns = ['badge', 'name', 'shift', 'day', 'month', 'year', 'group', 'lat', 'lon', 'area', 'time_to_travel', 'distance']
        data = []
        
        for i, policeman in enumerate(self.__policeman_data):
            name, _, _, group, badge = policeman
            shift = self.__turn
            day = self.__day
            month = self.__month
            year = self.__year 
            distance = dist[i]
            time_to_travel = round(distance / self.__avg_vel * 60.0, 1)
            
            cell_idx = dest[i]
            lat_idx = cell_idx // 31
            lon_idx = cell_idx % 31
            lat = lat_array[lat_idx]
            lon = lon_array[lon_idx]
            area = get_area_number(lat_idx, lon_idx)

            data.append([badge, name, shift, day, month, year, group, lat, lon, area, time_to_travel, distance])
              
        df = pd.DataFrame(data, columns=columns)
        if os.path.exists(self.__csv_name):
            df_existing = pd.read_csv(self.__csv_name)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(self.__csv_name, index=False)

    def week_allocation(self):
        self.__empty_csv()
        for i in range(Turn.NUM_WEEKLY_TURNS.value):
            self.__turn = i + 1
            self.__set_policeman_data()
            dest, dist = self.__allocate()
            self.__update_ui_csv(dest, dist)
            #ui.show_allocation_map(dest)
        pg = PolicemanGenerator()
        pg.update_shift_numbers()


if __name__ == "__main__":
    ui = UIAllocation()
    dest = ui.week_allocation()
    ui.show_allocation_map(dest)