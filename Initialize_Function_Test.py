import numpy as np
import csv

with open('arbirary_grid') as file:
    reader = csv.reader(file)
    for row in reader:
        x_c = int(row[0])
        y_c = int(row[1])



