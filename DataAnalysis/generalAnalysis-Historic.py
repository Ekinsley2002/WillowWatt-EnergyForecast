import pandas as pd 
import matplotlib as plt

building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")

building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")

building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")

# Modular cleanup time to match prototype
# prototype format - MM/DD/YYYY
# Historic data format - MM/DD/YY AM/PM
building_28['Time'] = pd.to_datetime(building_28['Date'])

building_54['Time'] = pd.to_datetime(building_54['Date'])

building_36['Time'] = pd.to_datetime(building_36['Date'])
