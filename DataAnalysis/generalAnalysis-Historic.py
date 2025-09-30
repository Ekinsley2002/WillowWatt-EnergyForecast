import pandas as pd 

building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")

building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")

building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")

print(building_28["Date"])

print(building_28["kWh"])