import pandas as pd 

dataframe = pd.read_excel("../Data/BLDG's 28-54-36.xlsx", sheet_name="15 minute Data", header=[0,1])

# Create smaller data frames from this that are each specific building
#building_28_kwh = dataframe[('28', 'kWh')]
#building_28_dates = dataframe[('28', 'Date')]

print("Column structure:")
print(dataframe.columns.tolist())
print("\nFirst few rows:")
print(dataframe.head())