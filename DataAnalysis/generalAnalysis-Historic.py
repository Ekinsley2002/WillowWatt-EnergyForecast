import pandas as pd 
import matplotlib.pyplot as plt

building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")

building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")

building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")


# Modular cleanup time to match prototype
# prototype format - MM/DD/YYYY
# Historic data format - MM/DD/YY AM/PM
building_28['Time'] = pd.to_datetime(building_28['Date'])

building_54['Time'] = pd.to_datetime(building_54['Date'])

building_36['Time'] = pd.to_datetime(building_36['Date'])

# Building 54 has a bunch of blank values..drop them
building_54 = building_54.dropna(subset=['Time', 'kWh']).reset_index(drop=True)

buildings = [building_28, building_54, building_36]

building_names = ['BLDG_28', 'BLDG_54', 'BLDG_36']

# plot this all this data for all three buildings, then save the figures.
for building, name in zip(buildings, building_names):
    plt.figure(figsize=(12, 6))
    plt.plot(building['Time'], building['kWh'])  # Specify which columns to plot
    plt.title(f'Consumption of Energy Over Time: {name}')
    plt.xlabel('Date')
    plt.ylabel('kWh energy consumption')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../DataAnalysis/Figures/{name}_TxkWh.png')
    plt.close()

# Debugging output for buildings 54 and 36 
'''
print(f"kWh range: {building_54['kWh'].min()} to {building_54['kWh'].max()}")
print(f"Most common kWh values:")
print(building_54['kWh'].value_counts().head(10))
print(f"Time range: {building_54['Time'].min()} to {building_54['Time'].max()}") 
'''

# Find row with maximum kWh value
# Find the row with the highest kWh value
max_kwh_row = building_54[building_54['kWh'] == building_54['kWh'].max()]
print("Date of maximum spike:")
print(max_kwh_row[['Time', 'kWh']])