# The purpose of this script is to find the datetime interval (Summer is suspected to be the cause).
import pandas as pd


# Pull data
df = pd.read_csv('../Data/09-06-2024 -- 09-06-2025.csv')

Time = df['Time']

# Divide by 1000 to get MW Multiply by 1000 to get Watts
averageEnergy = df['Average']/1000 # This is the average energy usage in 12 hour intervals

maximumEnergy = df['Maximum']/1000 # This is the maximum energy usage in 12 hour intervals

minimumEnergy = df['Minimum']/1000 # This is the minimum energy usage in 12 hour intervals

# Find where the energy drops significantly
# Calculate the difference between consecutive points
energy_diff = averageEnergy.diff()

# Find the largest negative drop (most significant decrease)
max_drop_idx = energy_diff.idxmin()
max_drop_value = energy_diff.min()


# Set thresholds
min_drop_magnitude = 15.0  # Must drop by at least 15 MW
min_recovery_magnitude = 15.0  # Must recover by at least 15 MW
min_duration = 7  # Must last at least 7 data points (3.5 days)

# Find potential drop-off points
energy_diff = averageEnergy.diff()
potential_drops = energy_diff[energy_diff < -min_drop_magnitude]

print("Analyzing potential drop-offs...")
print(f"Found {len(potential_drops)} potential drops of {min_drop_magnitude}+ MW")

for drop_idx in potential_drops.index:
    start_energy = averageEnergy.iloc[drop_idx-1]
    drop_energy = averageEnergy.iloc[drop_idx]
    actual_drop = start_energy - drop_energy
    
    print(f"\nChecking drop at index {drop_idx}:")
    print(f"  Drop magnitude: {actual_drop:.2f} MW")
    print(f"  Time before: {Time.iloc[drop_idx-1]}")
    print(f"  Time after: {Time.iloc[drop_idx]}")
    
    # Find where it recovers by at least 15 MW from the lowest point
    lowest_energy = averageEnergy.iloc[drop_idx:].min()
    recovery_threshold = lowest_energy + min_recovery_magnitude
    
    # Look for recovery points
    recovery_candidates = averageEnergy.iloc[drop_idx:].where(
        averageEnergy.iloc[drop_idx:] >= recovery_threshold
    )
    
    if not recovery_candidates.dropna().empty:
        recovery_idx = recovery_candidates.dropna().index[0]
        duration = recovery_idx - drop_idx
        recovery_energy = averageEnergy.iloc[recovery_idx]
        
        print(f"  Recovery found at index {recovery_idx}")
        print(f"  Duration: {duration} data points ({duration/2} days)")
        print(f"  Time at recovery: {Time.iloc[recovery_idx]}")
        print(f"  Energy at recovery: {recovery_energy:.2f} MW")
        print(f"  Recovery magnitude: {recovery_energy - lowest_energy:.2f} MW")
        
        if duration >= min_duration:
            print(f"  ✓ SUSTAINED DROP-OFF FOUND!")
            print(f"  Start time: {Time.iloc[drop_idx-1]}")
            print(f"  End time: {Time.iloc[recovery_idx]}")
            print(f"  Total duration: {duration/2} days")
            print(f"  Peak energy before: {start_energy:.2f} MW")
            print(f"  Lowest energy: {lowest_energy:.2f} MW")
            print(f"  Peak energy after: {recovery_energy:.2f} MW")
            break
    else:
        print(f"  No significant recovery found")