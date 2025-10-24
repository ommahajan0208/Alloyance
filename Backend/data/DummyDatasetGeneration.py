import numpy as np
import pandas as pd

N_SAMPLES = 25000
np.random.seed(42)

# Data mappings
data_mappings = {
    'Process Stage': ['Raw Material Extraction', 'Manufacturing', 'Use', 'Transport', 'End-of-Life'],
    'Technology': ['Conventional', 'Advanced', 'Emerging'],
    'Time Period': ['2010-2014', '2015-2019', '2020-2025'],
    'Location': ['North America', 'Europe', 'South America', 'Asia'],
    'Functional Unit': ['1 kg Aluminium Sheet', '1 kg Copper Wire', '1 m2 Aluminium Panel'],
    'Raw Material Type': ['Copper Ore', 'Copper Scrap', 'Aluminium Scrap', 'Aluminium Ore'],
    'Energy Input Type': ['Electricity', 'Coal', 'Natural Gas'],
    'Transport Mode': ['Rail', 'Ship', 'Truck'],
    'Fuel Type': ['Diesel', 'Electric', 'Heavy Fuel Oil'],
    'End-of-Life Treatment': ['Recycling', 'Landfill', 'Incineration', 'Reuse']
}

rows = []
for i in range(N_SAMPLES):
    stage = np.random.choice(data_mappings['Process Stage'])
    tech = np.random.choice(data_mappings['Technology'])
    loc = np.random.choice(data_mappings['Location'])
    raw_type = np.random.choice(data_mappings['Raw Material Type'])
    energy_type = np.random.choice(data_mappings['Energy Input Type'])
    transport_mode = np.random.choice(data_mappings['Transport Mode'])
    eol = np.random.choice(data_mappings['End-of-Life Treatment'])
    func_unit = np.random.choice(data_mappings['Functional Unit'])

    qty = 1000 if 'kg' in func_unit else 500
    is_primary = 'Ore' in raw_type
    recycled_fraction = 0.1 if is_primary else np.random.uniform(0.7, 0.99)

    energy_per_kg = np.random.normal(15 if is_primary else 1.5, 0.5)
    energy_input = energy_per_kg * qty

    if energy_type == "Coal":
        ghg_per_kg = np.random.uniform(10, 18) if is_primary else np.random.uniform(1, 3)
    elif energy_type == "Natural Gas":
        ghg_per_kg = np.random.uniform(6, 10)
    else:
        ghg_per_kg = np.random.uniform(1, 5)
    ghg_total = ghg_per_kg * qty

    material_cost = np.random.uniform(0.8, 2.0) * qty * (1.2 if is_primary else 1)
    processing_cost = material_cost * np.random.uniform(0.5, 1.0)

    air_emissions = ghg_total * np.random.uniform(0.03, 0.06)
    water_emissions = ghg_total * np.random.uniform(0.001, 0.003)

    transport_distance = np.random.uniform(50, 2000)
    transport_emission_factor = {'Truck': 0.1,'Rail': 0.03,'Ship': 0.015}[transport_mode]
    scope3_from_transport = transport_distance * transport_emission_factor * (qty / 1000)
    scope3_total = ghg_total * 0.2 + scope3_from_transport

    circularity_score = (recycled_fraction * 100) - (ghg_per_kg * 2) + np.random.uniform(-5, 5)
    circularity_score = np.clip(circularity_score, 0, 100)

    recycled_content = recycled_fraction * 100
    resource_eff = recycled_content + np.random.uniform(-5, 5)
    recovery_rate = np.clip(recycled_content + np.random.uniform(-10, 5), 0, 100)
    reuse_potential = np.clip(circularity_score + np.random.uniform(-10, 10), 0, 100)

    # ------------------ Strongly correlated Extended Product Life ------------------ #
    metal_quality = np.random.choice(['High','Medium','Low'])

    # Base life by functional unit
    if 'Aluminium Sheet' in func_unit:
        base_life = 10
    elif 'Copper Wire' in func_unit:
        base_life = 30
    elif 'Aluminium Panel' in func_unit:
        base_life = 40
    else:
        base_life = 15

    q_mult = {'High': 1.3, 'Medium': 1.0, 'Low': 0.75}[metal_quality]
    t_mult = {'Emerging': 1.25, 'Advanced': 1.15, 'Conventional': 1.0}[tech]
    circ_mult = 1 + (circularity_score / 200)  # more circular → longer life

    if eol == 'Reuse':
        eol_mult = 1.6
    elif eol == 'Recycling':
        eol_mult = 1.1
    elif eol == 'Landfill':
        eol_mult = 0.9
    elif eol == 'Incineration':
        eol_mult = 0.85
    else:
        eol_mult = 1.0

    extended_life = base_life * q_mult * t_mult * circ_mult * eol_mult
    extended_life *= np.random.normal(1.0, 0.05)  # small noise
    extended_life = float(np.clip(extended_life, 2, 120))

    # --- Adjust based on End-of-Life Treatment for circularity metrics --- #
    if eol == "Recycling":
        recovery_rate = np.random.uniform(80, 95)
        reuse_potential = np.random.uniform(20, 40)
        circularity_score = np.clip(circularity_score + np.random.uniform(15, 25), 0, 100)
    elif eol == "Reuse":
        reuse_potential = np.random.uniform(70, 95)
        recovery_rate = np.random.uniform(50, 70)
        circularity_score = np.clip(circularity_score + np.random.uniform(20, 30), 0, 100)
    elif eol == "Landfill":
        recovery_rate = np.random.uniform(0, 10)
        reuse_potential = np.random.uniform(0, 5)
        circularity_score = np.clip(circularity_score - np.random.uniform(20, 30), 0, 100)
    elif eol == "Incineration":
        recovery_rate = np.random.uniform(5, 15)
        reuse_potential = np.random.uniform(0, 2)
        circularity_score = np.clip(circularity_score - np.random.uniform(10, 20), 0, 100)

    rows.append({
        'Process Stage': stage,
        'Technology': tech,
        'Time Period': np.random.choice(data_mappings['Time Period']),
        'Location': loc,
        'Functional Unit': func_unit,
        'Raw Material Type': raw_type,
        'Raw Material Quantity (kg or unit)': qty,
        'Energy Input Type': energy_type,
        'Energy Input Quantity (MJ)': round(energy_input, 2),
        'Processing Method': tech,
        'Transport Mode': transport_mode,
        'Transport Distance (km)': round(transport_distance, 2),
        'Fuel Type': np.random.choice(data_mappings['Fuel Type']),
        'Metal Quality Grade': metal_quality,
        'Material Scarcity Level': np.random.choice(['Low','Medium','High']),
        'Material Cost (USD)': round(material_cost, 2),
        'Processing Cost (USD)': round(processing_cost, 2),
        'Emissions to Air CO2 (kg)': round(ghg_total * 0.6, 2),
        'Emissions to Air SOx (kg)': round(air_emissions * 0.1, 3),
        'Emissions to Air NOx (kg)': round(air_emissions * 0.08, 3),
        'Emissions to Air Particulate Matter (kg)': round(air_emissions * 0.05, 3),
        'Emissions to Water Acid Mine Drainage (kg)': round(water_emissions * 0.5, 4),
        'Emissions to Water Heavy Metals (kg)': round(water_emissions * 0.3, 4),
        'Emissions to Water BOD (kg)': round(water_emissions * 0.2, 4),
        'Greenhouse Gas Emissions (kg CO2-eq)': round(ghg_total, 2),
        'Scope 1 Emissions (kg CO2-eq)': round(ghg_total * 0.5, 2),
        'Scope 2 Emissions (kg CO2-eq)': round(ghg_total * 0.3, 2),
        'Scope 3 Emissions (kg CO2-eq)': round(scope3_total, 2),
        'End-of-Life Treatment': eol,
        'Environmental Impact Score': round(100 - circularity_score, 2),
        'Metal Recyclability Factor': round(recycled_fraction, 2),
        'Energy_per_Material': round(energy_per_kg, 2),
        'Total_Air_Emissions': round(air_emissions, 2),
        'Total_Water_Emissions': round(water_emissions, 3),
        'Transport_Intensity': round(transport_distance * transport_emission_factor, 3),
        'GHG_per_Material': round(ghg_per_kg, 2),
        'Time_Period_Numeric': np.random.choice([2012, 2017, 2023]),
        'Total_Cost': round(material_cost + processing_cost, 2),
        'Circularity_Score': round(circularity_score, 2),
        'Circular_Economy_Index': round(circularity_score/100, 2),
        'Recycled Content (%)': round(recycled_content, 2),
        'Resource Efficiency (%)': round(resource_eff, 2),
        'Extended Product Life (years)': round(extended_life, 1),
        'Recovery Rate (%)': round(recovery_rate, 2),
        'Reuse Potential (%)': round(reuse_potential, 2)
    })

df = pd.DataFrame(rows)
df.to_csv('lca_dataset.csv', index=False)
print(f"Synthetic dataset with {len(df)} rows saved as 'lca_dataset.csv'")