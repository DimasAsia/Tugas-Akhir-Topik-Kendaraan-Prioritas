import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Fungsi fuzzy_traffic_control harus sudah didefinisikan di sini
def fuzzy_traffic_control(density, vehicle_counter):
    # Fuzzy logic controller for determining the green light duration
    # based on the density and vehicle count.
    density_level = ctrl.Antecedent(np.arange(0, 3, 1), 'density')  # Mengubah range menjadi 0-3
    vehicle_count_level = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_count')
    green_light_duration = ctrl.Consequent(np.arange(0, 61, 1), 'green_light_duration')

    # Define membership functions for density
    density_level['Empty'] = fuzz.trimf(density_level.universe, [0, 0, 0.5])  # Empty (0)
    density_level['Low'] = fuzz.trimf(density_level.universe, [0, 0.5, 1.5])  # Low (1)
    density_level['Medium'] = fuzz.trimf(density_level.universe, [0.5, 1.5, 2.5])  # Medium (2)
    density_level['High'] = fuzz.trimf(density_level.universe, [1.5, 2.5, 3])  # High (3)


    # Define membership functions for vehicle count
    vehicle_count_level['none'] = fuzz.trimf(vehicle_count_level.universe, [0, 0, 1])
    vehicle_count_level['few'] = fuzz.trimf(vehicle_count_level.universe, [1, 5, 5])
    vehicle_count_level['moderate'] = fuzz.trimf(vehicle_count_level.universe, [6, 10, 15])
    vehicle_count_level['many'] = fuzz.trimf(vehicle_count_level.universe, [15, 21, 100])

    # Define membership functions for green light duration
    green_light_duration['short'] = fuzz.trimf(green_light_duration.universe, [1, 1, 5])
    green_light_duration['medium'] = fuzz.trimf(green_light_duration.universe, [6, 8, 10])
    green_light_duration['long'] = fuzz.trimf(green_light_duration.universe, [11, 20, 20])
    green_light_duration['very_short'] = fuzz.trimf(green_light_duration.universe, [0, 0, 1]) 
    
    # Define fuzzy rules
    
    rule1 = ctrl.Rule(density_level['Empty'] & vehicle_count_level['none'], green_light_duration['very_short'])
    rule2 = ctrl.Rule(density_level['Empty'] & vehicle_count_level['few'], green_light_duration['short'])
    rule3 = ctrl.Rule(density_level['Empty'] & vehicle_count_level['moderate'], green_light_duration['short'])
    rule4 = ctrl.Rule(density_level['Low'] & vehicle_count_level['few'], green_light_duration['short'])
    rule5 = ctrl.Rule(density_level['Low'] & vehicle_count_level['moderate'], green_light_duration['medium'])
    rule6 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['few'], green_light_duration['medium'])
    rule7 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['moderate'], green_light_duration['medium'])
    rule8 = ctrl.Rule(density_level['Medium'] & vehicle_count_level['many'], green_light_duration['long'])
    rule9 = ctrl.Rule(density_level['High'] & vehicle_count_level['few'], green_light_duration['short'])
    rule10 = ctrl.Rule(density_level['High'] & vehicle_count_level['moderate'], green_light_duration['long'])
    rule11 = ctrl.Rule(density_level['High'] & vehicle_count_level['many'], green_light_duration['long'])
    
    
    # Create and simulate the fuzzy control system
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
    traffic_controller = ctrl.ControlSystemSimulation(control_system)

    traffic_controller.input['density'] = density
    traffic_controller.input['vehicle_count'] = vehicle_counter

   # Crunch the numbers
    try:
        traffic_controller.compute()
        green_light_duration_output = round(traffic_controller.output['green_light_duration'])
    except Exception as e:
        green_light_duration_output = 0

    return green_light_duration_output

# Contoh pengujian sederhana untuk fuzzy_traffic_control
def main():
    test_cases = [
        (0, 0), 
        (0, 2), #rule 1
        (0, 6), #rule 2
        (1, 4), #rule 3
        (1, 10), #rule4
        (2, 5), #rule5
        (2, 12), #rule6
        (2, 16), #rule7
        (3, 4), #rule8
        (3, 11), #rule9
        (3, 17) #rule10
        
    ]

    for i, (density, vehicle_count) in enumerate(test_cases):
        result = fuzzy_traffic_control(density, vehicle_count)
        print(f"rule {i+1}: Density={density}, Vehicle Count={vehicle_count}, Green Light Duration={result}")

if __name__ == "__main__":
    main()
