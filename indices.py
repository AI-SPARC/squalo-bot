import numpy as np
import pandas as pd

def create_time_vector(vector):
    time_vector = []
    value = 0
    for val in vector:
        time_vector.append(value)
        value += 0.05
    return time_vector

scenario01 = pd.read_csv('images/scenario01/error_data.csv')
scenario02 = pd.read_csv('images/scenario03/error_data.csv')
scenario03 = pd.read_csv('images/scenario02/error_data.csv')

time_01 = create_time_vector(scenario01.values)
time_02 = create_time_vector(scenario02.values)
time_03 = create_time_vector(scenario03.values)

def iae(values):
    N = len(values)
    total_sum = 0
    for i in values:
        total_sum += np.abs(float(i))
    
    return total_sum/N

def ise(values):
    N = len(values)
    total_sum = 0
    for i in values:
        total_sum += float(i)**2
        
    return total_sum/N

def itae(values, time):
    N = len(values)
    for i in range(0, len(values)):
        total_sum = time[i] * np.abs(float(values[i]))

    return total_sum/N

scenario01_iae = iae(scenario01.values)
scenario01_ise = ise(scenario01.values)
scenario01_itae = itae(scenario01.values, time_01)

print('Scenario 01 Result: ')
print('IAE = ' + str(scenario01_iae))
print('ISE = ' + str(scenario01_ise))
print('ITAE = ' + str(scenario01_itae))

print('-'*20)

scenario02_iae = iae(scenario02.values)
scenario02_ise = ise(scenario02.values)
scenario02_itae = itae(scenario02.values, time_02)

print('Scenario 02 Result: ')
print('IAE = ' + str(scenario02_iae))
print('ISE = ' + str(scenario02_ise))
print('ITAE = ' + str(scenario02_itae))

print('-'*20)

scenario03_iae = iae(scenario03.values)
scenario03_ise = ise(scenario03.values)
scenario03_itae = itae(scenario03.values, time_03)

print('Scenario 03 Result: ')
print('IAE = ' + str(scenario03_iae))
print('ISE = ' + str(scenario03_ise))
print('ITAE = ' + str(scenario03_itae))




