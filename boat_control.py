from xml.dom import xmlbuilder
import numpy as np
import time
import math

from zmqRemoteApi import RemoteAPIClient

Ts = 0.05


def quat2euler(h):
    roll = np.arctan2(2*(h[0]*h[1] + h[2]*h[3]), 1 - 2*(h[1]**2 + h[2]**2))
    pitch = np.arcsin(2*(h[0]*h[2] - h[3]*h[1]))
    yaw = np.arctan2(2*(h[0]*h[3] + h[1]*h[2]), 1 - 2*(h[2]**2 + h[3]**2))

    return (roll, pitch, yaw)

def get_error(X_d, X_m):
    return X_d - X_m

def extract_error_information(error, X_m):
    v_x = error[0][0]
    v_y = error[1][0]
    yaw = X_m[5][0]
    return v_x, v_y, yaw

def extract_angular(v_x, v_y, yaw):
    value = np.arctan2(v_y, v_x) - yaw
    if value > np.pi:
        value = value - 2*np.pi
    return value

def extract_linear(v_x, v_y):
    return np.sqrt(v_x**2 + v_y**2)

def calculate_angular(right_motor, left_motor, distance):
    #Distance is 16cm (protoype) and 35-45cm for real scale
    return (right_motor - left_motor)/distance

def calculate_linear(right_motor, left_motor):
    return (right_motor+left_motor)/2

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()

boat_target = sim.getObject('/Boat_Target')

position = sim.getObjectPosition(boat_target, -1)
orientation = sim.getObjectQuaternion(boat_target, -1)
quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
(roll, pitch, yaw) = quat2euler(quat)

X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

boat = sim.getObject('/Boat')

boat_position = sim.getObjectPosition(boat, -1)
boat_orientation = sim.getObjectQuaternion(boat, -1)
quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
(roll, pitch, yaw) = quat2euler(quat)

X_m = np.array([[boat_position[0]], [boat_position[1]], [boat_position[2]], [roll], [pitch], [yaw]])

#First data
error = get_error(X_d, X_m)
v_x, v_y, yaw = extract_error_information(error, X_m)
angular = extract_angular(v_x, v_y, yaw)
linear = extract_linear(v_x, v_y)

boat_data_x = []
boat_data_y = []
target_position = [position[0], position[1]]
initial_position = [boat_position[0], boat_position[1]]
boat_data_x.append(boat_position[0])
boat_data_y.append(boat_position[1])

while abs(linear) > 0.3:
    print('Linear: ' + str(linear))
    sim.addForce(boat, [1,0,0] , [linear, 0, 0])
    sim.addForceAndTorque(boat, [0, 0, 0] , [0, 0, angular])

    time.sleep(Ts)

    #Recollect information
    #Target
    boat_target = sim.getObject('/Boat_Target')

    position = sim.getObjectPosition(boat_target, -1)
    orientation = sim.getObjectQuaternion(boat_target, -1)
    quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    (roll, pitch, yaw) = quat2euler(quat)

    X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

    #Boat
    boat_position = sim.getObjectPosition(boat, -1)
    boat_orientation = sim.getObjectQuaternion(boat, -1)
    quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    (roll, pitch, yaw) = quat2euler(quat)

    X_m = np.array([[boat_position[0]], [boat_position[1]], [boat_position[2]], [roll], [pitch], [yaw]])
    boat_data_x.append(boat_position[0])
    boat_data_y.append(boat_position[1])

    #Collect data
    error = get_error(X_d, X_m)
    v_x, v_y, yaw = extract_error_information(error, X_m)
    angular = extract_angular(v_x, v_y, yaw)
    linear = extract_linear(v_x, v_y)

print('Arrived')

from matplotlib import pyplot as plt

plt.scatter(boat_data_x, boat_data_y, color='blue')
plt.scatter(initial_position[0], initial_position[1], color='red')
plt.scatter(target_position[0], target_position[1], color='green')
plt.show()