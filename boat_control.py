from xml.dom import xmlbuilder
import numpy as np
import time
import math

from zmqRemoteApi import RemoteAPIClient

from mpc import MPC

Ts = 0.05
mass = 80
v_min = -5
v_max = 5
N = 5
N_c = 5

def quat2euler(h):
    roll = np.arctan2(2*(h[0]*h[1] + h[2]*h[3]), 1 - 2*(h[1]**2 + h[2]**2))
    pitch = np.arcsin(2*(h[0]*h[2] - h[3]*h[1]))
    yaw = np.arctan2(2*(h[0]*h[3] + h[1]*h[2]), 1 - 2*(h[2]**2 + h[3]**2))

    return (roll, pitch, yaw)

def get_error(X_d, X_m):
    return X_d - X_m

def extract_cartesian(velocity):
    linear = velocity[0][0]
    angular = velocity[-1][-1]
    x = linear*np.cos(angular)
    y = linear*np.sin(angular)

    return x, y

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

boat = sim.getObject('/Boat')
boat_target = sim.getObject('/Boat_Target')

position = sim.getObjectPosition(boat_target, boat)
orientation = sim.getObjectQuaternion(boat_target, boat)
quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
(roll, pitch, target_yaw) = quat2euler(quat)

X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [target_yaw]])

boat_position = sim.getObjectPosition(boat, -1)
boat_orientation = sim.getObjectQuaternion(boat, -1)
quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
(roll, pitch, yaw) = quat2euler(quat)

X_m = np.array([[boat_position[0]], [boat_position[1]], [boat_position[2]], [roll], [pitch], [yaw]])

#First data
boat_data_x = []
boat_data_y = []
target_position = [position[0], position[1]]
initial_position = [0, 0]
boat_data_x.append(boat_position[0])
boat_data_y.append(boat_position[1])

setpoint = np.ravel([[np.sqrt(position[0]**2 + position[1]**2), np.arctan2(position[1], position[0]), 0, 0] for i in range(0, N+1)])
mpc = MPC(mass, np.array(initial_position), v_min, v_max, N, N_c, Ts)
result = mpc.getNewForce(setpoint)
force = result[1]
print(force)

Fr = 0
Fl = 0

while True:
    print('Final Fr: ' + str(Fr))
    print('Final Fl: ' + str(Fl))
    sim.addForce(boat, [-0.3, -0.2, 0] , [Fr, 0, 0]) # Right Motor
    sim.addForce(boat, [-0.3, 0.2, 0] , [Fl, 0, 0]) # Left Motor
    #sim.addForceAndTorque(boat, [0, 0, 0] , [0, 0, angular])

    time.sleep(Ts)

    #Recollect information
    #Target

    position = sim.getObjectPosition(boat_target, boat)
    orientation = sim.getObjectQuaternion(boat_target, boat)
    quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    (roll, pitch, yaw) = quat2euler(quat)

    X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

    #Boat
    boat_position = sim.getObjectPosition(boat, -1)
    boat_orientation = sim.getObjectQuaternion(boat, -1)
    quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    (roll, pitch, yaw) = quat2euler(quat)
    boat_velocity = sim.getObjectVelocity(boat, -1)
    v_x, v_y = extract_cartesian(boat_velocity)
    v_linear = boat_velocity[0][0]
    v_angular = boat_velocity[-1][-1]


    X_m = np.array([[boat_position[0]], [boat_position[1]], [boat_position[2]], [roll], [pitch], [yaw]])
    boat_data_x.append(boat_position[0])
    boat_data_y.append(boat_position[1])

    #Collect data
    mpc.x_0 = np.array([0, 0, v_linear, v_angular])
    setpoint = np.ravel([[np.sqrt(position[0]**2 + position[1]**2), np.arctan2(position[1], position[0]), 0, 0] for i in range(0, N+1)])
    result = mpc.getNewForce(setpoint)
    force = result[1]
    print(force)

    Fr = force[0]
    Fl = force[1]

print('Arrived')

from matplotlib import pyplot as plt

plt.scatter(boat_data_x, boat_data_y, color='blue')
plt.scatter(initial_position[0], initial_position[1], color='red')
plt.scatter(target_position[0], target_position[1], color='green')
plt.show()