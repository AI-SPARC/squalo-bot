from xml.dom import xmlbuilder
import numpy as np
import time
import math

from zmqRemoteApi import RemoteAPIClient

from mpc_module import MPCClass

Ts = 0.05
mass = 80
F_min = -15
F_max = 15
N = 20

def quat2euler(h):
    roll = np.arctan2(2*(h[0]*h[1] + h[2]*h[3]), 1 - 2*(h[1]**2 + h[2]**2))
    pitch = np.arcsin(2*(h[0]*h[2] - h[3]*h[1]))
    yaw = np.arctan2(2*(h[0]*h[3] + h[1]*h[2]), 1 - 2*(h[2]**2 + h[3]**2))

    return (roll, pitch, yaw)

def extract_cartesian(velocity):
    linear = velocity[0][0]
    angular = velocity[-1][-1]
    x = linear*np.cos(angular)
    y = linear*np.sin(angular)
    return x, y

def extract_angular(v_x, v_y):
    #value = np.arctan2(v_y, v_x) - yaw
    value = np.arctan2(v_y, v_x)
    if value > np.pi:
        value = value - 2*np.pi
    return value

def extract_linear(v_x, v_y):
    return np.sqrt(v_x**2 + v_y**2)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

sim.startSimulation()

boat = sim.getObject('/Boat')
boat_target = sim.getObject('/Boat_Target')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#First data
Fr = [0]
Fl = [0]

while True:
    # Data acquisition
    
    #Target
    position = sim.getObjectPosition(boat_target, boat)
    orientation = sim.getObjectQuaternion(boat_target, boat)
    quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    (roll, pitch, target_yaw) = quat2euler(quat)

    theta = extract_angular(position[0], position[1])
    v = 0
    omega = 0
    reference = np.array([position[0], position[1], theta, v, omega])

    #Boat
    #boat_position = sim.getObjectPosition(boat, -1)
    #boat_orientation = sim.getObjectQuaternion(boat, -1)
    #quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
    #(roll, pitch, yaw) = quat2euler(quat)
    boat_velocity = sim.getObjectVelocity(boat, -1)

    #theta = extract_angular(boat_position[0], boat_position[1])
    v_boat = boat_velocity[0][0]
    omega_boat = boat_velocity[-1][-1]
    #actual_pos = np.array([boat_position[0], boat_position[1], theta, v, omega])
    actual_pos = np.array([0, 0, 0, v_boat, omega_boat])

    
    mpc = MPCClass(reference, Ts, actual_pos, N, F_max, F_min)
    

    # Computing control action
    force = mpc.run_mpc(actual_pos)

    Fr = force[0]
    Fl = force[1]


    # Actuation
    print('Final Fr: ' + str(Fr))
    print('Final Fl: ' + str(Fl))
    sim.addForce(boat, [-0.3, -0.2, 0] , [Fr[0], 0, 0]) # Right Motor
    sim.addForce(boat, [-0.3, 0.2, 0] , [Fl[0], 0, 0]) # Left Motor

    # Time step
    #time.sleep(Ts)
    client.step()
    #breakpoint()
    #Recollect information
