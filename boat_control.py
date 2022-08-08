from xml.dom import xmlbuilder
import numpy as np
import time
import math

import matplotlib.pyplot as plt
from zmqRemoteApi import RemoteAPIClient

from mpc_module import MPCClass


class BoatControl():
    def __init__(self):
        self.Ts = 0.05
        self.mass = 80
        self.F_min = -15
        self.F_max = 15
        self.N = 20

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.client.setStepping(True)

        self.sim.startSimulation()

        self.boat = self.sim.getObject('/Boat')
        self.boat_target = self.sim.getObject('/Boat_Target')

        self.theta_data = []
        self.omega_data = []
        self.Fr_data = []
        self.Fl_data = []
        self.x_boat = []
        self.y_boat = []
        self.x_target = -1
        self.y_target = -1
        self.boat_speed = []
        self.time_data = []

        self.theta_ref = []
        self.omega_ref = []
        self.v_ref = []

    def quat2euler(self, h):
        roll = np.arctan2(2*(h[0]*h[1] + h[2]*h[3]), 1 - 2*(h[1]**2 + h[2]**2))
        pitch = np.arcsin(2*(h[0]*h[2] - h[3]*h[1]))
        yaw = np.arctan2(2*(h[0]*h[3] + h[1]*h[2]), 1 - 2*(h[2]**2 + h[3]**2))

        return (roll, pitch, yaw)

    def extract_cartesian(self, velocity):
        linear = velocity[0][0]
        angular = velocity[-1][-1]
        x = linear*np.cos(angular)
        y = linear*np.sin(angular)
        return x, y

    def extract_angular(self, v_x, v_y):
        #value = np.arctan2(v_y, v_x) - yaw
        value = np.arctan2(v_y, v_x)
        if value > np.pi:
            value = value - 2*np.pi
        return value

    def extract_linear(self, v_x, v_y):
        return np.sqrt(v_x**2 + v_y**2)

    def collect_boat_info(self):
        boat_position = self.sim.getObjectPosition(self.boat, -1)
        #boat_orientation = sim.getObjectQuaternion(boat, -1)
        #quat = np.array([boat_orientation[3], boat_orientation[0], boat_orientation[1], boat_orientation[2]]) # Remember: getObjectQuaternion has real part as last element
        #(roll, pitch, yaw) = quat2euler(quat)
        boat_velocity = self.sim.getObjectVelocity(self.boat, -1)

        #theta = extract_angular(boat_position[0], boat_position[1])
        v_boat = boat_velocity[0][0]
        omega = boat_velocity[-1][-1]
        #actual_pos = np.array([boat_position[0], boat_position[1], theta, v, omega])
        actual_pos = np.array([0, 0, 0, v_boat, omega])

        self.boat_speed.append(v_boat)
        self.omega_data.append(omega)
        self.x_boat.append(boat_position[0])
        self.y_boat.append(boat_position[1])
        return actual_pos

    def collect_target_info(self):
        #Target
        position = self.sim.getObjectPosition(self.boat_target, self.boat)
        position_to_world = self.sim.getObjectPosition(self.boat_target, -1)
        #orientation = sim.getObjectQuaternion(boat_target, boat)
        #quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]]) # Remember: getObjectQuaternion has real part as last element
        #(roll, pitch, target_yaw) = quat2euler(quat)

        theta = self.extract_angular(position[0], position[1])
        v = 0
        omega = 0
        reference = np.array([position[0], position[1], theta, v, omega])

        self.theta_data.append(theta)
        self.x_target = position_to_world[0]
        self.y_target = position_to_world[1]

        self.theta_ref.append(theta)
        self.omega_ref.append(omega)
        self.v_ref.append(v)

        return reference

    def distance(self):
        target_position = self.sim.getObjectPosition(self.boat_target, -1)
        boat_position = self.sim.getObjectPosition(self.boat, -1)

        return math.dist([target_position[0], target_position[1]], [boat_position[0], boat_position[1]])

    def execute_control(self):
        steps = 0
        time_elapsed = 0
        max_steps = 300

        while self.distance() > 0.2:
            # Data acquisition
            boat_position = self.collect_boat_info()
            target_position = self.collect_target_info()
            mpc = MPCClass(target_position, self.Ts, boat_position, self.N, self.F_max, self.F_min)

            # Computing control action
            force = mpc.run_mpc(boat_position)

            Fr = force[0]
            Fl = force[1]

            # Actuation
            print('Final Fr: ' + str(Fr))
            print('Final Fl: ' + str(Fl))
            self.sim.addForce(self.boat, [-0.3, -0.2, 0] , [Fr[0], 0, 0]) # Right Motor
            self.sim.addForce(self.boat, [-0.3, 0.2, 0] , [Fl[0], 0, 0]) # Left Motor


            self.Fr_data.append(Fr[0])
            self.Fl_data.append(Fl[0])
            
            steps += 1
            time_elapsed += self.Ts
            self.time_data.append(time_elapsed)

            # Time step
            #time.sleep(Ts)
            self.client.step()
            #breakpoint()
            #Recollect information


if __name__ == '__main__':
    boat_control = BoatControl()
    boat_control.execute_control()

    plt.figure('Figure 1')
    plt.plot(boat_control.time_data, boat_control.boat_speed, label='Boat speed')
    plt.plot(boat_control.time_data, boat_control.v_ref, label='Reference')
    plt.title('Boat Speed History')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.savefig('images/figure_1.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()

    plt.figure('Figure 2')
    plt.plot(boat_control.time_data, boat_control.Fr_data, label='Right motor force')
    plt.plot(boat_control.time_data, boat_control.Fl_data, label='Left motor force')
    plt.title('Boat Forces History')
    plt.xlabel('Time')
    plt.ylabel('Fr and Fl')
    plt.savefig('images/figure_2.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()

    plt.figure('Figure 3')
    plt.plot(boat_control.x_boat, boat_control.y_boat, label='Boat Position')
    plt.plot([boat_control.x_target], [boat_control.y_target], marker="o", markersize=10, markerfacecolor="green", label='Target')
    plt.title('Boat Position History')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('images/figure_3.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()

    plt.figure('Figure 4')
    plt.plot(boat_control.time_data, boat_control.theta_data, label='Theta')
    plt.plot(boat_control.time_data, boat_control.theta_ref, label='Reference')
    plt.title('Theta History')
    plt.xlabel('Time')
    plt.ylabel('Theta')
    plt.savefig('images/figure_4.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()

    plt.figure('Figure 5')
    plt.plot(boat_control.time_data, boat_control.omega_data, label='Omega')
    plt.plot(boat_control.time_data, boat_control.omega_ref, label='Reference')
    plt.title('Omega History')
    plt.xlabel('Time')
    plt.ylabel('Omega')
    plt.savefig('images/figure_5.png', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
