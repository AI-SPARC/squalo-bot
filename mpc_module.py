import numpy as np
import do_mpc
from casadi import *

#Creating Model
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

'''
states 	                _x 	Required
inputs 	                _u 	Required
algebraic 	        _z 	Optional
parameter 	        _p 	Optional
timevarying_parameter 	_tvp 	Optional
'''

#Model Variables
r = model.set_variable(var_type='_x', var_name='r', shape=(1,1))
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
v = model.set_variable(var_type='_x', var_name='v', shape=(1,1))
omega = model.set_variable(var_type='_x', var_name='omega', shape=(1,1))

# Two states for the desired (set) motor position:
fr = model.set_variable(var_type='_u', var_name='fr')
fl = model.set_variable(var_type='_u', var_name='fl')

#Model Parameters
# As shown in the table above, we can use Long names or short names for the variable type.
J = model.set_variable('parameter', 'J')

#Right-hand-side equation

d = 0.3
l = 0.2
m = 80

v_dot = d*omega**2 + (1/m)*(fr+fl)
omega_dot = (l/(m*d**2 + J))*(fr-fl) - (m*d*v*omega)/(m*d**2 + J)


model.set_rhs('r', v)
model.set_rhs('theta', omega)
model.set_rhs('v', v_dot)
model.set_rhs('omega', omega_dot)

model.setup()

#Configuring MPC controller
mpc = do_mpc.controller.MPC(model)

#Optimizer parameters
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.05,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

#Objective function
mterm = r**2 + theta**2 + v**2 + omega**2
lterm = r**2 + theta**2 + v**2 + omega**2

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(
    fr=1e-2,
    fl=1e-2
)

#Constraints

# Lower bounds on inputs:
mpc.bounds['lower','_u', 'fr'] = -5
mpc.bounds['lower','_u', 'fl'] = -5
# Lower bounds on inputs:
mpc.bounds['upper','_u', 'fr'] = 5
mpc.bounds['upper','_u', 'fl'] = 5

#Scaling
mpc.scaling['_x', 'r'] = 1
mpc.scaling['_x', 'theta'] = 1

#Uncertain Parameters
J_value = 3.234*np.array([1., 0.9, 1.1])

mpc.set_uncertainty_values(
    J = J_value,
)

mpc.setup()

#Configuring the simulator
simulator = do_mpc.simulator.Simulator(model)

#Simulator Parameters
# Instead of supplying a dict with the splat operator (**), as with the optimizer.set_param(),
# we can also use keywords (and call the method multiple times, if necessary):
simulator.set_param(t_step = 0.1)

#Uncertain Parameters
p_template = simulator.get_p_template()

def p_fun(t_now):
    p_template['J'] = 2.25e-4
    return p_template

simulator.set_p_fun(p_fun)

simulator.setup()

#Creating the control loop
x0 = np.pi*np.array([1, 1, 1, 1]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

#Graphic
import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='r', axis=ax[0])
    g.add_line(var_type='_x', var_name='theta', axis=ax[0])
    g.add_line(var_type='_x', var_name='v', axis=ax[0])
    g.add_line(var_type='_x', var_name='omega', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='fr', axis=ax[1])
    g.add_line(var_type='_u', var_name='fl', axis=ax[1])


ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('motor angle [rad]')
ax[1].set_xlabel('time [s]')

#Running simulator

u0 = np.array([[10], [10]])
for i in range(200):
    simulator.make_step(u0)

sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
fig.show()

#Running Optimizer
u0 = mpc.make_step(x0)
sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
# Show the figure:
fig.show()


#Running Control Loop
simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

for i in range(20):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
fig.show()


#Saving Data
from do_mpc.data import save_results, load_results
save_results([mpc, simulator])



