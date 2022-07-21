import numpy as np
import do_mpc
from casadi import *


class MPCClass():
    def __init__(self, ref, ts, initial_position):
        self.Ts = ts
        self.reference = ref

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
        #r = model.set_variable(var_type='_x', var_name='r', shape=(1,1))
        x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
        theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(1,1))
        omega = model.set_variable(var_type='_x', var_name='omega', shape=(1,1))

        # Two states for the desired (set) motor position:
        fr = model.set_variable(var_type='_u', var_name='fr')
        fl = model.set_variable(var_type='_u', var_name='fl')

        #Model Parameters
        # As shown in the table above, we can use Long names or short names for the variable type.
        #J = model.set_variable('parameter', 'J')
        #J = 15
        #Right-hand-side equation

        d = 0.3
        l = 0.2
        m = 80

        J = 3.007 * m

        v_dot = (d*(omega**2)) + (1/m)*(fr+fl)
        omega_dot = (l/(m*(d**2) + J))*(fr-fl) - (m*d*v*omega)/(m*(d**2) + J)


        #model.set_rhs('r', v)
        model.set_rhs('x', v*np.cos(theta))
        model.set_rhs('y', v*np.sin(theta))
        model.set_rhs('theta', omega)
        model.set_rhs('v', v_dot)
        model.set_rhs('omega', omega_dot)

        model.setup()

        #Configuring MPC controller
        self.mpc = do_mpc.controller.MPC(model)

        #Optimizer parameters
        setup_mpc = {
            'n_horizon': 20,
            't_step': self.Ts,
            'n_robust': 0,
            'store_full_solution': False,
            'open_loop': True,
            'nlpsol_opts': {
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0
            }
        }
        self.mpc.set_param(**setup_mpc)

        #Objective function
        mterm = (self.reference[0] - x)**2 + \
                (self.reference[1] - y)**2 + \
                0*(self.reference[2] - theta)**2 + \
                (self.reference[3] - v)**2 + \
                (self.reference[4] - omega)**2
        #mterm = r**2 + theta**2 + v**2 + omega**2
        lterm = (self.reference[0] - x)**2 + \
                (self.reference[1] - y)**2 + \
                0*(self.reference[2] - theta)**2 + \
                0*(self.reference[3] - v)**2 + \
                0*(self.reference[4] - omega)**2
        #lterm = r**2 + theta**2 + v**2 + omega**2

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            fr=0,
            fl=0
        )
        #[4.5240309  0.23942829 0.05027771 0.02493158]
        #Constraints

        # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'fr'] = -20
        self.mpc.bounds['lower','_u', 'fl'] = -20
        # Lower bounds on inputs:
        self.mpc.bounds['upper','_u', 'fr'] = 20
        self.mpc.bounds['upper','_u', 'fl'] = 20

        #Scaling
        self.mpc.scaling['_x', 'x'] = 1
        self.mpc.scaling['_x', 'y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'omega'] = 1

        #Uncertain Parameters
        #J_value = 15*np.array([1., 0.9, 1.1])

        #self.mpc.set_uncertainty_values(
        #    J = J_value,
        #)
        self.mpc.x0 = initial_position
        self.mpc.setup()
        self.mpc.set_initial_guess()

    def run_mpc(self, actual_position):

        x0 = np.array(actual_position).reshape(-1,1)

        #Creating the control loop
        self.mpc.x0 = x0
        #Running Optimizer
        u0 = self.mpc.make_step(x0)
        return u0

