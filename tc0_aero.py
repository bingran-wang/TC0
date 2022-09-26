from csdl_om import Simulator
from csdl import Model, NonlinearBlockGS
import python_csdl_backend
import csdl
import numpy as np
import pandas as pd
from smt.surrogate_models import KRG
df = pd.read_excel(r'naca2412_input.xlsx')
class aerodynamics(Model):

    def define(self):
        # declare inputs with default values
        alpha = self.declare_variable('alpha',val = 4) #angle of attack (deg)
        AR = self.declare_variable('AR',val = 6) # aspect ratio
        S = self.declare_variable('S',val = 6) # wing area (m^2)
        rho = self.declare_variable('rho',val = 0.7) #density of air (kg/m^3)
        e = self.declare_variable('e',val = 0.8) #oswald coefficient
        V = self.declare_variable('V',val = 150) #velocity (m/s)
        # NACA Airfoil 2412
        # Reynold number = 1,000,000
        Cl0 = 0.2442
        Cla = 0.11095
        Cl = Cl0 + Cla*alpha
        Cd_min = 0.00547
        K = 0.002066

        Cd = Cd_min +1/(np.pi*e*AR)*Cl**2 + K*(Cl-Cl0)**2
        Cm = csdl.custom(alpha, op=AeroExplicit())
        Cl_max = 1.5820 + alpha - alpha

        self.register_output('Cl', Cl)
        self.register_output('Cd', Cd)
        self.register_output('Cm', Cm)
        self.register_output('Cl_max', Cl_max)

class AeroExplicit(csdl.CustomExplicitOperation):
    def initialize(self):
        # Surrogate modelling for Cm
        alpha_data = df.values[:,0]
        Cm_data = df.values[:,4]
        sm = KRG(theta0=[1e-2])
        sm.set_training_values(alpha_data, Cm_data)
        sm.train()
        self.sm = sm
        #Cm = sm.predict_values(np.array(1))
        #print('Cm value =', Cm)
    def define(self):
        # input: alpha
        self.add_input('alpha', shape=(1,))
        # output: Cm
        self.add_output('Cm', shape=(1,))
        self.declare_derivatives('Cm', 'alpha')

    def compute(self, inputs, outputs):

        # surrogate model
        Cm = self.sm.predict_values(inputs['alpha'])
        outputs['Cm'] = Cm[0]

    def compute_derivatives(self, inputs, derivatives):

        dm_da = self.sm.predict_derivatives(inputs['alpha'], 0)

        derivatives['Cm', 'alpha'] = 1*dm_da

sim = python_csdl_backend.Simulator(aerodynamics())
sim.run()

print('Cl = ', sim['Cl'])
print('Cd = ', sim['Cd'])
print('Cm = ', sim['Cm'])
print('Cl_max = ', sim['Cl_max'])
