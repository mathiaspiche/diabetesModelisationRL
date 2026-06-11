from collections import namedtuple
from simglucose.simulation.env import T1DSimEnv

PatientAction = namedtuple('patient_action', ['basal', 'bolus'])

class CustomT1DSimEnv(T1DSimEnv):
    def step(self, action, cho=0.0):
        basal = float(action[0])
        bolus = float(action[1])

        if cho > 0:
            self.patient._announce_meal(cho)
            self.patient._last_foodtaken = cho

        pump_action = PatientAction(basal=basal, bolus=bolus)
        return super().step(pump_action)