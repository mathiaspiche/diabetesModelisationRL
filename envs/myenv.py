from collections import namedtuple
from simglucose.simulation.env import T1DSimEnv

PatientAction = namedtuple('patient_action', ['basal', 'bolus'])

class CustomT1DSimEnv(T1DSimEnv):
    def step(self, action, cho=0.0):
        basal = float(action[0])
        bolus = float(action[1])


        meal = self.scenario.get_action(self.time).meal

        if meal == 0.0:
            bolus = 0.0

        if float(self.patient.observation.Gsub) < 70.0:
            basal = 0.0
            bolus = 0.0

        pump_action = PatientAction(basal=basal, bolus=bolus)

        step_result = super().step(pump_action)
        step_result.info['delivered_basal'] = basal
        step_result.info['delivered_bolus'] = bolus
        return step_result