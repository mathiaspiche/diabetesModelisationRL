from datetime import datetime, timedelta
from pathlib import Path
# simglucose core
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.base import Controller, Action
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim

RESULTS_DIR = Path("./simglucose_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Simulation start: midnight of today
now = datetime.now()
START_TIME = datetime(2018, 1, 1, 6, 0, 0)


class Case:
    def __init__(self, patient, sensor, pump, meals):
        self.patient = T1DPatient.withName(patient)
        self.sensor = CGMSensor.withName(sensor)
        self.pump = InsulinPump.withName(pump)
        self.meals = meals
        self.start_time = START_TIME

        self.scenario = CustomScenario(
            start_time=self.start_time,
            scenario=self.meals
        )

        self.env = T1DSimEnv(
            self.patient,
            self.sensor,
            self.pump,
            self.scenario
        )

        self.controller = BBController()

    def step(self):
        scenario = CustomScenario(
            start_time=self.start_time,
            scenario=self.meals
        )

        env = T1DSimEnv(
            self.patient,
            self.sensor,
            self.pump,
            scenario
        )

        simobj = SimObj(
            env,
            self.controller,
            sim_time=timedelta(hours=0.5),
            animate=False,
            path=str(RESULTS_DIR),
        )

        results = sim(simobj)
        print(results.to_string())

        self.start_time += timedelta(hours=0.5)

if __name__ == "__main__":
    case = Case("adult#002", "Dexcom", "Insulet",
                [(7, 45), (12, 70), (16, 20), (18, 80), (22, 15)])

    simobj = SimObj(
        case.env,
        case.controller,
        sim_time=timedelta(hours=50),
        animate=False,
        path=str(RESULTS_DIR),
    )

    results = sim(simobj)
    print(results.to_string())