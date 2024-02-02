from typing import Dict
from .aircraft.aircraft_arrival_setup import AircraftArrivalSetup
from .passenger_arrival_setup import PassengerArrivalSetup



def initiate_flow_entities(sim_setup: Dict) -> None:
    create_aircraft_arrival_process(sim_setup)
    create_passenger_arrival_process(sim_setup)

def create_aircraft_arrival_process(sim_setup: Dict):
    # Load or generate aircraft arrival schedule and start the aircraft arrival process
    flight_schedule_file_path = sim_setup.external_optimization_params.get('flight_schedule_file_path', None)
    charge_schedule_file_path = sim_setup.external_optimization_params.get('charge_schedule_file_path', None)

    # Load aircraft arrival schedule
    aircraft_arrival_process = AircraftArrivalSetup(
        env=sim_setup.env,
        network_simulation=sim_setup.sim_params['network_simulation'],
        vertiport_configs=sim_setup.network_and_demand_params['vertiports'],
        vertiport_layouts=sim_setup.vertiport_layouts,
        flight_schedule_file_path=flight_schedule_file_path,
        charge_schedule_file_path=charge_schedule_file_path,
        aircraft_params=sim_setup.aircraft_params,
        system_manager=sim_setup.system_manager,
        scheduler=sim_setup.scheduler,
        wind=sim_setup.wind,
        structural_entity_groups=sim_setup.structural_entity_groups,
        event_saver=sim_setup.event_saver,
        logger=sim_setup.logger,
        aircraft_logger=sim_setup.aircraft_logger,
        vertiport_logger=sim_setup.vertiport_logger,
        charging_strategy=sim_setup.get_charging_strategy()
    )

    aircraft_arrival_process.create_aircraft_arrival()

    if sim_setup.sim_params['verbose']:
        print("Success: Aircraft arrival process is created.")


def create_passenger_arrival_process(sim_setup: Dict):
    if sim_setup.sim_params['only_aircraft_simulation'] is False:
        # Load or generate passenger arrival schedule and start the passenger arrival process
        passenger_arrival_process = PassengerArrivalSetup(
            env=sim_setup.env,
            network_and_demand=sim_setup.network_and_demand_params,
            passenger_params=sim_setup.passenger_params,
            vertiport_ids=sim_setup.vertiport_ids,
            system_manager=sim_setup.system_manager,
            aircraft_params=None,
            network_simulation=sim_setup.sim_params['network_simulation'],
            passenger_logger=sim_setup.passenger_logger
            )
        sim_setup.env.process(
            passenger_arrival_process.create_passenger_arrival()
        )
        if sim_setup.sim_params['verbose']:
            print("Success: Passenger arrival process is created.")