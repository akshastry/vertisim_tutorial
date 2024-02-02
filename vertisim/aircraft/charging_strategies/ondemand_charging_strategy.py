from .base_charging_strategy import BaseChargingStrategy
from ...utils.units import ms_to_sec, sec_to_ms
from ...utils.helpers import check_magnitude_order, miliseconds_to_hms, careful_round
from ..aircraft import AircraftStatus

class OnDemandChargingStrategy(BaseChargingStrategy):
    def get_charging_time(self, aircraft):
        """
        Get charging time - Two options: Constant charging time or draw a number from distribution
        """
        if aircraft.time_charging is not None:
            return max(aircraft.time_passenger_embark_disembark, aircraft.time_charging)
        elif aircraft.target_soc_constant:
            charge_time = self.get_charge_time_from_soc(aircraft=aircraft, init_soc=aircraft.soc, target_soc=aircraft.target_soc_constant)
        else:
            charge_time = sec_to_ms(aircraft.system_manager.charging_time_distribution.pick_number_from_distribution())
            charge_time = max(aircraft.time_passenger_embark_disembark, charge_time)

        aircraft.event_saver.save_aircraft_charging_time(vertiport_id=aircraft.current_vertiport_id,
                                                     charging_time=charge_time)
        aircraft.logger.debug(f"Started Charging: Aircraft {aircraft.tail_number} will charge for {round(ms_to_sec(charge_time)/60,2)} mins at {aircraft.current_vertiport_id}. init_soc: {round(aircraft.soc, 2)}")
        
        return charge_time    
    
    def get_charge_time_from_soc(self, aircraft: object, init_soc: int, target_soc: int) -> float:
        """
        Calculates the charge time required to reach the target soc level.

        Parameters
        ----------
        init_soc: int
            Initial soc level of the aircraft.
        target_soc: int
            Target soc level of the aircraft.

        Returns
        -------
        float
            Charge time required to reach the target soc level.
        """
        if check_magnitude_order(init_soc, target_soc):
            df = aircraft.system_manager.aircraft_battery_models[aircraft.current_vertiport_id]
            init_soc_time = df[df.soc == round(init_soc)]['time_sec'].values[0]
            target_soc_time = df[df.soc == target_soc]['time_sec'].values[0]
            return sec_to_ms(target_soc_time - init_soc_time)
        else:
            aircraft.logger.error(f"Final soc level ({target_soc}) is greater than 100).")
            raise ValueError(f"Final soc level ({target_soc}) is greater than 100).")

    def charge_aircraft(self, aircraft: object, parking_space: object, shared_charger: bool = False) -> None:
        """
        If the charger configuration is fixed and share-limited, all of the chargers that a parking pad has access
        should be requested. We use chargers as a resource instead of simpy.Store or simpy.Filterstore because it's
        easier to follow which resource is being used by which aircraft.
        """
        if aircraft.soc >= aircraft.aircraft_params['min_reserve_soc']:
            return self._check_skip_charging(aircraft)
        aircraft.status = AircraftStatus.CHARGE

        # If the chargers are shared, request all of the chargers that the parking pad has access to.
        if shared_charger:
            used_charger_resource, selected_charging_request = self.handle_shared_charger(parking_space)
        else:
            used_charger_resource, selected_charging_request = self.request_charger(aircraft=aircraft)
            yield selected_charging_request

        self.start_charging_process(aircraft=aircraft)
        if not aircraft.charged_during_turnaround:
            idle_time = aircraft.env.now - aircraft.arrival_time
        else:
            idle_time = aircraft.env.now - aircraft.charging_end_time
        aircraft.idle_time += idle_time
        aircraft.save_process_time(event='idle', process_time=idle_time)
        aircraft.logger.info(f'Saved idle time of {miliseconds_to_hms(idle_time)} between arrival and charging for aircraft {aircraft.tail_number} at {aircraft.location}.')

        aircraft.detailed_status = 'charging'
        charge_time = self.get_charging_time(aircraft=aircraft)
        yield aircraft.env.timeout(round(charge_time))
        aircraft.save_process_time(event='charge', process_time=charge_time)
        aircraft.detailed_status = 'idle'
        self.end_charging_process(aircraft=aircraft)

        old_soc = aircraft.soc
        self.soc_charge_update(aircraft=aircraft, charge_time=charge_time)

        # Set the aircraft status to idle
        aircraft.status = AircraftStatus.IDLE 

        # Release the charger resource that is used by the aircraft and the charging request
        self.release_charger_resource(used_charger_resource, selected_charging_request)
        # Set the charged_during_turnaround to True to avoid charging the aircraft again during the turnaround
        aircraft.charged_during_turnaround = True
        aircraft.logger.debug(f"Finished Charging: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id}. Previous SoC: {round(old_soc, 2)}, New SoC: {careful_round(aircraft.soc, 2)}")

    def _check_skip_charging(self, aircraft):
        aircraft.set_charging_time_trackers()
        aircraft.event_saver.save_agent_state(agent=self, agent_type='aircraft', event='no_charge_needed')
        # After deboarding the passengers aircraft will be ready for the next flight.
        if len(aircraft.passengers_onboard) > 0:
            aircraft.detailed_status = 'embark_disembark'
            yield aircraft.env.timeout(round(aircraft.time_passenger_embark_disembark))
        else:
            yield aircraft.env.timeout(0)
        aircraft.save_process_time(event='embark_disembark', process_time=self.time_passenger_embark_disembark)
        aircraft.detailed_status = 'idle'
        return

