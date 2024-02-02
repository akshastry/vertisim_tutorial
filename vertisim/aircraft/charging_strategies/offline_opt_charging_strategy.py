from .base_charging_strategy import BaseChargingStrategy
from ..aircraft import AircraftStatus
from typing import Tuple, Generator, Any
from ...utils.units import ms_to_sec, sec_to_ms
from ...utils.helpers import check_magnitude_order, miliseconds_to_hms, careful_round, roundup_to_five_minutes

class OfflineOptimizationChargingStrategy(BaseChargingStrategy):
    """
    Strategy for charging aircraft in an offline optimization context.
    """    

    def get_charging_time(self, aircraft):
        charge_time = self.get_charge_time_from_soc(aircraft=aircraft,
                                                    init_soc=aircraft.soc,
                                                    target_soc=aircraft.target_soc)
        aircraft.event_saver.save_aircraft_charging_time(vertiport_id=aircraft.current_vertiport_id,
                                                     charging_time=charge_time)
        aircraft.logger.debug(f"Started Charging: Aircraft {aircraft.tail_number} will charge for {round(ms_to_sec(charge_time)/60,2)} mins at {aircraft.current_vertiport_id}. init_soc: {round(aircraft.soc, 2)}, target_soc: {aircraft.target_soc}")
        
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
        
    # def charge_aircraft(self, aircraft: object, parking_space: object, shared_charger: bool = False) -> None:
    #     """
    #     Handle the charging process for an aircraft.
    #     """
    #     aircraft.status = AircraftStatus.CHARGE
    #     used_charger_resource, selected_charging_request = self._request_charger(aircraft, parking_space, shared_charger)
    #     if not shared_charger:
    #         yield selected_charging_request
    #     self.start_charging_process(aircraft=aircraft)  
    #     self._update_idle_time(aircraft)
    #     yield aircraft.env.process(self._perform_charging(aircraft))
    #     self._finalize_charging(aircraft, used_charger_resource, selected_charging_request)

    def _request_charger(self, aircraft: object, parking_space: object, shared_charger: bool) -> Tuple[object, object]:
        """
        Request charger based on whether it is shared or not.
        """
        if shared_charger:
            return self.handle_shared_charger(parking_space)
        else:
            return self.request_charger(aircraft=aircraft)
        
    def _update_idle_time(self, aircraft: object) -> None:
        """
        Update the idle time of the aircraft.
        """
        idle_time = self._calculate_idle_time(aircraft)
        aircraft.idle_time += idle_time
        aircraft.save_process_time(event='idle', process_time=idle_time)
        aircraft.logger.info(f'Saved idle time of {miliseconds_to_hms(idle_time)} between arrival and charging for aircraft {aircraft.tail_number} at {aircraft.location}.')

    def _calculate_idle_time(self, aircraft: object) -> int:
        """
        Calculate idle time based on aircraft's charging history.
        """
        if not aircraft.charged_during_turnaround:
            return aircraft.env.now - aircraft.arrival_time
        else:
            return aircraft.env.now - aircraft.charging_end_time
        
    def _perform_charging(self, aircraft: object) -> None:
        """
        Perform the actual charging process.
        """
        aircraft.detailed_status = 'charging'    
        charge_time = self.get_charging_time()
        yield aircraft.env.timeout(round(charge_time))
        aircraft.save_process_time(event='charge', process_time=charge_time)
        old_soc = aircraft.soc
        self.soc_charge_update(charge_time)          
        aircraft.detailed_status = 'idle'
        aircraft.save_process_time(event='idle', process_time=self._compute_holding_after_charge(charge_time))
        
        yield aircraft.env.process(self.hold_for_charging_schedule_compatibility(aircraft=aircraft, charge_time=charge_time))

        aircraft.logger.info(f'Saved idle time of {miliseconds_to_hms(self._compute_holding_after_charge(charge_time))} for after charge holding for aircraft {aircraft.tail_number} at {aircraft.location}.')
        aircraft.logger.debug(f"Finished Charging: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id}. Previous SoC: {round(old_soc, 2)}, New SoC: {careful_round(aircraft.soc, 2)}")

    def _finalize_charging(self, aircraft: object, used_charger_resource: object, selected_charging_request: object) -> None:
        """
        Finalize the charging process, release resources, and update statuses.
        """
        self.end_charging_process()
        aircraft.status = AircraftStatus.IDLE    
        self.set_target_soc_none()
        self.release_charger_resource(used_charger_resource, selected_charging_request)
        aircraft.charged_during_turnaround = True

    def charge_aircraft(self, aircraft: object, parking_space: object, shared_charger: bool = False) -> None:
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
        charge_time = self.get_charging_time(aircraft)
        yield aircraft.env.timeout(round(charge_time))
        aircraft.save_process_time(event='charge', process_time=charge_time)
        aircraft.detailed_status = 'idle'
        aircraft.save_process_time(event='idle', process_time=self._compute_holding_after_charge(charge_time))
        yield aircraft.env.process(self.hold_for_charging_schedule_compatibility(aircraft=aircraft,
                                                                                 charge_time=charge_time))
        self.end_charging_process(aircraft=aircraft)
        aircraft.logger.info(f'Saved idle time of {miliseconds_to_hms(self._compute_holding_after_charge(charge_time))} for after charge holding for aircraft {aircraft.tail_number} at {aircraft.location}.')

        old_soc = aircraft.soc
        self.soc_charge_update(aircraft=aircraft, charge_time=charge_time)  

        # Set the aircraft status to idle
        aircraft.status = AircraftStatus.IDLE    

        # Reset the target_soc to None for the next flight
        self.set_target_soc_none(aircraft)
        # Release the charger resource that is used by the aircraft and the charging request
        self.release_charger_resource(used_charger_resource, selected_charging_request)
        # Set the charged_during_turnaround to True to avoid charging the aircraft again during the turnaround
        aircraft.charged_during_turnaround = True
        aircraft.logger.debug(f"Finished Charging: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id}. Previous SoC: {round(old_soc, 2)}, New SoC: {careful_round(aircraft.soc, 2)}")


    def hold_for_charging_schedule_compatibility(self, aircraft, charge_time):
        holding_time = self._compute_holding_after_charge(charge_time=charge_time)
        yield aircraft.env.timeout(holding_time)
    
    def _compute_holding_after_charge(self, charge_time):
        """
        If the charging schedule is fixed, the aircraft should hold until the next time step.
        """
        rounded_charge_time = roundup_to_five_minutes(charge_time)
        return rounded_charge_time - charge_time
        
    def set_target_soc_none(self, aircraft):
        aircraft.target_soc = None