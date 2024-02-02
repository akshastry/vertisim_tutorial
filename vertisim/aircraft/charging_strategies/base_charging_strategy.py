from abc import ABC, abstractmethod
from ...utils.helpers import get_key_from_value
from ...utils.units import ms_to_sec


class BaseChargingStrategy(ABC):
    
    @abstractmethod
    def get_charging_time(self, aircraft):
        pass
    
    @abstractmethod
    def charge_aircraft(self, aircraft: object, parking_space: object, shared_charger: bool = False) -> None:
        pass

    @abstractmethod
    def get_charge_time_from_soc(self, aircraft: object, init_soc:int, target_soc:int) -> float:
        pass

    def request_charger(self, aircraft):
        """
        This method is used when the charger configuration is normal (any parking pads can use any chargers).
        Returns the charger resource that is used by the aircraft and the charging request.
        """
        used_charger_resource = aircraft.system_manager.vertiports[aircraft.current_vertiport_id].charger_resource
        return used_charger_resource, used_charger_resource.request()

    def handle_shared_charger(self, aircraft: object, parking_space: object):
        charger_requests = aircraft.env.process(self.request_all_chargers(parking_space))
        charging_requests_list = self.get_first_filled_charger_request(charger_requests)
        selected_charging_request = charging_requests_list[0]
        used_charger_resource = get_key_from_value(charger_requests, selected_charging_request)
        # Find which charger request is being used
        self.release_unused_charger_resources(charger_requests, used_charger_resource, charging_requests_list)
        return used_charger_resource, selected_charging_request
    
    def request_all_chargers(self, aircraft: object, parking_space: object):
        # Request all of the chargers that the parking pad has access.
        charger_requests = {charger: charger.request() for charger in parking_space.charger_resources}
        # Yield all of the charger requests. When any_of used it fires all of the requests.
        yield aircraft.env.any_of(list(charger_requests.values()))
        return charger_requests

    def get_first_filled_charger_request(self, charger_requests: dict):
        # The selected charging request is the on top of the list
        return list(charger_requests.keys())

    def release_unused_charger_resources(self, charger_requests: dict, used_charger_resource: object, charging_requests_list: object):
        # Since any_of() used more than one request might be filled. Only one should be kept and the rest should be canceled.        
        for charger_res, charger_req in charger_requests.items():
            if charger_res != used_charger_resource:
                if len(charging_requests_list) > 1:
                    charger_res.release(charger_req)
                else:
                    charger_req.cancel()    

    def start_charging_process(self, aircraft: object):
        aircraft.charging_start_time = aircraft.env.now
        aircraft.event_saver.save_agent_state(agent=aircraft, agent_type='aircraft', event='charging_start')

    def end_charging_process(self, aircraft: object):
        aircraft.charging_end_time = aircraft.env.now
        aircraft.event_saver.save_agent_state(agent=aircraft, agent_type='aircraft', event='charging_end')

    def release_charger_resource(self, used_charger_resource: object, selected_charging_request: object):
        used_charger_resource.release(selected_charging_request)                    

    def calc_soc_from_charge_time(self, charge_time: int, initial_soc: int, df) -> int:
        """
        charge_time is the time between the current time and the charging start time.

        Parameters
        ----------
        charge_time: int
            Time between the current time and the charging start time in miliseconds.
        soc: int
            Initial soc level of the aircraft.
        df: pd.DataFrame
            Dataframe that contains the soc levels and the time information.

        Returns
        -------
        int
            The soc level of the aircraft after the charging time.
        """
        # Get the time at initial soc level
        initial_soc_time = df[df.soc == round(initial_soc)]['time_sec'].values[0]
        # Get the time at final soc level
        final_soc_time = initial_soc_time + ms_to_sec(charge_time)
        # Get the soc at final soc level
        idx_final_soc = df['time_sec'].sub(final_soc_time).abs().idxmin()
        return df.loc[idx_final_soc]['soc']
    
    def soc_charge_update(self, aircraft: object, charge_time: float) -> None:
        """
        Updates the charge level of the aircraft using current charge time.
        """
        charge_lookup_table = aircraft.system_manager.aircraft_battery_models[aircraft.current_vertiport_id]
        final_soc = self.calc_soc_from_charge_time(charge_time=charge_time, initial_soc=aircraft.soc, df=charge_lookup_table)
        # TODO: Might need attention. Previously it was self._soc = final_soc
        aircraft.soc = final_soc    
