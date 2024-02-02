import pandas as pd
from .units import sec_to_ms


def calc_required_charge_time_from_required_energy(mission_required_energy: float,
                                                   initial_soc: float,
                                                   min_reserve_soc: float,
                                                   df: pd.DataFrame) -> int:
    """
    Calculates the required charge time for a given mission required energy and initial SoC
    :param min_reserve_soc:
    :param mission_required_energy:
    :param initial_soc:
    :param df: Charging process dataframe
    :return:
    """
    # Get the energy in kWh at reserve SoC level (~20%)
    energy_at_reserve_soc = df[df.soc == min_reserve_soc]['cumulative_energy_kwh'].values[0]
    # Compute the required energy to complete the mission and still have target reserve soc
    total_required_energy = energy_at_reserve_soc + mission_required_energy
    initial_energy = df[df.soc == initial_soc]['cumulative_energy_kwh'].values[0]
    if initial_energy >= total_required_energy:
        return 0
    # Get the index of closest energy amount on df
    idx_req_energy = df['cumulative_energy_kwh'].sub(total_required_energy).abs().idxmin()
    # Return charge time in hrs
    return sec_to_ms(df.loc[idx_req_energy]['time'] - df[df.soc == initial_soc]['time'].values[0])
