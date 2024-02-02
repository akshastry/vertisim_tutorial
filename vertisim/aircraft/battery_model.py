# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class BatteryModel:
    def __init__(self, charger_model):
        self.charger_model = charger_model

    def charge_process(self, battery_capacity=160):
        """
        Computes the charging process for a given battery size and charger model. Each aircraft model has its own process
        :param battery_capacity:
        :return:
        """
        increments_size_in_soc = 0.5  # in %
        charge_rate_kw = self.charger_model.charger_max_charge_rate * self.charger_model.charger_efficiency
        prev_time = 0
        soc_initial = 0
        soc_final = 100
        total_charge = 0
        time = [prev_time]
        charge_rates = [charge_rate_kw]
        socs = [soc_initial]
        cumulative_energy_kwh = [0]

        for soc in np.arange(soc_initial, soc_final + 1, increments_size_in_soc):
            new_charge_rate_kw = self.charger_model.calc_charge_rate(self.charger_model.piecewise_soc, soc)
            average_kw = (charge_rate_kw + new_charge_rate_kw) / 2
            if average_kw == 0:
                break            

            charge_increment_kwh = round(increments_size_in_soc/100*battery_capacity, 4)  # Charge in the interval
            time_increment = round(charge_increment_kwh/average_kw, 4) # in hours
            current_time = prev_time + time_increment
            total_charge += charge_increment_kwh
            cumulative_energy_kwh.append(total_charge)
            time.append(current_time)
            charge_rates.append(average_kw)
            socs.append(soc)
            prev_time = current_time
            charge_rate_kw = average_kw
        
        charge_df = pd.DataFrame(data=list(zip(time, charge_rates, socs, cumulative_energy_kwh)),
                            columns=['time_hr', 'charge_rate', 'soc', 'cumulative_energy_kwh'], dtype=float)
        charge_df['time_sec'] = charge_df['time_hr']*3600
        charge_df = charge_df[charge_df.soc <= 100]
        return charge_df


# battery_model = BatteryModel()
#
# print(battery_model.calc_charge_rate(battery_model.piecewise_soc, 30))



    # def visualize_piecewise_soc_function(self, piecewise_soc):
    #     """
    #     Visualizes the piecewise function for the charge rate as a function of SoC.
    #     :param piecewise_soc:
    #     :return:
    #     """
    #     if self.save_battery_model_visual:
    #         x_axis = np.linspace(0, 101, 101)
    #         y_axis = [piecewise_soc.subs(x, i) for i in x_axis]
    #         plt.plot(x_axis, y_axis)
    #         plt.xlabel('SoC %')
    #         plt.ylabel('Charge Power (kW)')
    #         plt.grid('on', alpha=0.5)
    #         plt.savefig(f'{self.output_folder_path}/battery_model.png')