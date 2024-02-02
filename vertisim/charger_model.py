from sympy import Piecewise, var


class ChargerModel:
    def __init__(self, charger_max_charge_rate, charger_efficiency):
        self.charger_max_charge_rate = charger_max_charge_rate
        self.charger_efficiency = charger_efficiency
        self.piecewise_soc = self.calc_piecewise_soc_charge_rate_func()
        # self.visualize_piecewise_soc_function(self.piecewise_soc)
        # self.save_battery_model_visual = save_battery_model_visual
        # self.output_folder_path = output_folder_path

    @staticmethod
    def slope_at_soc_charge_rate(max_charge_rate):
        """
        Between 0-20% SoC charge rate is constant. After 20% SoC we observe a linear decrease in the charge rate.
        For more check paper TODO: add paper link and remove hardcoded value.
        """
        return max_charge_rate / 80

    def calc_piecewise_soc_charge_rate_func(self):
        """
        Calculates the piecewise function for the charge rate as a function of SoC.
        :return: piecewise linear function
        """
        elbow_soc = 20 # TODO: Remove hardcoded value
        var('x')
        max_charge_rate = self.charger_max_charge_rate * self.charger_efficiency
        piecewise_soc = Piecewise((max_charge_rate, x <= elbow_soc), (
            max_charge_rate - self.slope_at_soc_charge_rate(max_charge_rate) * (x - elbow_soc), x > elbow_soc))
        return piecewise_soc

    def calc_charge_rate(self, piecewise_soc, soc):
        return float(piecewise_soc.subs(x, soc))