import scipy.stats as stats
from typing import Dict, Union
from .units import sec_to_ms
from .helpers import set_seed
import random

# def pick_number(population, probabilities):
#     """
#     Picks a number from a distribution given population and probabilities.
#     :param population: np.ndarray of values in the distribution
#     :param probabilities: np.ndarray of probabilities of values in the distribution
#     :return:
#     """
#     time = random.choices(population, probabilities)[0]
#     time_sec = round(max(0, time))
#     return sec_to_ms(time_sec)


class DistributionGenerator:
    def __init__(self, distribution_params: Dict):
        self.distribution_params = distribution_params
        set_seed()
        self.distribution = getattr(stats, self.distribution_params['distribution_name'])
        # self.population, self.probabilities = self.generate_distribution()

    # def generate_distribution(self) -> int:
    #     """
    #     Generates a distribution given distribution name and parameters.
    #     :param distribution_params
    #     :param max_val_in_dist
    #     :return:
    #     population: np.ndarray of values in the distribution
    #     probabilities: np.ndarray of probabilities of values in the distribution
    #     """
    #     return self.distribution.rvs(**self.distribution_params['parameters'], size=1)[0]

    def pick_number_from_distribution(self):
        """
        Picks a number from the distribution.
        :return:
        """
        set_seed()
        time = self.distribution.rvs(**self.distribution_params['parameters'], size=1)[0]
        if self.distribution_params['max_val_in_dist'] is None:
            return round(max(0, time))
        return min(round(max(0, time)), self.distribution_params['max_val_in_dist'])




# def pick_number(population, probabilities):
#     """
#     Picks a number from a distribution given population and probabilities.
#     :param population: np.ndarray of values in the distribution
#     :param probabilities: np.ndarray of probabilities of values in the distribution
#     :return:
#     """
#     time = random.choices(population, probabilities)[0]
#     time_sec = round(max(0, time))
#     return sec_to_ms(time_sec)
# 
# 
# class DistributionGenerator:
#     def __init__(self, distribution_params: Dict, max_val_in_dist: Union[float, int], distribution_type: str):
#         self.distribution_params = distribution_params
#         self.max_val_in_dist = max_val_in_dist
#         self.distribution_type = distribution_type
#         self.population, self.probabilities = self.generate_distribution()
# 
#     def generate_distribution(self) -> [np.ndarray, np.ndarray]:
#         """
#         Generates a distribution given distribution name and parameters.
#         :param distribution_params
#         :param max_val_in_dist
#         :return:
#         population: np.ndarray of values in the distribution
#         probabilities: np.ndarray of probabilities of values in the distribution
#         """
#         dist = getattr(stats, self.distribution_params['distribution_name'])
#         population = np.linspace(0, self.max_val_in_dist, 1000)
#         try:
#             if self.distribution_type == 'discrete':
#                 probabilities = dist.pmf(x=population, **self.distribution_params['distribution_params'])
#             elif self.distribution_type == 'continuous':
#                 probabilities = dist.pdf(x=population, **self.distribution_params['distribution_params'])
#             return population, probabilities
#         except TypeError as e:
#             raise TypeError("Distribution parameters are not correct. Please check distribution type") from e
# 
#     def pick_number_from_distribution(self):
#         """
#         Picks a number from the distribution.
#         :return:
#         """
#         return pick_number(self.population, self.probabilities)