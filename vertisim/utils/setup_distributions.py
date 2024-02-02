from .distribution_generator import DistributionGenerator


def setup_passenger_distributions(**distributions_bundle) -> dict:
    """
    Using the input dictionaries, create distributions for each of them.
    :param args: The distribution dictionaries.
    :return: Nested dictionary of distribution objects.
    """
    distribution_functions = {}
    for distribution_input, value in distributions_bundle.items():
        if value is not None:
            distribution_functions[distribution_input] = DistributionGenerator(
                distribution_params=distributions_bundle[distribution_input],
                max_val_in_dist=distributions_bundle[distribution_input]['max_val_in_dist']
            )
    return distribution_functions
