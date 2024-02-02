def bundle_distributions(*args) -> dict:
    """
    Bundle distributions from the passenger parameters.
    :param args: The distribution dictionaries.
    :return: Nested dictionary of distribution dictionaries.
    """
    return {
        'car_to_entrance_walking_time_dist': args[0],
        'security_check_time_dist': args[1],
        'waiting_room_to_boarding_gate_walking_time_dist': args[2],
        'boarding_gate_to_aircraft_time_dist': args[3],
        'deboard_aircraft_and_walk_to_exit_dist': args[4],
    }
