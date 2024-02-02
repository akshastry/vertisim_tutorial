class DepartingPassenger:
    def __init__(self, env, passenger_id, flight_dest,
                 car_to_entrance_walk_time, securtiy_check_time, boarding_gate_to_aircraft_time):
        self.env = env
        self.passenger_id = passenger_id
        self.flight_dest = flight_dest
        self.car_to_entrance_walk_time = car_to_entrance_walk_time
        self.securtiy_check_time = securtiy_check_time
        self.boarding_gate_to_aircraft_time = boarding_gate_to_aircraft_time

        self.passenger_arrival = None
        self.vertiport_entrance_time = None
        self.security_check_end_time = None
        self.enter_waiting_room_time = None
        self.exit_waiting_room_time = None
        self.boarding_gate_arrival_time = None
        self.boarding_time = None
        self.take_off_time = None

    def walk_to_entrance(self):
        """ From car to the exit gate including time spent in the car."""
        yield self.env.timeout(random.randint(self.car_to_entrance_walk_time - 10, self.carToEntranceWalkingTime + 50))