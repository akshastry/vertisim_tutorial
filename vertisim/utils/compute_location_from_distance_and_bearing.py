import math

def compute_location_from_distance_and_bearing(origin, destination, velocity, time):
    R = 6371e3  # Earth's mean radius in meters

    # Convert velocity from m/s to km/h and time from seconds to hours
    velocity = velocity / 1000 * 3600  # m/s to km/h
    time = time / 3600  # seconds to hours

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])

    # Calculate the bearing
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)

    # Calculate the great-circle distance using the haversine formula
    d_lat = lat2 - lat1
    a = math.sin(d_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    # Calculate the distance the aircraft has traveled
    traveled_distance = velocity * time

    # If the aircraft hasn't reached the destination, calculate the new position
    if traveled_distance < distance:
        delta_sigma = traveled_distance / R
        new_lat = math.asin(math.sin(lat1) * math.cos(delta_sigma) + math.cos(lat1) * math.sin(delta_sigma) * math.cos(bearing))
        new_lon = lon1 + math.atan2(math.sin(bearing) * math.sin(delta_sigma) * math.cos(lat1), math.cos(delta_sigma) - math.sin(lat1) * math.sin(new_lat))

        # Convert the new latitude and longitude from radians to degrees
        new_lat, new_lon = math.degrees(new_lat), math.degrees(new_lon)
        return (new_lat, new_lon)

    else:
        return destination
