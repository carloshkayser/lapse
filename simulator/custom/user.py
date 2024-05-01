import copy


def immobile(user: object, parameters: dict = {}):
    """Defines the mobility model to static for a user

    Args:
        user (object): User whose mobility will be defined.
    """

    user.coordinates_trace.append(user.coordinates)


def user_step(self):
    """Method that executes the events involving the object at each time step."""
    pass


def user_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """

    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Base Station": f"{self.base_station} ({self.base_station.coordinates})" if self.base_station else None,
        "Delays": copy.deepcopy(self.delays),
        "Communication Paths": copy.deepcopy(self.communication_paths),
        "Making Requests": copy.deepcopy(self.making_requests),
    }
    return metrics
