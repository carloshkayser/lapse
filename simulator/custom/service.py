def service_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "label": self.label,
            "state": self.state,
            "_available": self._available,
            "cpu_demand": self.cpu_demand,
            "memory_demand": self.memory_demand,
            "mips_demand": self.mips_demand,
            "image_digest": self.image_digest,
            "input_event_size": self.input_event_size,
            "input_event_rate": self.input_event_rate,
            "level": self.level,
        },
        "relationships": {
            "application": {"class": type(self.application).__name__, "id": self.application.id},
            "server": {"class": type(self.server).__name__, "id": self.server.id} if self.server else None,
            "flows": [{"class": type(flow).__name__, "id": flow.id} for flow in self.flows],
        },
    }

    return dictionary
