def application_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "label": self.label,
            "processing_latency_sla": self.processing_latency_sla,
        },
        "relationships": {
            "services": [{"class": type(service).__name__, "id": service.id} for service in self.services],
            "users": [{"class": type(user).__name__, "id": user.id} for user in self.users],
        },
    }

    return dictionary
