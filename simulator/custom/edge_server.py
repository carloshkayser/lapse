def edge_server_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "available": self.available,
            "model_name": self.model_name,
            "codename": self.codename,
            "cpu": self.cpu,
            "memory": self.memory,
            "memory_demand": self.memory_demand,
            "disk": self.disk,
            "disk_demand": self.disk_demand,
            "mips": self.mips,
            "mips_demand": self.mips_demand,
            "coordinates": self.coordinates,
            "max_concurrent_layer_downloads": self.max_concurrent_layer_downloads,
            "active": self.active,
            "power_model_parameters": self.power_model_parameters,
        },
        "relationships": {
            "power_model": self.power_model.__name__ if self.power_model else None,
            "base_station": {"class": type(self.base_station).__name__, "id": self.base_station.id} if self.base_station else None,
            "network_switch": {"class": type(self.network_switch).__name__, "id": self.network_switch.id}
            if self.network_switch
            else None,
            "services": [{"class": type(service).__name__, "id": service.id} for service in self.services],
            "container_layers": [{"class": type(layer).__name__, "id": layer.id} for layer in self.container_layers],
            "container_images": [{"class": type(image).__name__, "id": image.id} for image in self.container_images],
            "container_registries": [{"class": type(reg).__name__, "id": reg.id} for reg in self.container_registries],
        },
    }

    return dictionary


def edge_server_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Available": self.available,
        "MIPS": self.mips,
        "CPU": self.cpu,
        "RAM": self.memory,
        "Disk": self.disk,
        "MIPS Demand": self.mips_demand,
        "CPU Demand": self.cpu_demand,
        "RAM Demand": self.memory_demand,
        "Disk Demand": self.disk_demand,
        "Ongoing Migrations": self.ongoing_migrations,
        "Services": [service.id for service in self.services],
        "Registries": [registry.id for registry in self.container_registries],
        "Layers": [layer.instruction for layer in self.container_layers],
        "Images": [image.name for image in self.container_images],
        "Download Queue": [f.metadata["object"].instruction for f in self.download_queue],
        "Waiting Queue": [layer.instruction for layer in self.waiting_queue],
        "Max. Concurrent Layer Downloads": self.max_concurrent_layer_downloads,
        "Power Consumption": self.get_power_consumption(),
    }

    return metrics
