# Importing EdgeSimPy components
from edge_sim_py import EdgeServer

# Custom methods
from simulator.helper_methods import *
from simulator.custom import *

# Importing Python libraries
import random


def storm(parameters: dict = {}):
    """The default resource-aware scheduler in Apache Storm can be likened to a round-robin algorithm. It operates
    by randomly selecting a server and allocating as many operators as possible until the computational resources
    reach their limits. After saturation, it proceeds to randomly select another server [1].

    [1] Xu, Jinlai, et al. "Amnis: Optimized stream processing for edge computing."
    Journal of Parallel and Distributed Computing 160 (2022): 49-64.

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    edge_servers = EdgeServer.all()

    # Select an edge server at random
    edge_server = random.choice(edge_servers)

    for application in Application.all():
        # Place as many services as possible on the edge server
        while not all(service.server for service in application.services):
            for service in application.services:
                if service.server:
                    continue

                if has_capacity_to_host(edge_server, service):
                    place(service=service, edge_server=edge_server)

            if not all(service.server for service in application.services):
                edge_server = random.choice(edge_servers)


def storm_la(parameters: dict = {}):
    """The default resource-aware scheduler in Apache Storm as in [1], but latency-aware.

    [1] Xu, Jinlai, et al. "Amnis: Optimized stream processing for edge computing."
    Journal of Parallel and Distributed Computing 160 (2022): 49-64.

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    for application in Application.all():
        sensor_location = application.users[0].base_station.network_switch

        # Get the closest edge server to the sensor
        closest_edge_servers = sorted(
            EdgeServer.all(),
            key=lambda edge_server: calculate_path_delay(sensor_location, edge_server.network_switch),
        )

        edge_server = closest_edge_servers.pop(0)

        for edge_server in closest_edge_servers:
            if has_capacity_to_host(edge_server, application.services[0]):
                place(service=application.services[0], edge_server=edge_server)
                break

        # Place as many services as possible on the edge server
        while not all(service.server for service in application.services):
            for service in application.services:
                if service.server:
                    continue

                if has_capacity_to_host(edge_server, service):
                    place(service=service, edge_server=edge_server)

            if not all(service.server for service in application.services):
                edge_server = closest_edge_servers.pop(0)
