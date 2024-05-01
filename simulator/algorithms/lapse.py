# Importing EdgeSimPy components
from edge_sim_py import *

# Importing backtracking and helper methods
from simulator.helper_methods import *


def get_all_shortest_path_between(origin_network_switch: object, target_network_switch: object) -> int:
    """Gets the distance (in terms of delay) between two network switches (origin and target).

    Args:
        origin_network_switch (object): Origin network switch.
        target_network_switch (object): Target network switch.

    Returns:
        delay (int): Delay between the origin and target network switches.
    """

    topology = origin_network_switch.model.topology

    paths = list(nx.all_shortest_paths(topology, source=origin_network_switch, target=target_network_switch, weight="delay"))

    return paths


def get_edge_servers_metadata(source: object, app: object, edge_servers: object = None) -> list:
    """Gets the metadata of the edge servers.

    Args:
        source (object): Application's data source.
        app (object): Application object.
        edge_servers (object, optional): List of edge servers to get metadata. Defaults to None.

    Returns:
        list: Edge servers metadata.
    """

    metadata = []

    edge_servers_list = edge_servers if edge_servers else EdgeServer.all()

    for edge_server in edge_servers_list:
        # Compute the percentage of services that can be hosted on the edge server
        app_demand = 0
        for service in app.services:
            if not service.server:
                app_demand += service.input_event_rate * service.mips_demand

        edge_server_attrs = {
            "object": edge_server,
            "path_delay_source": calculate_path_delay(source, edge_server.network_switch),
            "path_delay_sink": calculate_path_delay(app.services[-1].server.network_switch, edge_server.network_switch),
            "max_power_consumption": edge_server.power_model_parameters["max_power_consumption"],
        }

        metadata.append(edge_server_attrs)

    return metadata


def get_app_total_demand(application: Application) -> float:
    """Get the total demand of an application.

    Args:
        application (Application): Application object.

    Returns:
        float: Total demand of the application.
    """

    app_demand = 0
    for service in application.services:
        app_demand += service.input_event_rate * service.mips_demand

    return app_demand


def get_edge_servers_between(source: object, target: object) -> list:
    """Get the edge servers between the source and target network switches.

    Args:
        source (object): Source network switch.
        target (object): Target network switch.

    Returns:
        list: List of edge servers between the source and target network switches.
    """

    topology = source.model.topology

    best_path_servers = 0
    possible_edge_servers = None

    paths_between_sensor_and_target = get_all_shortest_path_between(source, target)

    for path in paths_between_sensor_and_target:
        edge_servers_in_path = []

        # search for edge servers in the path
        for switch in path:
            for es in switch.edge_servers:
                if es not in edge_servers_in_path:
                    edge_servers_in_path.append(es)

            # search for edge servers on neighboors network switches
            for neighbor in list(topology.neighbors(switch)):
                for es in neighbor.edge_servers:
                    if es not in edge_servers_in_path:
                        edge_servers_in_path.append(es)

        # Choose the path with the most edge servers
        if len(edge_servers_in_path) > best_path_servers:
            best_path_servers = len(edge_servers_in_path)
            possible_edge_servers = edge_servers_in_path

    return possible_edge_servers


def lapse(parameters: dict = {}):
    """A cost-based heuristic algorithm to optimize the placement of Data Stream Processing
    applications on heterogeneous edge computing infrastructures.

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    apps = Application.all()

    # Sorts applications based on their processing time SLA (from lowest to highest),
    # number of services (from highest to lowest), and input demand (from highest to lowest)
    apps = sorted(
        apps,
        key=lambda app: (
            -get_app_total_demand(app),
            app.processing_latency_sla,
        ),
    )

    for app in apps:
        source = app.users[0].base_station.network_switch
        sink = app.services[-1].server.network_switch

        possible_edge_servers = get_edge_servers_between(source, sink)

        for service in app.services:
            if service.server:
                continue

            while not service.server:
                edge_servers_metadata = get_edge_servers_metadata(source, app, edge_servers=possible_edge_servers)

                min_and_max = find_minimum_and_maximum(metadata=edge_servers_metadata)

                edge_servers_metadata = sorted(
                    edge_servers_metadata,
                    key=lambda m: (
                        get_norm(m, "path_delay_source", min=min_and_max["minimum"], max=min_and_max["maximum"])
                        + get_norm(m, "path_delay_sink", min=min_and_max["minimum"], max=min_and_max["maximum"])
                        + get_norm(m, "max_power_consumption", min=min_and_max["minimum"], max=min_and_max["maximum"]),
                    ),
                )

                for es_metadata in edge_servers_metadata:
                    edge_server = es_metadata["object"]

                    if has_capacity_to_host(edge_server, service):
                        place(service=service, edge_server=edge_server)
                        source = edge_server.network_switch
                        break

                if not service.server:
                    possible_edge_servers = EdgeServer.all()
