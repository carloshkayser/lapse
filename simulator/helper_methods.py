# Importing EdgeSimPy components
from edge_sim_py.components import *

# Importing Python libraries
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
import numpy as np
import logging
import random


logger = logging.getLogger("__main__")

convert_coord = lambda x: 2.0 * np.sin(np.radians(60)) * x / 3.0


def has_capacity_to_host(edge_server: object, service: object) -> bool:
    """Checks if the edge server has enough free resources to host a given service.

    Args:
        service (object): Service object that we are trying to host on the edge server.

    Returns:
        can_host (bool): Information of whether the edge server has capacity to host the service or not.
    """

    # Calculating the edge server's free resources
    free_mips = edge_server.mips - edge_server.mips_demand
    free_memory = edge_server.memory - edge_server.memory_demand

    mips_demand = service.mips_demand * service.input_event_rate
    memory_demand = (service.memory_demand + (service.input_event_rate * service.input_event_size)) / 1e6  # convert bytes to megabytes

    # Checking if the host would have resources to host the operator/service
    can_host = free_mips >= mips_demand and free_memory >= memory_demand

    return can_host


def place(service: object, edge_server: object):
    """Place an application's service on an edge server.

    Args:
        service (object): Service to be provisioned.
        edge_server (object): Edge server that will host the edge server.
    """

    operator_mips_demand = service.mips_demand * service.input_event_rate
    operator_memory_demand = (
        service.memory_demand + (service.input_event_rate * service.input_event_size)
    ) / 1e6  # convert bytes to megabytes

    # Updating the host's resource usage
    edge_server.mips_demand += operator_mips_demand
    edge_server.memory_demand += operator_memory_demand

    # Creating relationship between the host and the operator
    service.server = edge_server
    edge_server.services.append(service)

    application = service.application

    ### CREATING NETWORK FLOWS BETWEEN EDGE SERVERS ###

    for service in application.services:
        target = service.server

        if service.level == 0:
            source = application.users[0].base_station

        else:
            source = application.services[service.level - 1].server

        if not source or not target:
            continue

        if source.network_switch == target.network_switch:
            continue

        if len(service.flows) > 0:
            continue

        create_application_flows(source=source.network_switch, target=target.network_switch, service=service)


def create_application_flows(source, target, service):
    """Creates network flows between application's services

    Args:
        source: Source network switch
        target: Target network switch
        service: Service that will be connected by the network flow
    """

    topology = service.model.topology

    if source != target:
        path = nx.shortest_path(G=topology, source=source, target=target, weight="delay")

        data_to_transfer = service.input_event_size * service.input_event_rate

        # convert bytes to gigabits
        data_to_transfer /= 1.25e8

        flow = NetworkFlow(
            topology=topology,
            source=source,
            target=target,
            start=0,
            path=path,
            data_to_transfer=data_to_transfer,
            metadata={"type": "data_stream"},
        )

        service.flows.append(flow)

        flow.network_links = []
        for network_link_id in flow.bandwidth:
            flow.network_links.append(NetworkLink.find_by_id(network_link_id))

        logger.debug(
            "Creating NetworkFlow {} from {} to {} (path: {}) (Service: {})".format(
                flow.id, source.coordinates, target.coordinates, path, service.id
            )
        )

    else:
        logger.debug(f"Source: {source} and target: {target} are the same. No NetworkFlow created.")


def remove_service_from_server(service: object, edge_server: object):
    """Removes a service from an edge server.

    Args:
        service (object): Service to be removed.
        edge_server (object): Edge server where the service is hosted.
    """

    operator_mips_demand = service.mips_demand * service.input_event_rate
    operator_memory_demand = (
        service.memory_demand + (service.input_event_rate * service.input_event_size)
    ) / 1e6  # convert bytes to megabytes

    # Updating the host's resource usage
    edge_server.mips_demand -= operator_mips_demand
    edge_server.memory_demand -= operator_memory_demand

    # Removing relationship between the host and the operator/service
    service.server = None
    edge_server.services.remove(service)

    if len(service.flows) > 0:
        for flow in list(service.flows):
            for link in flow.network_links:
                link["active_flows"].remove(flow)

            NetworkFlow.remove(flow)

        service.flows = []


def estimate_processing_and_communication_latency(service: object, edge_server: object) -> float:
    """Estimates the communication latency of a service placed on an edge server.

    Args:
        service (object): Service to be placed.
        edge_server (object): Edge server where the service will be placed.

    Returns:
        e2e_proc_latency (float): End-to-end processing latency of the service.
    """

    if not has_capacity_to_host(edge_server=edge_server, service=service):
        return float("inf")

    # Placing the service temporarily on the edge server to compute the communication latency
    place(service=service, edge_server=edge_server)

    # Calling the network flow scheduling algorithm to compute the bandwidth available for each network flow
    service.model.network_flow_scheduling_algorithm(topology=service.model.topology, flows=NetworkFlow.all())

    # Computing the communication latency
    communication_latency = compute_operator_comm_latency(service=service)

    # Computing the processing latency
    processing_latency = compute_operator_processing_time(
        service=service,
        edge_server=edge_server,
    )

    # Removing the service from the edge server
    remove_service_from_server(service=service, edge_server=edge_server)

    e2e_proc_latency = communication_latency + processing_latency

    return e2e_proc_latency


def compute_operator_processing_time(service: Service, edge_server: EdgeServer) -> float:
    """Computes the processing time of a service placed on an edge server.

    Args:
        service (object): Service to be placed.
        edge_server (object): Edge server where the service will be placed.

    Returns:
        processing_time (float): Processing time of the service.
    """

    num_svcs = len([s for s in service.server.services])

    # How many MIPS are available for each service (since services share the same CPU)
    server_mips_for_each_service = edge_server.mips / num_svcs

    # How many events per second can be processed
    processing_rate = server_mips_for_each_service / service.mips_demand

    # How long does it take to process all events
    processing_time = service.input_event_rate / processing_rate

    logger.debug(
        "\tProcessing data on EdgeServer {} ({}). EdgeServer MIPS: {}. Computation time: {} seconds".format(
            service.server.id, service.server.coordinates, service.server.mips, processing_time
        )
    )

    return processing_time


def compute_operator_comm_latency(service: Service) -> float:
    """Computes the communication latency of a service.

    Args:
        service (object): Service which we want to compute the communication latency.

    Returns:
        float: Communication latency of the service.
    """

    communication_latency = 0

    # Check if the service has a flow, if not it means that the service is at the same server as the previous service
    if len(service.flows) > 0:
        for flow in service.flows:
            # Get the min value from the bandwidth dict
            min_bandwidth = min(flow.bandwidth.values())

            # Get links delay
            delay = 0
            for link in flow.network_links:
                delay += link["delay"]

            time_to_transfer = (flow.data_to_transfer / min_bandwidth) + (delay * service.input_event_rate)

            logger.debug(
                "\tMoving data from {} to {}. Time to transfer: {} seconds".format(
                    flow.source.coordinates, flow.target.coordinates, time_to_transfer
                )
            )

            logger.debug(
                "\tNetworkFlow {}. Data to transfer: {} Gigabits. Min bandwidth: {} Gbps. Delay: {} seconds".format(
                    flow.id, flow.data_to_transfer, min_bandwidth, delay
                )
            )

            logger.debug("\tCommunication Latency: {}. Time to transfer: {}\n".format(communication_latency, time_to_transfer))

            # Since a service can have multiple flows, we need to get the max communication latency from the slowest flow
            communication_latency = max(communication_latency, time_to_transfer)

    else:
        logger.debug(f"\tService {service.id} does not have NetworkFlow.")

    return communication_latency


def calculate_placement_processing_latency(application: Application) -> float:
    """Calculates the end-to-end processing latency of an application.

    Args:
        application (Application): Application object.

    Returns:
        processing_latency (float): Application end-to-end processing latency.
    """

    processing_latency = 0
    services = application.services

    if all(service.server for service in services):
        logger.debug(f"Application {application.id}")

        processing_latency_by_service = {}

        for service in services:
            # Compute communication time
            communication_time = compute_operator_comm_latency(service=service)

            # Compute computation time (i.e., processing latency)
            processing_time = compute_operator_processing_time(
                service=service,
                edge_server=service.server,
            )

            processing_latency_by_service[service] = processing_time + communication_time

        processing_latency = sum(processing_latency_by_service.values())

        logger.debug(f"\t=== Application {application.id} E2E Processing Latency: {processing_latency} seconds ===")

    return processing_latency


def uniform(n_items: int, valid_values: list, shuffle_distribution: bool = True) -> list:
    """Creates a list of size "n_items" with values from "valid_values" according to the uniform distribution.
    By default, the method shuffles the created list to avoid unbalanced spread of the distribution.

    Args:
        n_items (int): Number of items that will be created.
        valid_values (list): List of valid values for the list of values.
        shuffle_distribution (bool, optional): Defines whether the distribution is shuffled or not. Defaults to True.

    Raises:
        Exception: Invalid "valid_values" argument.

    Returns:
        uniform_distribution (list): List of values arranged according to the uniform distribution.
    """

    if not isinstance(valid_values, list) or isinstance(valid_values, list) and len(valid_values) == 0:
        raise Exception("You must inform a list of valid values within the 'valid_values' attribute.")

    # Number of occurrences that will be created of each item in the "valid_values" list
    distribution = [int(n_items / len(valid_values)) for _ in range(0, len(valid_values))]

    # List with size "n_items" that will be populated with "valid_values" according to the uniform distribution
    uniform_distribution = []

    for i, value in enumerate(valid_values):
        for _ in range(0, int(distribution[i])):
            uniform_distribution.append(value)

    # Computing leftover randomly to avoid disturbing the distribution
    leftover = n_items % len(valid_values)
    for i in range(leftover):
        random_valid_value = random.choice(valid_values)
        uniform_distribution.append(random_valid_value)

    # Shuffling distribution values in case 'shuffle_distribution' parameter is True
    if shuffle_distribution:
        random.shuffle(uniform_distribution)

    return uniform_distribution


def display_dataset(output_filename: str = "topology") -> None:
    """Creates a picture of the network topology (based on https://stackoverflow.com/a/67563903/7412570).

    Args:
        topology (object): EdgeSimPy topology object.
        output_filename (str, optional): Output file name. Defaults to "topology".
        service_labels (bool, optional): If you want to show service labels. Defaults to False.
        flow_labels (bool, optional): If you want to show network flow labels. Defaults to False.
    """

    sns.set_theme(style="whitegrid")

    colors = ["#F12B2E", "#BEC42E", "#25912E", "#3454D1", "#FF8019"]
    palette = sns.color_palette(colors, desat=0.75)
    sns.set_palette(palette)

    fig, ax = plt.subplots(1, figsize=(7, 7), dpi=100)
    ax.set_aspect("equal")

    map_coordinates = [bs.coordinates for bs in BaseStation.all()]
    vcoord, hcoord = prepare_coordinates(map_coordinates)

    # Add map hexagons
    for x, y in zip(hcoord, vcoord):
        hexagon = RegularPolygon(
            (x, y), numVertices=6, radius=2.0 / 3, orientation=np.radians(120), edgecolor=(0, 0, 0, 0.3), facecolor="#bcd6e8"
        )
        ax.add_patch(hexagon)

    # Add edge servers
    for server in EdgeServer.all():
        x, y = server.coordinates
        x, y = -y, convert_coord(x)

        if server.codename == "Model 1":
            c = palette.as_hex()[0]
        elif server.codename == "Model 2":
            c = palette.as_hex()[1]
        elif server.codename == "Model 3":
            c = palette.as_hex()[2]

        hexagon = RegularPolygon(
            (y, x), numVertices=6, radius=2.0 / 3, orientation=np.radians(120), edgecolor=(0, 0, 0, 0.8), facecolor=c
        )
        ax.add_patch(hexagon)

    sensor_coordinates = []
    sink_coordinates = []

    # Add sensors and sinks
    for app in Application.all():
        coord = app.users[0].coordinates

        for service in app.services:
            if service.server:
                sink_coord = service.server.coordinates

        sink_coordinates.append(sink_coord)
        sensor_coordinates.append(coord)

    # check if sensor and sink are in the same coordinates
    for i in range(len(sensor_coordinates)):
        for j in range(len(sink_coordinates)):
            if sensor_coordinates[i] == sink_coordinates[j]:
                sensor_coordinates[i] = (sensor_coordinates[i][0] + 0.3, sensor_coordinates[i][1])
                sink_coordinates[j] = (sink_coordinates[j][0] - 0.3, sink_coordinates[j][1])

    sensor_vcoord = [convert_coord(c[0]) for c in sensor_coordinates]
    sensor_hcoord = [c[1] for c in sensor_coordinates]

    for i in range(len(sensor_vcoord)):
        temp = sensor_vcoord[i]
        sensor_vcoord[i] = -sensor_hcoord[i]
        sensor_hcoord[i] = temp

    ax.scatter(sensor_hcoord, sensor_vcoord, alpha=1, marker="*", color="black", s=100)

    sink_vcoord = [convert_coord(c[0]) for c in sink_coordinates]
    sink_hcoord = [c[1] for c in sink_coordinates]

    for i in range(len(sink_vcoord)):
        temp = sink_vcoord[i]
        sink_vcoord[i] = -sink_hcoord[i]
        sink_hcoord[i] = temp

    ax.scatter(sink_hcoord, sink_vcoord, alpha=1, marker="X", color="black", s=100)

    handles = [
        Line2D([0], [0], linestyle="none", mfc="r", mec="k", marker="h", label="Model 1", ms=10),
        Line2D([0], [0], linestyle="none", mfc="y", mec="k", marker="h", label="Model 2", ms=10),
        Line2D([0], [0], linestyle="none", mfc="g", mec="k", marker="h", label="Model 3", ms=10),
        Line2D([0], [0], linestyle="none", mfc="black", mec="k", marker="*", label="Sensors", ms=10),
        Line2D([0], [0], linestyle="none", mfc="black", mec="k", marker="X", label="Sinks", ms=8),
    ]

    plt.legend(
        ncol=5,
        borderaxespad=-1,
        columnspacing=-10,
        prop={"size": 14},
        loc="lower left",
        mode="expand",
        handles=handles,
        markerscale=2,
        frameon=False,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
    )

    plt.autoscale(enable=True)
    plt.tight_layout()

    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rasterization_zorder(1)

    # save as pdf
    plt.savefig(f"{output_filename}.pdf", bbox_inches="tight", pad_inches=0.05, dpi=300, rasterized=True)


def prepare_coordinates(coordinates):
    # Vertical cartersian coords
    vcoord = [convert_coord(c[0]) for c in coordinates]

    # Horizontal cartesian coords
    hcoord = [c[1] for c in coordinates]

    for i in range(len(vcoord)):
        temp = vcoord[i]
        vcoord[i] = -hcoord[i]
        hcoord[i] = temp

    return vcoord, hcoord


def find_shortest_path(origin_network_switch: object, target_network_switch: object) -> int:
    """Finds the shortest path (delay used as weight) between two network switches (origin and target).

    Args:
        origin_network_switch (object): Origin network switch.
        target_network_switch (object): Target network switch.

    Returns:
        path (list): Shortest path between the origin and target network switches.
    """
    topology = origin_network_switch.model.topology
    path = []

    if not hasattr(topology, "delay_shortest_paths"):
        topology.delay_shortest_paths = {}

    key = (origin_network_switch, target_network_switch)

    if key in topology.delay_shortest_paths.keys():
        path = topology.delay_shortest_paths[key]
    else:
        path = nx.shortest_path(G=topology, source=origin_network_switch, target=target_network_switch, weight="delay")
        topology.delay_shortest_paths[key] = path

    return path


def calculate_path_delay(origin_network_switch: object, target_network_switch: object) -> int:
    """Gets the distance (in terms of delay) between two network switches (origin and target).

    Args:
        origin_network_switch (object): Origin network switch.
        target_network_switch (object): Target network switch.

    Returns:
        delay (int): Delay between the origin and target network switches.
    """
    topology = origin_network_switch.model.topology

    path = find_shortest_path(origin_network_switch=origin_network_switch, target_network_switch=target_network_switch)
    delay = topology.calculate_path_delay(path=path)

    return delay


def min_max_norm(x, min, max):
    """Normalizes a given value (x) using the Min-Max Normalization method.

    Args:
        x (any): Value that must be normalized.
        min (any): Minimum value known.
        max (any): Maximum value known.

    Returns:
        (any): Normalized value.
    """
    if min == max:
        return 1
    return (x - min) / (max - min)


def get_norm(metadata: dict, attr_name: str, min: dict, max: dict) -> float:
    """Wrapper to normalize a value using the Min-Max Normalization method.

    Args:
        metadata (dict): Dictionary that contains the metadata of the object whose values are being normalized.
        attr_name (str): Name of the attribute that must be normalized.
        min (dict): Dictionary that contains the minimum values of the attributes.
        max (dict): Dictionary that contains the maximum values of the attributes.

    Returns:
        normalized_value (float): Normalized value.
    """
    normalized_value = min_max_norm(x=metadata[attr_name], min=min[attr_name], max=max[attr_name])
    return normalized_value


def find_minimum_and_maximum(metadata: list):
    """Finds the minimum and maximum values of a list of dictionaries.

    Args:
        metadata (list): List of dictionaries that contains the analyzed metadata.

    Returns:
        min_and_max (dict): Dictionary that contains the minimum and maximum values of the attributes.
    """
    min_and_max = {
        "minimum": {},
        "maximum": {},
    }

    for metadata_item in metadata:
        for attr_name, attr_value in metadata_item.items():
            if attr_name != "object":
                # Updating the attribute's minimum value
                if (
                    attr_name not in min_and_max["minimum"]
                    or attr_name in min_and_max["minimum"]
                    and attr_value < min_and_max["minimum"][attr_name]
                ):
                    min_and_max["minimum"][attr_name] = attr_value

                # Updating the attribute's maximum value
                if (
                    attr_name not in min_and_max["maximum"]
                    or attr_name in min_and_max["maximum"]
                    and attr_value > min_and_max["maximum"][attr_name]
                ):
                    min_and_max["maximum"][attr_name] = attr_value

    return min_and_max
