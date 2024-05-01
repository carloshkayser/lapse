# Importing EdgeSimPy components
from edge_sim_py import *

# Importing helper methods
from simulator.helper_methods import *

# Importing Python libraries
from queue import Queue


def build_deployment_sequence() -> list:
    """Builds the deployment sequence of the services.

    Returns:
        list: Deployment sequence.
    """

    ds = []
    queue = Queue()

    sources = [app.services[0] for app in Application.all()]

    for source in sources:
        ds.append(source)

        for downstream in source.application.services[1:]:
            if downstream not in list(queue.queue):
                queue.put(downstream)

    while not queue.empty():
        operator = queue.get()

        upstreams = operator.application.services[: operator.application.services.index(operator)]

        if all(upstream in ds for upstream in upstreams):

            for downstream in operator.application.services[operator.application.services.index(operator) + 1 :]:
                if downstream not in list(queue.queue):
                    queue.put(downstream)

            ds.append(operator)
        else:
            queue.push(operator)

    return ds


def aels(parameters: dict = {}):
    """Aggregate End-to-End Latency Strategy (AELS) adapted version proposed in [1].

    [1] da Silva Veith, Alexandre, et al. "Latency-Aware Strategies for Deploying Data Stream Processing
    Applications on Large Cloud-Edge Infrastructure." IEEE transactions on cloud computing (2021).

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    ds = build_deployment_sequence()

    for operator in ds:
        if operator.server:
            continue

        edge_servers = sorted(
            EdgeServer.all(),
            key=lambda edge_server: estimate_processing_and_communication_latency(operator, edge_server),
        )

        for edge_server in edge_servers:
            if has_capacity_to_host(edge_server, operator):
                place(service=operator, edge_server=edge_server)
                break


def compute_cost(metadata: dict, min: dict, max: dict) -> float:
    """Compute the cost of a given edge server based on its metadata.

    Args:
        metadata (dict): Edge server metadata.
        min (dict): Minimum metadata values.
        max (dict): Maximum metadata values.

    Returns:
        float: Edge server cost.
    """

    reponse_time = metadata["reponse_time"]
    power_consumption = metadata["power_consumption"]

    if max["reponse_time"] - min["reponse_time"] == 0:
        reponse_time_cost = 0
    else:
        reponse_time_cost = (reponse_time - min["reponse_time"]) / (max["reponse_time"] - min["reponse_time"])

    if max["power_consumption"] - min["power_consumption"] == 0:
        power_consumption_cost = 0
    else:
        power_consumption_cost = (power_consumption - min["power_consumption"]) / (max["power_consumption"] - min["power_consumption"])

    return reponse_time_cost + power_consumption_cost


def aels_pa(parameters: dict = {}):
    """Aggregate End-to-End Latency Strategy Power Aware (AELS-PA) modified version proposed in [1].

    [1] da Silva Veith, Alexandre, et al. "Latency-Aware Strategies for Deploying Data Stream Processing
    Applications on Large Cloud-Edge Infrastructure." IEEE transactions on cloud computing (2021).

    Args:
        parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    ds = build_deployment_sequence()

    for operator in ds:
        if operator.server:
            continue

        edge_server_metadata = []
        for edge_server in EdgeServer.all():
            if not has_capacity_to_host(edge_server, operator):
                continue

            server_attrs = {
                "object": edge_server,
                "reponse_time": estimate_processing_and_communication_latency(operator, edge_server),
                "power_consumption": edge_server.power_model_parameters["max_power_consumption"],
            }

            edge_server_metadata.append(server_attrs)

        min_and_max = find_minimum_and_maximum(metadata=edge_server_metadata)

        edge_server_metadata = sorted(
            edge_server_metadata,
            key=lambda m: compute_cost(metadata=m, min=min_and_max["minimum"], max=min_and_max["maximum"]),
        )

        for m in edge_server_metadata:
            edge_server = m["object"]
            place(service=operator, edge_server=edge_server)
            break
