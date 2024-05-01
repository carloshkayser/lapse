from simulator.helper_methods import calculate_placement_processing_latency

from edge_sim_py import Application, EdgeServer, NetworkFlow


def compute_processing_latency():
    processing_latency = {}
    processing_latency_sla_violation = {}
    processing_latency_sla_violation_per_sla = {}
    processing_latency_sla_violation_per_app_chain_size = {}

    avg_event_processing_latency = {}

    for app in Application.all():
        processing_latency[app.id] = calculate_placement_processing_latency(app)
        avg_event_processing_latency[app.id] = processing_latency[app.id] / app.services[0].input_event_rate

        # Check processing latency SLA violation
        if avg_event_processing_latency[app.id] > app.processing_latency_sla:
            processing_latency_sla_violation[app.id] = True
            app.processing_latency_sla_violation = True

            if len(app.services) not in processing_latency_sla_violation_per_app_chain_size.keys():
                processing_latency_sla_violation_per_app_chain_size[len(app.services)] = 0
            processing_latency_sla_violation_per_app_chain_size[len(app.services)] += 1

            if app.processing_latency_sla not in processing_latency_sla_violation_per_sla.keys():
                processing_latency_sla_violation_per_sla[app.processing_latency_sla] = 0
            processing_latency_sla_violation_per_sla[app.processing_latency_sla] += 1

        else:
            processing_latency_sla_violation[app.id] = False
            app.processing_latency_sla_violation = False

    data = {}
    data["avg_event_processing_latency"] = avg_event_processing_latency
    data["processing_latency"] = processing_latency
    data["processing_latency_sla_violation"] = processing_latency_sla_violation
    data["overall_avg_event_processing_latency"] = sum(avg_event_processing_latency.values())
    data["overall_processing_latency"] = sum(processing_latency.values())
    data["number_of_processing_latency_sla_violation"] = sum(processing_latency_sla_violation.values())
    data["number_of_processing_latency_sla_violation_percentage"] = (
        sum(processing_latency_sla_violation.values()) / Application.count() * 100
    )

    data["chain_size"] = []
    for chain_size in set([len(app.services) for app in Application.all()]):
        data["chain_size"].append(
            {
                "chain_size": chain_size,
                "delay_sla_violations": processing_latency_sla_violation_per_app_chain_size.get(chain_size, None),
            }
        )

    data["processing_latency_sla"] = []
    for processing_latency_sla in set([app.processing_latency_sla for app in Application.all()]):
        data["processing_latency_sla"].append(
            {
                "processing_latency_sla": processing_latency_sla,
                "processing_latency_sla_violations": processing_latency_sla_violation_per_sla.get(processing_latency_sla, None),
            }
        )

    return data


def compute_edge_server_metrics():
    # Declaring infrastructure metrics
    overloaded_edge_servers = 0
    overall_occupation = 0
    occupation_per_model = {}
    overall_power_consumption = 0
    power_consumption_per_server_model = {}
    active_servers_per_model = {}
    max_power_consumption_possible = 0

    # Collecting infrastructure metrics
    for edge_server in EdgeServer.all():
        # Overall Occupation (using just CPU because it is the most constrained resource)
        capacity = edge_server.mips
        demand = edge_server.mips_demand
        edge_server_occupation = demand / capacity * 100

        overall_occupation += edge_server_occupation

        # Overall Consumption
        overall_power_consumption += edge_server.get_power_consumption()
        max_power_consumption_possible += edge_server.power_model_parameters["max_power_consumption"]

        # Number of overloaded edge servers
        free_cpu = edge_server.cpu - edge_server.cpu_demand
        free_mips = edge_server.mips - edge_server.mips_demand
        free_memory = edge_server.memory - edge_server.memory_demand
        free_disk = edge_server.disk - edge_server.disk_demand
        if free_mips < 0 or free_cpu < 0 or free_memory < 0 or free_disk < 0:
            overloaded_edge_servers += 1

        # Power consumption per Server Model
        if edge_server.codename not in power_consumption_per_server_model.keys():
            power_consumption_per_server_model[edge_server.codename] = []
        power_consumption_per_server_model[edge_server.codename].append(edge_server.get_power_consumption())

        # Occupation per Server Model
        if edge_server.codename not in occupation_per_model.keys():
            occupation_per_model[edge_server.codename] = []
        occupation_per_model[edge_server.codename].append(edge_server_occupation)

    # Aggregating overall metrics
    overall_occupation = overall_occupation / EdgeServer.count()

    for codename in occupation_per_model.keys():
        active_servers_per_model[codename] = len([item for item in occupation_per_model[codename] if item > 0])
        occupation_per_model[codename] = sum(occupation_per_model[codename]) / len(occupation_per_model[codename])

    data = {}
    data["overall_occupation"] = overall_occupation
    data["overall_power_consumption"] = overall_power_consumption
    data["overall_power_consumption_percentage"] = overall_power_consumption / max_power_consumption_possible * 100
    data["overloaded_edge_servers"] = overloaded_edge_servers

    data["model"] = []
    codenames = sorted(set([server.codename for server in EdgeServer.all()]))
    for model in codenames:
        data["model"].append(
            {
                "codename": model,
                "occupation": occupation_per_model[model],
                "power_consumption": sum(power_consumption_per_server_model[model]),
                "active_servers": active_servers_per_model[model],
            }
        )

    return data


def compute_application_path_size():
    path_size = {}
    path_size_by_sla = {}
    path_size_by_chain_size = {}

    for app in Application.all():
        path_size[app.id] = 0

        if app.processing_latency_sla not in path_size_by_sla.keys():
            path_size_by_sla[app.processing_latency_sla] = []

        if len(app.services) not in path_size_by_chain_size.keys():
            path_size_by_chain_size[len(app.services)] = []

        path_size_by_sla_ = 0
        path_size_by_chain_size_ = 0

        for service in app.services:
            if len(service.flows) > 0:
                for flow in service.flows:
                    path_size[app.id] += len(flow.network_links)
                    path_size_by_sla_ += len(flow.network_links)
                    path_size_by_chain_size_ += len(flow.network_links)

        path_size_by_sla[app.processing_latency_sla].append(path_size_by_sla_)
        path_size_by_chain_size[len(app.services)].append(path_size_by_chain_size_)

    return {"path_size": path_size, "path_size_by_sla": path_size_by_sla, "path_size_by_chain_size": path_size_by_chain_size}


def compute_network_metrics():
    # Declaring network metrics
    bandwidth_available_for_each_flow = {}
    bandwidth_available_for_each_flow_percentage = {}

    for flow in NetworkFlow.all():
        bandwidth_available_for_each_flow[flow.id] = min(flow.bandwidth.values())

        min_bandwidth = float("inf")
        for link in flow.network_links:
            min_bandwidth = min(min_bandwidth, link.bandwidth)

        bandwidth_available_for_each_flow_percentage[flow.id] = min(flow.bandwidth.values()) / min_bandwidth * 100

    return {
        "bandwidth_available_for_each_flow": bandwidth_available_for_each_flow,
        "bandwidth_available_for_each_flow_percentage": bandwidth_available_for_each_flow_percentage,
    }


def topology_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """

    self.model.network_flow_scheduling_algorithm(topology=self.model.topology, flows=NetworkFlow.all())

    application_metrics = compute_processing_latency()
    edge_server_metrics = compute_edge_server_metrics()
    application_path_size = compute_application_path_size()
    network_metrics = compute_network_metrics()

    metrics = {
        **application_metrics,
        **edge_server_metrics,
        **application_path_size,
        **network_metrics,
    }

    return metrics
