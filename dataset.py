"""Contains a script that generates a dataset for the simulations."""


# Importing EdgeSimPy components
from edge_sim_py import *

# Importing helper methods
from simulator.helper_methods import *

# Importing customized EdgeSimPy components
from simulator.custom import *

# Importing Python libraries
from sklearn.cluster import KMeans
import numpy as np
import argparse
import random

# Create a parser for command-line arguments
parser = argparse.ArgumentParser(description="Dataset generator for EdgeSimPy simulations.")

# name of the dataset
parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="Name of the dataset",
    default="dataset",
)

# seed
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    help="Seed for random number generation",
    default=1000,
)

args = parser.parse_args()
dataset_name = args.name
seed = args.seed

print("Dataset name:", dataset_name)
print("Seed:", seed)

random.seed(seed)

# Creating list of map coordinates
MAP_SIZE = 9
NUMBER_OF_LINKS = 208

# Infrastructure specifications
NUMBER_OF_EDGE_SERVERS = 25
LINK_LATENCY_VALUES_IN_SECONDS = [0.01, 0.02]
LINK_BW_VALUES_GBPS = [0.1]  # Gbps

# Application specifications
NUMBER_OF_APPLICATIONS = 8
NUMBER_OF_OPERATORS_BY_APPLICATION = [8, 12, 16]
APPLICATIONS_PROCESSING_LATENCY_SLAS = [0.08, 0.1]  # in seconds

# event_size in bytes
input_event_specifications = [
    {"event_size": 200000, "event_rate": 5000},
    {"event_size": 50000, "event_rate": 5000},
    {"event_size": 10000, "event_rate": 5000},
]

# cpu in MIPS
# mem in bytes
operator_demand_specifications = [
    {"cpu": 0.03, "mem": 7e7},
    {"cpu": 0.06, "mem": 8e7},
    {"cpu": 0.09, "mem": 9e7},
    {"cpu": 0.1, "mem": 1e8},
]

map_coordinates = hexagonal_grid(x_size=MAP_SIZE, y_size=MAP_SIZE)

for coordinates in map_coordinates:
    # Creating the base station object
    base_station = BaseStation()
    base_station.wireless_delay = 0
    base_station.coordinates = coordinates

    # Creating network switch object using the "sample_switch()" generator, which embeds built-in power consumption specs
    network_switch = sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)


delay_bandwidth_product = [(x, y) for x in LINK_LATENCY_VALUES_IN_SECONDS for y in LINK_BW_VALUES_GBPS]

link_distribution = uniform(
    n_items=NUMBER_OF_LINKS,
    valid_values=delay_bandwidth_product,
    shuffle_distribution=True,
)

specs = {}
for spec in link_distribution:
    if spec not in specs:
        specs[spec] = 0
    specs[spec] += 1

link_specifications = []
for (delay, bandwidth), v in specs.items():
    link_specifications.append({"number_of_objects": v, "delay": delay, "bandwidth": bandwidth})

# Creating a partially-connected mesh network topology
partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=link_specifications,
)

### Applications ###


def create_service(app: object, index: int, input_event_size: int, input_event_rate: int, operator_demand: list) -> None:
    operator = Service()
    operator.state = 0
    operator.image_digest = "A"
    operator.flows = []

    operator.input_event_size = input_event_size
    operator.input_event_rate = input_event_rate

    demand = operator_demand.pop()
    operator.mips_demand = demand["cpu"]
    operator.memory_demand = demand["mem"]

    operator._available = False
    operator.level = index

    app.connect_to_service(operator)


def create_linear_application_topology(user: object, spec: dict) -> None:
    """Creates a linear application topology

    Args:
        user (_type_): Application' user
        number_of_operators (int): Number of operators that compose the application
        application_processing_latency_sla (int): Application processing latency SLA
    """

    number_of_operators_by_application = spec["number_of_operators_by_application"]
    application_processing_latency_sla = spec["applications_processing_latency_slas"]
    operator_input_event = spec["operator_input_event"]
    operator_demand = spec["operator_demand"]

    app = Application()
    app.label = f"Linear Application {app.id}"
    app.processing_latency_sla = application_processing_latency_sla

    # Defining the relationship attributes between the user and its new application
    user.applications.append(app)
    app.users.append(user)

    print(f"Creating linear application {app}")
    print(f"\t# of operators: {number_of_operators_by_application}")
    print(f"\tDelay SLA: {application_processing_latency_sla}")

    for index in range(number_of_operators_by_application):
        create_service(app, index, operator_input_event["event_size"], operator_input_event["event_rate"], operator_demand)


print()

kmeans = KMeans(init="k-means++", n_init=100, n_clusters=NUMBER_OF_APPLICATIONS, random_state=seed, max_iter=1000).fit(
    [switch.coordinates for switch in NetworkSwitch.all()]
)

node_clusters = list(kmeans.labels_)

user_positions = []
for centroid in list(kmeans.cluster_centers_):
    node_closest_to_centroid = sorted(
        NetworkSwitch.all(), key=lambda switch: np.linalg.norm(np.array(switch.coordinates) - np.array([centroid[0], centroid[1]]))
    )[0]

    user_positions.append(node_closest_to_centroid.coordinates)

user_positions = random.sample(user_positions, len(user_positions))

print("User positions:", user_positions)

number_of_operators_by_application = uniform(
    n_items=NUMBER_OF_APPLICATIONS,
    valid_values=NUMBER_OF_OPERATORS_BY_APPLICATION,
    shuffle_distribution=True,
)

print("Number of operators by application:", number_of_operators_by_application)

# Defining user delay SLA values
applications_processing_latency_slas = uniform(
    n_items=NUMBER_OF_APPLICATIONS,
    valid_values=APPLICATIONS_PROCESSING_LATENCY_SLAS,
    shuffle_distribution=True,
)

print("Applications Processing Latency SLAs:", applications_processing_latency_slas)

application_types = uniform(
    n_items=NUMBER_OF_APPLICATIONS,
    valid_values=[create_linear_application_topology],
    shuffle_distribution=True,
)

print("Application types")
for type in application_types:
    print(f"\t{type.__name__}")

operator_input_event = uniform(
    n_items=NUMBER_OF_APPLICATIONS,
    valid_values=input_event_specifications,
    shuffle_distribution=True,
)

print("Operator input evet size:", operator_input_event)

operator_demand = uniform(
    n_items=sum(number_of_operators_by_application),
    valid_values=operator_demand_specifications,
    shuffle_distribution=True,
)

print("Operator MIPS:", operator_demand)

print()

for i, create_app in enumerate(application_types):
    user = User()
    user.mobility_model = immobile
    user._set_initial_position(user_positions[i])

    application_spec = {
        "number_of_operators_by_application": number_of_operators_by_application[i],
        "applications_processing_latency_slas": applications_processing_latency_slas[i],
        "operator_input_event": operator_input_event[i],
        "operator_demand": operator_demand,
    }

    create_app(user, application_spec)

### Edge Servers ###


def sgi_rackable_c2112_4g10() -> object:
    """Creates an EdgeServer object according to Ismail et al. [1].

    [1] Leila Ismail and Huned Materwala. 2021. ESCOVE: Energy-SLA-Aware Edge-Cloud Computation Offloading in Vehicular Networks. Sensors 21, 15 (2021).

    Returns:
        edge_server (object): Created EdgeServer object.
    """

    edge_server = EdgeServer()
    # edge_server.model_name = "SGI Rackable C2112-4G10"
    edge_server.model_name = "SGI"
    edge_server.codename = "Model 1"

    # Computational capacity (CPU in cores, RAM memory in megabytes, and disk in megabytes)
    edge_server.cpu = 32
    edge_server.memory = 32768
    edge_server.disk = 1048576
    edge_server.mips = 2750
    edge_server.mips_demand = 0

    # Power-related attributes
    edge_server.power_model_parameters = {
        "static_power_percentage": 265 / 1387,
        "max_power_consumption": 1387,
    }

    return edge_server


def proliant_dl360_gen9() -> object:
    """Creates an EdgeServer object according to Ismail et al. [1].

    [1] Leila Ismail and Huned Materwala. 2021. ESCOVE: Energy-SLA-Aware Edge-Cloud Computation Offloading in Vehicular Networks. Sensors 21, 15 (2021).

    Returns:
        edge_server (object): Created EdgeServer object.
    """

    edge_server = EdgeServer()
    # edge_server.model_name = "HPE ProLiant DL360 Gen9"
    edge_server.model_name = "HPE"
    edge_server.codename = "Model 3"

    # Computational capacity (CPU in cores, RAM memory in megabytes, and disk in megabytes)
    edge_server.cpu = 36
    edge_server.memory = 65536
    edge_server.disk = 1048576
    edge_server.mips = 3000
    edge_server.mips_demand = 0

    # Power-related attributes
    edge_server.power_model_parameters = {
        "static_power_percentage": 45 / 276,
        "max_power_consumption": 276,
    }

    return edge_server


def ar585_f1() -> object:
    """Creates an EdgeServer object according to Ismail et al. [1].

    [1] Leila Ismail and Huned Materwala. 2021. ESCOVE: Energy-SLA-Aware Edge-Cloud Computation Offloading in Vehicular Networks. Sensors 21, 15 (2021).

    Returns:
        edge_server (object): Created EdgeServer object.
    """

    edge_server = EdgeServer()
    # edge_server.model_name = "Acer AR585 F1"
    edge_server.model_name = "Acer"
    edge_server.codename = "Model 2"

    # Computational capacity (CPU in cores, RAM memory in megabytes, and disk in megabytes)
    edge_server.cpu = 48
    edge_server.memory = 65536
    edge_server.disk = 1048576
    edge_server.mips = 3500
    edge_server.mips_demand = 0

    # Power-related attributes
    edge_server.power_model_parameters = {
        "static_power_percentage": 127 / 559,
        "max_power_consumption": 559,
    }

    return edge_server


edge_server_specs = uniform(
    n_items=NUMBER_OF_EDGE_SERVERS,
    valid_values=[sgi_rackable_c2112_4g10, proliant_dl360_gen9, ar585_f1],
    shuffle_distribution=True,
)

kmeans = KMeans(init="k-means++", n_init=100, n_clusters=NUMBER_OF_EDGE_SERVERS, random_state=seed, max_iter=1000).fit(
    [switch.coordinates for switch in NetworkSwitch.all()]
)

node_clusters = list(kmeans.labels_)

edge_server_coordinates = []
for centroid in list(kmeans.cluster_centers_):
    node_closest_to_centroid = sorted(
        NetworkSwitch.all(), key=lambda switch: np.linalg.norm(np.array(switch.coordinates) - np.array([centroid[0], centroid[1]]))
    )[0]

    edge_server_coordinates.append(node_closest_to_centroid.coordinates)

edge_server_coordinates = random.sample(edge_server_coordinates, len(edge_server_coordinates))

for spec in edge_server_specs:
    # Creating the edge server object
    edge_server = spec()

    # Specifying the edge server's power model
    edge_server.power_model = CustomLinearServerPowerModel

    # Connecting the edge server to its base station
    base_station = BaseStation.find_by(attribute_name="coordinates", attribute_value=edge_server_coordinates[edge_server.id - 1])
    base_station._connect_to_edge_server(edge_server=edge_server)

### Sink Operators Placement ###

max_surrounding_coordinates = 6
min_surrounding_coordinates = 2

for application in Application.all():
    # Get sensor's location
    sensor_location = application.users[0].base_station.network_switch.coordinates

    # Get edge servers surrounding the sensor
    edge_servers = []
    for edge_server in EdgeServer.all():
        edge_server_coordinates = edge_server.base_station.network_switch.coordinates

        if sensor_location == edge_server_coordinates:
            continue

        # consider switchs around the sensor until reach the max_surrounding_coordinates and min_surrounding_coordinates
        if (
            edge_server_coordinates[0] >= sensor_location[0] - max_surrounding_coordinates
            and edge_server_coordinates[0] <= sensor_location[0] + max_surrounding_coordinates
            and edge_server_coordinates[1] >= sensor_location[1] - max_surrounding_coordinates / 2
            and edge_server_coordinates[1] <= sensor_location[1] + max_surrounding_coordinates / 2
        ):
            if (
                edge_server_coordinates[0] >= sensor_location[0] - min_surrounding_coordinates
                and edge_server_coordinates[0] <= sensor_location[0] + min_surrounding_coordinates
                and edge_server_coordinates[1] >= sensor_location[1] - min_surrounding_coordinates / 2
                and edge_server_coordinates[1] <= sensor_location[1] + min_surrounding_coordinates / 2
            ):
                continue

            edge_servers.append(edge_server)

    if len(edge_servers) == 0:
        raise Exception("No edge servers found for application's sink {}".format(application))

    # Select one at random
    edge_server = random.choice(edge_servers)

    place(service=application.services[-1], edge_server=edge_server)


### Registry ###

# As we are not considering the virtualization of container images, we can assume that each container image is a single layer with size 0
container_image_specifications = [
    {
        "name": "image",
        "tag": "latest",
        "digest": "A",
        "layers": [
            {
                "digest": "A",
                "size": 0,
            },
        ],
        "layers_digests": ["A"],
    },
]

container_registry_specifications = [
    {
        "number_of_objects": 1,
        "cpu_demand": 0,
        "memory_demand": 0,
        "images": [
            {"name": "image", "tag": "latest"},
        ],
    }
]

# Parsing the specifications for container images and container registries
container_registries = create_container_registries(
    container_registry_specifications=container_registry_specifications,
    container_image_specifications=container_image_specifications,
)

# Defining the initial placement for container images and container registries
for container_registry in container_registries:
    provision_container_registry(container_registry, random.choice(EdgeServer.all()))

# Loading custom EdgeSimPy components and methods
Application._to_dict = application_to_dict
EdgeServer._to_dict = edge_server_to_dict
Service._to_dict = service_to_dict

# Export scenario to dict
dataset = ComponentManager.export_scenario(file_name=dataset_name)

# Printing topology
display_dataset(output_filename=f"datasets/{dataset_name}")

print()

for app in Application.all():
    print(f"-> {app.label} ")

    for svc in app.services:
        print(
            "\tOperator {}. MIPS {:0.4f}. Mem {:0.4f}. Event Rate {:0.4f}. Event Size {:0.4f}".format(
                svc.id, svc.mips_demand, svc.memory_demand, svc.input_event_rate, svc.input_event_size
            )
        )

    print()

print()
print("##########################")
print("#### DATASET ANALYSIS ####")
print("##########################")
print()

applications_by_sla = {}
applications_by_chain_size = {}
number_of_apps = Application.count()
number_of_operators_by_application = sum([len(app.services) for app in Application.all()])

for app in Application.all():
    if len(app.services) not in applications_by_chain_size:
        applications_by_chain_size[len(app.services)] = 0
    applications_by_chain_size[len(app.services)] += 1

    if app.processing_latency_sla not in applications_by_sla:
        applications_by_sla[app.processing_latency_sla] = 0
    applications_by_sla[app.processing_latency_sla] += 1

print("==== SENSORS POSITIONS ====")

print(f"Sensors: {User.count()}")
for user in User.all():
    print(f"\t{user.applications[0]}. Sensor {user.id}. Position: {user.coordinates}.")

print()
print("==== SINK POSITIONS ====")

print(f"Sinks: {Application.count()}")
for app in Application.all():
    sink = app.services[-1]
    print(f"\t{app}. Sink {sink.id}. {sink.server}. Position: {sink.server.coordinates}.")

print()
print("==== APPLICATIONS OVERVIEW ====")
print()

print(f"Applications: {Application.count()}")
print(f"\t# of Operators: {number_of_operators_by_application}")
print(f"\t# of Applications by Chain Size: {applications_by_chain_size}")
print(f"\t# of SLAs by Chain Size: {applications_by_sla}")

# Calculating the infrastructure occupation and information about the services
edge_server_mips_capacity = 0
edge_server_memory_capacity = 0
service_mips_demand = 0
service_memory_demand = 0

for edge_server in EdgeServer.all():
    edge_server_mips_capacity += edge_server.mips
    edge_server_memory_capacity += edge_server.memory

for service in Service.all():
    service_mips_demand += service.mips_demand * service.input_event_rate
    service_memory_demand += (
        service.memory_demand + (service.input_event_rate * service.input_event_size)
    ) / 1e6  # convert bytes to megabytes

overall_mips_occupation = round((service_mips_demand / edge_server_mips_capacity) * 100, 1)
overall_memory_occupation = round((service_memory_demand / edge_server_memory_capacity) * 100, 1)

print()
print("==== INFRASTRUCTURE OCCUPATION OVERVIEW ====")
print()

print(f"Edge Servers: {EdgeServer.count()}")
print(f"\tCPU Capacity: {edge_server_mips_capacity}")
print(f"\tRAM Capacity: {edge_server_memory_capacity}")
print()

print(f"Services: {Service.count()}")
print(f"\tCPU Demand: {service_mips_demand}")
print(f"\tRAM Demand: {service_memory_demand}")
print()

print(f"Overall Occupation")
print(f"\tCPU: {service_mips_demand}. Percentage: {overall_mips_occupation}%")
print(f"\tRAM: {service_memory_demand}. Percentage: {overall_memory_occupation}%")
print()

topology = Topology.first()


# Calculating the network delay between users and edge servers (useful for defining reasonable delay SLAs)
users = []
for user in User.all():
    user_metadata = {"object": user, "all_delays": []}

    edge_servers = []
    for edge_server in EdgeServer.all():
        path = nx.shortest_path(
            G=Topology.first(), source=user.base_station.network_switch, target=edge_server.network_switch, weight="delay"
        )
        user_metadata["all_delays"].append(Topology.first().calculate_path_delay(path=path))

    user_metadata["min_delay"] = min(user_metadata["all_delays"])
    user_metadata["max_delay"] = max(user_metadata["all_delays"])
    user_metadata["avg_delay"] = sum(user_metadata["all_delays"]) / len(user_metadata["all_delays"])
    user_metadata["delays"] = {}

    for delay in sorted(list(set(user_metadata["all_delays"]))):
        user_metadata["delays"][delay] = user_metadata["all_delays"].count(delay)

    users.append(user_metadata)

print("==== NETWORK DISTANCE (DELAY) BETWEEN USERS AND EDGE SERVERS ====\n")

for user_metadata in users:
    user_attrs = {
        "object": user_metadata["object"],
        "sla": user_metadata["object"].applications[0].processing_latency_sla,
        "min": user_metadata["min_delay"],
        "max": user_metadata["max_delay"],
        "avg": round(user_metadata["avg_delay"]),
        "delays": user_metadata["delays"],
    }

    print(f"{user_attrs}")

    if user_attrs["min"] > user_attrs["sla"]:
        print(f"\n\nWARNING: {user_attrs['object']} delay SLA is not achievable!\n\n")

print()

max_aggregated_latency = -1
for node_u in topology.nodes():
    for node_v in topology.nodes():
        if node_u != node_v:
            shortest_path_length = nx.shortest_path_length(topology, source=node_u, target=node_v, weight="delay")
            max_aggregated_latency = max(max_aggregated_latency, shortest_path_length)

print("Maximum aggregated delay between nodes:", max_aggregated_latency)

max_aggregated_delay_between_sensor_and_edge_server = -1
min_aggregated_delay_between_sensor_and_edge_server = -1

for app in Application.all():
    source = app.users[0].base_station.network_switch

    for edge_server in EdgeServer.all():
        sink = edge_server.network_switch

        shortest_path_length = nx.shortest_path_length(topology, source=source, target=sink, weight="delay")

        if min_aggregated_delay_between_sensor_and_edge_server == -1:
            min_aggregated_delay_between_sensor_and_edge_server = shortest_path_length

        min_aggregated_delay_between_sensor_and_edge_server = min(
            min_aggregated_delay_between_sensor_and_edge_server, shortest_path_length
        )

        max_aggregated_delay_between_sensor_and_edge_server = max(
            max_aggregated_delay_between_sensor_and_edge_server, shortest_path_length
        )

print("Minimum aggregated delay between sensor and edge server:", min_aggregated_delay_between_sensor_and_edge_server)
print("Maximum aggregated delay between sensor and edge server:", max_aggregated_delay_between_sensor_and_edge_server)
