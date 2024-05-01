"""Contains a script that executes a list of experiments automatically."""

# Importing EdgeSimPy components
from edge_sim_py import *

# Importing placement strategies
from simulator.algorithms import *

# Importing customized EdgeSimPy components
from simulator.custom import *

# Importing Python libraries
import matplotlib.pyplot as plt
import argparse
import logging
import random
import time

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(seed_value: int, algorithm: str, dataset: str, parameters: dict = {}):
    """Executes a simulation experiment.

    Args:
        seed_value (int): Seed value.
        algorithm (str): Algorithm that will be executed.
        dataset (str): Dataset file.
        parameters (dict, optional): Algorithm parameters. Defaults to {}.

    Raises:
        Exception: Service was not placed.
    """

    # Setting a seed value to enable reproducibility
    random.seed(seed_value)

    logs_directory = f"logs/algorithm={algorithm};dataset={dataset.replace('datasets/', '').replace('.json', '')};"

    # Creating a Simulator object
    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=lambda model: model.schedule.steps == 1,
        resource_management_algorithm=eval(algorithm),
        resource_management_algorithm_parameters=parameters,
        network_flow_scheduling_algorithm=custom_equal_share,
        user_defined_functions=[
            immobile,
            CustomLinearServerPowerModel,
        ],
        dump_interval=1,
        logs_directory=logs_directory,
    )

    # Loading custom EdgeSimPy components and methods
    Topology.collect = topology_collect
    EdgeServer.collect = edge_server_collect
    NetworkFlow.collect = network_flow_collect
    NetworkFlow.step = network_flow_step
    User.collect = user_collect
    User.step = user_step

    # Loading the dataset
    simulator.initialize(input_file=dataset)

    start_time = time.time()

    # Executing the simulation
    simulator.run_model()

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    # Check if every services was placed
    for app in Application.all():
        for service in app.services:
            if not service.server:
                raise Exception(f"Service {service.id} was not placed!")


if __name__ == "__main__":
    # Parsing named arguments from the command line
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default="1")
    parser.add_argument("--dataset", "-d", help="Dataset file")
    parser.add_argument("--algorithm", "-a", help="Algorithm that will be executed")
    parser.add_argument("--debug", help="Turn debug mode on", default=False, action="store_true")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    parameters = {}
    main(seed_value=int(args.seed), algorithm=args.algorithm, dataset=args.dataset, parameters=parameters)
