""" Contains a server power model definition."""


class CustomLinearServerPowerModel:
    """Server power model based on [1], which assumes a linear correlation between a server's power consumption and demand.

    [1] Anton Beloglazov, and Rajkumar Buyya, "Optimal Online Deterministic Algorithms and Adaptive Heuristics for Energy and
    Performance Efficient Dynamic Consolidation of Virtual Machines in Cloud Data Centers", Concurrency and Computation: Practice
    and Experience (CCPE), Volume 24, Issue 13, Pages: 1397-1420, John Wiley & Sons, Ltd, New York, USA, 2012

    [2] Loukopoulos, Thanasis, Nikos Tziritas, Maria Koziri, George Stamoulis, Samee U. Khan, Cheng-Zhong Xu, and Albert Y. Zomaya. "Data stream processing at network edges." In 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pp. 657-665. IEEE, 2018.
    """

    @classmethod
    def get_power_consumption(cls, device: object) -> float:
        """Gets the power consumption of a server.

        Args:
            device (object): Server whose power consumption will be computed.

        Returns:
            power_consumption (float): Server's power consumption.
        """
        if device.active:
            static_power = (
                device.power_model_parameters["static_power_percentage"] * device.power_model_parameters["max_power_consumption"]
            )
            constant = (device.power_model_parameters["max_power_consumption"] - static_power) / 100

            # Computing sever power consumption based on MIPS demand and capacity as in [2]
            demand = device.mips_demand
            capacity = device.mips
            utilization = demand / capacity

            power_consumption = static_power + constant * utilization * 100

        else:
            power_consumption = 0

        return power_consumption
