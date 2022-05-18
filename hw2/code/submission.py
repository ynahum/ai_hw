from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        pas_0 = env.passengers[0]
        other_taxi_id = (taxi_id + 1) % 2
        other_taxi = env.get_taxi(other_taxi_id)
        pas_1 = env.passengers[1]

        gas_0 = env.gas_stations[0]
        gas_1 = env.gas_stations[1]

        man_dist_p0_to_d0 = manhattan_distance(pas_0.position, pas_0.destination)
        man_dist_p1_to_d1 = manhattan_distance(pas_1.position, pas_1.destination)

        man_dist_taxi_to_p0 = manhattan_distance(taxi.position, pas_0.position)
        man_dist_taxi_to_p1 = manhattan_distance(taxi.position, pas_1.position)

        cost_taxi_to_d0 = man_dist_taxi_to_p0 + man_dist_p0_to_d0
        cost_taxi_to_d1 = man_dist_taxi_to_p1 + man_dist_p1_to_d1

        man_dist_other_taxi_to_p0 = manhattan_distance(other_taxi.position, pas_0.position)
        man_dist_other_taxi_to_p1 = manhattan_distance(other_taxi.position, pas_1.position)

        taxi_to_p0_value = man_dist_p0_to_d0 - man_dist_taxi_to_p0
        taxi_to_p1_value = man_dist_p1_to_d1 - man_dist_taxi_to_p1

        #other_taxi_to_p0_value = man_dist_p0_to_d0 - man_dist_other_taxi_to_p0
        #other_taxi_to_p1_value = man_dist_p1_to_d1 - man_dist_other_taxi_to_p1

        p0_value_is_greater = taxi_to_p0_value > taxi_to_p1_value
        p0_closer_to_taxi_than_other_taxi = (man_dist_other_taxi_to_p0 > man_dist_taxi_to_p0)
        p1_closer_to_taxi_than_other_taxi = (man_dist_other_taxi_to_p1 > man_dist_taxi_to_p1)

        value = 0

        gas_fill_needed = False
        if cost_taxi_to_d0 > taxi.fuel and cost_taxi_to_d1 > taxi.fuel:
            gas_fill_needed = True

        # if gas fill is needed we set high value when the MD to gas station is small (inverse)
        if gas_fill_needed:
            man_dist_taxi_to_gas_0 = manhattan_distance(taxi.position, gas_0.position)
            man_dist_taxi_to_gas_1 = manhattan_distance(taxi.position, gas_1.position)
            if man_dist_taxi_to_gas_0 < man_dist_taxi_to_gas_1:
                value = 60 / man_dist_taxi_to_gas_0
            else:
                value = 60 / man_dist_taxi_to_gas_1
            return value

        # we can select which one to select
        if p0_closer_to_taxi_than_other_taxi and p1_closer_to_taxi_than_other_taxi:
            if p0_value_is_greater:
                value = taxi_to_p0_value
            else:
                value = taxi_to_p1_value
        elif p0_closer_to_taxi_than_other_taxi:
            value = taxi_to_p0_value
        elif p1_closer_to_taxi_than_other_taxi:
            value = taxi_to_p1_value

        return value


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
