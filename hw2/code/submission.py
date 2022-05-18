from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi_id = (taxi_id + 1) % 2
        other_taxi = env.get_taxi(other_taxi_id)
        gas_0 = env.gas_stations[0]
        gas_1 = env.gas_stations[1]

        cost_taxi_to_gas0 = manhattan_distance(taxi.position, gas_0.destination)
        cost_taxi_to_gas1 = manhattan_distance(taxi.position, gas_1.destination)
        if (cost_taxi_to_gas0 == (taxi.fuel - 1)) or (cost_taxi_to_gas1 == (taxi.fuel - 1)):
            if (cost_taxi_to_gas0 == (taxi.fuel - 1)):
                value = 60 / (cost_taxi_to_gas0 + 1)
            else:
                value = 60 / (cost_taxi_to_gas1 + 1)
            return value


        taxi_pas_selected = None
        man_dist_taxi_to_pass_selected = None

        if env.taxi_is_occupied(taxi_id):
            # our taxi agent is occupied
            taxi_pas_selected = taxi.passenger
            man_dist_taxi_to_pass_selected = 0
        else:
            if env.taxi_is_occupied(other_taxi_id):
                taxi_pas_selected = env.passengers[0]
                man_dist_taxi_to_pass_selected = \
                    manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
            else:
                # both taxies are not occupied
                opt_pas_1 = env.passengers[0]
                man_dist_opt_pas_1_to_dest =\
                    manhattan_distance(opt_pas_1.position, opt_pas_1.destination)
                man_dist_taxi_to_opt_pas_1 = manhattan_distance(taxi.position, opt_pas_1.position)
                opt_pas_2 = env.passengers[1]
                man_dist_opt_pas_2_to_dest =\
                    manhattan_distance(opt_pas_2.position, opt_pas_2.destination)
                man_dist_taxi_to_opt_pas_2 = manhattan_distance(taxi.position, opt_pas_2.position)
                cost_taxi_to_dest_1 = man_dist_taxi_to_opt_pas_1 + man_dist_opt_pas_1_to_dest
                cost_taxi_to_dest_2 = man_dist_taxi_to_opt_pas_2 + man_dist_opt_pas_2_to_dest

                taxi_to_dest_1_value = man_dist_opt_pas_1_to_dest - man_dist_taxi_to_opt_pas_1
                taxi_to_dest_2_value = man_dist_opt_pas_2_to_dest - man_dist_taxi_to_opt_pas_2

                man_dist_other_taxi_to_pas_1 = \
                    manhattan_distance(other_taxi.position, opt_pas_1.destination)
                man_dist_other_taxi_to_pas_2 = \
                    manhattan_distance(other_taxi.position, opt_pas_2.destination)
                taxi_to_dest_1_value_is_greater = taxi_to_dest_1_value > taxi_to_dest_2_value
                opt_p1_closer_to_taxi_than_other_taxi =\
                    (man_dist_other_taxi_to_pas_1 > man_dist_taxi_to_opt_pas_1)
                opt_p2_closer_to_taxi_than_other_taxi =\
                    (man_dist_other_taxi_to_pas_2 > man_dist_taxi_to_opt_pas_2)

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

        debug_prints = True
        if debug_prints:
            print("--------------------------------------------------")
            print(f"taxi.position: {taxi.position}")
            print(f"pas_0.position: {pas_0.position}")
            print(f"man_dist_taxi_to_p0: {man_dist_taxi_to_p0}")
            print(f"man_dist_p0_to_d0: {man_dist_p0_to_d0}")

            if (pas_1 is not None) and (taxi.position == pas_1.position):
                print(f"pas_1.position: {pas_1.position}")

            print(f"man_dist_taxi_to_p1: {man_dist_taxi_to_p1}")
            print(f"man_dist_p1_to_d1: {man_dist_p1_to_d1}")
            print(f"taxi_to_p0_value: {taxi_to_p0_value}")
            print(f"taxi_to_p1_value: {taxi_to_p1_value}")
            print(f"value={value}")
            print("--------------------------------------------------")

            if taxi.position == pas_0.position or ((pas_1 is not None) and (taxi.position == pas_1.position)):
                pass

        # for debug
        debug_prints = False
        if debug_prints:
            if taxi.position == pas_0.position or ((pas_1 is not None)  and (taxi.position == pas_1.position)):
                print(f"taxi_to_p0_value - {taxi_to_p0_value}")
                print(f"taxi_to_p1_value - {taxi_to_p1_value}")
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
