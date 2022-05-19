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

        cost_taxi_to_gas0 = manhattan_distance(taxi.position, gas_0.position)
        cost_taxi_to_gas1 = manhattan_distance(taxi.position, gas_1.position)
        if (cost_taxi_to_gas0 == (taxi.fuel - 1)) or (cost_taxi_to_gas1 == (taxi.fuel - 1)):
            if cost_taxi_to_gas0 == (taxi.fuel - 1):
                value = 60 / (cost_taxi_to_gas0 + 1)
            else:
                value = 60 / (cost_taxi_to_gas1 + 1)
            return value


        taxi_pas_selected = None
        md_taxi_to_pass_selected = None
        md_pas_to_dest_selected = None
        value_taxi_to_dest_selected = None
        
        if env.taxi_is_occupied(taxi_id):
            # our taxi agent is occupied
            taxi_pas_selected = taxi.passenger
            md_taxi_to_pass_selected = 0
            md_pas_to_dest_selected = \
                manhattan_distance(taxi.position, taxi_pas_selected.destination)
            value_taxi_to_dest_selected = 10 - md_pas_to_dest_selected - md_taxi_to_pass_selected
        else:
            if env.taxi_is_occupied(other_taxi_id):
                taxi_pas_selected = env.passengers[0]
                md_taxi_to_pass_selected = \
                    manhattan_distance(taxi.position, taxi_pas_selected.position)
                md_pas_to_dest_selected = \
                    manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
                value_taxi_to_dest_selected = md_pas_to_dest_selected - md_taxi_to_pass_selected
            else:
                # both taxies are not occupied
                opt_pas_1 = env.passengers[0]
                md_opt_pas_1_to_dest =\
                    manhattan_distance(opt_pas_1.position, opt_pas_1.destination)
                md_taxi_to_opt_pas_1 = manhattan_distance(taxi.position, opt_pas_1.position)
                cost_taxi_to_dest_1 = md_taxi_to_opt_pas_1 + md_opt_pas_1_to_dest
                taxi_to_dest_1_value = md_opt_pas_1_to_dest - md_taxi_to_opt_pas_1
                md_other_taxi_to_pas_1 = \
                    manhattan_distance(other_taxi.position, opt_pas_1.destination)

                opt_pas_2 = env.passengers[1]
                md_opt_pas_2_to_dest =\
                    manhattan_distance(opt_pas_2.position, opt_pas_2.destination)
                md_taxi_to_opt_pas_2 = manhattan_distance(taxi.position, opt_pas_2.position)
                cost_taxi_to_dest_2 = md_taxi_to_opt_pas_2 + md_opt_pas_2_to_dest
                taxi_to_dest_2_value = md_opt_pas_2_to_dest - md_taxi_to_opt_pas_2

                md_other_taxi_to_pas_2 = \
                    manhattan_distance(other_taxi.position, opt_pas_2.destination)

                taxi_to_dest_1_value_is_greater = taxi_to_dest_1_value > taxi_to_dest_2_value
                opt_p1_closer_to_taxi_than_other_taxi =\
                    (md_other_taxi_to_pas_1 > md_taxi_to_opt_pas_1)
                opt_p2_closer_to_taxi_than_other_taxi =\
                    (md_other_taxi_to_pas_2 > md_taxi_to_opt_pas_2)

                if opt_p1_closer_to_taxi_than_other_taxi and opt_p2_closer_to_taxi_than_other_taxi:
                    if taxi_to_dest_1_value_is_greater:
                        select_opt_p1 = True
                    else:
                        select_opt_p1 = False
                elif opt_p1_closer_to_taxi_than_other_taxi:
                    select_opt_p1 = True
                elif opt_p2_closer_to_taxi_than_other_taxi:
                    select_opt_p1 = False
                else:
                    # TODO: other agent has both passengers closer than our agent
                    # put a constant value?
                    print("************************TODO: other agent is closer to both")
                    #currently we select opt2 even though we cannot win the other agent
                    select_opt_p1 = False

                if select_opt_p1:
                    taxi_pas_selected = opt_pas_1
                    md_taxi_to_pass_selected = md_taxi_to_opt_pas_1
                    md_pas_to_dest_selected = md_opt_pas_1_to_dest
                    value_taxi_to_dest_selected = taxi_to_dest_1_value
                else:
                    taxi_pas_selected = opt_pas_2
                    md_taxi_to_pass_selected = md_taxi_to_opt_pas_2
                    md_pas_to_dest_selected = md_opt_pas_2_to_dest
                    value_taxi_to_dest_selected = taxi_to_dest_2_value

        total_value = value_taxi_to_dest_selected + taxi.cash

        debug_prints = True
        if debug_prints:
            print("--------------------------------------------------")
            print(f"taxi.position: {taxi.position}")
            print(f"taxi_pas_selected.position: {taxi_pas_selected.position}")
            print(f"taxi is occupied: {env.taxi_is_occupied(taxi_id)}")

            print(f"md_taxi_to_pass_selected: {md_taxi_to_pass_selected}")
            print(f"md_pas_to_dest_selected: {md_pas_to_dest_selected}")
            print(f"value_taxi_to_dest_selected: {value_taxi_to_dest_selected}")
            print(f"taxi.cash: {taxi.cash}")
            print(f"total_value: {total_value}")

            print("--------------------------------------------------")

        return value_taxi_to_dest_selected +


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
