from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        debug_prints = True

        taxi = env.get_taxi(taxi_id)
        if debug_prints:
            print(f"taxi.position: {taxi.position}")
            print(f"taxi is occupied: {env.taxi_is_occupied(taxi_id)}")
            print(f"taxi.fuel: {taxi.fuel}")

        other_taxi_id = (taxi_id + 1) % 2
        other_taxi = env.get_taxi(other_taxi_id)

        cash_value = taxi.cash - other_taxi.cash
        fuel_value = taxi.fuel - other_taxi.fuel
        total_value = cash_value + fuel_value

        survival_value = 0

        gas_0 = env.gas_stations[0]
        gas_1 = env.gas_stations[1]

        cost_taxi_to_gas0 = manhattan_distance(taxi.position, gas_0.position)
        cost_taxi_to_gas1 = manhattan_distance(taxi.position, gas_1.position)
        if (cost_taxi_to_gas0 > (taxi.fuel-2)) and (cost_taxi_to_gas1 > (taxi.fuel-2)):
            survival_value -= 10
            if debug_prints:
                print(f"cost_taxi_to_gas0: {cost_taxi_to_gas0}")
                print(f"cost_taxi_to_gas1: {cost_taxi_to_gas1}")
                print(f"pre total_value: {total_value}")
            total_value += survival_value
            if debug_prints:
                print(f"post total_value: {total_value}")
                print("--------------------------------------------------")
            return total_value

        taxi_pas_selected = None
        md_taxi_to_pass_selected = None
        md_pas_to_dest_selected = None
        cash_at_drop = None
        remaining_cost = None
        value_taxi_to_dest_selected = None

        select_dest = 0

        if env.taxi_is_occupied(taxi_id):
            # our taxi agent is occupied
            taxi_pas_selected = taxi.passenger
            md_taxi_to_pass_selected = 0
            md_pas_to_dest_selected = \
                manhattan_distance(taxi.position, taxi_pas_selected.destination)
            cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
            remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
            value_taxi_to_dest_selected = cash_at_drop - remaining_cost
        else:
            if env.taxi_is_occupied(other_taxi_id):
                taxi_pas_selected = env.passengers[0]
                md_taxi_to_pass_selected = \
                    manhattan_distance(taxi.position, taxi_pas_selected.position)
                md_pas_to_dest_selected = \
                    manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
                cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
                remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
                value_taxi_to_dest_selected = cash_at_drop - remaining_cost
            else:
                # both taxies are not occupied
                opt_pas_1 = env.passengers[0]
                md_opt_pas_1_to_dest =\
                    manhattan_distance(opt_pas_1.position, opt_pas_1.destination)
                md_taxi_to_opt_pas_1 = manhattan_distance(taxi.position, opt_pas_1.position)
                cost_taxi_to_dest_1 = md_taxi_to_opt_pas_1 + md_opt_pas_1_to_dest
                taxi_to_dest_1_value = (md_opt_pas_1_to_dest - md_taxi_to_opt_pas_1)/(cost_taxi_to_dest_1+1)
                md_other_taxi_to_pas_1 = \
                    manhattan_distance(other_taxi.position, opt_pas_1.destination)

                opt_pas_2 = env.passengers[1]
                md_opt_pas_2_to_dest =\
                    manhattan_distance(opt_pas_2.position, opt_pas_2.destination)
                md_taxi_to_opt_pas_2 = manhattan_distance(taxi.position, opt_pas_2.position)
                cost_taxi_to_dest_2 = md_taxi_to_opt_pas_2 + md_opt_pas_2_to_dest
                taxi_to_dest_2_value = (md_opt_pas_2_to_dest - md_taxi_to_opt_pas_2)/(cost_taxi_to_dest_2+1)
                md_other_taxi_to_pas_2 = \
                    manhattan_distance(other_taxi.position, opt_pas_2.destination)

                taxi_to_dest_1_value_is_greater = taxi_to_dest_1_value > taxi_to_dest_2_value
                opt_p1_closer_to_taxi_than_other_taxi =\
                    (md_other_taxi_to_pas_1 > md_taxi_to_opt_pas_1)
                opt_p2_closer_to_taxi_than_other_taxi =\
                    (md_other_taxi_to_pas_2 > md_taxi_to_opt_pas_2)

                # 1 - dest1, 2 - dest2
                select_dest = 0
                if opt_p1_closer_to_taxi_than_other_taxi and opt_p2_closer_to_taxi_than_other_taxi:
                    if taxi_to_dest_1_value_is_greater:
                        select_dest = 1
                    else:
                        select_dest = 2
                elif opt_p1_closer_to_taxi_than_other_taxi:
                    select_dest = 1
                elif opt_p2_closer_to_taxi_than_other_taxi:
                    select_dest = 2
                else:
                    select_dest = 0
                    print("*** no selection ** value is 0")

                if select_dest == 1:
                    taxi_pas_selected = opt_pas_1
                    md_taxi_to_pass_selected = md_taxi_to_opt_pas_1
                    md_pas_to_dest_selected = md_opt_pas_1_to_dest
                    cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
                    remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
                    value_taxi_to_dest_selected = cash_at_drop - remaining_cost
                elif select_dest == 2:
                    taxi_pas_selected = opt_pas_2
                    md_taxi_to_pass_selected = md_taxi_to_opt_pas_2
                    md_pas_to_dest_selected = md_opt_pas_2_to_dest
                    cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
                    remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
                    value_taxi_to_dest_selected = cash_at_drop - remaining_cost
                else:
                    value_taxi_to_dest_selected = 0

        total_value += value_taxi_to_dest_selected

        if debug_prints:
            if select_dest != 0:
                print(f"taxi_pas_selected.position: {taxi_pas_selected.position}")

            print(f"md_taxi_to_pass_selected: {md_taxi_to_pass_selected}")
            print(f"md_pas_to_dest_selected: {md_pas_to_dest_selected}")

            print(f"value_taxi_to_dest_selected: {value_taxi_to_dest_selected}")
            print(f"cash_value: {cash_value}")
            print(f"fuel_value: {fuel_value}")
            print(f"total_value: {total_value}")

            print("--------------------------------------------------")

        return total_value


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
