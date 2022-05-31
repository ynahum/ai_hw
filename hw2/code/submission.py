from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import time
import math


class DebugGlobals:
    def __init__(self):
        self.greedy_improved_debug_prints = False
        self.minimax_debug_prints = False
        self.ignore_time_limit = False
        self.max_depth = 20


MyDebug = DebugGlobals()


def greedy_improved_h(env: TaxiEnv, taxi_id: int):

    taxi = env.get_taxi(taxi_id)
    if MyDebug.greedy_improved_debug_prints:
        print(f"taxi.position: {taxi.position}")
        print(f"taxi is occupied: {env.taxi_is_occupied(taxi_id)}")
        print(f"taxi.fuel: {taxi.fuel}")

    other_taxi_id = (taxi_id + 1) % 2
    other_taxi = env.get_taxi(other_taxi_id)

    cash_value = 2 * (taxi.cash - other_taxi.cash)

    # check if we can win by not filling more fuel in some gas station
    # as we have cash and the other taxi stopped with less cash
    other_taxi_is_occupied = env.taxi_is_occupied(other_taxi_id)
    other_taxi_cannot_get_cash = (other_taxi.fuel == 0) or ((not other_taxi_is_occupied) and other_taxi.fuel < 3)
    win_condition = ((cash_value > 6) and (other_taxi.fuel <= 7)) or \
                    ((cash_value > 0) and other_taxi_cannot_get_cash)
    if win_condition:
        check_for_gas_refill = False
        fuel_value = 0
    else:
        check_for_gas_refill = True
        fuel_value = 2 * (taxi.fuel - other_taxi.fuel)

    survival_value = 0
    if check_for_gas_refill:
        if taxi.fuel <= 7:
            gas_0 = env.gas_stations[0]
            gas_1 = env.gas_stations[1]
            cost_taxi_to_gas0 = manhattan_distance(taxi.position, gas_0.position)
            cost_taxi_to_gas1 = manhattan_distance(taxi.position, gas_1.position)
            max_cost_taxi_to_gas = max(cost_taxi_to_gas0, cost_taxi_to_gas1)
            if max_cost_taxi_to_gas <= (taxi.fuel - 3):
                survival_value = 10 * (6 - max_cost_taxi_to_gas)
                if MyDebug.greedy_improved_debug_prints:
                    print(f"cost_taxi_to_gas0: {cost_taxi_to_gas0}")
                    print(f"cost_taxi_to_gas1: {cost_taxi_to_gas1}")

    taxi_pas_selected = None
    md_taxi_to_pass_selected = None
    md_pas_to_dest_selected = None
    cash_at_drop = None
    remaining_cost = None
    value_taxi_to_dest_selected = None

    if env.taxi_is_occupied(taxi_id):
        # our taxi agent is occupied
        taxi_pas_selected = taxi.passenger
        md_taxi_to_pass_selected = 0
        md_pas_to_dest_selected = manhattan_distance(taxi.position, taxi_pas_selected.destination)
        cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
        remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
        value_taxi_to_dest_selected = cash_at_drop - remaining_cost
    else:
        if other_taxi_is_occupied:
            taxi_pas_selected = env.passengers[0]
            md_taxi_to_pass_selected = manhattan_distance(taxi.position, taxi_pas_selected.position)
            md_pas_to_dest_selected = manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
            cash_at_drop = 2 * manhattan_distance(taxi_pas_selected.position, taxi_pas_selected.destination)
            remaining_cost = md_pas_to_dest_selected + md_taxi_to_pass_selected
            value_taxi_to_dest_selected = cash_at_drop - remaining_cost
        else:
            # both taxies are not occupied
            opt_pas_1 = env.passengers[0]
            md_opt_pas_1_to_dest = manhattan_distance(opt_pas_1.position, opt_pas_1.destination)
            md_taxi_to_opt_pas_1 = manhattan_distance(taxi.position, opt_pas_1.position)
            cost_taxi_to_dest_1 = md_taxi_to_opt_pas_1 + md_opt_pas_1_to_dest
            taxi_to_dest_1_value = (md_opt_pas_1_to_dest - md_taxi_to_opt_pas_1) / (cost_taxi_to_dest_1 + 1)
            md_other_taxi_to_pas_1 = manhattan_distance(other_taxi.position, opt_pas_1.destination)

            opt_pas_2 = env.passengers[1]
            md_opt_pas_2_to_dest = manhattan_distance(opt_pas_2.position, opt_pas_2.destination)
            md_taxi_to_opt_pas_2 = manhattan_distance(taxi.position, opt_pas_2.position)
            cost_taxi_to_dest_2 = md_taxi_to_opt_pas_2 + md_opt_pas_2_to_dest
            taxi_to_dest_2_value = (md_opt_pas_2_to_dest - md_taxi_to_opt_pas_2) / (cost_taxi_to_dest_2 + 1)
            md_other_taxi_to_pas_2 = manhattan_distance(other_taxi.position, opt_pas_2.destination)

            taxi_to_dest_1_value_is_greater = taxi_to_dest_1_value > taxi_to_dest_2_value
            opt_p1_closer_to_taxi_than_other_taxi = (md_other_taxi_to_pas_1 > md_taxi_to_opt_pas_1)
            opt_p2_closer_to_taxi_than_other_taxi = (md_other_taxi_to_pas_2 > md_taxi_to_opt_pas_2)

            # 1 - dest1, 2 - dest2
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
                if MyDebug.greedy_improved_debug_prints:
                    print("no winning path over other taxi")
                if taxi_to_dest_1_value_is_greater:
                    select_dest = 1
                else:
                    select_dest = 2

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
                print("error:shouldn't select other than 1 or 2 options to other")

    total_value = cash_value + fuel_value + survival_value + value_taxi_to_dest_selected

    if MyDebug.greedy_improved_debug_prints:
        print(f"win_condition: {win_condition}")
        print(f"taxi_pas_selected.position: {taxi_pas_selected.position}")

        print(f"md_taxi_to_pass_selected: {md_taxi_to_pass_selected}")
        print(f"md_pas_to_dest_selected: {md_pas_to_dest_selected}")

        print(f"survival_value: {survival_value}")
        print(f"cash_value: {cash_value}")
        print(f"fuel_value: {fuel_value}")
        print(f"value_taxi_to_dest_selected: {value_taxi_to_dest_selected}")
        print(f"total_value: {total_value}")

        print("--------------------------------------------------")

    return total_value


def rb_alpha_beta(env: TaxiEnv, taxi_id, h, is_max_turn, depth, alpha=None, beta=None):
    other_taxi_id = (taxi_id + 1) % 2
    if env.done():
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi(other_taxi_id)
        # as infinite reward in case we win
        return math.inf * (taxi.cash - other_taxi.cash)
    if depth == 0:
        return h(env, taxi_id)

    playing_taxi_id = taxi_id
    if not is_max_turn:
        playing_taxi_id = other_taxi_id

    operators = env.get_legal_operators(playing_taxi_id)
    children = [env.clone() for _ in operators]

    if is_max_turn:
        max_value = -math.inf
        for child, op in zip(children, operators):
            child.apply_operator(playing_taxi_id, op)
            if MyDebug.minimax_debug_prints:
                print(f"call minimax test op={op} from operators={operators}"
                      f" depth={depth} playing_taxi_id={playing_taxi_id}")
            child_minimax = rb_alpha_beta(child, taxi_id, h, not is_max_turn, depth - 1, alpha, beta)
            max_value = max([max_value, child_minimax])
            if alpha is not None:
                alpha = max([max_value, alpha])
            if beta is not None:
                if max_value >= beta:
                    max_value = math.inf
        if MyDebug.minimax_debug_prints:
            print(f"select maximized max_value={max_value} depth={depth} playing_taxi_id={playing_taxi_id}")
        return max_value
    else:
        min_value = math.inf
        for child, op in zip(children, operators):
            child.apply_operator(playing_taxi_id, op)
            if MyDebug.minimax_debug_prints:
                print(f"call minimax test op={op} from operators={operators}"
                      f" depth={depth} playing_taxi_id={playing_taxi_id}")
            child_minimax = rb_alpha_beta(child, taxi_id, h, not is_max_turn, depth - 1, alpha, beta)
            min_value = min([min_value, child_minimax])
            if beta is not None:
                beta = min([min_value, beta])
            if alpha is not None:
                if min_value <= alpha:
                    min_value = -math.inf
        if MyDebug.minimax_debug_prints:
            print(f"select minimized min_value={min_value} depth={depth} playing_taxi_id={playing_taxi_id}")
        return min_value


def taxi_run_step(env: TaxiEnv, taxi_id, time_limit, alpha_beta_pruning=False):
    start = time.time()

    operators = env.get_legal_operators(taxi_id)
    children = [env.clone() for _ in operators]
    for child, op in zip(children, operators):
        child.apply_operator(taxi_id, op)

    depth = 1
    while True:
        if MyDebug.minimax_debug_prints:
            print(f"depth={depth} call minimax from run_step")
            print("------------------------------")
        children_minimax = []
        for child, op in zip(children, operators):
            if MyDebug.minimax_debug_prints:
                print(f"call minimax test op={op} from operators={operators}")
            if alpha_beta_pruning:
                child_minimax = rb_alpha_beta(child, taxi_id, greedy_improved_h, False, depth - 1,alpha=-math.inf, beta=math.inf)
            else:
                child_minimax = rb_alpha_beta(child, taxi_id, greedy_improved_h, False, depth - 1)
            children_minimax.append(child_minimax)
        max_minimax = max(children_minimax)
        index_selected = children_minimax.index(max_minimax)
        if MyDebug.minimax_debug_prints:
            print(f"final maximize max_minimax={max_minimax} children_minimax={children_minimax}"
                  " index_selected={index_selected} op selected={operators[index_selected]}")

        end = time.time()
        process_time = end - start
        time_threshold = (time_limit/4)
        if MyDebug.minimax_debug_prints:
            time_threshold = (time_limit / 5)
            print(f"process_time={process_time}, time_threshold={time_threshold}")
            print("------------------------------")
        if not MyDebug.ignore_time_limit and process_time > time_threshold:
            break
        depth += 1
        if depth >= MyDebug.max_depth or depth > env.num_steps:
            break

    return operators[index_selected]


class AgentGreedyImproved(AgentGreedy):
    # section a : 3
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return greedy_improved_h(env, taxi_id)


class AgentMinimax(Agent):
    # section b : 1
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        return taxi_run_step(env, taxi_id, time_limit)


class AgentAlphaBeta(Agent):
    # section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        return taxi_run_step(env, agent_id, time_limit, alpha_beta_pruning=True)


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
