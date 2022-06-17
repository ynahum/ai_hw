""" You can import what ever you want """
from copy import deepcopy
import numpy as np
from value_and_policy_iteration import policy_iteration


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    num_of_different_policies = 1
    states_optimal_actions = deepcopy(U)
    action_to_prob_dict = {'UP':0, 'DOWN':1, 'RIGHT':2, 'LEFT':3}
    action_to_arrow = {'UP':'\u2191', 'DOWN':'\u2193', 'RIGHT':'\u2192', 'LEFT':'\u2190'}

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):

            state = (row, col)

            if mdp.board[row][col] == 'WALL' or state in mdp.terminal_states:
                continue

            values = {}
            next_states = {}
            for action in mdp.actions:
                next_states[action] = mdp.step(state, action)
            for actual_action in mdp.actions:
                expected_value = 0
                probs = mdp.transition_function[actual_action]
                for action in mdp.actions:
                    next_state = next_states[action]
                    next_state_value = U[next_state[0]][next_state[1]]
                    next_state_prob = probs[action_to_prob_dict[action]]
                    expected_value += next_state_value * next_state_prob
                values[actual_action] = float(mdp.board[row][col]) + mdp.gamma * expected_value
            max_action = max(values, key=values.get)
            max_value = values[max_action]
            arrows = [action_to_arrow[k] for k, v in values.items() if v == max_value]
            action_str = ""
            for c in arrows:
                action_str += c
            states_optimal_actions[row][col] = action_str
            num_of_different_policies *= len(states_optimal_actions[row][col])
    mdp.print_policy(states_optimal_actions)
    return num_of_different_policies
    # ========================


def set_reward(mdp, r):
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)
            if mdp.board[row][col] == 'WALL' or state in mdp.terminal_states:
                continue
            mdp.board[row][col] = str(r)


def find_r_list(mdp, a, a_policy, b, b_policy, epsilon=0.01):
    if b-a < epsilon:
        return [a]

    mid = (b+a)/2
    #print(f"a={a}, b={b}, mid={mid}")
    set_reward(mdp, mid)
    mid_policy = policy_iteration(mdp, a_policy)
    if mid_policy == a_policy:
        #print(f"mid_policy == a_policy")
        return find_r_list(mdp, mid, mid_policy, b, b_policy)
    if mid_policy == b_policy:
        #print(f"mid_policy == b_policy")
        return find_r_list(mdp, a, a_policy, mid, mid_policy)

    # mid is different from both a and b
    #print(f"call merge a={a}, b={b}, mid={mid}")
    low_list = find_r_list(mdp, a, a_policy, mid, mid_policy)
    #print(f"low_list={low_list}")
    high_list = find_r_list(mdp, mid, mid_policy, b, b_policy)
    #print(f"High_list={high_list}")
    merged_list = low_list + high_list
    return merged_list



def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]

    b = 10
    set_reward(mdp, b)
    b_policy = policy_iteration(mdp, policy)
    a = -10
    set_reward(mdp, a)
    a_policy = policy_iteration(mdp, policy)
    r_list = find_r_list(mdp, a, a_policy, b, b_policy)
    #print(f"r_list={r_list}")

    for i, r in enumerate(r_list):

        print(f"The MDP's optimal policy for", end = " ")
        if i == 0:
            print(f"r <= {r_list[i]}")
        elif i == len(r_list)-1:
            print(f"{r_list[i]} < r")
        else:
            print(f"{r_list[i-1]} < r <={r_list[i]}")
        set_reward(mdp, r_list[i])
        mdp.print_policy(policy_iteration(mdp, policy))

    # ========================
