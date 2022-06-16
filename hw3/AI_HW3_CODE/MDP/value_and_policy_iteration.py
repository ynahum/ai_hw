from copy import deepcopy
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    threshold = epsilon * (1-mdp.gamma)/mdp.gamma

    U_current = deepcopy(U_init)
    U_next = deepcopy(U_init)
    #print(f"U_init={U_init}")
    #print(f"board={mdp.board}")
    #print(f"terminal_states={mdp.terminal_states}")
    #print(f"probs={mdp.transition_function}")
    action_to_prob_dict = {'UP':0, 'DOWN':1, 'RIGHT':2, 'LEFT':3}

    iteration_counter = 0
    while True:
        delta = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):

                state = (row, col)

                if mdp.board[row][col] == 'WALL':
                    U_next[row][col] = 'WALL'
                    continue

                if state in mdp.terminal_states:
                    U_next[row][col] = float(mdp.board[row][col])
                else:
                    values = {}
                    next_states = {}
                    for action in mdp.actions:
                        next_states[action] = mdp.step(state, action)
                    for actual_action in mdp.actions:
                        expected_value = 0
                        probs = mdp.transition_function[actual_action]
                        for action in mdp.actions:
                            next_state = next_states[action]
                            next_state_value = U_current[next_state[0]][next_state[1]]
                            next_state_prob = probs[action_to_prob_dict[action]]
                            expected_value +=  next_state_value * next_state_prob
                        values[actual_action] = float(mdp.board[row][col]) + mdp.gamma * expected_value
                    max_action = max(values, key=values.get)
                    max_value = values[max_action]
                    U_next[row][col] = max_value

                abs_diff = abs(U_next[row][col] - U_current[row][col])
                if abs_diff > delta:
                    delta = abs_diff
        if mdp.gamma == 1:
            #print(f"gamma={mdp.gamma}, check delta == 0")
            assert(threshold == 0)
            if delta == threshold:
                break
        else:
            #print(f"gamma={mdp.gamma}, check delta < threshold={threshold}")
            if delta < threshold:
                break
        U_current = deepcopy(U_next)
        iteration_counter +=1

    return U_next
    # ========================


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(U)
    action_to_prob_dict = {'UP':0, 'DOWN':1, 'RIGHT':2, 'LEFT':3}

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
                values[actual_action] = mdp.gamma * expected_value
            max_action = max(values, key=values.get)
            policy[row][col] = max_action
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
