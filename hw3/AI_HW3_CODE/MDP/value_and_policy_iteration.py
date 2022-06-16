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
                values[actual_action] = float(mdp.board[row][col]) + mdp.gamma * expected_value
            max_action = max(values, key=values.get)
            policy[row][col] = max_action
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    num_of_states = mdp.num_row * mdp.num_col

    row_col_to_id = lambda row, col, num_col: row * num_col + col
    #id_to_row_col = lambda id, num_col: (id/num_col, id % num_col)
    action_to_prob_dict = {'UP':0, 'DOWN':1, 'RIGHT':2, 'LEFT':3}

    I = np.eye(num_of_states)
    P = np.zeros((num_of_states, num_of_states))
    r = np.zeros((num_of_states, 1))
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)
            state_index = row_col_to_id(row, col, mdp.num_col)

            if mdp.board[row][col] == 'WALL':
                # for inversion
                P[state_index][state_index] = 1
                continue

            # r vector setup
            r[state_index] = float(mdp.board[row][col])

            if state in mdp.terminal_states:
                continue

            policy_action = policy[row][col]
            next_state_probs = mdp.transition_function[policy_action]
            for action in mdp.actions:
                next_state = mdp.step(state, action)
                next_state_index = row_col_to_id(next_state[0], next_state[1], mdp.num_col)
                P[state_index][next_state_index] += next_state_probs[action_to_prob_dict[action]]

    U_vec = np.linalg.inv(I - mdp.gamma * P) @ r
    U = deepcopy(policy)
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == 'WALL':
                U[row][col] = 'WALL'
                continue
            state_index = row_col_to_id(row,col,mdp.num_col)
            U[row][col] = float(U_vec[state_index])

    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    current_policy = deepcopy(policy_init)
    while True:
        # policy evaluation
        U_pi = policy_evaluation(mdp, current_policy)

        # policy improvement (greedy relative to current U_pi)
        next_policy = get_policy(mdp,U_pi)

        if current_policy == next_policy:
            optimal_policy = deepcopy(current_policy)
            break

        current_policy = deepcopy(next_policy)

    return optimal_policy

    # ========================
