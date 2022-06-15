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

    U_current = U_init
    print(f"U_init={U_init}")
    print(f"board={mdp.board}")
    print(f"terminal_states={mdp.terminal_states}")
    while True:
        delta = 0
        U_next = []
        for row in range(mdp.num_row):
            U_next.append([])
            for col in range(mdp.num_col):
                U_next[row].append(0)
                state = (row, col)
                if mdp.board[row][col] == 'WALL':
                    U_next[row][col] = 'WALL'
                    continue
                if mdp.board[row][col] in mdp.terminal_states:
                    U_next[row][col] = float(mdp.board[row][col])
                    continue
                values = {}
                for action in mdp.actions:
                    next_state = mdp.step(state, action)
                    values[action] = U_current[next_state[0]][next_state[1]]
                max_action = max(values, key=values.get)
                max_value = values[max_action]
                U_next[row][col] = float(mdp.board[row][col]) + max_value
                abs_diff = abs(U_next[row][col] - U_current[row][col])
                if abs_diff > delta:
                    delta = abs_diff
        if delta <= threshold:
            break

        # deep copy
        U_current = []
        for row in range(mdp.num_row):
            U_current.append([])
            for col in range(mdp.num_col):
                U_current[row].append(U_next[row][col])
    return U_next
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
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
