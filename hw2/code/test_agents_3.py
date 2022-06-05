import random
import time

from TaxiEnv import TaxiEnv
import argparse
import submission
import Agent

agents = {
    "random": Agent.AgentRandom(),
    "greedy": Agent.AgentGreedy(),
    "greedy_improved": submission.AgentGreedyImproved(),
    "minimax": submission.AgentMinimax(),
    "alphabeta": submission.AgentAlphaBeta(),
    "expectimax": submission.AgentExpectimax()
}


def run_game(agent_names, seed, count_steps, time_limit= 0.5):

    # agent_names = sys.argv
    env = TaxiEnv()

    env.generate(seed, 2*count_steps)

    #print('start a game')
    print_game = False
    if print_game:
        print('initial board:')
        env.print()

    for _ in range(count_steps):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i, time_limit)
            end = time.time()
            if end - start > time_limit:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
            if print_game:
                print('taxi ' + str(i) + ' chose ' + op)
                env.print()
        if env.done():
            break
    balances = env.get_balances()
    #print('end the game')
    print(balances)
    if balances[0] == balances[1]:
        return 2
    else:
        return balances.index(max(balances))


def run_games(agents_names, num_of_games=10, seed_offset=0):
    agents_wins = [0,0]
    for seed in range(0+seed_offset,num_of_games+seed_offset):
        print(f"game {seed}:")
        result = run_game(agent_names=agents_names, seed=seed, count_steps=50, time_limit=0.5)
        if result != 2:
            agents_wins[result] += 1

    print("----------------------------------------------------------")
    print(f"taxi {agents_names[0]} won {agents_wins[0]}/{num_of_games} games!")
    print(f"taxi {agents_names[1]} won {agents_wins[1]}/{num_of_games} games!")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    run_games(["alphabeta", "greedy_improved"], num_of_games=100)
    #----------------------------------------------------------
    #taxi alphabeta won 41 / 100 games!
    #taxi greedy_improved won 27 / 100 games!
    ----------------------------------------------------------

    #run_games(["expectimax", "greedy_improved"], num_of_games=100)
    #----------------------------------------------------------
    #taxi expectimax won 49 / 100 games!
    #taxi greedy_improved won 41 / 100 games!
    #----------------------------------------------------------

    #run_games(["minimax", "random"])
    #run_games(["greedy_improved", "random"])

    '''
    if result == 2:
        print('draw')
    else:
        print(f"taxi {agent_names[result]} wins!")
    '''
