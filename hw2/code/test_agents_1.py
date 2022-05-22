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


def run_agents(agent_names, seed, count_steps, time_limit= 0.5):

    # agent_names = sys.argv
    env = TaxiEnv()

    env.generate(seed)

    for _ in range(count_steps):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i, time_limit)
            end = time.time()
            if end - start > time_limit:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
        if env.done():
            break
    balances = env.get_balances()
    print(balances)
    if balances[0] == balances[1]:
        print('draw')
    else:
        print(f"taxi {agent_names[balances.index(max(balances))]} wins!")


if __name__ == "__main__":
    run_agents(agent_names=["greedy_improved", "greedy"], seed=1, count_steps=100)
    run_agents(agent_names=["greedy_improved", "greedy"], seed=2, count_steps=100)
    run_agents(agent_names=["greedy_improved", "greedy"], seed=3, count_steps=100)
