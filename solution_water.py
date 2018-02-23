"""Solution to the bonus task #3 about water containers.
"""
from itertools import permutations


def moves(state, capacities, goal_state):
    n = len(capacities)
    moves = []
    for i, j in permutations(range(n),2):
        if state[i] == 0:  # přeléváme z prázdného džbánu
            continue
        if state[j] == capacities[j]:  # přeléváme do plného džbánu
            continue
        diff = capacities[j] - state[j]  # kolik můžeme přelít
        new_state = list(state)  # viz měnitelný seznam vs neměnitelná N-tice
        new_state[j] = state[j] + min(state[i], diff)
        new_state[i] = state[i] - min(state[i], diff)
        moves.append(((i,j),tuple(new_state)))
#     return moves
    # cenová funkce, která upřednostňuje výběr méně vzdálených stavů
    return sorted(moves, key=lambda k: sum(map(lambda i: abs(i[0]-i[1]),zip(k[1],goal_state))))


def solve(initial_state, goal_state, capacities):
    solved = {initial_state}
    def solve_rec(state):
        solved.add(state)
        if state == goal_state:
            return []
        for step, next_state in moves(state, capacities, goal_state):
            if next_state in solved:
                continue
            steps = solve_rec(next_state)
            if steps is not None:
                return [(step, next_state)] + steps
        return None
    return solve_rec(initial_state)

i = (2, 5, 4, 1, 1)
g = (7, 3, 2, 1, 0)
c = (10, 5, 5, 2, 1)

solve(i, g, c)
