"""Reseni ukolu z notebooku.
"""
from collections import deque
from math import inf
from search import is_goal, actions, move, log_search_step


def greedy_search(initial_state):
    state = initial_state
    plan = ''
    while not is_goal(state):
        available_actions = actions(state)
        if not available_actions:
            break # Failed to find a path.
        # Choose the first available action (greedy choice).
        action = available_actions[0]
        state = move(state, action)
        plan += action
    # Return a complete or a partial path.
    return plan


# Nasledujici algoritmus neprochazi vrcholy v korektnim DFS poradi,
# protoze pridava do zasobniku vsechny zkoumaneho stavu najednou
# a nikdy podruhe (pro zachovani liearni velikosti zasobniku).
# Pri hledani cesty k cilovemu stavu to nicemu nevadi, ale je
# dobre vedet, ze to neni presne DFS a pro nektere jine aplikace
# (napr. detekce orientovanych cyklu) by byl kod potreba upravit.
def dfs(initial_state):
    stack = [initial_state]
    plans = {initial_state: ''}
    while stack:
        state = stack.pop()
        if is_goal(state):
            log_search_step(state, stack)
            return plans[state]
        for action in reversed(actions(state)):
            next_state = move(state, action)
            if next_state not in plans:  # jeste jsme ho nevideli
                stack.append(next_state)
                plans[next_state] = plans[state] + action
        log_search_step(state, stack)


def bfs(initial_state):
    if is_goal(initial_state):
        return ''
    queue = deque([initial_state])
    plans = {initial_state: ''}
    while queue:
        state = queue.popleft()
        for action in actions(state):
            next_state = move(state, action)
            if next_state not in plans:
                queue.append(next_state)
                plans[next_state] = plans[state] + action
            # Cilovy test lze provadet uz pri zarazovani do fronty.
            if is_goal(next_state):
                log_search_step(state, queue)
                return plans[next_state]
        log_search_step(state, queue)


ACTION_COSTS = {'l': 3, 'f': 2, 'r': 3}

def ucs(initial_state):
    fringe = {initial_state}
    costs = {initial_state: 0}
    plans = {initial_state: ''}
    while fringe:
        # Vybirame stav z okraje s nejnizsi cenou:
        state = min(fringe, key=lambda s: costs[s])
        fringe.remove(state)
        if is_goal(state):
            log_search_step(state, fringe, costs)
            return plans[state]
        for action in actions(state):
            next_state = move(state, action)
            new_cost = costs[state] + ACTION_COSTS[action]
            old_cost = costs.get(next_state, inf)
            if new_cost < old_cost:
                fringe.add(next_state)
                costs[next_state] = new_cost
                plans[next_state] = plans[state] + action
        log_search_step(state, fringe, costs)


def heuristic_distance(state):
    # Cilovy radek ma hodnotu 0, radek pod nim 1, atd.
    vertical_distance = state.spaceship.row
    # Jaka by byla cena, kdyby raketka mohla letet porad rovne.
    return vertical_distance * ACTION_COSTS['f']


def a_star(initial_state):
    fringe = {initial_state}
    costs = {initial_state: 0}
    heuristic = {initial_state: heuristic_distance(initial_state)}
    plans = {initial_state: ''}
    while fringe:
        state = min(fringe, key=lambda s: costs[s] + heuristic[s])
        fringe.remove(state)
        if is_goal(state):
            log_search_step(state, fringe, costs, heuristic)
            return plans[state]
        for action in actions(state):
            next_state = move(state, action)
            new_cost = costs[state] + ACTION_COSTS[action]
            old_cost = costs.get(next_state, inf)
            if new_cost < old_cost:
                fringe.add(next_state)
                costs[next_state] = new_cost
                plans[next_state] = plans[state] + action
            if next_state not in heuristic:
                heuristic[next_state] = heuristic_distance(next_state)
        log_search_step(state, fringe, costs, heuristic)


# ----------------------------------------------------------------------------

# Obecne schema stromoveho prohledavani.
# Je parametrizovane typem okraje (Fringe), ktery
# popisuje strategii pro vyber stavu k prozkoumani.
def tree_search(initial_state, Fringe=set):
    fringe = Fringe([initial_state])
    plans = {initial_state: ''}
    while fringe:
        # Vyber jednoho stavu z okraje.
        state = fringe.pop()
        # Pokud je tento stav cilovy, muzeme prohledavani ukoncit.
        if is_goal(state):
            log_search_step(state, fringe)
            return plans[state]
        # Pokud neni, expandujeme tento stav, tj. pridame na okraj
        # vsechny jeho nasledniky.
        for action in actions(state):
            next_state = move(state, action)
            plans[next_state] = plans[state] + action
            fringe.add(next_state)
        log_search_step(state, fringe)


# Rekurzivni DFS pro stromy (nehlida zacykleni)
def recursive_dfs(state):
    if is_goal(state):
        return [state]
    for action in actions(state):
        next_state = move(state, action)
        path = recursive_dfs(next_state)
        if path:
            return [state] + path
    return None  # no path found


# Rekurzivni DFS pro grafy (hlida zacykleni)
def recursive_graph_dfs(start_state):
    explored = set()
    def dfs(state):
        explored.add(state)
        if is_goal(state):
            return [state]
        for action in actions(state):
            next_state = move(state, action)
            if next_state in explored:
                continue
            path = dfs(next_state)
            if path:
                return [state] + path
        return None  # no path found
    return dfs(start_state)
