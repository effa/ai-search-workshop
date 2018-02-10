"""Reseni ukolu z notebooku.
"""
from collections import deque
from itertools import product
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


def all_plans_of_length(n):
    return [''.join(plan) for plan in product('lfr', repeat=n)]


def is_correct(initial_state, plan):
    state = initial_state
    for action in plan:
        state = move(state, action)
    return is_goal(state)


def generate_and_test(initial_state):
    for plan in all_plans_of_length(initial_state.n-1):
        if is_correct(initial_state, plan):
            return plan


# Nasledujici algoritmus neprochazi vrcholy v korektnim DFS poradi,
# protoze pridava do zasobniku vsechny nasledniky zkoumaneho stavu najednou
# a nikdy podruhe (pro zachovani liearni velikosti zasobniku).
# Pri hledani cesty k cilovemu stavu to nicemu nevadi, ale je
# dobre vedet, ze to neni presne DFS a pro nektere jine aplikace
# (napr. detekce orientovanych cyklu) by byl kod potreba upravit.
def dfs(initial_state):
    stack = [initial_state]
    plans = {initial_state: ''}
    log_search_step(None, stack, plans)
    while stack:
        state = stack.pop()
        if is_goal(state):
            log_search_step(state, stack, plans)
            return plans[state]
        for action in reversed(actions(state)):
            next_state = move(state, action)
            if next_state not in plans:  # jeste jsme ho nevideli
                stack.append(next_state)
                plans[next_state] = plans[state] + action
        log_search_step(state, stack, plans)


# Tohle je "temer-korektni" iterativni implementace, ktera objevuje vzdy jedineho
# naslednika a stavy oznacuje za prozkoumane az po objeveni vsech nasledniku.
# Takto je prezentovan napr. v Algoritmech a datovych strukturach I.
# Pro optimalni casovou slozitost by se do zasobniku mela ukladat informace,
# ktereho naslednika budeme objevovat priste, cimz se vyhneme opakovanemu
# hledani prvniho neobjevenoho naslednika (to by vsak vyzadovalo upravit funkci
# pro vizualizaci algoritmu).
def dfs_correct(initial_state):
    stack = [initial_state]
    plans = {initial_state: ''}
    log_search_step(None, stack, plans)
    while stack:
        state = stack.pop()
        if is_goal(state):
            log_search_step(state, stack, plans)
            return plans[state]
        explored = True
        for action in actions(state):
            next_state = move(state, action)
            if next_state not in plans:
                stack.append(state)
                stack.append(next_state)
                plans[next_state] = plans[state] + action
                explored = False
                break
        log_search_step(state if explored else None, stack, plans)


# Varianta DFS s nasobnym ukladanim na zasobnik, ktera dosahuje korektniho
# poradi za cenu trochu komplikovanejsiho kodu a vetsi pametove slozitosti
# (zasobnik muze mit velikost poctu vsech hran).
def dfs_multistore(initial_state):
    stack = [initial_state]
    plans = {initial_state: ''}
    explored = set()
    log_search_step(None, stack, plans)
    while stack:
        state = stack.pop()
        if is_goal(state):
            log_search_step(state, stack, plans)
            return plans[state]
        # kontrola, zda jsme cil uz neprozkoumali
        if state in explored:
            continue
        explored.add(state)
        for action in reversed(actions(state)):
            next_state = move(state, action)
            # nasobne ukladani (neprozkoumane stavy se muzou na zasobniku
            # objevit vickrat)
            if next_state not in explored:
                stack.append(next_state)
                plans[next_state] = plans[state] + action
        log_search_step(state, stack, plans)


def bfs(initial_state):
    queue = deque([initial_state])
    plans = {initial_state: ''}
    log_search_step(None, queue, plans)
    while queue:
        state = queue.popleft()
        # Cilovy test by u BFS slo provadet uz pri zarazovani do fronty.
        if is_goal(state):
            log_search_step(state, queue, plans)
            return plans[state]
        for action in actions(state):
            next_state = move(state, action)
            if next_state not in plans:
                queue.append(next_state)
                plans[next_state] = plans[state] + action
        log_search_step(state, queue, plans)


ACTION_COSTS = {'l': 3, 'f': 2, 'r': 3}

def ucs(initial_state):
    fringe = {initial_state}
    costs = {initial_state: 0}
    plans = {initial_state: ''}
    log_search_step(None, fringe, plans, costs)
    while fringe:
        # Vybirame stav z okraje s nejnizsi cenou:
        state = min(fringe, key=lambda s: costs[s])
        fringe.remove(state)
        if is_goal(state):
            log_search_step(state, fringe, plans, costs)
            return plans[state]
        for action in actions(state):
            next_state = move(state, action)
            new_cost = costs[state] + ACTION_COSTS[action]
            old_cost = costs.get(next_state, inf)
            if new_cost < old_cost:
                fringe.add(next_state)
                costs[next_state] = new_cost
                plans[next_state] = plans[state] + action
        log_search_step(state, fringe, plans, costs)


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
    log_search_step(None, fringe, plans, costs, heuristic)
    while fringe:
        state = min(fringe, key=lambda s: costs[s] + heuristic[s])
        fringe.remove(state)
        if is_goal(state):
            log_search_step(state, fringe, plans, costs, heuristic)
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
        log_search_step(state, fringe, plans, costs, heuristic)


# ----------------------------------------------------------------------------

# Obecne schema stromoveho prohledavani.
# Je parametrizovane typem okraje (Fringe), ktery
# popisuje strategii pro vyber stavu k prozkoumani.
def tree_search(initial_state, Fringe=set):
    fringe = Fringe([initial_state])
    plans = {initial_state: ''}
    log_search_step(None, fringe, plans)
    while fringe:
        # Vyber jednoho stavu z okraje.
        state = fringe.pop()
        # Pokud je tento stav cilovy, muzeme prohledavani ukoncit.
        if is_goal(state):
            log_search_step(state, fringe, plans)
            return plans[state]
        # Pokud neni, expandujeme tento stav, tj. pridame na okraj
        # vsechny jeho nasledniky.
        for action in actions(state):
            next_state = move(state, action)
            plans[next_state] = plans[state] + action
            fringe.add(next_state)
        log_search_step(state, fringe, plans)


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
