from collections import namedtuple, defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
from itertools import accumulate
import ipywidgets as widgets
import matplotlib.pyplot as plt


Position = namedtuple('Position', ['row', 'col'])

class State:
    def __init__(self, world, spaceship):
        # TODO: enforce immutability
        # TODO: allow for state[position]
        self.spaceship = spaceship
        self.world = world  # no spaceship there!
        # assume immutable state, precompute properties
        self.n = max(pos.row for pos in world) + 1
        self.m = max(pos.col for pos in world) + 1
        # cache wormholes
        wormholes = defaultdict(list)
        for pos, field in world.items():
            if field in 'WXYZ':
                wormholes[field].append(pos)
        self.wormhole_positions = []
        self.wormhole_end = {}
        for letter, positions in wormholes.items():
            if len(positions) != 2:
                continue
                #raise ValueError('Expects exectly 2 wormholes of given type.')
            self.wormhole_positions.extend(positions)
            self.wormhole_end[positions[0]] = positions[1]
            self.wormhole_end[positions[1]] = positions[0]

    @property
    def x(self):
        col = self.spaceship.col
        return col + 0.5

    @property
    def y(self):
        row = self.spaceship.row
        return self.n - row - 0.5

    def is_goal(self):
        return self.spaceship.row == 0

    def show(self):
        show_state(self)

    def is_wormhole(self, position):
        return position in self.wormhole_positions

    def get_wormhole_end(self, position):
        return self.wormhole_end[position]

    def __eq__(self, other):
        return self.world == other.world and self.spaceship == other.spaceship

    def __hash__(self):
        return hash(self.spaceship)

    def __str__(self):
        # TODO: add __repr__ for debugging (State('''...''')
        fields = [[
            self.world[(Position(row, col))]
            for col in range(self.m)]
                for row in range(self.n)]
        fields[self.spaceship.row][self.spaceship.col] = 'S'
        text = '\n'.join('|{inside}|'.format(inside='|'.join(row)) for row in fields)
        #text = text.replace(' ', '.')
        return text

    def __repr__(self):
        x = self.spaceship.col
        y = self.m - self.spaceship.row - 1
        return '{x}-{y}'.format(x=x,y=y)

    def _repr_png_(self):
        # Used by jupyter to render the output
        # (http://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)
        # Assumes matplotlib-inline mode
        self.show()


def parse_state(text):
    rows = text.strip().split('||')
    fields = [row.strip('|').split('|') for row in rows]
    world = {}
    spaceship = None
    for r, row in enumerate(fields):
        for c, field in enumerate(row):
            world[Position(r, c)] = field
            if field == 'S':
                spaceship = Position(r, c)
                world[spaceship] = ' '
    return State(world, spaceship)


def move_spaceship(spaceship, action):
    dy = -1
    dx = -1 if action == 'l' else 0 if action == 'f' else 1
    new_spaceship = Position(
        row=spaceship.row + dy,
        col=spaceship.col + dx)
    return new_spaceship


def move(state, action):
    # TODO: thow an error if the resulting state is dead (?)
    # (or allow for representation of dead states)
    world = state.world.copy()
    spaceship = move_spaceship(state.spaceship, action)
    if state.is_wormhole(spaceship):
        spaceship = state.get_wormhole_end(spaceship)
#     world[state.spaceship] = ' '
#     world[spaceship] = 'S'
    return State(world, spaceship)


def actions(state):
    """Return actions that don't lead to dead state.
    """
    return [
        a for a in 'lfr'
        if state.world.get(move_spaceship(state.spaceship, a), 'A') != 'A']


def is_goal(state):
    # DISQ: Rozmyslet, zde je pro zacatecniky vhodnejsi
    # methoda state.is_goal() nebo funkce is_goal(state).
    return state.is_goal()


# --------------------------------------------------------------------------
# TODO: Factor out visualization into its own module

IMAGES = {
    name: plt.imread('img/{name}.png'.format(name=name))
    for name in [
        'spaceship', 'asteroid', 'wormhole', 'wormhole2',
        'wormhole3', 'wormhole4', 'background-blue-goal']}


def show_state(state):
    # Requires %matplotlib inline mode to work
    width, height = state.m, state.n
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.grid(True)
    ax.set_xticks(range(width+1))
    ax.set_yticks(range(height+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.patch.set_facecolor('black')

    def put_img(name, pos):
        x = pos.col
        y = height - pos.row - 1
        ax.imshow(IMAGES[name], extent=[x, x+1, y, y+1])

    for x in range(width):
        ax.imshow(
            IMAGES['background-blue-goal'],
            extent=[x, x+1, height-1, height])
    for pos, field in state.world.items():
        if field == 'A':
            put_img('asteroid', pos)
        if field == 'W':
            put_img('wormhole', pos)
        if field == 'X':
            put_img('wormhole2', pos)
        if field == 'Y':
            put_img('wormhole3', pos)
        if field == 'Z':
            put_img('wormhole4', pos)
    put_img('spaceship', state.spaceship)


def draw_move(start, end, style='o-', color='tab:green'):
    xs = [start.x, end.x]
    ys = [start.y, end.y]
    if end.is_wormhole(end.spaceship):
        midpos = end.get_wormhole_end(end.spaceship)
        xs.insert(1, midpos.col + 0.5)
        ys.insert(1, end.n - midpos.row - 0.5 )
    plt.plot(xs, ys, style, color=color, markevery=[0, len(xs)-1])


def show_plan(state, plan, interactive=False):
    plan = plan or ''
    if type(plan) != str:
        raise ValueError("Plan musi byt retezec akci.")
    if any(a not in 'lfr' for a in plan):
        raise ValueError("Povolene akce jsou 'l', 'f', 'r'.")
    states = list(accumulate([state] + list(plan), move))
    path = [s.spaceship for s in states]

    def show_plan_step(step):
        state = states[step]
        state.show()
        message = "Plan: '{plan}'".format(plan=plan)
        plt.text(0, -0.4, message,
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center',
            bbox={'facecolor': 'tab:green'})
        for i in range(len(states) - 1):
            draw_move(states[i], states[i+1], 'o-', 'tab:green')

    if interactive:
        step = widgets.IntSlider(
            min=0, max=len(plan),
            value=0,
            description='Krok')
        return widgets.interact(show_plan_step, step=step)
    else:
        show_plan_step(step=0)




# --------------------------------------------------------------------------
# TODO: Factor out debugging tools into own module

def get_root(parents):
    for child, parent in parents.items():
        if parent is None:
            return child

#def costs_as_state(costs, heuristic=None):
#    heuristic = heuristic or (lambda x: 0)
#    world = next(iter(costs.keys())).world.copy()
#    for state in costs.keys():
#        total_cost = costs[state] + heuristic(state)
#        world[state.spaceship] = str(total_cost)
#    return State(world)


def show_search_tree(tree, fringe, explored, costs=None):
    state = explored[-1] if explored else get_root(tree)
    show_state(state)
    for child, parent in tree.items():
        if not parent:
            continue
        draw_move(parent, child, style='o-', color='tab:blue')
    # mark explored
    xs = [s.spaceship.col + 0.5 for s in explored]
    ys = [state.n - s.spaceship.row - 0.5 for s in explored]
    if costs:
        labels = [
            '{order}\nc={cost}'.format(order=i, cost=costs[s])
            for i, s in enumerate(explored, start=1)]
    else:
        labels = [str(i) for i in range(1,len(explored)+1)]
    for label, x, y in zip(labels, xs, ys):
        #plt.annotate(label, xy=(x, y))
        plt.text(
            x, y, label,
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor': 'tab:blue', 'pad': 5, 'alpha': 0.95})
    # mark fringe
    xs = [s.spaceship.col + 0.5 for s in fringe]
    ys = [state.n - s.spaceship.row - 0.5 for s in fringe]
    if costs:
        labels = [
            '{order}\nc={cost}'.format(order='?', cost=costs[s])
            for s in fringe]
        for label, x, y in zip(labels, xs, ys):
            plt.text(
                x, y, label,
                horizontalalignment='center',
                verticalalignment='center',
                bbox={'facecolor': 'tab:red', 'pad': 5})
    else:
        plt.plot(xs, ys, 's', color='tab:red')


def create_tree_search_widget(explored_states, trees, fringes,
                              costs=None, interactive=False):
    if not trees:
        print('Zadne stromy k zobrazeni.')
        return
    def show_search_tree_at(step):
        tree = trees[step]
        fringe = fringes[step]
        #print('fringe at', step, 'is', str(fringe))
        show_search_tree(
            tree,
            fringe=fringe,
            costs=costs[step] if costs else None,
            explored=explored_states[:step])

    if interactive:
        step = widgets.IntSlider(
            min=0, max=len(explored_states),
            value=0,  #len(explored_states),
            description='Krok')
        return widgets.interact(show_search_tree_at, step=step)
    else:
        show_search_tree_at(step=len(explored_states))


class Logger:
    def __init__(self):
        self.output_text = False
        self.output_widget = True

    def set_output(self, text=False, widget=False):
        self.output_text = text
        self.output_widget = widget

    def start_search(self, state, costs=False):
        self.explored_states = []
        self.trees = [{state: None}]
        self.fringes = [set([state])]
        self.costs = [{state: 0}] if costs else None
        self.log('start search')

    def end_search(self, interactive=False):
        create_tree_search_widget(
            self.explored_states, self.trees, self.fringes,
            costs=self.costs,
            interactive=interactive)

    def log(self, message):
        step = len(self.explored_states)
        if self.output_text:
            print('{step}: {message}'.format(
                step=step, message=message))

    def log_search_step(self, explored_state, fringe, costs=None):
        # TODO: check type of states from fringe
        fringe = set(state for state in fringe)
        last_tree = self.trees[-1]
        last_fringe = self.fringes[-1]
        tree = deepcopy(last_tree)
        #new_explored_states = last_fringe - fringe
        for action in actions(explored_state):
            next_state = move(explored_state, action)
            if next_state in fringe:
                tree[next_state] = explored_state
        self.explored_states.append(explored_state)
        self.trees.append(tree)
        self.fringes.append(fringe)
        if self.costs:
            if costs is None:
                raise ValueError('Zadejte i ceny stavu.')
            self.costs.append(deepcopy(costs))
        self.log(str(fringe))


LOGGER = Logger()
LOGGER.set_output(text=False, widget=True)

@contextmanager
def visualize_search(state, interactive=False, costs=False):
    LOGGER.start_search(state, costs=costs)
    yield
    LOGGER.end_search(interactive=interactive)


def log_search_step(explored_state, fringe, costs=None):
    LOGGER.log_search_step(explored_state, fringe, costs=costs)
