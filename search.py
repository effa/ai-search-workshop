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

    @property
    def height(self):
        return self.n

    @property
    def row(self):
        # Tohle je desny hack, aby slo heuristiku pro A* zadat pochopitelne.
        # TODO: zkonzistentnit ukladani a prezentaci radku
        return self.height - self.spaceship.row

    def is_goal(self):
        return not self.is_dead() and self.spaceship.row == 0

    def is_dead(self):
        return self.world.get(self.spaceship, 'A') == 'A'

    def show(self):
        show_state(self)

    def is_wormhole(self, position):
        return position in self.wormhole_positions

    def get_wormhole_end(self, position):
        return self.wormhole_end[position]

    def at(self, position):
        """Return state with the spaceship at different location

        Args:
            position: chess like e.g. 'c2'
        """
        world = self.world.copy()
        spaceship = Position(
            row=self.n - int(position[1]),
            col=ord(position[0]) - ord('a'))
        return State(world, spaceship)


    def __eq__(self, other):
        return self.world == other.world and self.spaceship == other.spaceship

    def __hash__(self):
        return hash(self.spaceship)

    def __str__(self):
        return repr(self)
        #fields = [[
        #    self.world[(Position(row, col))]
        #    for col in range(self.m)]
        #        for row in range(self.n)]
        #fields[self.spaceship.row][self.spaceship.col] = 'S'
        #text = '\n'.join('|{inside}|'.format(inside='|'.join(row)) for row in fields)
        ##text = text.replace(' ', '.')
        #return text

    def __repr__(self):
        x = self.spaceship.col
        letter = chr(ord('a') + x)
        y = self.n - self.spaceship.row
        return '{x}{y}'.format(x=letter,y=y)

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
    # no action allowed in dead state
    if state.is_dead():
        return state
    world = state.world.copy()
    spaceship = move_spaceship(state.spaceship, action)
    if state.is_wormhole(spaceship):
        spaceship = state.get_wormhole_end(spaceship)
    return State(world, spaceship)


def actions(state):
    """Return actions that don't lead to dead state.
    """
    return [a for a in 'lfr' if not move(state, a).is_dead()]


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
    ax.set_xticklabels([])
    ax.set_xticks([x+0.5 for x in range(width)], minor=True)
    ax.set_xticklabels(
        [chr(ord('a')+i) for i in range(width)], minor=True)
    ax.set_yticks(range(height+1))
    ax.set_yticklabels([])
    ax.set_yticks([y+0.5 for y in range(height)], minor=True)
    ax.set_yticklabels(
        [str(i) for i in range(1, height+1)], minor=True)
    ax.tick_params(axis='both', which='both',length=0)
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
        plt.text(0, -0.6, message,
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

def draw_plans_as_edges(initial_state, plans):
    for plan in plans.values():
        state = initial_state
        for action in plan:
            next_state = move(state, action)
            draw_move(state, next_state, style='o-', color='tab:blue')
            state = next_state


def draw_explored_states(explored_states, costs=None, heuristic=None):
    explored_states = [s for s in  explored_states if s is not None]
    xs = [s.spaceship.col + 0.5 for s in explored_states]
    ys = [s.n - s.spaceship.row - 0.5 for s in explored_states]
    if costs:
        if heuristic:
            labels = [
                '{order}\n{g}+{h}={f}'.format(
                    order=i,
                    g=costs[s],
                    h=heuristic[s],
                    f=costs[s]+heuristic[s])
                for i, s in enumerate(explored_states, start=1)]
        else:
            labels = [
                '{order}\nc={cost}'.format(order=i, cost=costs[s])
                for i, s in enumerate(explored_states, start=1)]
    else:
        labels = [str(i) for i in range(1,len(explored_states)+1)]
    for label, x, y in zip(labels, xs, ys):
        #plt.annotate(label, xy=(x, y))
        plt.text(
            x, y, label,
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor': 'tab:blue', 'pad': 5, 'alpha': 0.95})


def draw_fringe_states(fringe, costs=None, heuristic=None):
    xs = [s.spaceship.col + 0.5 for s in fringe]
    ys = [s.n - s.spaceship.row - 0.5 for s in fringe]
    if costs:
        labels = [
            '{order}\nc={cost}'.format(order='?', cost=costs[s])
            for s in fringe]
        if heuristic:
            labels = [
                '{order}\n{g}+{h}={f}'.format(
                    order='?',
                    g=costs[s],
                    h=heuristic[s],
                    f=costs[s]+heuristic[s])
                for s in fringe]
        for label, x, y in zip(labels, xs, ys):
            plt.text(
                x, y, label,
                horizontalalignment='center',
                verticalalignment='center',
                bbox={'facecolor': 'tab:red', 'pad': 5})
    else:
        plt.plot(xs, ys, 's', color='tab:red')


def show_search_tree(initial_state, explored_states, fringe, plans,
                     costs=None, heuristic=None):
    current_state = explored_states[-1] or initial_state
    show_state(current_state)
    draw_plans_as_edges(initial_state, plans)
    draw_explored_states(explored_states, costs, heuristic)
    draw_fringe_states(fringe, costs, heuristic)


def create_tree_search_widget(initial_state, explored_states,
                              plans, fringes,
                              costs=None, heuristic=None,
                              interactive=False):
    if not explored_states:
        print('Zadne kroky k zobrazeni.')
        return

    def show_search_tree_at(step):
        #print('fringe at', step, 'is', str(fringe))
        show_search_tree(
            initial_state=initial_state,
            explored_states=explored_states[:step+1],
            fringe=fringes[step],
            plans=plans[step],
            costs=costs[step] if costs else None,
            heuristic=heuristic[step] if heuristic else None)

    if interactive:
        step = widgets.IntSlider(
            min=0, max=len(explored_states)-1,
            value=0,  #len(explored_states),
            description='Krok')
        return widgets.interact(show_search_tree_at, step=step)
    else:
        show_search_tree_at(step=len(explored_states)-1)


def diff(new_dict, old_dict=None):
    changes = [
        (k,v) for k,v in new_dict.items()
        if not old_dict or k not in old_dict or old_dict[k] != v]
    if len(changes) == 0:
        return ''
    if len(changes) < len(new_dict):
        template = '+ {changes}'
    else:
        template = '{changes}'
    message = template.format(
        changes=', '.join(
            '{k}:{v}'.format(
                k=str(k), v=str(v) or "''") for k, v in changes))
    return message


class Logger:
    def __init__(self):
        self.output_text = True
        self.output_image = True
        self.output_interactive = False

    def set_output(self, text=False, image=True, interactive=False):
        self.output_text = text
        self.output_image = image
        self.output_interactive = interactive

    def start_search(self, initial_state, costs=False, heuristic=False):
        self.initial_state = initial_state
        self.explored_states = []
        self.plans = []
        self.fringes = []
        self.costs = [] if costs else None
        self.heuristic = [] if heuristic else None
        #self.log('start search')
        self.log_header()

    def end_search(self):
        if self.output_image:
            create_tree_search_widget(
                initial_state=self.initial_state,
                explored_states=self.explored_states,
                plans=self.plans,
                fringes=self.fringes,
                costs=self.costs,
                heuristic=self.heuristic,
                interactive=self.output_interactive)

    @property
    def row_format(self):
        row_format = '{explored:<9} {fringe:<20} {plans:<23}'
        if self.costs is not None:
            row_format += ' {costs:<20}'
        if self.heuristic is not None:
            row_format += ' {heuristic:<5}'
        return row_format

    def log_header(self):
        if self.output_text:
            header = self.row_format.format(
                explored='Explored:',
                fringe='Fringe:',
                plans='Plans:',
                costs='Costs:',
                heuristic='Heuristic:')
            print(header)

    def log(self, state, fringe, plans, costs=None, heuristic=None):
        step = str(len(self.explored_states)-1) + ':'
        if self.output_text:
            message = self.row_format.format(
                explored=step+' '+str(state), fringe=fringe, plans=plans,
                costs=costs, heuristic=heuristic)
            print(message)

    def log_search_step(self, explored_state, fringe, plans,
                        costs=None, heuristic=None):
        if len(self.fringes) >= 1:
            prev_state = self.explored_states[-1]
            prev_plans = self.plans[-1]
            prev_costs = self.costs[-1] if costs else None
            prev_heuristic = self.heuristic[-1] if heuristic else None
        else:
            prev_state = None
            prev_plans = None
            prev_costs = None
            prev_heuristic = None
        fringe = deepcopy(fringe)
        self.fringes.append(fringe)
        plans = deepcopy(plans)
        self.plans.append(plans)
        self.explored_states.append(deepcopy(explored_state))
        if self.costs is not None:
            if costs is None:
                raise ValueError('Zadejte i ceny stavu.')
            self.costs.append(deepcopy(costs))
        if self.heuristic is not None:
            if heuristic is None:
                raise ValueError('Zadejte i heuristiky.')
            self.heuristic.append(deepcopy(heuristic))
        self.log(
            state=repr(explored_state) if explored_state else '-',
            fringe=str(list(fringe)),
            plans=diff(plans, prev_plans),
            costs=diff(costs, prev_costs) if costs else None,
            heuristic=diff(heuristic, prev_heuristic) if heuristic else None)



LOGGER = Logger()
#LOGGER.set_output(text=False, image=True)

@contextmanager
def visualize_search(state, text=False, image=True, interactive=False,
                     costs=False, heuristic=False):
    LOGGER.set_output(text=text, image=image, interactive=interactive)
    LOGGER.start_search(state, costs=costs, heuristic=heuristic)
    yield
    LOGGER.end_search()


def log_search_step(state, fringe, plans, costs=None, heuristic=None):
    LOGGER.log_search_step(
        state, fringe=fringe, plans=plans,
        costs=costs, heuristic=heuristic)
