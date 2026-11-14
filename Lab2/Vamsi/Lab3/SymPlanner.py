from typing import List, Tuple, Dict, Optional, Iterable, Set
from dataclasses import dataclass, field
import heapq

# DO NOT MODIFY - start
# -------------------------------------------------------------------------------

Predicate = Tuple[str, Tuple[str, ...]]  # e.g., ('at', ('robot','A'))
Subst = Dict[str, str]

def is_var(x: str) -> bool:
    return isinstance(x, str) and x.startswith("?")


def substitute(pred: Predicate, subst: Subst) -> Predicate:
    name, args = pred
    return (name, tuple(subst.get(a, a) for a in args))


def unify(a: Predicate, b: Predicate, subst: Optional[Subst] = None) -> Optional[Subst]:
    """Unify predicate a (may contain vars) with ground predicate b (state fact)."""
    if subst is None:
        subst = {}
    if a[0] != b[0] or len(a[1]) != len(b[1]):
        return None
    theta = dict(subst)
    for av, bv in zip(a[1], b[1]):
        if is_var(av):
            if av in theta and theta[av] != bv:
                return None
            theta[av] = bv
        else:
            if av != bv:
                return None
    return theta


def unify_all(preconds: List[Predicate], state: Set[Predicate]) -> List[Subst]:
    """All substitutions that satisfy all preconditions in the given ground state."""
    sols = []
    preconds = list(preconds)

    def backtrack(i: int, theta: Subst):
        if i == len(preconds):
            sols.append(theta.copy())
            return
        p = preconds[i]
        for s in state:
            if s[0] == p[0] and len(s[1]) == len(p[1]):
                new_theta = unify(substitute(p, theta), s, theta)
                if new_theta is not None:
                    backtrack(i + 1, new_theta)

    backtrack(0, {})
    # Deduplicate
    uniq, seen = [], set()
    for th in sols:
        key = tuple(sorted(th.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(th)
    return uniq

@dataclass(frozen=True)
class GroundAction:
    name: str
    args: Tuple[str, ...]
    add: Tuple[Predicate, ...]
    delete: Tuple[Predicate, ...]

    def label(self) -> str:
        return f"{self.name}({', '.join(self.args)})"
    
@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: Tuple[str, ...]  # variable names like ?x
    preconds: Tuple[Predicate, ...]  # positive literals only (STRIPS)
    add_effects: Tuple[Predicate, ...]
    del_effects: Tuple[Predicate, ...] = field(default_factory=tuple)

    def ground(self, subst: Subst):
        return GroundAction(
            name=self.name,
            args=tuple(subst.get(v, v) for v in self.parameters),
            add=tuple(substitute(p, subst) for p in self.add_effects),
            delete=tuple(substitute(p, subst) for p in self.del_effects),
        )

# DO NOT MODIFY - end
# -------------------------------------------------------------------------------

def reconstruct_plan(current_state, lookup_table):
    plan = []
    while current_state in lookup_table:
        parent_state, action = lookup_table[current_state]
        if action is not None:
            plan.append(action)
        current_state = parent_state
    plan.reverse()
    return plan



class Planner:
    def __init__(self, actions: List[ActionSchema]):
        self.actions = actions  # Initialize with all possible actions

    # Get applicable actions for a given state
    def applicable_actions(self, state: Set[Predicate]) -> List[GroundAction]:
        out = []
        for schema in self.actions:
            for theta in unify_all(list(schema.preconds), state):
                out.append(schema.ground(theta))
        return out

    # Get new state from a given state and action
    def apply(self, state: Set[Predicate], action: GroundAction) -> Set[Predicate]:
        new_state = set(state)
        for d in action.delete:
            if d in new_state:
                new_state.remove(d)
        for a in action.add:
            new_state.add(a)
        return new_state

    # Simple goal-count heuristic - count goals that have not been reached
    def heuristic(self, state: Set[Predicate], goal: Set[Predicate]) -> int:
        return sum(1 for g in goal if g not in state)

    # Search for a plan from 'init' state to 'goal' state within 'max_expansions' steps
    def plan(
        self,
        init: Iterable[Predicate],
        goal: Iterable[Predicate],
        max_expansions: int = 50000, # DO NOT MODIFY THIS
    ):
        init_state = frozenset(init)
        goal_set = set(goal)

        # Setup A* over state space
        frontier = []
        g_cost = {init_state: 0}
        parent = {init_state: (None, None)}  # parent_state + action = new_state
        h0 = self.heuristic(set(init_state), goal_set)
        heapq.heappush(frontier, (h0, 0, init_state))

        # TODO 1.2: Implement A* search and return list[GroundAction] as the plan if found, else None
        # HINT: Refer to lecture slides for pseudocode
        num_expansions = 0
        while len(frontier) > 0 and num_expansions < max_expansions:
            f_cost, g_cost_current, current_state = heapq.heappop(frontier)

            if(goal_set.issubset(current_state)):
                return reconstruct_plan(current_state, parent)

            for potential_action in self.applicable_actions(current_state):
                new_state = frozenset(self.apply(current_state, potential_action))
                new_g_cost = g_cost_current + 1  # Assuming each action has a cost of 1

                if new_state not in g_cost.keys() or new_g_cost < g_cost[new_state]:
                    g_cost[new_state] = new_g_cost
                    h_new = self.heuristic(new_state, goal_set)
                    total_cost = new_g_cost + h_new
                    heapq.heappush(frontier, (total_cost, new_g_cost, new_state))
                    parent[new_state] = (current_state, potential_action)
            num_expansions += 1

        return None  # No plan found within max_expansions
