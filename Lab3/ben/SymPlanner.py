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
        max_expansions: int = 500
    ):
        init_state = frozenset(init)
        goal_set = set(goal)

        # Setup A* over state space
        frontier = []
        g_cost = {init_state: 0}
        parent = {init_state: (None, None)}  # parent_state + action = new_state
        h0 = self.heuristic(init_state, goal_set)
        heapq.heappush(frontier, (h0, 0, init_state))
        visited = {init_state}

        target_goal = None
        n_exp = 0
        while len(frontier) > 0 and n_exp < max_expansions:
            heuristic, cost, state = heapq.heappop(frontier)
            if all(g in state for g in goal_set):
                target_goal = state
                break

            next_cost = cost + 1
            for action in self.applicable_actions(state):
                next_state = frozenset(self.apply(state, action))
                if g_cost.get(next_state, next_cost + 1) > next_cost:
                    parent[next_state] = (state, action)
                    g_cost[next_state] = next_cost
                if next_state not in visited:
                    heapq.heappush(frontier, (self.heuristic(next_state, goal_set) + next_cost, next_cost, next_state))
                    visited.add(next_state)
            n_exp += 1

        if target_goal is None:
            return None
        rev_plan = []
        current_state = target_goal
        while current_state != init_state:
            current_state, action = parent[current_state]
            rev_plan.append(action)
        return reversed(rev_plan)
