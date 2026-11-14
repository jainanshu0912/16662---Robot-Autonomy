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
        self.actions = actions

    def applicable_actions(self, state) -> List[GroundAction]:
        out = []
        for schema in self.actions:
            for theta in unify_all(list(schema.preconds), state):
                out.append(schema.ground(theta))
        return out

    def apply(self, state, action: GroundAction) -> Set[Predicate]:
        new_state = set(state)
        for d in action.delete:
            new_state.discard(d)
        for a in action.add:
            new_state.add(a)
        return new_state

    def heuristic(self, state, goal: Set[Predicate]) -> int:
        return sum(1 for g in goal if g not in state)

    def plan(self, init: Iterable[Predicate], goal: Iterable[Predicate],
             max_expansions: int = 50000) -> Optional[List[GroundAction]]:
        """A* over symbolic states (unit action cost). Returns a list of GroundActions or None."""
        init_state = frozenset(init)
        goal_set = set(goal)

        counter = 0  # tie-breaker so the heap never compares frozensets
        frontier = [(self.heuristic(init_state, goal_set), 0, counter, init_state)]
        g_cost = {init_state: 0}
        parent = {init_state: (None, None)}
        closed = set()
        expansions = 0

        while frontier:
            _, g, _, state = heapq.heappop(frontier)
            if state in closed:
                continue
            if goal_set <= state:
                plan = []
                while parent[state][0] is not None:
                    prev, act = parent[state]
                    plan.append(act)
                    state = prev
                return plan[::-1]

            closed.add(state)
            expansions += 1
            if expansions > max_expansions:
                return None

            for act in self.applicable_actions(state):
                child = frozenset(self.apply(state, act))
                new_g = g + 1
                if child not in g_cost or new_g < g_cost[child]:
                    g_cost[child] = new_g
                    parent[child] = (state, act)
                    counter += 1
                    heapq.heappush(frontier, (new_g + self.heuristic(child, goal_set), new_g, counter, child))
        return None
