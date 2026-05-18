from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    # Compute unsatisfied goal fluents
    unsatisfied = goal - state
    
    # If all goal fluents are already satisfied, cost is 0
    if not unsatisfied:
        return 0.0
    
    # Generate all groundings (preconditions are ignored in this relaxation)
    from planning.pddl import get_all_groundings
    all_actions = get_all_groundings(domain, objects)
    
    # Greedy set cover: repeatedly pick the action that covers the most unsatisfied fluents
    count = 0
    while unsatisfied:
        # Find the action that covers the most unsatisfied fluents
        best_action = None
        best_coverage = 0
        
        for action in all_actions:
            coverage = len(action.add_list & unsatisfied)  # Intersection: fluents added that are unsatisfied
            if coverage > best_coverage:
                best_coverage = coverage
                best_action = action
        
        # If no action covers any unsatisfied fluent, we're stuck (shouldn't happen in valid problems)
        if best_action is None or best_coverage == 0:
            break
        
        # Remove covered fluents from unsatisfied set
        unsatisfied = unsatisfied - best_action.add_list
        count += 1
    
    return float(count)
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    from planning.pddl import get_applicable_actions
    
    # Start from the given state in the relaxed problem
    relaxed_state = state
    count = 0
    
    # Hill-climbing: repeatedly pick the action that adds the most unsatisfied goal fluents
    while not goal.issubset(relaxed_state):
        unsatisfied = goal - relaxed_state
        
        # Get all applicable actions in the relaxed state
        applicable = get_applicable_actions(relaxed_state, domain, objects)
        
        # Find the action that adds the most unsatisfied goal fluents
        best_action = None
        best_progress = 0
        
        for action in applicable:
            progress = len(action.add_list & unsatisfied)  # Fluents added that satisfy goal
            if progress > best_progress:
                best_progress = progress
                best_action = action
        
        # If no action makes progress, we're stuck
        if best_action is None or best_progress == 0:
            break
        
        # Apply the action in the relaxed problem (only add fluents, never delete)
        relaxed_state = relaxed_state | best_action.add_list
        count += 1
    
    return float(count)
    ### End of your code ###
