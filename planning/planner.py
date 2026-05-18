from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    frontera = Queue()
    visitados = set()
    estado_inicial = problem.getStartState()

    frontera.push((estado_inicial, []))

    while not frontera.isEmpty():
        estado_actual, plan_actual = frontera.pop()

        if estado_actual in visitados:
            continue
        visitados.add(estado_actual)
        if problem.isGoalState(estado_actual):
            return plan_actual
        for siguiente_estado, accion, costo in problem.getSuccessors(estado_actual):

            if siguiente_estado not in visitados:
                nuevo_plan = plan_actual + [accion]
                frontera.push((siguiente_estado, nuevo_plan))
    return []

# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    ### Your code here ###
    if action.add_list.isdisjoint(goal_set):
        return None
    if not action.del_list.isdisjoint(goal_set):
        return None
    nuevo_objetivo = (goal_set - action.add_list) | action.precond_pos
    return frozenset(nuevo_objetivo)
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    ### Your code here ###
    from collections import deque

    estado_inicial = problem.getStartState()
    acciones = get_all_groundings(problem.domain, problem.objects)

    predicados_estaticos = {"MedicalPost", "Adjacent", "Pickable"}

    def limpiar(subgoal):
        resultado = set()
        for fluente in subgoal:
            if fluente[0] in predicados_estaticos:
                if fluente in estado_inicial:
                    continue
                return None
            resultado.add(fluente)
        return frozenset(resultado)

    meta = limpiar(problem.goal)
    if meta is None:
        return []
    if meta.issubset(estado_inicial):
        return []

    cola = deque()
    cola.append((meta, []))
    visitados = {meta}

    while cola:
        sub_meta, plan = cola.popleft()
        problem._expanded += 1

        faltantes = sub_meta - estado_inicial

        for accion in acciones:
            if accion.add_list.isdisjoint(faltantes):
                continue

            nueva_meta = regress(sub_meta, accion)
            if nueva_meta is None:
                continue

            nueva_meta = limpiar(nueva_meta)
            if nueva_meta is None or nueva_meta in visitados:
                continue

            nuevo_plan = [accion] + plan

            if nueva_meta.issubset(estado_inicial):
                return nuevo_plan

            visitados.add(nueva_meta)
            cola.append((nueva_meta, nuevo_plan))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    initial_state = problem.getStartState()
    goal = problem.goal
    domain = problem.domain
    objects = problem.objects
    
    # Priority queue: (priority, state, plan)
    frontier = PriorityQueue()
    
    # Initial heuristic estimate
    h_initial = heuristic(initial_state, goal, domain, objects)
    frontier.push((initial_state, []), 0 + h_initial)
    
    # Track the best g-cost (accumulated cost) for each state
    best_g = {initial_state: 0}
    
    while not frontier.isEmpty():
        state, plan = frontier.pop()
        
        # Check if this is a goal state
        if problem.isGoalState(state):
            return plan
        
        # Explore successors
        for next_state, action, cost in problem.getSuccessors(state):
            # Calculate the new accumulated cost (g-value)
            g_next = best_g[state] + cost
            
            # Only add to frontier if we found a better path to next_state
            if next_state not in best_g or g_next < best_g[next_state]:
                best_g[next_state] = g_next
                
                # Calculate the heuristic estimate (h-value)
                h_next = heuristic(next_state, goal, domain, objects)
                
                # Calculate priority f(n) = g(n) + h(n)
                priority = g_next + h_next
                
                new_plan = plan + [action]
                frontier.push((next_state, new_plan), priority)
    
    # No plan found
    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
