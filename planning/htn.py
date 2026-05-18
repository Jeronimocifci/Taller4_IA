from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    from planning.utils import Queue
    
    # Start with a plan containing only the root HLA
    if not hlas:
        return []
    
    root_hla = hlas[0]
    frontier = Queue()
    frontier.push([root_hla])
    
    while not frontier.isEmpty():
        plan = frontier.pop()
        problem._expanded += 1
        
        # Check if the plan is fully primitive
        if is_plan_primitive(plan):
            # Try to execute the plan from the initial state
            current_state = problem.initial_state
            valid = True
            
            for action in plan:
                if not is_applicable(current_state, action):
                    valid = False
                    break
                current_state = apply_action(current_state, action)
            
            # If the plan executed successfully and reached the goal, return it
            if valid and problem.isGoalState(current_state):
                return plan
        else:
            # Find the first non-primitive step
            for i, step in enumerate(plan):
                if not is_primitive(step):
                    # Try each refinement of this HLA
                    hla = step
                    for refinement in hla.refinements:
                        # Create a new plan by replacing the HLA with the refinement
                        new_plan = plan[:i] + refinement + plan[i+1:]
                        frontier.push(new_plan)
                    break  # Only refine the first non-primitive step
    
    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    from planning.pddl import get_all_groundings
    
    initial_state = problem.initial_state
    objects = problem.objects
    domain = problem.domain
    
    # Helper: find all adjacent cells from a cell
    def find_adjacent(cell, state):
        adjacent = []
        for fluent in state:
            if fluent[0] == "Adjacent" and fluent[1] == cell:
                adjacent.append(fluent[2])
        return adjacent
    
    # Helper: find path between two cells using BFS
    def find_path(from_cell, to_cell, state):
        """Find a path from from_cell to to_cell, returns list of cells including start and end."""
        from collections import deque
        
        if from_cell == to_cell:
            return [from_cell]
        
        queue = deque([(from_cell, [from_cell])])
        visited = {from_cell}
        
        while queue:
            current, path = queue.popleft()
            neighbors = find_adjacent(current, state)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == to_cell:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return None  # No path found
    
    # Helper: Create a Navigate HLA with Move sequence refinement
    def create_navigate_hla(from_cell, to_cell):
        """Create a Navigate HLA with refinements (Move action sequences)."""
        path = find_path(from_cell, to_cell, initial_state)
        if not path or len(path) < 2:
            return None
        
        # Create refinement: sequence of Move actions
        refinement = []
        for i in range(len(path) - 1):
            current_cell = path[i]
            next_cell = path[i + 1]
            move_action = Action(
                f"Move(robot,{current_cell},{next_cell})",
                [("At", "robot", current_cell), ("Adjacent", current_cell, next_cell), ("Free", next_cell)],
                [],
                [("At", "robot", next_cell), ("Free", current_cell)],
                [("At", "robot", current_cell), ("Free", next_cell)],
            )
            refinement.append(move_action)
        
        hla = HLA(f"Navigate({from_cell},{to_cell})", [refinement])
        return hla
    
    # Helper: Get object's initial position
    def get_initial_position(obj_name, state):
        for fluent in state:
            if fluent[0] == "At" and fluent[1] == obj_name:
                return fluent[2]
        return None
    
    # Helper: Create PrepareSupplies HLA
    def create_prepare_supplies_hla(supplies_name, medical_post):
        """Create a PrepareSupplies HLA."""
        robot_pos = (1, 4)  # Default initial position
        for fluent in initial_state:
            if fluent[0] == "At" and fluent[1] == "robot":
                robot_pos = fluent[2]
                break
        
        supplies_pos = get_initial_position(supplies_name, initial_state)
        if not supplies_pos:
            return None
        
        # Refinement: Navigate to supplies, PickUp, Navigate to medical post, SetupSupplies
        refinement = []
        
        # Navigate to supplies
        nav_to_supplies = create_navigate_hla(robot_pos, supplies_pos)
        if nav_to_supplies:
            refinement.extend(nav_to_supplies.refinements[0])
        
        # PickUp supplies
        pickup_action = Action(
            f"PickUp(robot,{supplies_name},{supplies_pos})",
            [("At", "robot", supplies_pos), ("At", supplies_name, supplies_pos), 
             ("HandsFree", "robot"), ("Pickable", supplies_name)],
            [],
            [("Holding", "robot", supplies_name)],
            [("At", supplies_name, supplies_pos), ("HandsFree", "robot")],
        )
        refinement.append(pickup_action)
        
        # Navigate to medical post
        nav_to_post = create_navigate_hla(supplies_pos, medical_post)
        if nav_to_post:
            refinement.extend(nav_to_post.refinements[0])
        
        # SetupSupplies at medical post
        setup_action = Action(
            f"SetupSupplies(robot,{supplies_name},{medical_post})",
            [("At", "robot", medical_post), ("MedicalPost", medical_post), 
             ("Holding", "robot", supplies_name)],
            [],
            [("SuppliesReady", medical_post), ("HandsFree", "robot")],
            [("Holding", "robot", supplies_name)],
        )
        refinement.append(setup_action)
        
        hla = HLA(f"PrepareSupplies({supplies_name},{medical_post})", [refinement])
        return hla
    
    # Helper: Create ExtractPatient HLA
    def create_extract_patient_hla(patient_name, medical_post):
        """Create an ExtractPatient HLA."""
        patient_pos = get_initial_position(patient_name, initial_state)
        if not patient_pos:
            return None
        
        # Refinement: Navigate to patient, PickUp, Navigate to medical post, PutDown
        refinement = []
        
        # Navigate to patient
        # Start navigation from medical_post (after PrepareSupplies finishes)
        nav_to_patient = create_navigate_hla(medical_post, patient_pos)
        if nav_to_patient:
            refinement.extend(nav_to_patient.refinements[0])
        
        # PickUp patient
        pickup_action = Action(
            f"PickUp(robot,{patient_name},{patient_pos})",
            [("At", "robot", patient_pos), ("At", patient_name, patient_pos),
             ("HandsFree", "robot"), ("Pickable", patient_name)],
            [],
            [("Holding", "robot", patient_name)],
            [("At", patient_name, patient_pos), ("HandsFree", "robot")],
        )
        refinement.append(pickup_action)
        
        # Navigate to medical post
        nav_to_post = create_navigate_hla(patient_pos, medical_post)
        if nav_to_post:
            refinement.extend(nav_to_post.refinements[0])
        
        # PutDown patient
        putdown_action = Action(
            f"PutDown(robot,{patient_name},{medical_post})",
            [("At", "robot", medical_post), ("Holding", "robot", patient_name)],
            [],
            [("At", patient_name, medical_post), ("HandsFree", "robot")],
            [("Holding", "robot", patient_name)],
        )
        refinement.append(putdown_action)
        
        hla = HLA(f"ExtractPatient({patient_name},{medical_post})", [refinement])
        return hla
    
    # Helper: Create FullRescueMission HLA
    def create_full_rescue_mission(supplies_name, patient_name, medical_post):
        """Create a FullRescueMission HLA."""
        prep_hla = create_prepare_supplies_hla(supplies_name, medical_post)
        extract_hla = create_extract_patient_hla(patient_name, medical_post)
        
        if not prep_hla or not extract_hla:
            return None
        
        # Refinement: PrepareSupplies, ExtractPatient, Rescue
        refinement = [prep_hla, extract_hla]
        
        # Add Rescue action
        rescue_action = Action(
            f"Rescue(robot,{patient_name},{medical_post})",
            [("At", "robot", medical_post), ("At", patient_name, medical_post),
             ("MedicalPost", medical_post), ("SuppliesReady", medical_post)],
            [],
            [("Rescued", patient_name)],
            [("At", patient_name, medical_post)],
        )
        refinement.append(rescue_action)
        
        hla = HLA(f"FullRescueMission({supplies_name},{patient_name},{medical_post})", [refinement])
        return hla
    
    # Build the HLA hierarchy
    hlas_list = []
    
    # Get medical posts and patients/supplies
    medical_posts = objects.get("medical_posts", [])
    patients = objects.get("patients", [])
    supplies = objects.get("supplies", [])
    
    if medical_posts and patients and supplies:
        # Create FullRescueMission HLA(s) for each patient with first supply and first medical post
        medical_post = medical_posts[0]
        
        for patient in patients:
            supply = supplies[0] if supplies else None
            if supply:
                full_mission = create_full_rescue_mission(supply, patient, medical_post)
                if full_mission:
                    hlas_list.append(full_mission)
    
    return hlas_list
    ### End of your code ###
