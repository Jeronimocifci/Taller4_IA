from __future__ import annotations

from planning.pddl import Problem
from planning.domain import DOMAIN
from world.rescue_layout import RescueLayout
from world.rescue_rules import build_initial_state


# ===========================================================================
# Punto 1a (parte b) - Definicion de los estados objetivo (goals)
# ===========================================================================
#
# Un "goal" en PDDL es un conjunto de fluentes que TODOS deben estar presentes
# en el estado final para considerar resuelto el problema.
#
# En esta implementacion el goal se representa como un frozenset de fluentes,
# y la funcion Problem.isGoalState(state) verifica si goal.issubset(state).
# ===========================================================================


class SimpleRescueProblem(Problem):
    """
    Problema de rescate con UN solo paciente.

    Goal: Rescued(patient_0)

    Para alcanzarlo, el robot debe:
        1. Recoger los suministros y llevarlos al puesto medico (SetupSupplies).
        2. Llevar al paciente hasta el puesto medico.
        3. Ejecutar la accion Rescue sobre el paciente.
    """

    def __init__(self, layout: RescueLayout) -> None:
        # build_initial_state convierte el layout en un estado PDDL
        # (conjunto de fluentes) y un diccionario de objetos por tipo.
        initial_state, objects = build_initial_state(layout)

        # El estado objetivo solo exige que patient_0 este rescatado.
        # build_initial_state nombra a los pacientes como patient_0,
        # patient_1, ..., asi que en el caso simple basta con patient_0.
        goal = frozenset({("Rescued", "patient_0")})

        super().__init__(initial_state, goal, DOMAIN, objects)
        self.layout = layout


class MultiRescueProblem(Problem):
    """
    Problema de rescate con MULTIPLES pacientes.

    Goal: Rescued(patient_0) AND Rescued(patient_1) AND ... AND Rescued(patient_n)

    El robot debe rescatar a TODOS los pacientes presentes en el layout.
    """

    def __init__(self, layout: RescueLayout) -> None:
        initial_state, objects = build_initial_state(layout)

        # Construimos un fluente ("Rescued", p) por cada paciente del layout.
        # objects["patients"] es la lista ["patient_0", "patient_1", ...].
        goal = frozenset({("Rescued", p) for p in objects["patients"]})

        super().__init__(initial_state, goal, DOMAIN, objects)
        self.layout = layout
