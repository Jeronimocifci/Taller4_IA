from __future__ import annotations

from planning.pddl import ActionSchema

# ===========================================================================
# Punto 1a - Esquemas de accion (PDDL) para el dominio de rescate
# ===========================================================================
#
# Cada accion se representa como un ActionSchema con:
#   - parameters : nombres de variables (placeholders) que luego se sustituyen
#                  por constantes reales del problema (objetos, celdas, etc.)
#   - precond_pos: fluentes que DEBEN estar presentes en el estado
#   - precond_neg: fluentes que NO deben estar presentes en el estado
#   - add_list   : fluentes que se anaden al estado al ejecutar la accion
#   - del_list   : fluentes que se eliminan del estado al ejecutar la accion
#
# Convencion de nombres de variables usada por get_applicable_actions():
#   "r"         -> robot
#   "from_cell" -> celda origen        "to_cell" -> celda destino
#   "loc"       -> celda actual del robot
#   "obj"       -> cualquier objeto recogible (suministros o pacientes)
#   "s"         -> suministros medicos
#   "p"         -> paciente
#
# Semantica del operador RESULT(s, a):
#       s' = (s  -  DEL(a))  union  ADD(a)
# Es decir: primero se borran los fluentes del delete-list y luego se anaden
# los del add-list. Los fluentes no mencionados quedan igual.
# ===========================================================================


# ---------------------------------------------------------------------------
# Move(r, from_cell, to_cell)
#   El robot se desplaza una celda hasta un vecino libre.
#   Esta accion ya viene dada como ejemplo.
# ---------------------------------------------------------------------------
MOVE: ActionSchema = ActionSchema(
    name="Move",
    parameters=["r", "from_cell", "to_cell"],
    precond_pos=[
        ("At", "r", "from_cell"),       # el robot esta en la celda origen
        ("Adjacent", "from_cell", "to_cell"),  # las celdas son adyacentes
        ("Free", "to_cell"),            # la celda destino esta libre
    ],
    precond_neg=[],
    add_list=[
        ("At", "r", "to_cell"),         # ahora el robot esta en la destino
        ("Free", "from_cell"),          # la celda origen queda libre
    ],
    del_list=[
        ("At", "r", "from_cell"),       # el robot deja de estar en origen
        ("Free", "to_cell"),            # la destino deja de estar libre
    ],
)


# ---------------------------------------------------------------------------
# PickUp(r, obj, loc)
#   El robot recoge un objeto recogible que esta en su misma celda.
#   Despues de recogerlo: el objeto deja de estar en la celda y el robot
#   ya no tiene las manos libres (ahora sostiene el objeto).
# ---------------------------------------------------------------------------
PICKUP: ActionSchema = ActionSchema(
    name="PickUp",
    parameters=["r", "obj", "loc"],
    precond_pos=[
        ("At", "r", "loc"),             # el robot esta en loc
        ("At", "obj", "loc"),           # el objeto esta en loc
        ("HandsFree", "r"),             # el robot tiene las manos libres
        ("Pickable", "obj"),            # el objeto es recogible
    ],
    precond_neg=[],
    add_list=[
        ("Holding", "r", "obj"),        # el robot ahora carga el objeto
    ],
    del_list=[
        ("At", "obj", "loc"),           # el objeto ya no esta en la celda
        ("HandsFree", "r"),             # el robot deja de tener manos libres
    ],
)


# ---------------------------------------------------------------------------
# PutDown(r, obj, loc)
#   El robot deposita en su celda actual un objeto que esta cargando.
#   Despues: el objeto vuelve a estar en la celda y el robot recupera
#   sus manos libres.
# ---------------------------------------------------------------------------
PUTDOWN: ActionSchema = ActionSchema(
    name="PutDown",
    parameters=["r", "obj", "loc"],
    precond_pos=[
        ("At", "r", "loc"),             # el robot esta en loc
        ("Holding", "r", "obj"),        # el robot esta cargando el objeto
    ],
    precond_neg=[],
    add_list=[
        ("At", "obj", "loc"),           # el objeto queda en la celda
        ("HandsFree", "r"),             # el robot recupera manos libres
    ],
    del_list=[
        ("Holding", "r", "obj"),        # el robot deja de cargar el objeto
    ],
)


# ---------------------------------------------------------------------------
# Rescue(r, p, loc)
#   El robot rescata a un paciente que esta junto a el en un puesto medico
#   donde ya hay suministros listos.
#   Despues: el paciente queda marcado como Rescued y deja de estar en loc.
# ---------------------------------------------------------------------------
RESCUE: ActionSchema = ActionSchema(
    name="Rescue",
    parameters=["r", "p", "loc"],
    precond_pos=[
        ("At", "r", "loc"),             # el robot esta en loc
        ("At", "p", "loc"),             # el paciente esta en loc
        ("MedicalPost", "loc"),         # loc es un puesto medico
        ("SuppliesReady", "loc"),       # los suministros estan listos
    ],
    precond_neg=[],
    add_list=[
        ("Rescued", "p"),               # el paciente queda rescatado
    ],
    del_list=[
        ("At", "p", "loc"),             # ya no esta fisicamente en loc
    ],
)


# ---------------------------------------------------------------------------
# SetupSupplies(r, s, loc)
#   El robot instala los suministros que esta cargando en un puesto medico.
#   Importante: NO se exige At(s, loc) como precondicion porque el robot
#   esta cargando s, y el fluente At(s, loc) fue eliminado al hacer PickUp.
#   Despues: el puesto queda con SuppliesReady y el robot recupera manos libres.
# ---------------------------------------------------------------------------
SETUP_SUPPLIES: ActionSchema = ActionSchema(
    name="SetupSupplies",
    parameters=["r", "s", "loc"],
    precond_pos=[
        ("At", "r", "loc"),             # el robot esta en loc
        ("MedicalPost", "loc"),         # loc es un puesto medico
        ("Holding", "r", "s"),          # el robot carga los suministros
    ],
    precond_neg=[],
    add_list=[
        ("SuppliesReady", "loc"),       # el puesto queda con suministros listos
        ("HandsFree", "r"),             # el robot recupera manos libres
    ],
    del_list=[
        ("Holding", "r", "s"),          # el robot deja de cargar los suministros
    ],
)


# Lista de todos los esquemas que componen el dominio de planificacion.
# Esta lista es la que utilizan el planificador y la funcion get_all_groundings.
DOMAIN: list[ActionSchema] = [MOVE, PICKUP, PUTDOWN, RESCUE, SETUP_SUPPLIES]
