import numpy as np

SEMILLA = 42

def ensure_state_in_q(agent, state):
    """
    Garantiza que un estado exista en las tablas del agente Monte Carlo antes de usarlo.

    Esta función inicializa, si el estado no existe todavía:
    - agent.q_table[state]: estimaciones de valor de acción Q(s,a)
    - agent.c_table[state]: acumuladores/pesos (o recuentos) usados en las
      actualizaciones Monte Carlo off-policy


    La política de comportamiento epsilon-greedy puede intentar consultar Q(s,a)
    para un estado que aún no ha aparecido en la tabla. Como el agente Monte Carlo
    actualiza los valores al final del episodio, puede ocurrir que un estado se use
    durante la selección de acción antes de haber sido inicializado. Esta función evita
    errores (por ejemplo, KeyError) y garantiza una entrada válida inicializada a cero.

    Parámetros
    agent : AgentMonteCarlo
        Instancia del agente Monte Carlo que contiene q_table y c_table.
    state : tuple
        Estado de Blackjack con la forma:
        (suma_jugador, carta_visible_dealer, usable_ace).
    """
    if state not in agent.q_table:
        n_actions = agent.env.action_space.n
        agent.q_table[state] = np.zeros(n_actions, dtype=np.float32)
        agent.c_table[state] = np.zeros(n_actions, dtype=np.float32)


def q_to_v_and_policy(Q):
    """
    Convierte una tabla de valores de acción Q(s,a) en:
    - una función de valor de estado V(s)
    - una política greedy pi(s)

    Para cada estado s se calcula:
    - V(s) = max_a Q(s,a)
    - pi(s) = argmax_a Q(s,a)

    Parámetros
    Q : dict
        Diccionario que mapea estado -> array/lista de valores de acción.
        Ejemplo:
        Q[(suma_jugador, carta_visible_dealer, usable_ace)] = [Q_stick, Q_hit]

    Devuelve
    V : dict
        Diccionario estado -> valor de estado, calculado como el máximo valor Q.
    policy : dict
        Diccionario estado -> acción greedy (índice de la acción con mayor valor).
        En Blackjack:
        0 = STICK, 1 = HIT.
    """
    V = {}
    policy = {}
    for s, qvals in Q.items():
        qvals = np.asarray(qvals, dtype=np.float32)
        best_a = int(np.argmax(qvals))
        policy[s] = best_a
        V[s] = float(np.max(qvals))
    return V, policy