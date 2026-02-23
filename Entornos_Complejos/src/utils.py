import numpy as np

SEMILLA = 42


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