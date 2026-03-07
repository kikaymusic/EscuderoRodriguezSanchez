import time
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Mapeo de acciones específico para MountainCar-v0
ACTION_NAMES = {0: "IZQUIERDA", 1: "NADA", 2: "DERECHA"}


def evaluate_mountaincar_visual(env, agent, n_episodes=5, seed_base=1234, delay=0.02):
    """
    Evaluación visual de un agente en MountainCar-v0.
    Muestra los episodios frame a frame en un entorno Jupyter/Colab.

    IMPORTANTE: El entorno (env) debe ser creado con render_mode="rgb_array":
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    """

    # Diccionario auxiliar por si no está definido globalmente
    actions_map = globals().get('ACTION_NAMES', {0: "Izquierda", 1: "Nada", 2: "Derecha"})

    # Marcadores adaptados a MountainCar
    successes = 0
    failures = 0
    total_rewards = []
    total_steps = []

    # Salidas de UI
    out_plot = widgets.Output()
    out_info = widgets.Output()

    def render_frame(title, extra_lines=None):
        if extra_lines is None:
            extra_lines = []

        with out_plot:
            # Limpiamos solo el contenido del widget de imagen
            clear_output(wait=True)
            
            # Capturamos el frame antes de crear la figura
            frame = env.render()
            
            # Creamos la figura explícitamente para controlarla
            fig = plt.figure(figsize=(7, 4.5))
            
            if frame is not None:
                plt.imshow(frame)
                plt.axis("off")
            else:
                plt.text(0.5, 0.5, "Error: Crea el env con render_mode='rgb_array'", 
                         ha="center", va="center")
                plt.axis("off")
            
            plt.title(title)
            plt.show()
            # CERRAMOS la figura inmediatamente para evitar que se duplique 
            # o se acumule en la memoria del notebook
            plt.close(fig)

        with out_info:
            clear_output(wait=True)
            print("=" * 70)
            print("MOUNTAIN CAR LIVE EVALUATION")
            print("=" * 70)
            print(f"Episodios objetivo : {n_episodes}")
            print(f"Delay por frame     : {delay:.3f}s")
            print("-" * 70)
            print(f"Éxitos (Llegó)     : {successes}")
            print(f"Fallos (Timeout)   : {failures}")

            played = successes + failures
            success_rate = (successes / played) * 100 if played > 0 else 0.0
            print(f"Tasa de éxito      : {success_rate:.2f}%")
            print("-" * 70)

            for line in extra_lines:
                print(line)
            print("=" * 70)

    # Mostrar contenedor estático una vez
    display(widgets.VBox([out_plot, out_info]))

    # Bucle principal de evaluación
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed_base + ep)
        done = False
        step_idx = 0
        ep_reward = 0.0
        
        while not done:
            # Desempaquetar el estado continuo de MountainCar
            position, velocity = state

            # Obtenemos la acción del agente
            action = int(agent.get_action(state))

            extra_lines = [
                f"Episodio            : {ep + 1}/{n_episodes}",
                f"Paso                : {step_idx}",
                f"Estado              : Pos={position:.4f}, Vel={velocity:.4f}",
                f"Acción elegida      : {actions_map.get(action, action)} ({action})",
            ]

            render_frame(
                title=f"Episodio {ep + 1}/{n_episodes} | Paso {step_idx} | Pos: {position:.2f}",
                extra_lines=extra_lines
            )

            # Pausa para poder ver la animación
            time.sleep(delay)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            state = next_state
            ep_reward += reward
            step_idx += 1

        # Resultado del episodio
        if step_idx < 200 and not truncated:
            successes += 1
            result_label = "¡ÉXITO! LLEGÓ A LA CIMA"
        else:
            failures += 1
            result_label = "FALLO (Timeout 200 pasos)"

        total_rewards.append(ep_reward)
        total_steps.append(step_idx)

        extra_lines = [
            f"Episodio            : {ep + 1}/{n_episodes}",
            f"Resultado final     : {result_label}",
            f"Recompensa total    : {ep_reward:.0f}",
            f"Pasos totales       : {step_idx}",
        ]

        render_frame(
            title=f"Episodio {ep + 1}/{n_episodes} FIN | {result_label}",
            extra_lines=extra_lines
        )

        time.sleep(max(delay, 1.0))  # Pausa más larga al acabar el episodio

    # Resumen final
    played = successes + failures
    success_rate = (successes / played) * 100 if played > 0 else 0.0
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    render_frame(
        title="Evaluación Completada",
        extra_lines=[
            f"Episodios jugados   : {played}",
            f"Tasa de éxito final : {success_rate:.2f}%",
            f"Recompensa media    : {avg_reward:.2f}",
            f"Pasos medios        : {avg_steps:.2f}"
        ]
    )

    return {
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps
    }