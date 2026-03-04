import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from unittest.mock import patch
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict
from matplotlib.ticker import NullLocator
plt.gca().yaxis.set_minor_locator(NullLocator())

def plot_blackjack_values(V):
    def get_Z(x, y, usable_ace):
        return V.get((x, y, usable_ace), 0)

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.set_zlabel("State Value")
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()
    plt.close()

def plot_policy(policy):
    def get_Z(x, y, usable_ace):
        return policy.get((x, y, usable_ace), 1)  # default to HIT

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.grid(color='w', linestyle='-', linewidth=1)
        plt.gca().invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()
    plt.close()

def plot_q_heatmap(Q, state_filter_fn, label):
    """Plot Q[state][action] for filtered states."""
    import seaborn as sns
    states = [s for s in Q.keys() if state_filter_fn(s)]
    hit_vals = [Q[s][1] for s in states]
    stick_vals = [Q[s][0] for s in states]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(hit_vals, color="red", label="HIT", kde=True, stat="density", ax=ax)
    sns.histplot(stick_vals, color="blue", label="STICK", kde=True, stat="density", ax=ax)
    ax.set_title(f"Q-Value Distribution for Filtered States {label}")
    ax.set_xlabel("Q(s,a)")
    ax.legend()
    plt.show()
    plt.close()

def plot_rewards(reward_list, length_list, window=4000):
    """Plot reward and episode lengths (moving average)."""
    rewards = np.array(reward_list)
    lengths = np.array(length_list)
    
    # Calcular promedios móviles para recompensas y longitudes
    reward_means = np.convolve(rewards, np.ones(window) / window, mode='valid')
    length_means = np.convolve(lengths, np.ones(window) / window, mode='valid')
    
    # Crear subgráficas
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    
    # Graficar recompensas
    axs[0].set_title(f"Episode Rewards (Moving Average, Window = {window})")
    axs[0].plot(reward_means, label="Average Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Average Reward")
    axs[0].grid(True)
    
    # Graficar longitudes de los episodios
    axs[1].set_title(f"Episode Lengths (Moving Average, Window = {window})")
    axs[1].plot(length_means, label="Average Length", color='orange')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Length")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close()



def capture_figure(plot_fn, *args, **kwargs):
    captured = []
    surfaces_by_ax = {}

    original_plot_surface = Axes3D.plot_surface

    def patched_plot_surface(self, X, Y, Z, **kw):
        ax_id = id(self)
        if ax_id not in surfaces_by_ax:
            surfaces_by_ax[ax_id] = []
        surfaces_by_ax[ax_id].append((X.copy(), Y.copy(), Z.copy(), kw))
        return original_plot_surface(self, X, Y, Z, **kw)

    def capture_show():
        fig = plt.gcf()
        fig._surfaces_by_ax = surfaces_by_ax
        captured.append(fig)

    with patch("matplotlib.pyplot.show", side_effect=capture_show), \
         patch("matplotlib.pyplot.close", return_value=None), \
         patch.object(Axes3D, "plot_surface", patched_plot_surface):
        plot_fn(*args, **kwargs)

    return captured[0] if captured else None


def compare_plots(plot_fn, agents_data, agent_names=None, **kwargs):
    if agent_names is None:
        agent_names = [f"Agent {i+1}" for i in range(len(agents_data))]

    def get_main_axes(fig):
        return [ax for ax in fig.axes if
                len(ax.get_images()) > 0 or
                len(ax.get_lines()) > 0 or
                hasattr(ax, 'get_zlim')]

    def extract_surfaces(ax, fig):
        surfaces_by_ax = getattr(fig, '_surfaces_by_ax', {})
        return surfaces_by_ax.get(id(ax), [])

    captured_data = []
    for data in agents_data:
        args = data if isinstance(data, tuple) else (data,)
        fig = capture_figure(plot_fn, *args, **kwargs)

        axes_data = []
        for old_ax in get_main_axes(fig):
            is_3d = hasattr(old_ax, 'get_zlim')
            axes_data.append({
                'title':    old_ax.get_title(),
                'xlabel':   old_ax.get_xlabel(),
                'ylabel':   old_ax.get_ylabel(),
                'xlim':     old_ax.get_xlim(),
                'ylim':     old_ax.get_ylim(),
                'xticks':   old_ax.get_xticks(),
                'yticks':   old_ax.get_yticks(),
                'is_3d':    is_3d,
                'zlim':     old_ax.get_zlim()   if is_3d else None,
                'zlabel':   old_ax.get_zlabel() if is_3d else None,
                'elev':     old_ax.elev         if is_3d else None,
                'azim':     old_ax.azim         if is_3d else None,
                'surfaces': extract_surfaces(old_ax, fig) if is_3d else [],
                'lines':   [(l.get_xdata(), l.get_ydata(), l.get_label(), l.get_color())
                            for l in old_ax.get_lines()],
                'images':  [(im.get_array(), im.get_cmap(), im.norm.vmin, im.norm.vmax, im.get_extent())
                            for im in old_ax.get_images()],
            })
        captured_data.append(axes_data)
        plt.close(fig)

    n_cols = len(captured_data[0])
    n_rows = len(captured_data)
    fig_compare = plt.figure(figsize=(7 * n_cols, 6 * n_rows))

    for row, (axes_data, name) in enumerate(zip(captured_data, agent_names)):
        for col, data in enumerate(axes_data):
            subplot_idx = row * n_cols + col + 1

            if data['is_3d']:
                ax = fig_compare.add_subplot(n_rows, n_cols, subplot_idx, projection='3d')
                for X, Y, Z, kw in data['surfaces']:
                    ax.plot_surface(X, Y, Z, **kw)
                ax.set_xlim(data['xlim'])
                ax.set_ylim(data['ylim'])
                ax.set_zlim(data['zlim'])
                ax.set_xlabel(data['xlabel'])
                ax.set_ylabel(data['ylabel'])
                ax.set_zlabel(data['zlabel'])
                ax.view_init(data['elev'], data['azim'])
            else:
                ax = fig_compare.add_subplot(n_rows, n_cols, subplot_idx)
                for xdata, ydata, label, color in data['lines']:
                    ax.plot(xdata, ydata, label=label, color=color)
                for arr, cmap, vmin, vmax, extent in data['images']:
                    shown = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                                      extent=extent, aspect='auto')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = plt.colorbar(shown, ticks=[0, 1], cax=cax)
                    cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])
                ax.set_xlim(data['xlim'])
                ax.set_ylim(data['ylim'])
                ax.set_xticks(data['xticks'])
                ax.set_yticks(data['yticks'])
                ax.grid(True)
                if data['lines']:
                    ax.legend()

            ax.set_title(f"[{name}] {data['title']}")
            ax.set_xlabel(data['xlabel'])
            ax.set_ylabel(data['ylabel'])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_steps_per_episode(all_lengths: List[np.ndarray], labels: List[str], title: str = "Pasos por Episodio", window: int = 1):
    """
    Genera una gráfica de Pasos por Episodio en escala logarítmica corregida.
    
    Args:
        all_lengths: Lista de arrays con los pasos por episodio.
        labels: Nombres de los algoritmos.
        title: Título de la gráfica.
        window: Tamaño de la ventana para media móvil (1 = sin suavizado).
    """
    plt.figure(figsize=(11, 6))
    
    for lengths, label in zip(all_lengths, labels):
        data = np.array(lengths)
        
        # Aplicar media móvil si window > 1 para reducir el ruido visual
        if window > 1:
            data = np.convolve(data, np.ones(window)/window, mode='valid')
        
        plt.plot(data, label=label, linewidth=1.2, alpha=0.8)

    # 1. Configurar escala logarítmica
    plt.yscale('log')
    
    # 2. CORRECCIÓN CLAVE: Ajustar límites del eje Y para que la escala no se comprima.
    # Mountain Car suele fallar a los 200 o 500 pasos; ajustamos el rango visual.
    plt.ylim(80, 220) 
    
    # 3. Forzar los ticks para que coincidan con la imagen de referencia
    y_ticks = [50, 100, 150, 200]
    plt.yticks(y_ticks, [str(t) for t in y_ticks])
    
    # Estética y rejilla
    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Pasos por episodio (escala log)', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Rejilla sutil solo en los ticks principales para limpieza visual
    plt.grid(True, which="major", axis="y", ls="-", alpha=0.3)
    plt.grid(True, which="major", axis="x", ls="-", alpha=0.1)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_cost_to_go(pos_space, vel_space, V_grid, episode_num="Final", agent_name="Agent"):
    """
    Dibuja la superficie 3D del Cost-to-Go para MountainCar.
    Basado en las representaciones clásicas de Sutton & Barto.
    """
    # El cost-to-go es el valor negativo de V(s)
    cost_to_go = -V_grid

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Crear la malla (meshgrid) para Matplotlib
    P, V = np.meshgrid(pos_space, vel_space)

    # Dibujar la superficie 3D.
    # Usamos cost_to_go.T (transpuesta) para alinear correctamente los ejes X e Y con la matriz
    surf = ax.plot_surface(P, V, cost_to_go.T, cmap='viridis',
                           edgecolor='none', alpha=0.9)

    ax.set_xlabel('Posición', fontsize=12)
    ax.set_ylabel('Velocidad', fontsize=12)
    ax.set_zlabel('Superficie de valores (Pasos estimados)', fontsize=12)
    ax.set_title(f'[{agent_name}] Superficie de valores', fontsize=14)

    # Añadir barra de color
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1, label='Pasos')

    # Ajustar el ángulo de visión para que se parezca al del libro
    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.show()

def plot_policy_comparison(agents, labels, env, resolution=100):
    """
    Compara las políticas de diferentes agentes para Mountain Car.
    """
    fig, axes = plt.subplots(1, len(agents), figsize=(14, 6), sharey=True)
    if len(agents) == 1: axes = [axes]
    
    pos_range = np.linspace(env.observation_space.low[0], env.observation_space.high[0], resolution)
    vel_range = np.linspace(env.observation_space.low[1], env.observation_space.high[1], resolution)
    
    for idx, agent in enumerate(agents):
        z = np.zeros((resolution, resolution))
        
        for i, p in enumerate(pos_range):
            for j, v in enumerate(vel_range):
                state = np.array([p, v])
                
                # Adaptación para tu código específico:
                # Usamos _get_all_q_values para obtener los valores Q aproximados
                # y seleccionamos la acción con el valor más alto (greedy).
                if hasattr(agent, '_get_all_q_values'):
                    q_values = agent._get_all_q_values(state)
                    z[j, i] = np.argmax(q_values)
                else:
                    # Caso genérico por si el agente de DQL usa otro método (ej: predict)
                    # Aquí podrías añadir una condición para tu agente DQL
                    try:
                        z[j, i] = agent.get_best_action(state)
                    except AttributeError:
                        z[j, i] = agent.get_action(state) # Cuidado: esto podría incluir epsilon

        # Colores: Rojo (Izquierda), Gris (Nada), Verde (Derecha)
        axes[idx].contourf(pos_range, vel_range, z, levels=[-0.5, 0.5, 1.5, 2.5], 
                                colors=['#e74c3c', '#bdc3c7', '#2ecc71'])
        
        axes[idx].set_title(f"[{labels[idx]}] Elecciones del agente", fontsize=14)
        axes[idx].set_xlabel("Posición")
        if idx == 0: axes[idx].set_ylabel("Velocidad")

    # Leyenda
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#e74c3c', lw=4),
                    Line2D([0], [0], color='#bdc3c7', lw=4),
                    Line2D([0], [0], color='#2ecc71', lw=4)]
    fig.legend(custom_lines, ['Izquierda (0)', 'Nada (1)', 'Derecha (2)'], 
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.show()

def compare_rewards(agents_data, agent_names, window=4000):
    plt.figure(figsize=(10, 5))
    for (rewards, _), name in zip(agents_data, agent_names):
        means = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(means, label=name)

    plt.title(f"Episode Rewards (Moving Average, Window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_lengths(agents_data, agent_names, window=4000):
    plt.figure(figsize=(10, 5))
    for (_, lengths), name in zip(agents_data, agent_names):
        means = np.convolve(lengths, np.ones(window) / window, mode='valid')
        plt.plot(means, label=name)

    plt.title(f"Episode Lengths (Moving Average, Window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Length")
    plt.grid(True)
    plt.legend()
    plt.show()