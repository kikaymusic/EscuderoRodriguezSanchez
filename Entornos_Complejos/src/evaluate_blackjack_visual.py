import time
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

ACTION_NAMES = {0: "STICK", 1: "HIT"}  # Blackjack-v1 usual mapping

def evaluate_blackjack_visual(env, agent, n_episodes=20, seed_base=1234, delay=0.6, greedy_eval=True):
    """
    Visual evaluation of an agent in Blackjack.
    Shows episodes one by one with a running scoreboard.
    No training is performed.
    """

    # Scoreboard
    wins = 0
    losses = 0
    draws = 0

    # UI outputs
    out_plot = widgets.Output()
    out_info = widgets.Output()

    def choose_action(state):
        # Greedy evaluation from Q-table (preferred for comparing before/after training)
        if greedy_eval and hasattr(agent, "q_table") and state in agent.q_table:
            q = np.array(agent.q_table[state], dtype=float)
            max_q = np.max(q)
            best_actions = np.where(q == max_q)[0]
            return int(np.random.choice(best_actions))

        # Fallback to agent policy
        return int(agent.get_action(state))

    def render_frame(title, extra_lines=None):
        if extra_lines is None:
            extra_lines = []

        with out_plot:
            clear_output(wait=True)
            frame = env.render()
            plt.figure(figsize=(7, 4.5))
            if frame is not None:
                plt.imshow(frame)
                plt.axis("off")
            else:
                plt.text(0.5, 0.5, "No frame available from env.render()", ha="center", va="center")
                plt.axis("off")
            plt.title(title)
            plt.show()

        with out_info:
            clear_output(wait=True)
            print("=" * 70)
            print("BLACKJACK LIVE EVALUATION")
            print("=" * 70)
            print(f"Mode                : {'Greedy (from Q-table)' if greedy_eval else 'Agent policy'}")
            print(f"Episodes target      : {n_episodes}")
            print(f"Delay                : {delay:.2f}s")
            print("-" * 70)
            print(f"Wins                 : {wins}")
            print(f"Losses               : {losses}")
            print(f"Draws                : {draws}")
            played = wins + losses + draws
            win_rate = (wins / played) * 100 if played > 0 else 0.0
            print(f"Win rate             : {win_rate:.2f}%")
            print("-" * 70)
            if hasattr(agent, "q_table"):
                print(f"States in Q-table    : {len(agent.q_table)}")
                nonzero = sum(
                    1 for _, qv in agent.q_table.items()
                    if np.any(np.abs(np.array(qv, dtype=float)) > 1e-12)
                )
                print(f"Non-zero Q states    : {nonzero}")
                print("-" * 70)

            for line in extra_lines:
                print(line)
            print("=" * 70)

    # Display static container once
    display(widgets.VBox([out_plot, out_info]))

    # Main evaluation loop
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed_base + ep)
        done = False
        step_idx = 0
        total_reward = 0.0
        last_action = None

        while not done:
            player_sum, dealer_card, usable_ace = state
            q_vals = None
            if hasattr(agent, "q_table") and state in agent.q_table:
                q_vals = np.array(agent.q_table[state], dtype=float)

            action = choose_action(state)
            last_action = action

            extra_lines = [
                f"Episode             : {ep+1}/{n_episodes}",
                f"Step                : {step_idx}",
                f"State               : player_sum={player_sum}, dealer_card={dealer_card}, usable_ace={usable_ace}",
                f"Action              : {ACTION_NAMES.get(action, action)} ({action})",
            ]
            if q_vals is None:
                extra_lines.append("Q(state)             : not initialized")
            else:
                extra_lines.append(f"Q(state)             : {q_vals}")
                if len(q_vals) >= 2:
                    extra_lines.append(
                        f"Q(STICK), Q(HIT)     : {q_vals[0]:.4f}, {q_vals[1]:.4f}"
                    )

            render_frame(
                title=f"Episode {ep+1}/{n_episodes} | Step {step_idx} | State={state}",
                extra_lines=extra_lines
            )

            time.sleep(delay)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            state = next_state
            total_reward += reward
            step_idx += 1

        # Episode result
        if total_reward > 0:
            wins += 1
            result_label = "WIN"
        elif total_reward < 0:
            losses += 1
            result_label = "LOSS"
        else:
            draws += 1
            result_label = "DRAW"

        extra_lines = [
            f"Episode             : {ep+1}/{n_episodes}",
            f"Final result        : {result_label}",
            f"Total reward        : {total_reward}",
            f"Steps               : {step_idx}",
            f"Last action         : {ACTION_NAMES.get(last_action, last_action) if last_action is not None else None}",
            f"Final state         : {state}",
        ]

        render_frame(
            title=f"Episode {ep+1}/{n_episodes} FINISHED | {result_label} | reward={total_reward}",
            extra_lines=extra_lines
        )

        time.sleep(max(delay, 0.8))

    # Final summary
    played = wins + losses + draws
    win_rate = (wins / played) * 100 if played > 0 else 0.0
    render_frame(
        title="Evaluation finished",
        extra_lines=[
            f"Completed episodes   : {played}",
            f"Final wins           : {wins}",
            f"Final losses         : {losses}",
            f"Final draws          : {draws}",
            f"Final win rate       : {win_rate:.2f}%",
        ]
    )

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "episodes": played,
    }