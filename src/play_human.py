import cv2
from tetris import Tetris
from time import sleep
import time
import os
import numpy as np
import datetime

#KEY_LEFT = 2   # Left arrow (user system)
#KEY_RIGHT = 3  # Right arrow (user system)
#KEY_DOWN = 1   # Down arrow (user system)
#KEY_UP = 0     # Up arrow (user system)
KEY_LEFT = 81   # Left arrow (for KBD)
KEY_RIGHT = 83  # Right arrow (for KBD)
KEY_DOWN = 84   # Down arrow (for KBD)
KEY_UP = 82     # Up arrow (for KBD)
KEY_ESC = 27   # Escape
KEY_R = ord('r')

# Tunable gravity delay in milliseconds
GRAVITY_DELAY = 500  # Lower is faster gravity (e.g., 500ms)

# At the top, after imports
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
data = []

def main():
    env = Tetris()
    done = False
    print('Controls: ← (left), → (right), ↓ (down), ↑ (rotate), r (reset), esc (quit)')
    print(f'Gravity: piece moves down every {GRAVITY_DELAY} ms')

    last_gravity_time = time.time()
    env.render()  # Initial render

    while True:
        key = cv2.waitKey(1)  # Fast polling for input
        # print(f"Key pressed: {key}")  # Config: print the key code

        now = time.time()
        gravity_applied = False
        if not done and (now - last_gravity_time) * 1000 >= GRAVITY_DELAY:
            # Gravity: move piece down
            env.current_pos[1] += 1
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
                # Place piece
                env.board = env._add_piece_to_board(env._get_rotated_piece(), env.current_pos)
                lines_cleared, env.board = env._clear_lines(env.board)
                env.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
                env._new_round()
                if env.game_over:
                    print('Game Over! Press r to reset or esc to quit.')
                    done = True
            last_gravity_time = now
            env.render()
            gravity_applied = True

        if key == -1:
            continue

        if key == KEY_ESC:
            print('Exiting...')
            break
        if key == KEY_R:
            print('Resetting game...')
            env.reset()
            done = False
            last_gravity_time = time.time()
            env.render()
            continue
        if done:
            continue

        # Copy current state
        x, rotation = env.current_pos[0], env.current_rotation
        state_changed = False

        if key == KEY_LEFT:
            x = max(0, x - 1)
            state_changed = True
        elif key == KEY_RIGHT:
            x = min(Tetris.BOARD_WIDTH - 1, x + 1)
            state_changed = True
        elif key == KEY_DOWN:
            # Drop piece by one
            env.current_pos[1] += 1
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
            state_changed = True
        elif key == KEY_UP:
            # Rotate piece
            rotation = (rotation + 90) % 360
            state_changed = True
        else:
            continue

        # Play move if not down
        if key in [KEY_LEFT, KEY_RIGHT, KEY_UP]:
            # Try to move/rotate, check collision
            env.current_pos[0] = x
            env.current_rotation = rotation
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                # Undo move if collision
                if key == KEY_LEFT:
                    env.current_pos[0] += 1
                elif key == KEY_RIGHT:
                    env.current_pos[0] -= 1
                elif key == KEY_UP:
                    env.current_rotation = (rotation - 90) % 360
        # If down, try to move down, if can't, place piece
        elif key == KEY_DOWN:
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
                # Place piece
                env.board = env._add_piece_to_board(env._get_rotated_piece(), env.current_pos)
                lines_cleared, env.board = env._clear_lines(env.board)
                env.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
                env._new_round()
                if env.game_over:
                    print('Game Over! Press r to reset or esc to quit.')
                    done = True
            # else: piece moved down by one
        if state_changed:
            env.render()
            # --- Data collection for behavioral cloning ---
            # Get the current state (use env._get_board_props)
            state = env._get_board_props(env.board)
            # Map key to action index
            if key == KEY_LEFT:
                action = 0
            elif key == KEY_RIGHT:
                action = 1
            elif key == KEY_DOWN:
                action = 2
            elif key == KEY_UP:
                action = 3
            else:
                action = -1
            if action != -1:
                data.append((state, action))

    if data:
        try:
            states = np.array([item[0] for item in data])
            actions = np.array([item[1] for item in data])

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            np.save(os.path.join(DATA_DIR, f'human_demo_states_{timestamp}.npy'), states)
            np.save(os.path.join(DATA_DIR, f'human_demo_actions_{timestamp}.npy'), actions)
            print(f"Saved {len(data)} data points to separate state/action files")

        except ValueError as e:
            print(f"Could not save as numpy arrays: {e}")
            print("States shape info:", [np.array(item[0]).shape for item in data[:5]])
            import pickle
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'human_demo_{timestamp}.pkl'
            with open(os.path.join(DATA_DIR, filename), 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved as pickle file: {filename}")

    #if data:
    #    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #    filename = f'human_demo_{timestamp}.npy'
    #    np.save(os.path.join(DATA_DIR, filename), data, allow_pickle=True)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
