import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle

def load_data_safely(filepath):
    """Safely load data from .npy or .pkl files with error handling."""
    try:
        if filepath.endswith('.npy'):
            data = np.load(filepath, allow_pickle=True)
            # Check if the loaded data is empty
            if data.size == 0:
                print(f"Warning: {filepath} is empty, skipping...")
                return []
            return data.tolist() if hasattr(data, 'tolist') else data
        else:
            return []
    except (EOFError, ValueError, pickle.UnpicklingError) as e:
        return []
    except Exception as e:
        return []

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
all_data = []
for fname in os.listdir(DATA_DIR):
    if (fname.startswith('human_demo') or fname.startswith('rl_demo')) and (fname.endswith('.npy')):
        filepath = os.path.join(DATA_DIR, fname)
        file_data = load_data_safely(filepath)
        if file_data:
            if isinstance(file_data, (list, tuple)):
                all_data.extend(file_data)
            else:
                all_data.append(file_data)

if not all_data:
    raise RuntimeError('No demonstration data found in data directory.')

data = all_data
states = []
raw_actions = []

for i, sample in enumerate(data):
    try:
        if len(sample) >= 2:  # Ensure it's a (state, action) pair
            state, action = sample[0], sample[1]
            states.append(state)
            raw_actions.append(action)
        else:
            print(f"Warning: Sample {i} has invalid format: {sample}")
    except Exception as e:
        print(f"Warning: Could not process sample {i}: {e}")

def tuple_to_action(prev_x, prev_rot, x, rot):
    if x < prev_x:
        return 0  # left
    elif x > prev_x:
        return 1  # right
    elif rot != prev_rot:
        return 3  # rotate
    else:
        return 2  # down

actions = []
prev_x, prev_rot = None, None
invalid_actions = []

for i, act in enumerate(raw_actions):
    action_value = None
    if isinstance(act, tuple):
        if len(act) >= 2:
            if prev_x is not None and prev_rot is not None:
                action_value = tuple_to_action(prev_x, prev_rot, act[0], act[1])
            else:
                action_value = 2  # Default to down for the first move
            prev_x, prev_rot = act[0], act[1]
        else:
            action_value = 2  # Default
    else:
        # Handle scalar actions
        try:
            action_value = int(act)
            prev_x, prev_rot = None, None  # Reset for human data
        except (ValueError, TypeError):
            print(f"Warning: Could not convert action to int at index {i}: {act}")
            action_value = 2  # Default
    
    # Validate action is in valid range [0, 3]
    if action_value is not None:
        if 0 <= action_value <= 3:
            actions.append(action_value)
        else:
            print(f"Warning: Invalid action value {action_value} at index {i}, clamping to valid range")
            invalid_actions.append((i, action_value))
            # Clamp to valid range
            action_value = max(0, min(3, action_value))
            actions.append(action_value)
    else:
        print(f"Warning: None action at index {i}, using default")
        actions.append(2)

print(f"Processed {len(actions)} actions")
if invalid_actions:
    print(f"Found {len(invalid_actions)} invalid actions that were clamped:")
    for idx, val in invalid_actions[:10]:  # Show first 10
        print(f"  Index {idx}: {val}")
    if len(invalid_actions) > 10:
        print(f"  ... and {len(invalid_actions) - 10} more")

# Additional validation
unique_actions = np.unique(actions)
print(f"Unique action values found: {unique_actions}")
if np.any(unique_actions < 0) or np.any(unique_actions > 3):
    print("ERROR: Still have invalid actions after processing!")
    # Filter out invalid actions and corresponding states
    valid_indices = [i for i, a in enumerate(actions) if 0 <= a <= 3]
    actions = [actions[i] for i in valid_indices]
    states = [states[i] for i in valid_indices]
    print(f"Filtered to {len(actions)} valid samples")

# Convert states to consistent format
print("Converting states to numpy arrays...")
processed_states = []
for i, state in enumerate(states):
    try:
        # Handle different state formats
        if isinstance(state, (list, tuple)):
            state_array = np.array(state, dtype=np.float32).flatten()
        elif isinstance(state, np.ndarray):
            state_array = state.astype(np.float32).flatten()
        else:
            state_array = np.array([float(state)], dtype=np.float32)
        
        processed_states.append(state_array)
        
    except Exception as e:
        print(f"Warning: Could not process state {i}: {e}")
        continue

if not processed_states:
    raise RuntimeError('No valid states could be processed.')

# Find the maximum state size to pad smaller states
max_state_size = max(len(state) for state in processed_states)
print(f"Maximum state size: {max_state_size}")

# Pad states to consistent size
final_states = []
final_actions = []

for i, (state, action) in enumerate(zip(processed_states, actions)):
    if len(state) < max_state_size:
        # Pad with zeros
        padded_state = np.zeros(max_state_size, dtype=np.float32)
        padded_state[:len(state)] = state
        final_states.append(padded_state)
    else:
        final_states.append(state[:max_state_size])  # Truncate if too long
    
    final_actions.append(action)

# Convert to numpy arrays
states = np.array(final_states, dtype=np.float32)
actions = np.array(final_actions, dtype=np.int32)

print(f"Final data shape: states={states.shape}, actions={actions.shape}")
print(f"Action value range: min={actions.min()}, max={actions.max()}")
print(f"Action distribution: {np.bincount(actions)}")

# Final validation check
invalid_action_mask = (actions < 0) | (actions > 3)
if np.any(invalid_action_mask):
    print(f"ERROR: Found {np.sum(invalid_action_mask)} invalid actions in final dataset!")
    print(f"Invalid values: {actions[invalid_action_mask][:10]}")  # Show first 10
    raise ValueError("Dataset contains invalid action labels. Please check your data preprocessing.")

print("✓ All actions are in valid range [0, 3]")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Build a simple policy network
model = keras.Sequential([
    keras.layers.Input(shape=(states.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')  # 4 possible actions
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Train with validation
print("Starting training...")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the trained policy
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'policy_bc.keras')
model.save(model_path)
print(f"Model saved to: {model_path}")

# Save training history
history_path = os.path.join(model_dir, 'training_history.npy')
np.save(history_path, history.history)
print(f"Training history saved to: {history_path}")
