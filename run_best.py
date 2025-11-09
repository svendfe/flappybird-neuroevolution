import numpy as np
import json
from ple.games.flappybird import FlappyBird
from ple import PLE
import time

# --- Configuration ---
CHECKPOINT_FILE = "best_academic_solution.json"  # Change to your checkpoint file
NUM_GAMES = 5  # Number of games to play

# Network configuration (must match training)
num_inputs = 8
num_hidden = 6
num_outputs = 1

# --- Load the solution ---
print("="*70)
print("LOADING BEST EVOLVED NETWORK")
print("="*70)

try:
    with open(CHECKPOINT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"\nLoaded: {CHECKPOINT_FILE}")
    print(f"Training fitness: {data['fitness']:.2f}")
    print(f"Total connections: {data['analysis']['total_connections']}")
    print(f"  - Direct: {data['analysis']['direct_connections']}")
    print(f"  - Through hidden: {data['analysis']['hidden_paths']}")
    print(f"Active hidden neurons: {data['analysis']['active_hidden_neurons']}")
    print(f"Total weights: {data['analysis']['num_weights']}")
    
except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{CHECKPOINT_FILE}'")
    print("\nAvailable options:")
    print("1. Run the training script first to generate a checkpoint")
    print("2. Change CHECKPOINT_FILE to point to an existing checkpoint")
    print("3. Check that the file is in the same directory as this script")
    exit(1)

# --- Helper Functions (copied from training script) ---

def get_active_connections(topology_genes, I, H, O):
    """Decode binary topology genes into active connections."""
    connections = {
        'direct': [],
        'through_hidden': []
    }
    
    num_basic_arch = I * O * (H + 1)
    
    for idx in range(min(len(topology_genes), num_basic_arch)):
        if topology_genes[idx] == 1:
            # Decode this basic architecture
            architectures_per_output = I * (H + 1)
            output_idx = idx // architectures_per_output
            remainder = idx % architectures_per_output
            input_idx = remainder // (H + 1)
            hidden_idx = remainder % (H + 1) - 1
            
            if hidden_idx == -1:
                connections['direct'].append((input_idx, output_idx))
            else:
                connections['through_hidden'].append((input_idx, hidden_idx, output_idx))
    
    return connections

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def forward_pass_with_topology(inputs, connections, weights, I, H, O):
    """Forward pass through network with arbitrary topology."""
    inputs = inputs.flatten()
    hidden_activations = np.zeros(H)
    outputs = np.zeros(O)
    
    # Input → Hidden
    for idx, (input_idx, hidden_idx, output_idx) in enumerate(connections['through_hidden']):
        weight = weights['input_hidden_weights'][idx]
        hidden_activations[hidden_idx] += inputs[input_idx] * weight
    
    # Apply ReLU
    hidden_activations = relu(hidden_activations)
    
    # Hidden → Output
    for idx, (input_idx, hidden_idx, output_idx) in enumerate(connections['through_hidden']):
        weight = weights['hidden_output_weights'][idx]
        outputs[output_idx] += hidden_activations[hidden_idx] * weight
    
    # Direct connections
    for idx, (input_idx, output_idx) in enumerate(connections['direct']):
        weight = weights['direct_weights'][idx]
        outputs[output_idx] += inputs[input_idx] * weight
    
    # Apply sigmoid
    outputs = sigmoid(outputs)
    
    return outputs.reshape(1, O)

def get_network_inputs(state):
    """Convert game state to normalized network inputs."""
    return np.array([
        (state['player_y'] - 256) / 256.0,
        state['player_vel'] / 10.0,
        (state['next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_pipe_top_y'] - 256) / 256.0,
        (state['next_pipe_bottom_y'] - 256) / 256.0,
        (state['next_next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_next_pipe_top_y'] - 256) / 256.0,
        (state['next_next_pipe_bottom_y'] - 256) / 256.0,
    ]).reshape(1, num_inputs)

def decode_weights(weight_genes, connections):
    """Distribute weight genes to connections."""
    weights = {
        'direct_weights': [],
        'input_hidden_weights': [],
        'hidden_output_weights': []
    }
    
    idx = 0
    
    # Direct connection weights
    for _ in connections['direct']:
        if idx < len(weight_genes):
            weights['direct_weights'].append(weight_genes[idx])
            idx += 1
        else:
            weights['direct_weights'].append(0.0)
    
    # Hidden connection weights
    for _ in connections['through_hidden']:
        # Input→Hidden
        if idx < len(weight_genes):
            weights['input_hidden_weights'].append(weight_genes[idx])
            idx += 1
        else:
            weights['input_hidden_weights'].append(0.0)
        
        # Hidden→Output
        if idx < len(weight_genes):
            weights['hidden_output_weights'].append(weight_genes[idx])
            idx += 1
        else:
            weights['hidden_output_weights'].append(0.0)
    
    return weights

# --- Decode the Network ---

topology_genes = data['topology_bits']
weight_genes = data['weights']

connections = get_active_connections(topology_genes, num_inputs, num_hidden, num_outputs)
weights = decode_weights(weight_genes, connections)

print("\n" + "="*70)
print("NETWORK TOPOLOGY")
print("="*70)

print("\nDirect Connections (Input → Output):")
if connections['direct']:
    for input_idx, output_idx in connections['direct']:
        print(f"  i{input_idx} → o{output_idx}")
else:
    print("  None")

print("\nConnections Through Hidden Layer:")
if connections['through_hidden']:
    # Group by hidden neuron for better visualization
    by_hidden = {}
    for input_idx, hidden_idx, output_idx in connections['through_hidden']:
        if hidden_idx not in by_hidden:
            by_hidden[hidden_idx] = []
        by_hidden[hidden_idx].append((input_idx, output_idx))
    
    for hidden_idx in sorted(by_hidden.keys()):
        print(f"\n  Hidden Neuron h{hidden_idx}:")
        for input_idx, output_idx in by_hidden[hidden_idx]:
            print(f"    i{input_idx} → h{hidden_idx} → o{output_idx}")
else:
    print("  None")

# --- Play Multiple Games ---

print("\n" + "="*70)
print(f"PLAYING {NUM_GAMES} GAMES")
print("="*70)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=True)
p.init()

scores = []
frames_survived = []

for game_num in range(NUM_GAMES):
    print(f"\n--- Game {game_num + 1}/{NUM_GAMES} ---")
    
    p.reset_game()
    frames = 0
    
    while not p.game_over():
        frames += 1
        
        state = p.getGameState()
        inputs_np = get_network_inputs(state)
        
        # Forward pass
        prediction = forward_pass_with_topology(inputs_np, connections, weights,
                                               num_inputs, num_hidden, num_outputs)
        
        # Decide action
        action = None
        if prediction[0, 0] > 0.5:
            action = 119  # Flap
        
        p.act(action)
        time.sleep(1/30.0)
    
    score = p.score()
    scores.append(score)
    frames_survived.append(frames)
    
    print(f"  Score: {score}")
    print(f"  Frames survived: {frames}")
    
    # Wait a bit between games
    time.sleep(1)

# --- Statistics ---

print("\n" + "="*70)
print("PERFORMANCE STATISTICS")
print("="*70)

print(f"\nScores across {NUM_GAMES} games:")
print(f"  Best: {max(scores)}")
print(f"  Worst: {min(scores)}")
print(f"  Average: {np.mean(scores):.2f}")
print(f"  Median: {np.median(scores):.2f}")
print(f"  Std Dev: {np.std(scores):.2f}")

print(f"\nFrames survived:")
print(f"  Best: {max(frames_survived)}")
print(f"  Worst: {min(frames_survived)}")
print(f"  Average: {np.mean(frames_survived):.2f}")

print(f"\nAll scores: {scores}")

print("\n" + "="*70)
print("TESTING COMPLETE! 🎮")
print("="*70)