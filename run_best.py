import pygad
import pygad.nn
import pygad.gann
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE

# --- 1. Helper Functions ---
def get_network_inputs(state):
    return np.array([
        (state['player_y'] - 256) / 256.0,
        (state['player_vel'] / 10.0),
        (state['next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_pipe_top_y'] - 256) / 256.0,
        (state['next_pipe_bottom_y'] - 256) / 256.0,
        (state['next_next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_next_pipe_top_y'] - 256) / 256.0,
        (state['next_next_pipe_bottom_y'] - 256) / 256.0,
    ]).reshape(1, 8)

def update_network_weights(last_layer, weights_vector):
    """
    Manually maps a flat gene vector to the network layers.
    """
    layers = []
    current = last_layer
    
    # Traverse backwards from Output -> Input to get all layers
    while current is not None:
        layers.append(current)
        current = getattr(current, "previous_layer", None)
    
    layers.reverse() # Arrange Input -> Output
    
    start_idx = 0
    
    for layer in layers:
        # Skip layers without weights (like InputLayer)
        if not hasattr(layer, "initial_weights"):
            continue
            
        w_shape = layer.initial_weights.shape
        w_size = np.prod(w_shape)
        
        # Slice the weights from the vector
        w_chunk = weights_vector[start_idx : start_idx + w_size]
        
        # Assign to trained_weights
        layer.trained_weights = w_chunk.reshape(w_shape)
        
        start_idx += w_size

# --- 2. Load the Solution ---
solution = np.load("best_weights.npy")
print("Solution loaded!")

# --- 3. RECREATE THE NETWORK STRUCTURE ---
# CRITICAL: These numbers must match your training script exactly!
# If you used [5] hidden neurons in training, use [5] here.
num_inputs = 8
num_neurons_hidden_layers = [16, 8, 4]
num_outputs = 1

# Create a temporary GANN instance just to generate the network objects
# We don't need the GA part, just the Neural Network container.
gann_instance = pygad.gann.GANN(
    num_solutions=2, 
    num_neurons_input=num_inputs,
    num_neurons_output=num_outputs,
    num_neurons_hidden_layers=num_neurons_hidden_layers,
    output_activation="sigmoid",
    hidden_activations="relu"
)

# Get the network structure (this is a pygad.nn.DenseLayer object)
network = gann_instance.population_networks[0]

# --- 4. Inject the Loaded Weights ---
# Now we put your loaded DNA into the network body
update_network_weights(network, solution)

# --- 5. Run the Game ---
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False) # set force_fps=True to watch at normal speed
p.init()
p.reset_game()

frames_lived = 0
max_frames = 10000 

while not p.game_over() and frames_lived < max_frames:
    frames_lived += 1

    state = p.getGameState()
    inputs_np = get_network_inputs(state)
    
    # Predict using the NETWORK OBJECT (not the solution array)
    prediction = pygad.nn.predict(last_layer=network,
                                  data_inputs=inputs_np,
                                  problem_type="regression")
    
    action = None
    # Adjust threshold if needed (e.g., 0.5 for Sigmoid)
    if prediction[0][0] > 0.5:
        action = 119 
        
    p.act(action)

print(f"Game Over. Lasted {frames_lived} frames.")