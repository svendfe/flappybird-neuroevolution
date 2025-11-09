import pygad
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import json

# --- 1. Global Variables ---
game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()

# Fixed input/output sizes
num_inputs = 8
num_outputs = 1

# Architecture search space
MIN_LAYERS = 1
MAX_LAYERS = 3
MIN_NEURONS_PER_LAYER = 8
MAX_NEURONS_PER_LAYER = 24

# --- Manual Neural Network Implementation ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def forward_pass(inputs, weights_list, architecture):
    """
    Perform forward pass through the network.
    
    Args:
        inputs: Input array of shape (1, num_inputs)
        weights_list: List of weight matrices for each layer
        architecture: List of hidden layer sizes
    
    Returns:
        Output of the network
    """
    activation = inputs
    
    # Hidden layers with ReLU
    for i, weight_matrix in enumerate(weights_list[:-1]):
        activation = np.dot(activation, weight_matrix)
        activation = relu(activation)
    
    # Output layer with Sigmoid
    activation = np.dot(activation, weights_list[-1])
    activation = sigmoid(activation)
    
    return activation

# --- Helper Functions ---
def get_network_inputs(state):
    """Convert game state to normalized network inputs."""
    return np.array([
        (state['player_y'] - 256) / 256.0,
        (state['player_vel'] / 10.0),
        (state['next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_pipe_top_y'] - 256) / 256.0,
        (state['next_pipe_bottom_y'] - 256) / 256.0,
        (state['next_next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_next_pipe_top_y'] - 256) / 256.0,
        (state['next_next_pipe_bottom_y'] - 256) / 256.0,
    ]).reshape(1, num_inputs)


def decode_architecture(arch_genes):
    """
    Decode architecture genes into a list of hidden layer sizes.
    
    arch_genes format: [num_layers, layer1_size, layer2_size, layer3_size]
    """
    num_layers = int(np.clip(arch_genes[0], MIN_LAYERS, MAX_LAYERS))
    
    layers = []
    for i in range(num_layers):
        layer_size = int(np.clip(arch_genes[i + 1], 
                                  MIN_NEURONS_PER_LAYER, 
                                  MAX_NEURONS_PER_LAYER))
        layers.append(layer_size)
    
    return layers


def genes_to_weights(weight_genes, architecture):
    """
    Convert flat weight genes into a list of weight matrices.
    
    Args:
        weight_genes: Flat array of weights
        architecture: List of hidden layer sizes
    
    Returns:
        List of weight matrices [input->hidden1, hidden1->hidden2, ..., hiddenN->output]
    """
    layer_sizes = [num_inputs] + architecture + [num_outputs]
    weights_list = []
    idx = 0
    
    for i in range(len(layer_sizes) - 1):
        rows = layer_sizes[i]
        cols = layer_sizes[i + 1]
        num_weights = rows * cols
        
        # Extract weights for this layer
        layer_weights = weight_genes[idx:idx + num_weights]
        
        # Pad with zeros if we don't have enough weights
        if len(layer_weights) < num_weights:
            layer_weights = np.concatenate([
                layer_weights, 
                np.zeros(num_weights - len(layer_weights))
            ])
        
        # Reshape to matrix
        weight_matrix = layer_weights.reshape(rows, cols)
        weights_list.append(weight_matrix)
        idx += num_weights
    
    return weights_list


def calculate_num_weights(architecture):
    """Calculate total number of weights needed for an architecture."""
    layer_sizes = [num_inputs] + architecture + [num_outputs]
    total = sum(layer_sizes[i] * layer_sizes[i + 1] 
                for i in range(len(layer_sizes) - 1))
    return total


def play_game_with_weights(weights_list, architecture):
    """
    Play a full game with the given weights and return the fitness.
    """
    global p
    
    p.reset_game()
    
    frames_lived = 0
    max_frames = 10000

    while not p.game_over() and frames_lived < max_frames:
        frames_lived += 1

        state = p.getGameState()
        inputs_np = get_network_inputs(state)

        # Forward pass through our manual network
        prediction = forward_pass(inputs_np, weights_list, architecture)

        action = None
        if prediction[0, 0] > 0.5:
            action = 119

        p.act(action)

    score = p.score()
    survival_bonus = frames_lived / 100.0
    fitness = (score * 100) + survival_bonus
    fitness = max(0, fitness)
    return fitness


# --- Architecture + Weight Fitness Function ---

def fitness_func_neuroevolution(ga_instance, solution, sol_idx):
    """
    Fitness function that evaluates both architecture and weights.
    
    Solution format:
    [arch_gene1, arch_gene2, arch_gene3, arch_gene4, weight1, weight2, ...]
    """
    try:
        # Decode architecture from first 4 genes
        arch_genes = solution[:4]
        architecture = decode_architecture(arch_genes)
        
        # Get the weight genes (everything after architecture genes)
        weight_genes = solution[4:]
        
        # Convert genes to weight matrices
        weights_list = genes_to_weights(weight_genes, architecture)
        
        # Play the game and get fitness
        fitness = play_game_with_weights(weights_list, architecture)
        
        # Add a small penalty for network complexity
        complexity_penalty = sum(architecture) * 0.01
        fitness -= complexity_penalty
        
        return fitness
        
    except Exception as e:
        # If something goes wrong, return very low fitness
        print(f"Error evaluating solution {sol_idx}: {e}")
        return -1000


# --- Generation Callback ---

def on_generation(ga_instance):
    """Track progress and save the best architectures found."""
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    avg_fit = np.mean(ga_instance.last_generation_fitness)
    
    # Decode the best architecture
    best_solution = ga_instance.best_solution()[0]
    best_arch = decode_architecture(best_solution[:4])
    num_weights = calculate_num_weights(best_arch)
    
    print(f"Gen {gen:3d} | Best: {best_fit:7.2f} | Avg: {avg_fit:7.2f} | "
          f"Arch: {best_arch} | Weights: {num_weights}")

    # Save checkpoint every 10 generations
    if gen % 10 == 0:
        checkpoint_data = {
            'solution': best_solution.tolist(),
            'architecture': best_arch,
            'fitness': float(best_fit),
            'generation': gen,
            'num_weights': num_weights
        }
        with open(f"checkpoint_gen_{gen}.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  → Checkpoint saved: checkpoint_gen_{gen}.json")


# --- 6. Calculate Maximum Possible Weights ---

# For the largest possible network (3 layers of 32 neurons each)
max_total_neurons = [num_inputs] + [MAX_NEURONS_PER_LAYER] * MAX_LAYERS + [num_outputs]
max_weights = sum(max_total_neurons[i] * max_total_neurons[i + 1] 
                  for i in range(len(max_total_neurons) - 1))

print("--- Neuroevolution Setup ---")
print(f"Architecture search space:")
print(f"  Layers: {MIN_LAYERS}-{MAX_LAYERS}")
print(f"  Neurons per layer: {MIN_NEURONS_PER_LAYER}-{MAX_NEURONS_PER_LAYER}")
print(f"  Maximum possible weights: {max_weights}")

# Total genes = 4 architecture genes + max possible weights
num_genes = 4 + max_weights
print(f"  Total genes per solution: {num_genes}")

# --- 7. Setup PyGAD GA for Neuroevolution ---

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=8,
    fitness_func=fitness_func_neuroevolution,
    sol_per_pop=40,
    num_genes=num_genes,
    # Gene ranges
    gene_space=[
        # Architecture genes
        {'low': MIN_LAYERS, 'high': MAX_LAYERS},  # num_layers
        {'low': MIN_NEURONS_PER_LAYER, 'high': MAX_NEURONS_PER_LAYER},  # layer 1
        {'low': MIN_NEURONS_PER_LAYER, 'high': MAX_NEURONS_PER_LAYER},  # layer 2
        {'low': MIN_NEURONS_PER_LAYER, 'high': MAX_NEURONS_PER_LAYER},  # layer 3
        # Weight genes (remaining genes can be any value)
    ] + [{'low': -0.5, 'high': 0.5}] * max_weights,
    
    on_generation=on_generation,
    mutation_type="random",
    mutation_percent_genes=5,
    parent_selection_type="tournament",
    crossover_type="single_point",
    keep_parents=3,
    save_best_solutions=False,  # Changed to False to avoid memory issues
)

print("\n--- Starting Neuroevolution ---")
print("This will take longer than standard training...")
print("The algorithm is evolving both architecture AND weights!\n")
start_time = time.time()

# Run the genetic algorithm
ga_instance.run()

end_time = time.time()
print(f"\n--- Training Finished in {end_time - start_time:.2f} seconds ---")

# --- 8. Results ---

best_solution, best_fitness, best_idx = ga_instance.best_solution()
best_architecture = decode_architecture(best_solution[:4])
best_num_weights = calculate_num_weights(best_architecture)

print(f"\n{'='*60}")
print(f"BEST SOLUTION FOUND")
print(f"{'='*60}")
print(f"Fitness: {best_fitness:.2f}")
print(f"Architecture: {best_architecture}")
print(f"Total neurons in hidden layers: {sum(best_architecture)}")
print(f"Total weights: {best_num_weights}")

# Save the complete solution
final_data = {
    'solution': best_solution.tolist(),
    'architecture': best_architecture,
    'fitness': float(best_fitness),
    'num_weights': best_num_weights,
    'training_time_seconds': end_time - start_time
}

with open("best_neuroevolution_solution.json", 'w') as f:
    json.dump(final_data, f, indent=2)

np.save("best_neuroevolution_weights.npy", best_solution)
print(f"\nSolution saved to: best_neuroevolution_solution.json")
print(f"Weights saved to: best_neuroevolution_weights.npy")

# --- 9. Watch the Best Network Play ---

print("\n--- Running Best Network ---")
print("Close the game window to exit.\n")

# Re-initialize with display
p = PLE(game, fps=30, display_screen=True, force_fps=True)
p.init()
p.reset_game()

# Convert best solution to weights
best_weight_genes = best_solution[4:]
best_weights_list = genes_to_weights(best_weight_genes, best_architecture)

# Play the game
while not p.game_over():
    state = p.getGameState()
    inputs_np = get_network_inputs(state)

    # Forward pass
    prediction = forward_pass(inputs_np, best_weights_list, best_architecture)
    
    action = None
    if prediction[0, 0] > 0.5:
        action = 119

    p.act(action)
    time.sleep(1/30.0)

print(f"\nFinal Score: {p.score()}")
print(f"Best Architecture: {best_architecture}")
print("\nNeuroevolution complete! 🧬🎮")