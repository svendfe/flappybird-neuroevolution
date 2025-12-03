import pygad
import pygad.nn
import pygad.gann
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time

# --- 1. Global Variables ---
game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=True)  # No display during training
p.init()
num_solutions = 100

# Define the Neural Network architecture
# Inputs: 8 (player_y, player_vel, next_pipe_dist_x, next_pipe_top_y, next_pipe_bottom_y,
#            next_next_pipe_dist_x, next_next_pipe_top_y, next_next_pipe_bottom_y)
# Output: 1 (Sigmoid activation: > 0.5 = flap, <= 0.5 = do nothing)
num_inputs = 8
num_outputs = 1

# Neural network architecture - can experiment with these
num_neurons_hidden_layers = [16, 8, 4]  # Two hidden layers

# --- 2. Helper Function for Input Preprocessing ---

def get_network_inputs(state):
    """
    Convert game state to normalized network inputs.
    Normalization helps the network learn more effectively.
    """
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

def update_network_weights(last_layer, weights_vector):
    """
    Updates the weights of a PyGAD network using a 1D solution vector.
    Traverses the linked list of layers to update them in correct order.
    """
    layers = []
    current = last_layer
    
    # 1. Traverse backwards from Output -> Input
    while current is not None:
        layers.append(current)
        # Safely get previous_layer. If it doesn't exist (e.g., InputLayer), returns None.
        current = getattr(current, "previous_layer", None)
    
    # 2. Reverse to get Input -> Output order
    layers.reverse()
    
    start_idx = 0
    
    # 3. Iterate over the ordered layers
    for layer in layers:
        # Skip layers that don't have weights (like InputLayer)
        if not hasattr(layer, "initial_weights"):
            continue
            
        # Calculate the number of weights in this layer
        w_shape = layer.initial_weights.shape
        w_size = np.prod(w_shape)
        
        # Slice the 1D gene vector to get weights for this layer
        w_chunk = weights_vector[start_idx : start_idx + w_size]
        
        # Reshape and assign to the layer
        layer.trained_weights = w_chunk.reshape(w_shape)
        
        start_idx += w_size

        
# --- 3. Fitness Function ---

def fitness_func(ga_instance, solution, sol_idx):
    global gann_instance, p
    
    # 1. Use the first network as a 'template' structure
    # We don't care which one we pick, because we are about to overwrite its weights
    network = gann_instance.population_networks[0]
    
    # 2. LOAD THE GENES into the network
    # This fixes the error and ensures we are testing the MUTATED weights
    update_network_weights(network, solution)
    
    p.reset_game()
    frames_lived = 0
    max_frames = 10000 
    total_distance_penalty = 0

    while not p.game_over() and frames_lived < max_frames:
        frames_lived += 1
        state = p.getGameState()
        
        gap_center = (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) / 2
        dist = abs(state['player_y'] - gap_center)
        total_distance_penalty += dist

        inputs_np = get_network_inputs(state)
        
        # 3. Predict using the updated network
        prediction = pygad.nn.predict(last_layer=network,
                                    data_inputs=inputs_np,
                                    problem_type="regression")
        
        action = None
        if prediction[0] > 0.5:
            action = 119 
        p.act(action)

    score = p.score()
    
    avg_distance = total_distance_penalty / frames_lived if frames_lived > 0 else 256
    fitness = ((score + 5) * 1000) + (frames_lived) - (avg_distance * 1.5)
    
    return max(0, fitness)

# --- 4. Generation Callback ---

def on_generation(ga_instance):
    """
    This function is called after each generation.
    Updates the GANN networks with evolved weights and tracks progress.
    """
    global gann_instance

    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    mean_fit = np.mean(ga_instance.last_generation_fitness)
    
    print(f"Gen {gen:3d} | Best: {best_fit:7.2f} | mean: {mean_fit:7.2f}")

    # Save checkpoints every 10 generations
    if gen % 10 == 0:
        np.save(f"checkpoint_gen_{gen}.npy", ga_instance.best_solution())
        print(f"  → Checkpoint saved: checkpoint_gen_{gen}.npy")

    # Update the GANN population with the new weights from the GA
    population_matrices = pygad.gann.population_as_matrices(
        population_networks=gann_instance.population_networks,
        population_vectors=ga_instance.population
    )
    gann_instance.update_population_trained_weights(population_matrices)

# Crossover custom
def blx_alpha_crossover(parents, offspring_size, ga_instance):
    # Dynamic alpha: Starts at 0.5, decays to 0.0 by the last generation
    current_gen = ga_instance.generations_completed
    max_gens = ga_instance.num_generations
    # Linear decay formula
    alpha = 0.5 * (1 - (current_gen / max_gens)) 
    
    offspring = []
    for _ in range(offspring_size[0]):
        parent1_idx = np.random.randint(0, parents.shape[0])
        parent2_idx = np.random.randint(0, parents.shape[0])
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        
        child = np.zeros(offspring_size[1])
        for gene_idx in range(offspring_size[1]):
            min_val = min(parent1[gene_idx], parent2[gene_idx])
            max_val = max(parent1[gene_idx], parent2[gene_idx])
            I = max_val - min_val
            
            # Slide 17: Interval definition [cite: 343]
            C_min = min_val - alpha * I
            C_max = max_val + alpha * I
            
            child[gene_idx] = np.random.uniform(C_min, C_max)
        offspring.append(child)
    return np.array(offspring)

# --- 5. Setup PyGAD GANN (Neural Network) ---
print("--- Initializing Neural Networks ---")
gann_instance = pygad.gann.GANN(
    num_solutions=num_solutions, 
    num_neurons_input=num_inputs,
    num_neurons_output=num_outputs,
    num_neurons_hidden_layers=num_neurons_hidden_layers,
    output_activation="sigmoid",
    hidden_activations="relu"
)

# Get the initial population of weights as 1D vectors
population_vectors = pygad.gann.population_as_vectors(
    population_networks=gann_instance.population_networks
)

# --- 6. Setup PyGAD GA (Genetic Algorithm) ---
num_genes = len(population_vectors)
print(f"Network has {num_genes} weights (genes)")

# Create the GA instance with optimized hyperparameters
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=13,
    fitness_func=fitness_func,
    sol_per_pop=num_solutions,
    num_genes=num_genes,
    initial_population=population_vectors,
    on_generation=on_generation,
    mutation_type="adaptive",
    mutation_probability=[0.03, 0.01],
    parent_selection_type="tournament",
    K_tournament=4,
    crossover_type=blx_alpha_crossover,
    crossover_probability=0.8,
    keep_parents=6, 
    init_range_low=-1.0,
    init_range_high=1.0,
    save_best_solutions=True
)

print("\n--- Starting Training ---")
print("This may take a while depending on your hardware...")
start_time = time.time()

# Run the genetic algorithm
ga_instance.run()

end_time = time.time()
print(f"\n--- Training Finished in {end_time - start_time:.2f} seconds ---")

# --- 7. Results ---

# Get the best solution (best set of weights) found
best_solution, best_fitness, best_idx = ga_instance.best_solution()
print(f"\nBest solution fitness: {best_fitness:.2f}")
print(f"Best solution index: {best_idx}")

# Save the best weights to a file
np.save("best_weights.npy", best_solution)
print("Best weights saved to: best_weights.npy")

# --- 8. Watch the Best Network Play ---

solution, best_fitness, best_idx = ga_instance.best_solution()

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)  # No display during training
p.init()
# 1. Use the first network as a 'template' structure
network = gann_instance.population_networks[0]

# 2. LOAD THE GENES into the network
update_network_weights(network, solution)

p.reset_game()
frames_lived = 0
max_frames = 10000 
total_distance_penalty = 0

while not p.game_over() and frames_lived < max_frames:
    frames_lived += 1
    state = p.getGameState()
    

    inputs_np = get_network_inputs(state)
    
    # 3. Predict using the updated network
    prediction = pygad.nn.predict(last_layer=network,
                                data_inputs=inputs_np,
                                problem_type="regression")
    
    action = None
    if prediction[0] > 0.5:
        action = 119 
        
    p.act(action)

print(f"Game Over. Lasted {frames_lived} frames.")