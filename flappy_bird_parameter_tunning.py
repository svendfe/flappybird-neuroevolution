import pygad
import pygad.nn
import pygad.gann
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time

# --- 1. Global Variables ---
game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=False)  # No display during training
p.init()

# Define the Neural Network architecture
# Inputs: 8 (player_y, player_vel, next_pipe_dist_x, next_pipe_top_y, next_pipe_bottom_y,
#            next_next_pipe_dist_x, next_next_pipe_top_y, next_next_pipe_bottom_y)
# Output: 1 (Sigmoid activation: > 0.5 = flap, <= 0.5 = do nothing)
num_inputs = 8
num_outputs = 1

# Neural network architecture - can experiment with these
num_neurons_hidden_layers = [16, 8]  # Two hidden layers for better learning

# --- 2. Helper Function for Input Preprocessing ---

def get_network_inputs(state):
    """
    Convert game state to normalized network inputs.
    Normalization helps the network learn more effectively.
    """
    return np.array([
        (state['player_y'] - 256) / 256.0,  # Center and normalize
        state['player_vel'] / 10.0,
        (state['next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_pipe_top_y'] - 256) / 256.0,
        (state['next_pipe_bottom_y'] - 256) / 256.0,
        (state['next_next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_next_pipe_top_y'] - 256) / 256.0,
        (state['next_next_pipe_bottom_y'] - 256) / 256.0,
    ]).reshape(1, num_inputs)

# --- 3. Fitness Function ---

def fitness_func(ga_instance, solution, sol_idx):
    """
    Calculates the fitness of a single solution (a set of network weights).
    Playing a full game of Flappy Bird using the network and returning fitness score.
    
    Fitness = (pipes_passed * 100) + (frames_lived / 100)
    This rewards both passing pipes and survival time.
    """
    global gann_instance, p
    
    # Get the specific network corresponding to this solution's index
    network = gann_instance.population_networks[sol_idx]
    
    # Reset the game for a new run
    p.reset_game()
    
    frames_lived = 0
    max_frames = 10000  # Set a max lifetime to avoid infinite loops

    # --- Game Loop ---
    while not p.game_over() and frames_lived < max_frames:
        frames_lived += 1

        # 1. Get the current game state
        state = p.getGameState()
        
        # 2. Format the state as input for the neural network
        inputs_np = get_network_inputs(state)

        # 3. Get the network's decision
        prediction = pygad.nn.predict(last_layer=network,
                                    data_inputs=inputs_np,
                                    problem_type="regression")

        # 4. Translate the output into a game action
        action = None
        if prediction[0] > 0.5:
            action = 119  # 119 is the ASCII code for 'w' (flap) used by PLE

        # 5. Perform the action
        p.act(action)

    # Calculate fitness: reward pipes passed heavily, plus survival bonus
    score = p.score()
    survival_bonus = frames_lived / 100.0
    fitness = (score * 100) + survival_bonus
    
    return fitness

# --- 4. Generation Callback ---

def on_generation(ga_instance):
    """
    This function is called after each generation.
    Updates the GANN networks with evolved weights and tracks progress.
    """
    global gann_instance

    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    avg_fit = np.mean(ga_instance.last_generation_fitness)
    
    print(f"Gen {gen:3d} | Best: {best_fit:7.2f} | Avg: {avg_fit:7.2f}")

    # Save checkpoints every 10 generations
    if gen % 10 == 0:
        np.save(f"checkpoint_gen_{gen}.npy", ga_instance.best_solution()[0])
        print(f"  → Checkpoint saved: checkpoint_gen_{gen}.npy")

    # Update the GANN population with the new weights from the GA
    population_matrices = pygad.gann.population_as_matrices(
        population_networks=gann_instance.population_networks,
        population_vectors=ga_instance.population
    )
    gann_instance.update_population_trained_weights(population_matrices)


# --- 5. Setup PyGAD GANN (Neural Network) ---
print("--- Initializing Neural Networks ---")
gann_instance = pygad.gann.GANN(
    num_solutions=50,  # Must match sol_per_pop
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
num_genes = population_vectors.shape[1]
print(f"Network has {num_genes} weights (genes)")

# Create the GA instance with optimized hyperparameters
ga_instance = pygad.GA(
    num_generations=100,  # More generations for better results
    num_parents_mating=8,  # More parents = more genetic diversity
    fitness_func=fitness_func,
    sol_per_pop=50,  # Larger population
    num_genes=num_genes,
    initial_population=population_vectors,
    on_generation=on_generation,
    mutation_type="random",
    mutation_percent_genes=10,  # Higher mutation for exploration
    parent_selection_type="sss",  # Steady-state selection
    crossover_type="single_point",
    keep_parents=2,  # Keep more good solutions
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

print("\n--- Running Best Network ---")
print("Close the game window to exit.")

# Re-initialize with display ON and forced FPS for smooth playback
p = PLE(game, fps=30, display_screen=True, force_fps=True)
p.init()
p.reset_game()

# Update the GANN instance one last time to ensure the
# network at best_idx has the final best_solution weights
population_matrices = pygad.gann.population_as_matrices(
    population_networks=gann_instance.population_networks,
    population_vectors=ga_instance.population
)
gann_instance.update_population_trained_weights(population_matrices)

# Get the best network
best_network = gann_instance.population_networks[best_idx]

# Game loop for demonstration
while not p.game_over():
    # Get state
    state = p.getGameState()
    
    # Get normalized inputs (using our helper function)
    inputs_np = get_network_inputs(state)

    # Get prediction
    prediction = pygad.nn.predict(last_layer=best_network,
                                  data_inputs=inputs_np,
                                  problem_type="regression")
    
    # Act
    action = None
    if prediction[0] > 0.5:
        action = 119  # flap

    p.act(action)
    
    # Small delay so we can watch it
    time.sleep(1/30.0) 

print(f"\nFinal Score of Best Network: {p.score()}")
print("Training complete!")