🧬 EvoFlappy

EvoFlappy is a neuroevolution experiment that uses Genetic Programming (GP) to automatically evolve both the architecture and weights of a Neural Network (NN) capable of playing Flappy Bird.

Instead of traditional gradient-based learning, this project uses evolutionary search (via PyGAD
) to find the best-performing neural network through generations of simulated gameplay.

🚀 Features

Evolves neural network architectures (layers and neurons) and weights simultaneously.

Uses ReLU activations for hidden layers and Sigmoid for output.

Implements a custom fitness function based on score and survival time.

Saves checkpoints and the best evolved model.

Visualizes the best network playing Flappy Bird at the end.

🧩 Requirements

Make sure you have Python 3.8+ installed, then install the dependencies:

pip install pygad pygame ple numpy


⚠️ ple (PyGame Learning Environment) may require manual installation depending on your OS.
See https://github.com/ntasfi/PyGame-Learning-Environment

🕹️ How to Run

Run the script directly:

python evoflappy.py


The program will:

Initialize the Flappy Bird environment.

Randomly generate neural networks.

Evolve them using genetic algorithms over multiple generations.

Save checkpoints and the best-performing network.

Display the best bird playing the game.

📁 Output Files

checkpoint_gen_*.json — Best solution every 10 generations.

best_neuroevolution_solution.json — Final best solution (architecture, fitness, etc.).

best_neuroevolution_weights.npy — Raw NumPy array of the best genome.

🧠 Example Fitness Output
Gen  10 | Best:  523.50 | Avg:  312.45 | Arch: [16, 12] | Weights: 205
  → Checkpoint saved: checkpoint_gen_10.json

🧩 Project Structure
evoflappy.py               # Main neuroevolution script
best_neuroevolution_*.json # Saved best solutions
best_neuroevolution_*.npy  # Saved weight matrices
