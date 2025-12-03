import pygad
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import json

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()

# Arquitectura fija de entrada/salida
I = 8  # Neuronas de entrada (fijo)
O = 1  # Neuronas de salida (fijo)
H_MAX = 16  # Número MÁXIMO de neuronas ocultas a explorar


def build_correspondence_table(I, H, O):
    """
    Construye la tabla que mapea cada bit a una arquitectura básica.
    
    Returns:
        list: Tabla con arquitecturas básicas
        int: Número total de bits necesarios
    """
    table = []
    bit_idx = 0
    
    # Tipo 1: Conexiones directas i_r -> o_p (cardinal = 1)
    # Diapositiva 21: b = {(i_r, o_p)}
    for i in range(I):
        for o in range(O):
            table.append({
                'type': 'direct',
                'from_layer': 'input',
                'from_idx': i,
                'to_layer': 'output',
                'to_idx': o,
                'bit': bit_idx
            })
            bit_idx += 1
    
    # Tipo 2: Conexiones a través de neurona oculta i_r -> h_s -> o_p (cardinal = 2)
    # Diapositiva 21: b = {(i_r, h_s), (h_s, o_p)}
    for i in range(I):
        for h in range(H):
            for o in range(O):
                table.append({
                    'type': 'hidden',
                    'from_layer': 'input',
                    'from_idx': i,
                    'hidden_idx': h,
                    'to_layer': 'output',
                    'to_idx': o,
                    'bit': bit_idx
                })
                bit_idx += 1
    
    return table, bit_idx

# Construir tabla de correspondencias
correspondence_table, NUM_BITS = build_correspondence_table(I, H_MAX, O)
print(f"{'='*70}")
print(f"MÉTODO DE ARQUITECTURAS BÁSICAS")
print(f"{'='*70}")
print(f"Inputs: {I}, Hidden Max: {H_MAX}, Outputs: {O}")
print(f"#B_{{I,H,O}} = I·O·(H+1) + 1 = {I}·{O}·({H_MAX}+1) + 1 = {I*O*(H_MAX+1) + 1}")
print(f"Bits necesarios: {NUM_BITS}")
print(f"Espacio de búsqueda: 2^{NUM_BITS} = {2**NUM_BITS:.2e} arquitecturas válidas")
print(f"{'='*70}\n")

# =============================================================================
# DECODIFICACIÓN DE ARQUITECTURAS
# =============================================================================

def decode_architecture(binary_genome):
    """
    Decodifica una cadena binaria en una arquitectura válida.
    Diapositiva 31: "El proceso de decodificación es muy eficiente"
    
    Proceso:
    1. Cada bit = 1 representa una arquitectura básica
    2. Se superponen todas las arquitecturas básicas (operación ⊕)
    3. Resultado: arquitectura válida v ∈ V_{I,H,O}
    
    Returns:
        dict: {
            'direct': [(in_idx, out_idx), ...],
            'hidden': [(in_idx, h_idx, out_idx), ...],
            'active_hidden': [h1, h2, ...],
            'num_connections': int
        }
    """
    direct_connections = []
    hidden_connections = []
    active_hidden = set()
    
    # Diapositiva 31: "La posición de cada 1 representa una arquitectura básica"
    for i, bit in enumerate(binary_genome):
        if bit == 1 and i < len(correspondence_table):
            arch_basic = correspondence_table[i]
            
            if arch_basic['type'] == 'direct':
                # Conexión directa input -> output
                conn = (arch_basic['from_idx'], arch_basic['to_idx'])
                if conn not in direct_connections:
                    direct_connections.append(conn)
            
            elif arch_basic['type'] == 'hidden':
                # Conexión input -> hidden -> output
                conn = (arch_basic['from_idx'], 
                       arch_basic['hidden_idx'], 
                       arch_basic['to_idx'])
                if conn not in hidden_connections:
                    hidden_connections.append(conn)
                    active_hidden.add(arch_basic['hidden_idx'])
    
    return {
        'direct': direct_connections,
        'hidden': hidden_connections,
        'active_hidden': sorted(list(active_hidden)),
        'num_connections': len(direct_connections) + len(hidden_connections) * 2
    }

# =============================================================================
# RED NEURONAL MANUAL
# =============================================================================

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def build_weight_matrices(architecture, weights_flat):
    """
    Construye matrices de pesos para una arquitectura específica.
    
    Args:
        architecture: Arquitectura decodificada
        weights_flat: Array plano con todos los pesos
    
    Returns:
        dict: {
            'input_to_hidden': matriz (I x H_active),
            'hidden_to_output': matriz (H_active x O),
            'input_to_output': matriz (I x O)
        }
    """
    H_active = len(architecture['active_hidden'])
    
    # Calcular número de pesos necesarios
    n_ih = I * H_active if H_active > 0 else 0
    n_ho = H_active * O if H_active > 0 else 0
    n_io = I * O
    total_weights = n_ih + n_ho + n_io
    
    # Asegurar que tenemos suficientes pesos
    if len(weights_flat) < total_weights:
        weights_flat = np.concatenate([weights_flat, 
                                      np.zeros(total_weights - len(weights_flat))])
    
    idx = 0
    weights = {}
    
    # Input -> Hidden
    if H_active > 0:
        weights['input_to_hidden'] = weights_flat[idx:idx+n_ih].reshape(I, H_active)
        idx += n_ih
        
        # Hidden -> Output
        weights['hidden_to_output'] = weights_flat[idx:idx+n_ho].reshape(H_active, O)
        idx += n_ho
    else:
        weights['input_to_hidden'] = None
        weights['hidden_to_output'] = None
    
    # Input -> Output (conexiones directas)
    weights['input_to_output'] = weights_flat[idx:idx+n_io].reshape(I, O)
    
    return weights

def forward_pass(inputs, architecture, weights):
    """
    Forward pass con arquitectura específica.
    
    Args:
        inputs: (1, I) array
        architecture: Arquitectura decodificada
        weights: Diccionario de matrices de pesos
    """
    # Salida directa input -> output
    output = np.dot(inputs, weights['input_to_output'])
    
    # Si hay neuronas ocultas activas
    if len(architecture['active_hidden']) > 0:
        # Input -> Hidden (con ReLU)
        hidden = np.dot(inputs, weights['input_to_hidden'])
        hidden = relu(hidden)
        
        # Hidden -> Output
        hidden_contribution = np.dot(hidden, weights['hidden_to_output'])
        output += hidden_contribution
    
    # Activación final (sigmoid)
    output = sigmoid(output)
    return output

# =============================================================================
# FUNCIÓN DE JUEGO
# =============================================================================

def get_network_inputs(state):
    """Normalizar inputs del juego."""
    return np.array([
        (state['player_y'] - 256) / 256.0,
        (state['player_vel'] / 10.0),
        (state['next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_pipe_top_y'] - 256) / 256.0,
        (state['next_pipe_bottom_y'] - 256) / 256.0,
        (state['next_next_pipe_dist_to_player'] - 256) / 256.0,
        (state['next_next_pipe_top_y'] - 256) / 256.0,
        (state['next_next_pipe_bottom_y'] - 256) / 256.0,
    ]).reshape(1, I)

def play_game(architecture, weights_flat, max_frames=10000):
    """
    Juega Flappy Bird con una arquitectura y pesos dados.
    
    Returns:
        float: Fitness del juego
    """
    global p
    p.reset_game()
    
    # Construir matrices de pesos
    weight_matrices = build_weight_matrices(architecture, weights_flat)
    
    frames_lived = 0
    
    while not p.game_over() and frames_lived < max_frames:
        frames_lived += 1
        
        state = p.getGameState()
        inputs_np = get_network_inputs(state)
        
        # Forward pass
        prediction = forward_pass(inputs_np, architecture, weight_matrices)
        
        action = None
        if prediction[0, 0] > 0.5:
            action = 119  # Flap
        
        p.act(action)
    
    score = p.score()
    return score, frames_lived

# =============================================================================
# FASE 1: BÚSQUEDA DE ARQUITECTURA (AAGG BINARIO)
# =============================================================================

def fitness_func_architecture(ga_instance, solution, sol_idx):
    """
    Función de fitness para búsqueda de arquitectura.
    Diapositiva 9: f = K1·MSE + K2·(C_a/C_t) + K3·(I_a/I_max)
    
    En esta fase:
    - Entrenamos con pesos ALEATORIOS (inicialización rápida)
    - Evaluamos múltiples veces para robustez
    - Penalizamos complejidad excesiva
    """
    try:
        binary_genome = [int(bit) for bit in solution]
        architecture = decode_architecture(binary_genome)
        
        # Verificar que no sea arquitectura nula
        if architecture['num_connections'] == 0:
            return -1000
        
        # Evaluar con pesos aleatorios (3 juegos para promedio)
        num_weights = calculate_num_weights(architecture)
        scores = []
        
        for _ in range(3):  # 3 evaluaciones con pesos diferentes
            random_weights = np.random.uniform(-0.5, 0.5, num_weights)
            score, frames = play_game(architecture, random_weights, max_frames=5000)
            fitness = score * 100 + frames / 100.0
            scores.append(fitness)
        
        avg_fitness = np.mean(scores)
        
        # Penalización por complejidad (Diapositiva 9)
        H_active = len(architecture['active_hidden'])
        max_connections = I * H_MAX + H_MAX * O + I * O
        complexity_ratio = architecture['num_connections'] / max_connections
        complexity_penalty = complexity_ratio * 50  # Penalización moderada
        
        final_fitness = avg_fitness - complexity_penalty
        
        return max(0, final_fitness)
        
    except Exception as e:
        print(f"Error en evaluación {sol_idx}: {e}")
        return -1000

def calculate_num_weights(architecture):
    """Calcula número total de pesos para una arquitectura."""
    H_active = len(architecture['active_hidden'])
    n_ih = I * H_active
    n_ho = H_active * O
    n_io = I * O
    return n_ih + n_ho + n_io

def on_generation_phase1(ga_instance):
    """Callback para Fase 1."""
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    avg_fit = np.mean(ga_instance.last_generation_fitness)
    
    # Decodificar mejor solución
    best_sol = ga_instance.best_solution()[0]
    best_arch = decode_architecture([int(b) for b in best_sol])
    
    print(f"[FASE 1] Gen {gen:3d} | Best: {best_fit:8.2f} | Avg: {avg_fit:8.2f} | "
          f"Hidden: {len(best_arch['active_hidden'])} | Conn: {best_arch['num_connections']}")

print("\n" + "="*70)
print("FASE 1: BÚSQUEDA DE ARQUITECTURA (AAGG BINARIO)")
print("="*70)

ga_architecture = pygad.GA(
    num_generations=30,  # Menos generaciones, búsqueda exploratoria
    num_parents_mating=6,
    fitness_func=fitness_func_architecture,
    sol_per_pop=20,  # Población más pequeña
    num_genes=NUM_BITS,
    gene_type=int,
    gene_space=[0, 1],  # Binario
    
    on_generation=on_generation_phase1,
    
    mutation_type="random",
    mutation_percent_genes=10,  # Mayor mutación para exploración
    parent_selection_type="tournament",
    crossover_type="uniform",  # Uniform crossover para arquitecturas
    keep_parents=2,
)

print("Evolucionando arquitecturas con pesos aleatorios...\n")
start_phase1 = time.time()
ga_architecture.run()
end_phase1 = time.time()

# Obtener mejor arquitectura
best_arch_genome, best_arch_fitness, _ = ga_architecture.best_solution()
best_architecture = decode_architecture([int(b) for b in best_arch_genome])

print(f"\n{'='*70}")
print(f"MEJOR ARQUITECTURA ENCONTRADA (Fase 1)")
print(f"{'='*70}")
print(f"Fitness: {best_arch_fitness:.2f}")
print(f"Neuronas ocultas activas: {best_architecture['active_hidden']}")
print(f"Conexiones totales: {best_architecture['num_connections']}")
print(f"  - Directas (I→O): {len(best_architecture['direct'])}")
print(f"  - A través de hidden (I→H→O): {len(best_architecture['hidden'])}")
print(f"Tiempo: {end_phase1 - start_phase1:.2f} segundos")
print(f"{'='*70}\n")

# Guardar arquitectura
arch_data = {
    'genome': best_arch_genome.tolist(),
    'architecture': {
        'active_hidden': best_architecture['active_hidden'],
        'num_connections': best_architecture['num_connections'],
        'direct_connections': best_architecture['direct'],
        'hidden_connections': best_architecture['hidden']
    },
    'fitness': float(best_arch_fitness),
    'phase1_time': end_phase1 - start_phase1
}

with open("best_architecture_phase1.json", 'w') as f:
    json.dump(arch_data, f, indent=2)

# =============================================================================
# FASE 2: ENTRENAMIENTO DE PESOS (AAGG REAL)
# =============================================================================

NUM_WEIGHTS_PHASE2 = calculate_num_weights(best_architecture)

print(f"{'='*70}")
print(f"FASE 2: ENTRENAMIENTO DE PESOS (AAGG REAL)")
print(f"{'='*70}")
print(f"Arquitectura FIJA: {best_architecture['active_hidden']}")
print(f"Número de pesos a optimizar: {NUM_WEIGHTS_PHASE2}")
print(f"{'='*70}\n")

def fitness_func_weights(ga_instance, solution, sol_idx):
    """
    Función de fitness para entrenamiento de pesos.
    Arquitectura ya está fija, solo optimizamos pesos.
    Diapositiva 33-34: Codificación real de parámetros
    """
    try:
        weights = solution
        score, frames = play_game(best_architecture, weights, max_frames=10000)
        
        # Fitness = score principal + bonus supervivencia
        fitness = score * 100 + frames / 100.0
        return max(0, fitness)
        
    except Exception as e:
        print(f"Error en evaluación {sol_idx}: {e}")
        return -1000

def on_generation_phase2(ga_instance):
    """Callback para Fase 2."""
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    avg_fit = np.mean(ga_instance.last_generation_fitness)
    
    print(f"[FASE 2] Gen {gen:3d} | Best: {best_fit:8.2f} | Avg: {avg_fit:8.2f}")
    
    # Guardar checkpoints
    if gen % 10 == 0:
        best_sol = ga_instance.best_solution()[0]
        checkpoint = {
            'weights': best_sol.tolist(),
            'architecture': arch_data['architecture'],
            'fitness': float(best_fit),
            'generation': gen
        }
        with open(f"checkpoint_phase2_gen{gen}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  → Checkpoint guardado")

# Inicialización de pesos con Xavier/He
init_scale = np.sqrt(2.0 / (I + O))

ga_weights = pygad.GA(
    num_generations=100,  # Más generaciones para refinamiento
    num_parents_mating=10,
    fitness_func=fitness_func_weights,
    sol_per_pop=40,  # Población más grande para convergencia
    num_genes=NUM_WEIGHTS_PHASE2,
    gene_type=float,
    
    # Rango de pesos (Diapositiva 34: codificación real)
    init_range_low=-init_scale,
    init_range_high=init_scale,
    gene_space={'low': -2.0, 'high': 2.0},
    
    on_generation=on_generation_phase2,
    
    mutation_type="random",
    mutation_percent_genes=5,  # Menor mutación, búsqueda local
    parent_selection_type="tournament",
    crossover_type="single_point",
    keep_parents=3,
)

print("Optimizando pesos para arquitectura fija...\n")
start_phase2 = time.time()
ga_weights.run()
end_phase2 = time.time()

# =============================================================================
# RESULTADOS FINALES
# =============================================================================

best_weights, best_fitness, _ = ga_weights.best_solution()

print(f"\n{'='*70}")
print(f"SOLUCIÓN FINAL (Dos Fases)")
print(f"{'='*70}")
print(f"Fitness final: {best_fitness:.2f}")
print(f"Arquitectura: {best_architecture['active_hidden']}")
print(f"Pesos optimizados: {NUM_WEIGHTS_PHASE2}")
print(f"\nTiempos:")
print(f"  Fase 1 (Arquitectura): {end_phase1 - start_phase1:.2f}s")
print(f"  Fase 2 (Pesos): {end_phase2 - start_phase2:.2f}s")
print(f"  Total: {end_phase1 - start_phase1 + end_phase2 - start_phase2:.2f}s")
print(f"{'='*70}\n")

# Guardar solución completa
final_solution = {
    'architecture': arch_data['architecture'],
    'weights': best_weights.tolist(),
    'fitness': float(best_fitness),
    'num_weights': NUM_WEIGHTS_PHASE2,
    'phase1_time': end_phase1 - start_phase1,
    'phase2_time': end_phase2 - start_phase2,
    'total_time': end_phase1 - start_phase1 + end_phase2 - start_phase2
}

with open("final_two_phase_solution.json", 'w') as f:
    json.dump(final_solution, f, indent=2)

np.save("final_weights.npy", best_weights)

print("Archivos guardados:")
print("  - best_architecture_phase1.json")
print("  - final_two_phase_solution.json")
print("  - final_weights.npy")

# =============================================================================
# VISUALIZACIÓN DEL MEJOR AGENTE
# =============================================================================

print(f"\n{'='*70}")
print("EJECUTANDO MEJOR AGENTE")
print(f"{'='*70}")
print("Cierra la ventana del juego para terminar.\n")

# Re-inicializar con display
p = PLE(game, fps=30, display_screen=True, force_fps=True)
p.init()
p.reset_game()

weight_matrices = build_weight_matrices(best_architecture, best_weights)

while not p.game_over():
    state = p.getGameState()
    inputs_np = get_network_inputs(state)
    
    prediction = forward_pass(inputs_np, best_architecture, weight_matrices)
    
    action = None
    if prediction[0, 0] > 0.5:
        action = 119
    
    p.act(action)
    time.sleep(1/30.0)

print(f"\nScore final: {p.score()}")
print(f"Arquitectura: {best_architecture['active_hidden']}")
print("\n¡Neuroevolución en dos fases completada! 🧬🎮")
print("Método de Arquitecturas Básicas + Codificación Real")