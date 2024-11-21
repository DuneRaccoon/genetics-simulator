import random
import tkinter as tk
from tkinter import ttk
import numpy as np
import heapq

from config import Config

# Generate gene names (will be updated after configuration)
gene_names = []
traits_list = []

class Creature:
    def __init__(self, chromosomes=None, position=None):
        self.num_chromosomes = Config.NUM_CHROMOSOMES
        self.genes_per_chromosome = Config.GENES_PER_CHROMOSOME
        self.chromosomes = chromosomes if chromosomes else self.generate_chromosomes()
        self.traits = {}
        self.age = 0
        self.position = position
        self.alive = True
        self.hunger = 0  # Hunger level (0 to 100)
        self.thirst = 0  # Thirst level (0 to 100)
        self.calculate_traits()
        self.color = self.calculate_color()
        self.fitness = self.calculate_fitness()
        self.target = None  # Target position for pathfinding
        self.path = []      # Path to follow

    def generate_chromosomes(self):
        # Each gene is a continuous value between 0 and 1
        return [
            {gene_names[i + c * self.genes_per_chromosome]: random.uniform(0, 1)
             for i in range(self.genes_per_chromosome)}
            for c in range(self.num_chromosomes)
        ]

    def calculate_traits(self):
        # Initialize traits
        self.traits = {trait: 0.0 for trait in traits_list}
        # Sum contributions from all genes
        for chromosome in self.chromosomes:
            for gene, gene_value in chromosome.items():
                if gene in gene_trait_mapping:
                    for trait, weight in gene_trait_mapping[gene].items():
                        self.traits[trait] += gene_value * weight
        # Normalize traits to be between 0 and 1
        for trait in self.traits:
            # Assuming traits can be negative due to negative weights
            self.traits[trait] = max(min(self.traits[trait], 1), 0)

    def calculate_color(self):
        # Map selected traits to RGB values
        r_trait = 'aggression'
        g_trait = 'camouflage'
        b_trait = 'vision'
        r = int(50 + self.traits[r_trait] * 205)   # Scale to 50-255
        g = int(50 + self.traits[g_trait] * 205)
        b = int(50 + self.traits[b_trait] * 205)
        # Ensure RGB values are within 0-255
        r = min(max(0, r), 255)
        g = min(max(0, g), 255)
        b = min(max(0, b), 255)
        return f'#{r:02x}{g:02x}{b:02x}'

    def calculate_fitness(self):
        # Weighted fitness function
        fitness = sum(self.traits[trait] * trait_weights[trait] for trait in self.traits)
        return fitness

    def age_creature(self):
        self.age += 1
        # Increase hunger and thirst
        self.hunger += 5
        self.thirst += 5
        if self.hunger >= 100 or self.thirst >= 100:
            self.alive = False  # Dies from starvation or dehydration
            return True
        # Creatures die naturally after exceeding their 'longevity' trait scaled lifespan
        lifespan = int(self.traits['longevity'] * 10) + 1  # Ensure at least lifespan of 1
        return self.age > lifespan
    

    def act(self, simulation):
        # Decide action based on traits and needs
        if not self.alive:
            return

        # Use different pathfinding algorithms based on intelligence
        intelligence = self.traits['intelligence']
        vision_range = int(self.traits['vision'] * 5) + 1  # Vision range between 1 and 6

        # Check hunger and thirst levels; seek resources if necessary
        if self.hunger > 50 or self.thirst > 50:
            resource_type = 'food' if self.hunger > self.thirst else 'water'
            target = self.find_resource(simulation, resource_type, vision_range)
            if target:
                self.move_towards_target(simulation, target, intelligence)
                return

        # High aggression creatures seek targets
        if self.traits['aggression'] > 0.7:
            target = self.find_target(simulation, vision_range)
            if target:
                self.move_towards_target(simulation, target.position, intelligence, attack_target=target)
                return
        # General movement
        self.move(simulation, intelligence)

    def find_resource(self, simulation, resource_type, vision_range):
        # Find the nearest resource of the given type within vision range
        min_distance = None
        target = None
        x0, y0 = self.position
        for (x1, y1), res_type in simulation.resources.items():
            if res_type == resource_type:
                distance = abs(x1 - x0) + abs(y1 - y0)
                if distance <= vision_range:
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        target = (x1, y1)
        return target

    def find_target(self, simulation, vision_range):
        # Find the nearest creature to attack within vision range
        min_distance = None
        target = None
        x0, y0 = self.position
        for creature in simulation.population:
            if creature != self and creature.alive:
                x1, y1 = creature.position
                distance = abs(x1 - x0) + abs(y1 - y0)
                if distance <= vision_range:
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        target = creature
        return target

    def move_towards_target(self, simulation, target_position, intelligence, attack_target=None):
        # Determine pathfinding method based on intelligence
        if intelligence > 0.7:
            # High intelligence: Use A* pathfinding
            path = self.a_star_pathfinding(simulation, target_position)
        elif intelligence > 0.4:
            # Medium intelligence: Greedy movement
            path = self.greedy_move(simulation, target_position)
        else:
            # Low intelligence: Random movement
            path = [self.random_adjacent_position(simulation)]
        if path:
            next_position = path[0]
            self.position = next_position
            # Check if reached target
            if self.position == target_position:
                if attack_target:
                    self.attack(attack_target, simulation)
                else:
                    self.consume_resource(simulation, target_position)
        else:
            # Can't find path, stay in place
            pass

    def move(self, simulation, intelligence):
        # General movement based on intelligence
        if intelligence > 0.7:
            # High intelligence: Move towards a goal (e.g., center of the grid)
            goal = (Config.GRID_SIZE // 2, Config.GRID_SIZE // 2)
            path = self.a_star_pathfinding(simulation, goal)
            if path:
                self.position = path[0]
        elif intelligence > 0.4:
            # Medium intelligence: Greedy movement towards goal
            goal = (Config.GRID_SIZE // 2, Config.GRID_SIZE // 2)
            path = self.greedy_move(simulation, goal)
            if path:
                self.position = path[0]
        else:
            # Low intelligence: Random movement
            self.position = self.random_adjacent_position(simulation)

    def attack(self, target, simulation):
        # Simple attack logic
        attack_power = self.traits['aggression'] * self.traits['strength']
        defense_power = target.traits['endurance'] * target.traits['camouflage']
        if attack_power > defense_power:
            target.alive = False  # Target dies
            simulation.events.append(f"Creature at {self.position} killed creature at {target.position}")
        else:
            simulation.events.append(f"Creature at {self.position} failed to kill creature at {target.position}")

    def consume_resource(self, simulation, position):
        resource_type = simulation.resources.pop(position)
        if resource_type == 'food':
            self.hunger = max(self.hunger - 50, 0)
        elif resource_type == 'water':
            self.thirst = max(self.thirst - 50, 0)
        simulation.events.append(f"Creature at {self.position} consumed {resource_type}")

    def random_adjacent_position(self, simulation):
        x, y = self.position
        adjacent_positions = simulation.get_adjacent_positions(x, y)
        if adjacent_positions:
            return random.choice(adjacent_positions)
        else:
            return self.position

    def greedy_move(self, simulation, goal):
        # Move towards the goal if adjacent cell is free
        x, y = self.position
        gx, gy = goal
        dx = 1 if gx > x else -1 if gx < x else 0
        dy = 1 if gy > y else -1 if gy < y else 0
        new_positions = [(x + dx, y), (x, y + dy), (x + dx, y + dy)]
        for nx, ny in new_positions:
            nx %= Config.GRID_SIZE
            ny %= Config.GRID_SIZE
            if simulation.is_cell_free(nx, ny):
                return [(nx, ny)]
        # If can't move towards goal, stay in place
        return None

    def a_star_pathfinding(self, simulation, goal):
        start = self.position
        grid = simulation.grid
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            for neighbor in simulation.get_adjacent_positions(*current):
                tentative_g_score = g_score[current] + 1  # Assuming cost = 1
                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue
                if not simulation.is_cell_free(*neighbor) and neighbor != goal:
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        # Path not found
        return None

    def heuristic(self, pos, goal):
        # Use Manhattan distance as heuristic
        x0, y0 = pos
        x1, y1 = goal
        return abs(x1 - x0) + abs(y1 - y0)

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        # Exclude the starting position
        return total_path[1:]

class Simulation:
    def __init__(self):
        self.population_size = Config.POPULATION_SIZE
        self.population = []
        self.wave = 0
        self.environmental_factors = self.generate_environmental_factors()
        self.grid = self.create_grid()
        self.obstacles = set()
        self.resources = {}
        self.events = []  # List to store events
        self.place_obstacles()
        self.place_resources()
        self.create_population()

    def generate_environmental_factors(self):
        # Environmental factors can affect the fitness calculation
        factors = {
            'temperature': random.uniform(0, 1),  # 0: cold, 1: hot
            'predation': random.uniform(0, 1),    # Level of predation
            'resource_abundance': random.uniform(0, 1)  # Availability of resources
        }
        return factors

    def create_grid(self):
        # Create a grid with empty cells
        return [[None for _ in range(Config.GRID_SIZE)] for _ in range(Config.GRID_SIZE)]

    def place_obstacles(self):
        # Randomly place obstacles in the grid
        num_obstacles = int(Config.GRID_SIZE ** 2 * Config.OBSTACLES_PERCENTAGE)
        while len(self.obstacles) < num_obstacles:
            x = random.randint(0, Config.GRID_SIZE - 1)
            y = random.randint(0, Config.GRID_SIZE - 1)
            self.obstacles.add((x, y))

    def place_resources(self):
        # Randomly place resources in the grid
        num_resources = int(Config.GRID_SIZE ** 2 * Config.RESOURCES_PERCENTAGE)
        resource_types = ['food', 'water']
        while len(self.resources) < num_resources:
            x = random.randint(0, Config.GRID_SIZE - 1)
            y = random.randint(0, Config.GRID_SIZE - 1)
            if (x, y) not in self.obstacles and (x, y) not in self.resources:
                res_type = random.choice(resource_types)
                self.resources[(x, y)] = res_type

    def create_population(self):
        # Ensure creatures don't spawn on obstacles or resources
        for _ in range(self.population_size):
            creature = Creature()
            self.population.append(creature)

    def run_wave(self):
        self.wave += 1
        self.events.append(f"--- Wave {self.wave} ---")
        for _ in range(Config.REVOLUTIONS_PER_WAVE):
            self.run_revolution()
        self.cull_population()
        self.reproduce_population()
        self.age_population()
        # Update environmental factors occasionally
        if self.wave % 10 == 0:
            self.environmental_factors = self.generate_environmental_factors()
        # Update stats for GUI
        self.update_stats()

    def run_revolution(self):
        # Clear events for this revolution
        self.events.append(f"Revolution in Wave {self.wave}")
        # Update grid with current positions
        self.update_grid()
        for creature in self.population:
            if creature.alive:
                creature.act(self)
        # Remove dead creatures from grid
        self.update_grid()

    def update_grid(self):
        self.grid = [[None for _ in range(Config.GRID_SIZE)] for _ in range(Config.GRID_SIZE)]
        for x, y in self.obstacles:
            self.grid[y][x] = 'obstacle'
        for (x, y), res_type in self.resources.items():
            self.grid[y][x] = res_type
        for creature in self.population:
            if creature.alive:
                x, y = creature.position
                self.grid[y][x] = creature

    def is_cell_free(self, x, y):
        x %= Config.GRID_SIZE
        y %= Config.GRID_SIZE
        cell = self.grid[y][x]
        return cell is None or cell in ['food', 'water']

    def get_adjacent_positions(self, x, y):
        positions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % Config.GRID_SIZE
                ny = (y + dy) % Config.GRID_SIZE
                if self.is_cell_free(nx, ny):
                    positions.append((nx, ny))
        return positions

    def cull_population(self):
        # Remove dead creatures
        self.population = [creature for creature in self.population if creature.alive]
        if not self.population:
            self.events.append("All creatures have died.")
            return
        # Calculate average fitness
        avg_fitness = sum(creature.fitness for creature in self.population) / len(self.population)
        fitness_threshold = avg_fitness * 0.5  # Threshold is 50% of average fitness

        # Environmental adjustments to fitness threshold
        temp_factor = self.environmental_factors['temperature']
        fitness_threshold *= (1 + (temp_factor - 0.5) * 0.2)  # Adjust based on temperature

        # Cull creatures below fitness threshold
        initial_population = len(self.population)
        self.population = [
            creature for creature in self.population if creature.fitness >= fitness_threshold
        ]
        culled = initial_population - len(self.population)
        self.events.append(f"Culled {culled} creatures due to low fitness.")

    def reproduce_population(self):
        offspring = []
        while len(self.population) + len(offspring) < self.population_size:
            parent1, parent2 = self.select_mates()
            child_chromosomes = self.reproduce(parent1, parent2)
            child = Creature(chromosomes=child_chromosomes)
            offspring.append(child)
        self.population.extend(offspring)
        self.events.append(f"Reproduced {len(offspring)} offspring.")

    def select_mates(self):
        # Mate selection based on fertility and compatibility
        eligible_parents = [creature for creature in self.population if creature.traits['fertility'] > 0.5]
        if len(eligible_parents) >= 2:
            parent1 = random.choice(eligible_parents)
            # Select mate based on compatibility (similar traits)
            compatibilities = [
                (self.calculate_compatibility(parent1, creature), creature)
                for creature in eligible_parents if creature != parent1
            ]
            compatibilities.sort(reverse=True)
            parent2 = compatibilities[0][1]  # Most compatible mate
            return parent1, parent2
        else:
            # If not enough eligible parents, pick randomly
            return random.sample(self.population, 2)

    def calculate_compatibility(self, creature1, creature2):
        # Compatibility based on similarity of key traits
        key_traits = ['social_behavior', 'adaptability', 'stress_tolerance']
        compatibility = sum(
            1 - abs(creature1.traits[trait] - creature2.traits[trait])
            for trait in key_traits
        ) / len(key_traits)
        return compatibility

    def reproduce(self, parent1, parent2):
        # Sexual reproduction with crossover and mutation
        child_chromosomes = []
        for i in range(Config.NUM_CHROMOSOMES):
            chromosome1 = parent1.chromosomes[i]
            chromosome2 = parent2.chromosomes[i]
            # Crossover point
            crossover_point = random.randint(1, Config.GENES_PER_CHROMOSOME - 1)
            child_chromosome = {}
            # Combine genes from both parents
            gene_keys = list(chromosome1.keys())
            for j in range(Config.GENES_PER_CHROMOSOME):
                gene = gene_keys[j]
                if j < crossover_point:
                    gene_value = chromosome1[gene]
                else:
                    gene_value = chromosome2[gene]
                # Mutation
                if random.random() < Config.MUTATION_RATE:
                    gene_value += random.uniform(-0.1, 0.1)
                    gene_value = max(min(gene_value, 1), 0)
                child_chromosome[gene] = gene_value
            child_chromosomes.append(child_chromosome)
        return child_chromosomes

    def age_population(self):
        # Age creatures and remove those that die of old age or starvation
        survivors = []
        for creature in self.population:
            if creature.alive and not creature.age_creature():
                survivors.append(creature)
        self.population = survivors

    def simulate(self):
        for _ in range(Config.NUM_WAVES):
            self.run_wave()
            if len(self.population) == 0:
                self.events.append("All creatures have died. Simulation ended.")
                break

    def update_stats(self):
        # Calculate stats for GUI display
        self.total_population = len(self.population)
        if self.population:
            self.average_fitness = sum(creature.fitness for creature in self.population) / self.total_population
        else:
            self.average_fitness = 0

class SimulationGUI:
    def __init__(self):
        self.simulation = None
        self.root = tk.Tk()
        self.root.title("Evolution Simulator")
        self.create_configuration_interface()

    def create_configuration_interface(self):
        # Create GUI elements to set configuration
        self.config_frame = tk.Frame(self.root)
        self.config_frame.pack(pady=10)
        
        self.population_size_var = tk.IntVar(value=Config.POPULATION_SIZE)
        self.num_chromosomes_var = tk.IntVar(value=Config.NUM_CHROMOSOMES)
        self.genes_per_chromosome_var = tk.IntVar(value=Config.GENES_PER_CHROMOSOME)
        self.num_waves_var = tk.IntVar(value=Config.NUM_WAVES)
        self.revolutions_per_wave_var = tk.IntVar(value=Config.REVOLUTIONS_PER_WAVE)
        self.mutation_rate_var = tk.DoubleVar(value=Config.MUTATION_RATE)
        self.grid_size_var = tk.IntVar(value=Config.GRID_SIZE)
        self.obstacles_percentage_var = tk.DoubleVar(value=Config.OBSTACLES_PERCENTAGE)
        self.resources_percentage_var = tk.DoubleVar(value=Config.RESOURCES_PERCENTAGE)

        self.create_config_field("Population Size:", self.population_size_var)
        self.create_config_field("Number of Chromosomes:", self.num_chromosomes_var)
        self.create_config_field("Genes per Chromosome:", self.genes_per_chromosome_var)
        self.create_config_field("Number of Waves:", self.num_waves_var)
        self.create_config_field("Revolutions per Wave:", self.revolutions_per_wave_var)
        self.create_config_field("Mutation Rate (0-1):", self.mutation_rate_var)
        self.create_config_field("Grid Size:", self.grid_size_var)
        self.create_config_field("Obstacles Percentage:", self.obstacles_percentage_var)
        self.create_config_field("Resources Percentage:", self.resources_percentage_var)

        self.start_button = tk.Button(self.root, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(pady=10)

    def create_config_field(self, label_text, variable):
        frame = tk.Frame(self.config_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)
        label = tk.Label(frame, text=label_text, width=25, anchor='w')
        label.pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def start_simulation(self):
        # Update configuration with user inputs
        Config.POPULATION_SIZE = self.population_size_var.get()
        Config.NUM_CHROMOSOMES = self.num_chromosomes_var.get()
        Config.GENES_PER_CHROMOSOME = self.genes_per_chromosome_var.get()
        Config.NUM_WAVES = self.num_waves_var.get()
        Config.REVOLUTIONS_PER_WAVE = self.revolutions_per_wave_var.get()
        Config.MUTATION_RATE = self.mutation_rate_var.get()
        Config.GRID_SIZE = self.grid_size_var.get()
        Config.OBSTACLES_PERCENTAGE = self.obstacles_percentage_var.get()
        Config.RESOURCES_PERCENTAGE = self.resources_percentage_var.get()
        Config.TOTAL_GENES = Config.NUM_CHROMOSOMES * Config.GENES_PER_CHROMOSOME

        # Regenerate gene names and traits
        self.generate_gene_names_and_traits()

        # Hide configuration interface
        self.config_frame.destroy()
        self.start_button.destroy()

        # Initialize simulation
        global simulation
        simulation = Simulation()
        self.simulation = simulation
        self.canvas_size = 600

        # Create main GUI frames
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas for the grid
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Bind mouse events for mouseover functionality
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Create stats display
        self.stats_label = tk.Label(self.stats_frame, text="Simulation Stats", font=("Helvetica", 14))
        self.stats_label.pack(pady=10)
        self.population_label = tk.Label(self.stats_frame, text="")
        self.population_label.pack()
        self.fitness_label = tk.Label(self.stats_frame, text="")
        self.fitness_label.pack()
        self.creature_info_label = tk.Label(self.stats_frame, text="", justify=tk.LEFT)
        self.creature_info_label.pack(pady=20)

        # Create event log window
        self.event_log_window = tk.Toplevel(self.root)
        self.event_log_window.title("Event Log")
        self.event_log_text = tk.Text(self.event_log_window, state='disabled', width=60, height=20)
        self.event_log_text.pack()

        self.draw_population()
        self.root.after(1000, self.next_wave)

    def generate_gene_names_and_traits(self):
        global gene_names, traits_list, gene_trait_mapping, trait_weights
        gene_names = [f'gene{i+1}' for i in range(Config.TOTAL_GENES)]
        # Define traits
        traits_list = [
            'strength', 'speed', 'intelligence', 'endurance', 'aggression',
            'vision', 'camouflage', 'fertility', 'longevity', 'temperature_resistance',
            'hearing', 'immunity', 'metabolism', 'reproductive_rate', 'disease_resistance',
            'water_conservation', 'night_vision', 'social_behavior', 'adaptability', 'stress_tolerance'
        ]
        # Ensure we have enough genes to affect traits
        if Config.TOTAL_GENES < len(traits_list):
            print("Not enough genes to cover all traits")
            self.root.destroy()
            return
        # Map genes to traits
        gene_trait_mapping = {}
        for gene in gene_names:
            # Each gene affects 1 to 3 traits
            num_traits = random.randint(1, 3)
            affected_traits = random.sample(traits_list, num_traits)
            gene_trait_mapping[gene] = {}
            for trait in affected_traits:
                # Assign a random weight between -1 and 1
                weight = random.uniform(-1, 1)
                gene_trait_mapping[gene][trait] = weight
        # Trait weights for fitness calculation
        trait_weights = {trait: random.uniform(0.5, 1.5) for trait in traits_list}

    def draw_population(self):
        self.canvas.delete("all")
        cell_size = self.canvas_size // Config.GRID_SIZE

        # Draw obstacles
        for y in range(Config.GRID_SIZE):
            for x in range(Config.GRID_SIZE):
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                if (x, y) in self.simulation.obstacles:
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, fill='gray', outline=""
                    )
                elif (x, y) in self.simulation.resources:
                    res_type = self.simulation.resources[(x, y)]
                    color = 'yellow' if res_type == 'food' else 'blue'
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, fill=color, outline=""
                    )
        # Draw creatures
        for creature in self.simulation.population:
            if creature.alive:
                x, y = creature.position
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=creature.color, outline=""
                )
        # Update stats display
        self.update_stats_display()

        # Update event log
        self.update_event_log()

    def on_mouse_move(self, event):
        # Calculate grid position from mouse coordinates
        cell_size = self.canvas_size // Config.GRID_SIZE
        x = event.x // cell_size
        y = event.y // cell_size
        x = min(max(x, 0), Config.GRID_SIZE - 1)
        y = min(max(y, 0), Config.GRID_SIZE - 1)
        # Check if there's a creature at this position
        creature = None
        for c in self.simulation.population:
            if c.alive and c.position == (x, y):
                creature = c
                break
        if creature:
            info = f"Creature at ({x}, {y}):\n"
            for trait, value in creature.traits.items():
                info += f"{trait.capitalize()}: {value:.2f}\n"
            info += f"Hunger: {creature.hunger}\n"
            info += f"Thirst: {creature.thirst}\n"
            info += f"Age: {creature.age}\n"
            self.creature_info_label.config(text=info)
        else:
            self.creature_info_label.config(text="")

    def next_wave(self):
        if self.simulation:
            self.simulation.run_wave()
            self.draw_population()
            if len(self.simulation.population) > 0:
                self.root.after(1000, self.next_wave)
            else:
                print("All creatures have died. Simulation ended.")
                self.root.destroy()

    def update_stats_display(self):
        self.population_label.config(text=f"Population Size: {self.simulation.total_population}")
        self.fitness_label.config(text=f"Average Fitness: {self.simulation.average_fitness:.2f}")

    def update_event_log(self):
        self.event_log_text.config(state='normal')
        self.event_log_text.delete(1.0, tk.END)
        for event in self.simulation.events[-20:]:
            self.event_log_text.insert(tk.END, event + "\n")
        self.event_log_text.config(state='disabled')

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = SimulationGUI()
    gui.run()
