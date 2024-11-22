from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple, Union
import heapq

from config import Config

# Generate gene names (will be updated after configuration)
gene_names = []
traits_list = []
gene_trait_mapping = {}
trait_weights = {}

def split(x, y):
    return [x // y + (1 if x % y > i else 0) for i in range(y)]

class Creature:
    def __init__(
        self,
        simulation: Simulation,
        *,
        chromosomes=None, 
        position=None
    ):
        self.simulation = simulation
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
        self.known_resources: Dict[Tuple[int, int], Dict[str, any]] = {}  # Resources known to the creature
        self.reproduction_cooldown = 0  # Cooldown period after reproduction
        self.mate: Union[Creature, None] = None  # Current mate target
        self.reproduction_counter = 0  # Number of successful reproductions
        self.current_thought = ""  # Current action or thought
        self.thoughts = []  # Log of actions and thoughts
        self.cause_of_death = ""  # Reason for death

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
        # Define traits for each color channel
        r_traits = ['aggression', 'strength', 'speed']
        g_traits = ['camouflage', 'endurance', 'longevity']
        b_traits = ['vision', 'intelligence', 'hearing']

        # Calculate average trait values for each channel
        r_avg = sum(self.traits[trait] for trait in r_traits) / len(r_traits)
        g_avg = sum(self.traits[trait] for trait in g_traits) / len(g_traits)
        b_avg = sum(self.traits[trait] for trait in b_traits) / len(b_traits)

        # Scale averages to RGB values (50-255)
        r = int(50 + r_avg * 205)
        g = int(50 + g_avg * 205)
        b = int(50 + b_avg * 205)

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
        
        # Decrease reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
            
        # Increase hunger and thirst
        self.hunger += 5
        self.thirst += 5
        if self.hunger >= 100 or self.thirst >= 100:
            self.alive = False  # Dies from starvation or dehydration
            self.cause_of_death = "Starvation/Dehydration"
            self.simulation.events.append(f"Creature at {self.position} died of hunger or thirst.")
            self.simulation.add_to_graveyard(self)
            return True
        
        # Creatures die naturally after exceeding their 'longevity' trait scaled lifespan
        lifespan = int(self.traits['longevity'] * 10) + 1  # Ensure at least lifespan of 1
        died = self.age > lifespan
        if died:
            self.alive = False
            self.cause_of_death = "Old Age"
            self.simulation.events.append(f"Creature at {self.position} died of old age.")
            self.simulation.add_to_graveyard(self)
            
        return died

    def is_culled(self, fitness_threshold):
        """
        Other culling criteria can be added here, such as disease, natural disasters, etc.

        If the creature is culled, returns True, otherwise False.
        """
        # Check if creature should be culled based on fitness
        if self.fitness < fitness_threshold:
            self.alive = False
            self.cause_of_death = "Culled due to Low Fitness"
            self.simulation.events.append(f"Creature at {self.position} culled due to low fitness.")
            self.simulation.add_to_graveyard(self)
            return True
        return False

    def act(self):
        # Decide action based on traits and needs
        if not self.alive:
            return

        # Scan surroundings to update known resources
        self.scan_surroundings()

        # Use different pathfinding algorithms based on intelligence
        intelligence = self.traits['intelligence']
        vision_range = int(self.traits['vision'] * 5) + 1  # Vision range between 1 and 6

        # Check hunger and thirst levels; seek resources if necessary
        if self.hunger > 50 or self.thirst > 50:
            resource_type = 'food' if self.hunger > self.thirst else 'water'
            target = self.find_resource(resource_type)
            if target:
                self.move_towards_target(target, intelligence)
                self.current_thought = f"Seeking {resource_type}"
                return

        # Attempt reproduction if not hungry/thirsty and not in cooldown
        if self.reproduction_cooldown == 0 and self.traits['fertility'] > 0.5:
            mate = self.find_mate(vision_range)
            if mate:
                self.mate = mate
                self.move_towards_target(mate.position, intelligence, mate_target=mate)
                self.current_thought = f"Seeking mate at {mate.position}"
                return

        # High aggression creatures seek targets
        if self.traits['aggression'] > 0.7:
            target = self.find_target(vision_range)
            if target:
                self.move_towards_target(target.position, intelligence, attack_target=target)
                self.current_thought = f"Attacking creature at {target.position}"
                return

        # General movement
        self.move(simulation, intelligence)

    def find_resource(self, simulation, resource_type, vision_range):
        # Find the nearest resource of the given type within vision range
        min_distance = None
        target = None
        x0, y0 = self.position
        for (x1, y1), res in self.known_resources.items():
            if res['type'] == resource_type and not res.get('claimed', False):
                distance = abs(x1 - x0) + abs(y1 - y0)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    target = (x1, y1)
        # Claim the resource if found
        if target:
            self.simulation.resources[target]['claimed'] = True
            self.target_resource = target
            # Update GUI immediately to reflect claimed resource
            if self.simulation.gui:
                self.simulation.gui.draw_population()
        else:
            self.target_resource = None
        return target

    def find_target(self, vision_range):
        # Find the nearest creature to attack within vision range
        min_distance = None
        target = None
        x0, y0 = self.position
        for creature in self.simulation.population:
            if creature != self and creature.alive:
                x1, y1 = creature.position
                distance = abs(x1 - x0) + abs(y1 - y0)
                if distance <= vision_range:
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        target = creature
        return target
    
    def find_mate(self, vision_range):
        # Find the nearest eligible mate within vision range
        min_distance = None
        potential_mate = None
        x0, y0 = self.position
        for creature in self.simulation.population:
            if creature != self and creature.alive and creature.reproduction_cooldown == 0:
                if creature.traits['fertility'] > 0.5 and creature.hunger <= 50 and creature.thirst <= 50:
                    x1, y1 = creature.position
                    distance = abs(x1 - x0) + abs(y1 - y0)
                    if distance <= vision_range:
                        if min_distance is None or distance < min_distance:
                            min_distance = distance
                            potential_mate = creature
        return potential_mate

    def move_towards_target(self, target_position, intelligence, attack_target: Creature=None, mate_target: Creature=None):
        # Check if target is still valid before moving
        if attack_target and not attack_target.alive:
            # Target is dead, abandon action
            return
        elif mate_target and (not mate_target.alive or mate_target.reproduction_cooldown > 0):
            # Mate is no longer available
            self.mate = None
            return
        elif not attack_target and not mate_target and self.target_resource:
            # Check if resource still exists and is not claimed
            resource = self.simulation.resources.get(self.target_resource)
            if not resource or resource.get('claimed', False):
                # Resource no longer available
                self.target_resource = None
                return

        # Determine pathfinding method based on intelligence
        if intelligence > 0.6:
            # High intelligence: Use A* pathfinding
            path = self.a_star_pathfinding(target_position)
        elif intelligence > 0.35:
            # Medium intelligence: Greedy movement
            path = self.greedy_move(target_position)
        else:
            # Low intelligence: Random movement
            path = [self.random_adjacent_position()]
        if path:
            next_position = path[0]
            self.position = next_position
            # Check if reached target
            if self.position == target_position:
                if attack_target:
                    self.attack(attack_target)
                elif mate_target:
                    self.attempt_reproduction(mate_target)
                else:
                    self.consume_resource(target_position)
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
            self.simulation.events.append(f"Creature at {self.position} failed to kill creature at {target.position}")

    def consume_resource(self, simulation, position):
        resource_type = simulation.resources.pop(position)
        if resource_type == 'food':
            self.hunger = max(self.hunger - 50, 0)
        elif resource_type == 'water':
            self.thirst = max(self.thirst - 50, 0)
        simulation.events.append(f"Creature at {self.position} consumed {resource_type}")

    def random_adjacent_position(self, simulation):
        x, y = self.position
        adjacent_positions = self.simulation.get_adjacent_positions(x, y)
        if adjacent_positions:
            return random.choice(adjacent_positions)
        else:
            return self.position

    def greedy_move(self, goal):
        # Move towards the goal if adjacent cell is free
        x, y = self.position
        gx, gy = goal
        dx = 1 if gx > x else -1 if gx < x else 0
        dy = 1 if gy > y else -1 if gy < y else 0
        new_positions = [(x + dx, y), (x, y + dy), (x + dx, y + dy)]
        random.shuffle(new_positions)  # Shuffle to add some randomness
        for nx, ny in new_positions:
            nx %= Config.GRID_SIZE
            ny %= Config.GRID_SIZE
            if self.simulation.is_cell_free(nx, ny):
                return [(nx, ny)]
        # If can't move towards goal, stay in place
        return None

    def a_star_pathfinding(self, goal):
        start = self.position
        grid = self.simulation.grid
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            for neighbor in self.simulation.get_adjacent_positions(*current):
                tentative_g_score = g_score[current] + 1  # Assuming cost = 1
                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue
                if not self.simulation.is_cell_free(*neighbor) and neighbor != goal:
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
        self.population: List[Creature] = []
        self.new_offspring: List[Creature] = []  # Offspring to be added after revolution
        self.wave = 0
        self.revolution = 0
        self.environmental_factors = self.generate_environmental_factors()
        self.grid = self.create_grid()
        self.obstacles = set()
        self.resources = {}
        self.events = []  # List to store events
        self.place_obstacles()
        self.place_resources()
        self.create_population()
        self.total_revolutions = 0  # Total number of revolutions run

    def generate_environmental_factors(self):
        # Environmental factors can affect the fitness calculation
        factors = {
            'temperature': random.uniform(0, 1),  # 0: cold, 1: hot
            'predation': random.uniform(0, 1),    # Level of predation
            'resource_abundance': random.uniform(0, 1),  # Availability of resources
            'disease': random.uniform(0, 1),      # Disease prevalence
            'natural_disasters': random.uniform(0, 1)  # Frequency of natural disasters
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
                self.resources[(x, y)] = {
                    "type": res_type,
                    "claimed": False
                }

    def create_population(self):
        # Ensure creatures don't spawn on obstacles or resources
        for _ in range(self.population_size):
            position = self.random_free_position()
            creature = Creature(self, position=position)
            self.population.append(creature)

    def random_free_position(self):
        while True:
            x = random.randint(0, Config.GRID_SIZE - 1)
            y = random.randint(0, Config.GRID_SIZE - 1)
            if (x, y) not in self.obstacles and (x, y) not in self.resources and not any(c.position == (x, y) for c in self.population):
                return (x, y)
            
    def find_adjacent_free_cell(self, position):
        x, y = position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = (x + dx) % Config.GRID_SIZE
                ny = (y + dy) % Config.GRID_SIZE
                if self.is_cell_free(nx, ny):
                    return (nx, ny)
        return None

    def run_wave(self):
        # Prepare for a new wave
        self.wave += 1
        self.revolution = 0
        self.events.append(f"--- Wave {self.wave} ---")
        if self.wave % 10 == 0:
            self.environmental_factors = self.generate_environmental_factors()
            labelled_factors = "\n".join(f"{k}: {v:.2f}" for k, v in self.environmental_factors.items())
            self.events.append("Environmental factors have changed: \n" + labelled_factors)

    def run_revolution(self):
        self.revolution += 1
        self.events.append(f"Revolution {self.revolution} in Wave {self.wave}")
        self.total_revolutions += 1
        self.update_stats()
        # Process each creature individually and update the grid after each
        for creature in self.population:
            if creature.alive:
                creature.act()
                self.update_grid()
        # Remove consumed resources
        self.resources = {pos: res for pos, res in self.resources.items() if not res.get('consumed', False)}
        # Add new offspring to the population
        if self.new_offspring:
            self.population.extend(self.new_offspring)
            self.new_offspring = []
        # Update grid after adding offspring
        self.update_grid()

    def is_wave_complete(self):
        return self.revolution >= Config.REVOLUTIONS_PER_WAVE

    def add_to_graveyard(self, creature: Creature):
        self.graveyard.append(creature)
        
    def is_simulation_complete(self):
        return self.wave >= Config.NUM_WAVES or len(self.population) == 0
        
    def update_grid(self):
        self.grid = [[None for _ in range(Config.GRID_SIZE)] for _ in range(Config.GRID_SIZE)]
        for x, y in self.obstacles:
            self.grid[y][x] = 'obstacle'
        for (x, y), res in self.resources.items():
            self.grid[y][x] = res['type']
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

        # Disease adjustments to fitness threshold
        disease_factor = self.environmental_factors['disease']
        fitness_threshold *= (1 - disease_factor * 0.2)  # Adjust based on disease prevalence

        # Cull creatures below fitness threshold
        initial_population = len(self.population)
        survivors = []
        for creature in self.population:
            if not creature.is_culled(fitness_threshold):
                survivors.append(creature)

        self.population = survivors
        culled = initial_population - len(self.population)
        self.events.append(f"Culled {culled} creatures due to low fitness.")
        
    def age_population(self):
        # Age creatures and remove those that die of old age or starvation
        survivors = []
        for creature in self.population:
            if creature.alive and not creature.age_creature():
                survivors.append(creature)
        self.population = survivors

    def simulate(self):
        for _ in range(Config.NUM_WAVES):
            if self.gui and self.gui.paused:
                break  # Break if simulation is paused
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

    def set_gui(self, gui):
        self.gui: SimulationGUI = gui

class SimulationGUI:
    def __init__(self):
        self.simulation = None
        self.root = tk.Tk()
        self.root.title("Evolution Simulator")
        self.paused = False
        self.selected_creature = None  # Add this line
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
        
        # Hide configuration interface
        self.config_frame.destroy()
        self.start_button.destroy()

        # Initialize simulation
        global simulation
        simulation = Simulation()
        simulation.set_gui(self)
        self.simulation = simulation
        self.canvas_size = 600

        # Create main GUI frames using notebook tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Simulation Tab
        self.simulation_frame = tk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="Simulation")

        # Graveyard Tab
        self.graveyard_frame = tk.Frame(self.notebook)
        self.notebook.add(self.graveyard_frame, text="Graveyard")

        # Create canvas for the grid in simulation tab
        self.canvas = tk.Canvas(self.simulation_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.LEFT)

        # Bind mouse events for mouseover and click functionality
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)  # Bind left-click event

        # Create stats display
        self.stats_frame = tk.Frame(self.simulation_frame)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.stats_label = tk.Label(self.stats_frame, text="Simulation Stats", font=("Helvetica", 14))
        self.stats_label.pack(pady=10)
        self.population_label = tk.Label(self.stats_frame, text="")
        self.population_label.pack()
        self.fitness_label = tk.Label(self.stats_frame, text="")
        self.fitness_label.pack()
        self.wave_label = tk.Label(self.stats_frame, text="")
        self.wave_label.pack()
        self.revolution_label = tk.Label(self.stats_frame, text="")
        self.revolution_label.pack()

        self.creature_info_label = tk.Label(self.stats_frame, text="", justify=tk.LEFT)
        self.creature_info_label.pack(pady=10)

        # Add label for selected creature information
        self.selected_creature_info_label = tk.Label(self.stats_frame, text="", justify=tk.LEFT, fg="blue")
        self.selected_creature_info_label.pack(pady=10)

        # Create pause/play buttons
        self.button_frame = tk.Frame(self.stats_frame)
        self.button_frame.pack(pady=10)
        self.pause_button = tk.Button(self.button_frame, text="Pause", command=self.pause_simulation)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.play_button = tk.Button(self.button_frame, text="Play", command=self.play_simulation)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.restart_button = tk.Button(self.button_frame, text="Restart", command=self.restart_simulation)
        self.restart_button.pack(side=tk.LEFT, padx=5)

        # Create event log window
        self.event_log_window = tk.Toplevel(self.root)
        self.event_log_window.title("Event Log")
        self.event_log_text = tk.Text(self.event_log_window, state='disabled', width=60, height=20)
        self.event_log_text.pack()

        # Graveyard Listbox in the graveyard tab
        self.graveyard_listbox = tk.Listbox(self.graveyard_frame, width=80)
        self.graveyard_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for the graveyard listbox
        self.graveyard_scrollbar = tk.Scrollbar(self.graveyard_frame)
        self.graveyard_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.graveyard_listbox.config(yscrollcommand=self.graveyard_scrollbar.set)
        self.graveyard_scrollbar.config(command=self.graveyard_listbox.yview)

        self.draw_population()
        self.root.after(1000, self.run_simulation)  # Start simulation after 1 second

    def restart_simulation(self):
        # Reset the simulation and GUI
        self.simulation = None
        self.selected_creature = None
        self.paused = False
        self.notebook.destroy()
        self.create_configuration_interface()
    
    def draw_population(self):
        self.canvas.delete("all")
        cell_size = self.canvas_size // Config.GRID_SIZE

        # Draw obstacles, resources, and creatures
        for y in range(Config.GRID_SIZE):
            for x in range(Config.GRID_SIZE):
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                if (x, y) in self.simulation.obstacles:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='gray', outline="")
                elif (x, y) in self.simulation.resources:
                    resource = self.simulation.resources[(x, y)]
                    color = 'yellow' if resource["type"] == 'food' else 'blue'
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                    
        # Draw creatures
        for creature in self.simulation.population:
            if creature.alive:
                x, y = creature.position
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                if creature == self.selected_creature:
                    outline_color = 'red'
                    outline_width = 2
                else:
                    outline_color = ''
                    outline_width = 0
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=creature.color, outline=outline_color, width=outline_width
                )
                
        # Update stats and event log
        self.update_stats_display()
        self.update_event_log()
        self.update_selected_creature_info()  # Update selected creature info

    def on_mouse_click(self, event):
        # Calculate grid position from mouse coordinates
        cell_size = self.canvas_size // Config.GRID_SIZE
        x = event.x // cell_size
        y = event.y // cell_size
        x = min(max(x, 0), Config.GRID_SIZE - 1)
        y = min(max(y, 0), Config.GRID_SIZE - 1)
        # Check if a creature is at this position
        for c in self.simulation.population:
            if c.alive and c.position == (x, y):
                self.selected_creature = c
                # Update selected creature info label
                self.update_selected_creature_info()
                break
        else:
            self.selected_creature = None
            self.selected_creature_info_label.config(text="")
        # Redraw the population to show the selection
        self.draw_population()

    def update_selected_creature_info(self):
        c = self.selected_creature
        if c and c.alive:
            info = f"Selected Creature at ({c.position[0]}, {c.position[1]}):\n"
            info += f"Current Thought: {c.current_thought}\n"
            for trait, value in c.traits.items():
                info += f"{trait.capitalize()}: {value:.2f}\n"
            info += f"Hunger: {c.hunger}\n"
            info += f"Thirst: {c.thirst}\n"
            info += f"Age: {c.age}\n"
            self.selected_creature_info_label.config(text=info)
        else:
            self.selected_creature_info_label.config(text="Selected creature is no longer alive.")
            self.selected_creature = None

    def on_mouse_move(self, event):
        # Calculate grid position from mouse coordinates
        cell_size = self.canvas_size // Config.GRID_SIZE
        x = event.x // cell_size
        y = event.y // cell_size
        x = min(max(x, 0), Config.GRID_SIZE - 1)
        y = min(max(y, 0), Config.GRID_SIZE - 1)
        # Check what's at this position
        info = ""
        for c in self.simulation.population:
            if c.alive and c.position == (x, y):
                # Creature info
                info += f"Creature at ({x}, {y}):\n"
                info += f"Current Thought: {c.current_thought}\n"
                for trait, value in c.traits.items():
                    info += f"{trait.capitalize()}: {value:.2f}\n"
                info += f"Hunger: {c.hunger}\n"
                info += f"Thirst: {c.thirst}\n"
                info += f"Age: {c.age}\n"
                break
        else:
            if (x, y) in self.simulation.resources:
                resource = self.simulation.resources[(x, y)]
                info += f'Resource at ({x}, {y}): {resource["type"].capitalize()}\n'
            elif (x, y) in self.simulation.obstacles:
                info += f"Obstacle at ({x}, {y})\n"
            else:
                info += f"Empty cell at ({x}, {y})\n"
        self.creature_info_label.config(text=info)

    def run_simulation(self):
        if not self.paused and self.simulation:
            if self.simulation.is_wave_complete():
                # End-of-wave processing
                self.simulation.age_population()
                self.simulation.cull_population()
                if self.simulation.is_simulation_complete():
                    print("Simulation ended.")
                    self.paused = True  # Pause the simulation
                    # Update the graveyard tab
                    self.update_graveyard()
                    return
                else:
                    self.simulation.run_wave()
            else:
                self.simulation.run_revolution()
                self.draw_population()
        if self.simulation and self.simulation.is_simulation_complete():
            print("Simulation ended.")
            self.paused = True
            self.update_graveyard()
        else:
            self.root.after(500, self.run_simulation)  # Schedule the next simulation step

    def update_graveyard(self):
        # Clear the graveyard listbox
        self.graveyard_listbox.delete(0, tk.END)
        for idx, creature in enumerate(self.simulation.graveyard, 1):
            info = f"{idx}. Age: {creature.age}, Cause of Death: {creature.cause_of_death}, Traits: "
            traits_info = ', '.join(f"{trait}: {value:.2f}" for trait, value in creature.traits.items())
            self.graveyard_listbox.insert(tk.END, info + traits_info)

    def pause_simulation(self):
        self.paused = True

    def play_simulation(self):
        if self.paused:
            self.paused = False
            self.run_simulation()

    def update_stats_display(self):
        self.population_label.config(text=f"Population Size: {self.simulation.total_population}")
        self.fitness_label.config(text=f"Average Fitness: {self.simulation.average_fitness:.2f}")
        self.wave_label.config(text=f"Current Wave: {self.simulation.wave}")
        self.revolution_label.config(text=f"Current Revolution: {self.simulation.revolution}")

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
