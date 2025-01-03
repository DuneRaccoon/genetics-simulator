from __future__ import annotations

import random
import heapq
import math
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Literal, Tuple, Union

from config import Config

# Generate gene names (will be updated after configuration)
gene_names = []  # e.g. ['gene1', 'gene2', 'gene3', ...]
traits_list = []  # e.g. ['strength', 'speed', 'intelligence', ...]
gene_trait_mapping = {}  # e.g. {'gene1': {'strength': 0.5, 'speed': -0.3}, ...}
trait_weights = {}  # e.g. {'strength': 1.2, 'speed': 0.8, ...}

MAX_MEMORY_CAPACITIY = 20
MIN_MEMORY_CAPACITIY = 3
DIVERSITY = 5 # Diversity of gene pool amongst population e.g. 5 means splitting population in 5 groups of n

def split_n(x: int, y: int):
    # Split x into y parts
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
        self.sex = random.choice(['male', 'female'])
        self.position = position
        self.alive = True
        self.hunger = 0  # Hunger level (0 to 100)
        self.thirst = 0  # Thirst level (0 to 100)
        self.energy = 100  # Energy level (0 to 100)
        self.health = 100.0
        self.calculate_traits()
        self.phenotype = self.express_phenotype()
        self.fitness = self.calculate_fitness()
        self.target = None  # Target position for pathfinding
        self.path = []      # Path to follow
        self.known_resources: Dict[Tuple[int, int], Resource] = {}
        self.reproduction_cooldown = 0  # Cooldown period after reproduction
        self.mate: Union[Creature, None] = None  # Current mate target
        self.reproduction_counter = 0  # Number of successful reproductions
        self.current_thought = ""  # Current action or thought
        self.thoughts = []  # Log of actions and thoughts
        self.cause_of_death = ""
        self.target_resource = None  # Current target resource
        
        self.memory_resources: Dict[Tuple[int, int], Dict[str, any]] = {}
        self.memory_creatures: Dict[Tuple[int, int], Dict[str, any]] = {}

        # For females, initialize reproductive cycle
        if self.sex == 'female':
            self.cycle_length = 15  # Total length of the cycle in revolutions
            self.fertile_window = (7, 15)  # Days in cycle when the female is fertile
            self.cycle_day = random.randint(1, self.cycle_length)  # Start at a random day in the cycle

    @property
    def vision_range(self) -> int:
        """
        Scale vision range based on vision trait, between 1 and 6 cells around self
        """
        return int(self.traits['vision'] * 5) + 1
    
    @property
    def memory_capacity(self):
        # Scale memory capacity based on intelligence trait
        return int(self.traits['intelligence'] * (MAX_MEMORY_CAPACITIY - MIN_MEMORY_CAPACITIY)) + MIN_MEMORY_CAPACITIY
    
    @property
    def defensive_power(self):
        endurance = self.traits.get('endurance') or 0.0
        camouflage = self.traits.get('camouflage') or 0.0
        speed = self.traits.get('speed') or 0.0
        return (endurance * 0.6 + camouflage * 0.3 + speed * 0.1) * 50
    
    @property
    def attack_power(self):
        strength = self.traits.get('strength') or 0.0
        aggression = self.traits.get('aggression') or 0.0
        speed = self.traits.get('speed') or 0.0
        return (strength * 0.6 + aggression * 0.3 + speed * 0.1) * 50
    
    @property
    def critical_chance(self):
        intelligence = self.traits.get('intelligence') or 0.0
        aggression = self.traits.get('aggression') or 0.0
        return (intelligence * 0.5 + aggression * 0.5) * 0.2  # Max 20% chance

    @property
    def evade_chance(self):
        speed = self.traits.get('speed') or 0.0
        intelligence = self.traits.get('intelligence') or 0.0
        return (speed * 0.7 + intelligence * 0.3) * 0.5  # Max 50%
    
    @property
    def hit_chance(self):
        aggression = self.traits.get('aggression') or 0.0
        intelligence = self.traits.get('intelligence') or 0.0
        return (aggression * 0.7 + intelligence * 0.3) * 0.8
    
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

        # Traits that should always be present, as they are crucial for survival
        always_present_traits = ['intelligence', 'strength', 'speed', 'endurance', 'vision']

        # For always present traits, use a normal distribution (to simulate a bell curve)
        for trait in always_present_traits:
            # Sum contributions from multiple genes to get a normal distribution
            gene_contributions = []
            for chromosome in self.chromosomes:
                for gene, gene_value in chromosome.items():
                    if gene in gene_trait_mapping and trait in gene_trait_mapping[gene]:
                        weight = gene_trait_mapping[gene][trait]
                        gene_contributions.append(gene_value * weight)

            # If there are contributions, calculate the trait value
            if gene_contributions:
                mean = sum(gene_contributions) / len(gene_contributions)
                variance = sum((x - mean) ** 2 for x in gene_contributions) / len(gene_contributions)
                std_dev = math.sqrt(variance) if variance > 0 else 0
                # Normalize the trait value to be between 0 and 1
                trait_value = max(min(mean + std_dev, 1), 0)
                self.traits[trait] = trait_value
            else:
                # If no genes affect the trait, set a default mid value
                self.traits[trait] = 0.5

        # For other traits, use the previous system
        for trait in self.traits:
            if trait not in always_present_traits:
                # Sum contributions from all genes
                for chromosome in self.chromosomes:
                    for gene, gene_value in chromosome.items():
                        if gene in gene_trait_mapping and trait in gene_trait_mapping[gene]:
                            weight = gene_trait_mapping[gene][trait]
                            self.traits[trait] += gene_value * weight
                # Normalize traits to be between 0 and 1
                self.traits[trait] = max(min(self.traits[trait], 1), 0)
                
        
        # # Potentially normalise traits?
        # trait_values = list(self.traits.values())
        # mean = sum(trait_values) / len(trait_values)
        # variance = sum((x - mean) ** 2 for x in trait_values) / len(trait_values)
        # std_dev = math.sqrt(variance) if variance > 0 else 0.0001

        # for trait in self.traits:
        #     z = (self.traits[trait] - mean) / std_dev
        #     self.traits[trait] = 0.5 + z * 0.15
        #     self.traits[trait] = max(min(self.traits[trait], 1), 0)

    def express_phenotype(self):
        """
        Convert traits to a color phenotype for visualization.
        """
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
    
    def analyse_phenotype(self, creature: Creature):
        """
        Extract traits from a given creature's phenotype.
        
        Can be used by creatures with >= mid intelligence to analyze other creatures' traits.
        """
        
        if self.traits['intelligence'] < 0.5:
            return None
        
        r = int(creature.phenotype[1:3], 16)
        g = int(creature.phenotype[3:5], 16)
        b = int(creature.phenotype[5:7], 16)

        # Scale RGB values to 0-1 range
        r_avg = (r - 50) / 205
        g_avg = (g - 50) / 205
        b_avg = (b - 50) / 205

        # Calculate trait values based on RGB channels
        r_traits = ['aggression', 'strength', 'speed']
        g_traits = ['camouflage', 'endurance', 'longevity']
        b_traits = ['vision', 'intelligence', 'hearing']

        traits = {}
        for trait, value in zip(r_traits + g_traits + b_traits, [r_avg, g_avg, b_avg]):
            traits[trait] = value

        return traits

    def calculate_fitness(self):
        # Weighted fitness function
        fitness = sum(self.traits[trait] * trait_weights[trait] for trait in self.traits)
        return fitness
    
    def have_thought(self, thought: str):
        self.current_thought = thought
        self.thoughts.append(thought)
        
        if len(self.thoughts) > self.memory_capacity:
            # Forget old thoughts
            self.thoughts.pop(0)
            
    def die(self, cause: str):
        self.alive = False
        self.cause_of_death = cause
        self.simulation.events.append(f"Creature at {self.position} died of {cause}.")
        self.simulation.add_to_graveyard(self)
        return True

    def increase_hunger_thirst(self):
        # Increase hunger and thirst
        metabolism_rate = self.traits.get('metabolism') or 1.0
        hunger_increase = 5 * metabolism_rate
        thirst_increase = 5 * metabolism_rate
        self.hunger += hunger_increase
        self.thirst += thirst_increase
        return (self.hunger >= 100 or self.thirst >= 100) and self.die("Starvation/Dehydration")

    def age_creature(self):
        self.age += 1
        # Calculate lifespan based on longevity trait
        lifespan = int(self.traits['longevity'] * (Config.MAX_LIFESPAN - Config.MIN_LIFESPAN)) + Config.MIN_LIFESPAN
        return self.age >= lifespan and self.die("Old Age")

    def calculate_energy_consumption(self):
        # Base energy consumption per revolution
        base_energy_consumption = 2

        # Adjust based on metabolism and activity
        metabolism_rate = self.traits.get('metabolism') or 1.0
        endurance = self.traits.get('endurance') or 0.5
        activity_level = 1 if self.energy > 0 else 0  # Active if energy > 0

        # Energy consumption increases with higher metabolism and activity
        energy_consumption = base_energy_consumption * metabolism_rate * activity_level
        # Energy consumption decreases with higher endurance
        energy_consumption /= (endurance + 0.1)  # Avoid division by zero

        return energy_consumption

    def is_culled(self, fitness_threshold):
        """
        Other culling criteria can be added here, such as disease, natural disasters, etc.

        If the creature is culled, returns True, otherwise False.
        """
        return self.fitness < fitness_threshold and self.die("Fitness below threshold")

    def act(self):
        """
        Each "frame" or "revolution" of the simulation, we call the act method to decide the creature's next "thought" and action.
        
        ALl actions should initiate from this method.
        """
        if not self.alive:
            return

        # Adjust behavior based on circadian rhythm traits and time of day
        time_of_day = self.simulation.get_time_of_day()
        activity_level = self.determine_activity_level(time_of_day)

        if activity_level < 0.3 and self.energy < 25:
            # Less active during this time
            self.have_thought(f"Resting during {time_of_day}.")
            self.rest()
            return

        # Rest if out of energy
        if self.energy <= 0:
            self.have_thought("Resting to regain energy.")
            self.rest()
            return
        
        # Scan surroundings to update known resources
        self.scan_surroundings()
        
        # Check hunger and thirst levels; seek resources if necessary
        if self.hunger > 50 or self.thirst > 50:
            resource_type = 'food' if self.hunger > self.thirst else 'water'
            target = self.find_resource(resource_type)
            if target:
                self.have_thought(f"Seeking {resource_type}")
                self.move_towards_target(target)
                return
            else:
                 # Use memory to find resources
                memory_target = self.find_resource_in_memory(resource_type)
                if memory_target:
                    self.have_thought(f"Recalling location of {resource_type} from memory")
                    self.move_towards_target(memory_target)
                    return
                else:
                    self.have_thought(f"Searching for {resource_type}")
                    self.move()
                    return
            
        # Adjust reproduction behavior based on sex and cycle
        if self.sex == 'female':
            # Increment cycle day
            self.cycle_day += 1
            if self.cycle_day > self.cycle_length:
                self.cycle_day = 1  # Reset cycle

            # Check if in fertile window
            if self.fertile_window[0] <= self.cycle_day <= self.fertile_window[1]:
                can_reproduce = True
            else:
                can_reproduce = False
        else:
            # Males can attempt to reproduce at any time
            can_reproduce = True

        # Attempt reproduction if not hungry/thirsty and not in cooldown
        if can_reproduce and self.reproduction_cooldown == 0 and self.traits['fertility'] > 0.5:
            mate = self.find_mate()
            if mate:
                self.mate = mate
                self.move_towards_target(mate.position, mate_target=mate)
                self.have_thought(f"Seeking mate at {mate.position}")
                return
            else:
                # Use memory to find mates
                memory_mate = self.find_mate_in_memory()
                if memory_mate:
                    self.mate = memory_mate
                    self.move_towards_target(memory_mate['position'], mate_target=None)
                    self.have_thought(f"Recalling mate from memory at {memory_mate['position']}")
                    return

        # High aggression creatures seek targets
        if self.traits['aggression'] > 0.7:
            target = self.find_target()
            if target:
                self.move_towards_target(target.position, attack_target=target)
                self.have_thought(f"Attacking creature at {target.position}")
                return
            else:
                # Use memory to find targets
                memory_target = self.find_target_in_memory()
                if memory_target:
                    self.move_towards_target(memory_target['position'], attack_target=None)
                    self.have_thought(f"Recalling target from memory at {memory_target['position']}")
                    return

        # General movement
        self.move()
        
    def find_resource_in_memory(self, resource_type: Literal['food', 'water']):
        # Find the nearest remembered resource of the given type
        min_distance = None
        target = None
        x0, y0 = self.position
        current_time = self.simulation.total_revolutions

        # Clean up outdated memories (optional)
        self.cleanup_memory_resources()

        for (x1, y1), resource in self.memory_resources.items():
            if resource['resource'].type == resource_type:
                distance = abs(x1 - x0) + abs(y1 - y0)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    target = (x1, y1)
                    
        return target

    def find_mate_in_memory(self):
        # Find a remembered potential mate
        min_distance = None
        potential_mate = None
        x0, y0 = self.position

        # Clean up outdated memories (optional)
        self.cleanup_memory_creatures()

        for (x1, y1), creature_info in self.memory_creatures.items():
            if self.sex != creature_info['sex']:
                # Additional mate suitability checks can be added here
                distance = abs(x1 - x0) + abs(y1 - y0)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    potential_mate = creature_info
        return potential_mate
    
    def find_target_in_memory(self):
        # Find a remembered target creature
        min_distance = None
        target = None
        x0, y0 = self.position

        # Clean up outdated memories (optional)
        self.cleanup_memory_creatures()

        for (x1, y1), creature_info in self.memory_creatures.items():
            distance = abs(x1 - x0) + abs(y1 - y0)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                target = creature_info
        return target
    
    def cleanup_memory_resources(self):
        # Optionally remove resources that are likely consumed or outdated
        # For example, remove memories older than a certain number of revolutions
        
        for position, resource in list(self.memory_resources.items()):
            if resource["resource"].consumed:
                self.memory_resources.pop(position)
            elif resource.get('last_seen', 0) < self.simulation.total_revolutions - 10:
                self.memory_resources.pop(position)

    def cleanup_memory_creatures(self):
        # Optionally remove creature memories that are outdated
        for position, creature_info in list(self.memory_creatures.items()):
            if creature_info.get('last_seen', 0) < self.simulation.total_revolutions - 10:
                self.memory_creatures.pop(position)
                
    def decay_memory(self):
        # Reduce the strength of memories over time or remove old ones
        # For simplicity, we'll remove memories older than a certain threshold
        memory_lifespan = int(self.traits['intelligence'] * 50)  # Adjust as needed
        current_time = self.simulation.total_revolutions

        # Remove old resource memories
        to_remove = []
        for position, resource in self.memory_resources.items():
            if current_time - resource.get('last_seen', current_time) > memory_lifespan:
                to_remove.append(position)
        for position in to_remove:
            del self.memory_resources[position]

        # Remove old creature memories
        to_remove = []
        for position, creature_info in self.memory_creatures.items():
            if current_time - creature_info.get('last_seen', current_time) > memory_lifespan:
                to_remove.append(position)
        for position in to_remove:
            del self.memory_creatures[position]

    def rest(self):
        # Regain some energy while resting
        rest_efficiency = self.traits.get('rest_efficiency', 0.5)
        energy_gain = 5 + rest_efficiency * 5  # Base gain plus efficiency bonus
        self.energy += energy_gain
        if self.energy > 100:
            self.energy = 100

    def determine_activity_level(self, time_of_day: Literal['day', 'night', 'twilight']):
        # Activity level is influenced by circadian traits and time of day
        if time_of_day == 'day':
            activity_trait = self.traits.get('diurnal_activity', 0.5)
        elif time_of_day == 'night':
            activity_trait = self.traits.get('nocturnal_activity', 0.5)
        else:
            activity_trait = self.traits.get('crepuscular_activity', 0.5)

        # Additional modifiers can be added here (e.g., environmental factors)
        return activity_trait

    def scan_surroundings(self):
        """
        Scan the surroundings to update known resources and creatures within vision range.
        """
        vision_range = self.vision_range
        x0, y0 = self.position

        # List to collect observed resources and creatures
        observed_resources = []
        observed_creatures = []

        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                x = (x0 + dx) % Config.GRID_SIZE
                y = (y0 + dy) % Config.GRID_SIZE

                # Check for resources
                if (x, y) in self.simulation.resources:
                    resource = self.simulation.resources[(x, y)]
                    if not resource.consumed:
                        resource_info = {
                            'resource': resource,
                            'last_seen': self.simulation.total_revolutions
                        }
                        observed_resources.append(((x, y), resource_info))
                        self.known_resources[(x, y)] = resource

                # Check for creatures
                cell: Creature = self.simulation.grid[y][x]
                if isinstance(cell, Creature) and cell != self and cell.alive:
                    creature_info = {
                        'position': (x, y),
                        'sex': cell.sex,
                        'traits': cell.traits.copy(),
                        'age': cell.age,
                        'last_seen': self.simulation.total_revolutions
                    }
                    observed_creatures.append(((x, y), creature_info))

        # Update memory with observed resources
        for position, resource in observed_resources:
            self.memory_resources[position] = resource

        # Update memory with observed creatures
        for position, creature_info in observed_creatures:
            self.memory_creatures[position] = creature_info

        # Ensure memory does not exceed capacity
        self.manage_memory_capacity()

    def manage_memory_capacity(self):
        """
        Ensure memory does not exceed capacity by removing oldest memories.
        """
        while len(self.memory_resources) > self.memory_capacity:
            # Remove the oldest memory (FIFO)
            self.memory_resources.pop(next(iter(self.memory_resources)))

        while len(self.memory_creatures) > self.memory_capacity:
            # Remove the oldest memory
            self.memory_creatures.pop(next(iter(self.memory_creatures)))
            
    def find_resource(self, resource_type: Literal['food', 'water']):
        # Clean up known_resources by removing resources that no longer exist
        for pos in list(self.known_resources.keys()):
            if pos not in self.simulation.resources:
                del self.known_resources[pos]
                
        # Find the nearest known resource of the given type
        min_distance = None
        target = None
        x0, y0 = self.position
        for (x1, y1), resource in self.known_resources.items():
            if resource.type == resource_type and not resource.claimed and not resource.consumed:
                distance = abs(x1 - x0) + abs(y1 - y0)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    target = (x1, y1)
                    
        # Claim the resource if found
        if target:
            self.simulation.resources[target].claim()
            self.target_resource = target
        else:
            self.target_resource = None
            
        return target

    def find_target(self):
        vision_range = self.vision_range
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

    def find_mate(self):
        vision_range = self.vision_range
        # Find the nearest eligible mate within vision range
        min_distance = None
        potential_mate = None
        x0, y0 = self.position
        for creature in self.simulation.population:
            if creature != self and creature.alive and creature.reproduction_cooldown == 0:
                # Check for opposite sex
                if self.sex != creature.sex:
                    # For females, ensure they are in the fertile window
                    if creature.sex == 'female':
                        if not (creature.fertile_window[0] <= creature.cycle_day <= creature.fertile_window[1]):
                            continue  # Female is not fertile
                    # Check fertility trait and hunger/thirst levels
                    if creature.traits['fertility'] > 0.5 and creature.hunger <= 50 and creature.thirst <= 50:
                        x1, y1 = creature.position
                        distance = abs(x1 - x0) + abs(y1 - y0)
                        if distance <= vision_range:
                            if min_distance is None or distance < min_distance:
                                min_distance = distance
                                potential_mate = creature
        return potential_mate

    def move_towards_target(self, target_position: Tuple[int, int], attack_target: Creature = None, mate_target: Creature = None):
        # Check if target is still valid before moving
        if attack_target and not attack_target.alive:
            # Target is dead, abandon action
            return self.have_thought("Target is dead")
        elif mate_target and (not mate_target.alive or mate_target.reproduction_cooldown > 0):
            # Mate is no longer available
            self.have_thought("Mate no longer available")
            self.mate = None
            return
        elif not attack_target and not mate_target:
            # Check if resource at target_position still exists
            resource = self.simulation.resources.get(target_position)
            if not resource or resource.consumed:
                # Resource no longer available
                self.have_thought("Resource no longer available")
                if target_position in self.known_resources:
                    del self.known_resources[target_position]
                if target_position in self.memory_resources:
                    del self.memory_resources[target_position]
                self.target_resource = None
                return

        # If moving towards a resource from memory, check if resource is still there upon arrival
        if not attack_target and not mate_target and self.position == target_position:
            if (self.position in self.simulation.resources) and not self.simulation.resources[self.position].consumed:
                self.consume_resource(self.position)
            else:
                # Resource is not there; remove it from memory
                if self.position in self.memory_resources:
                    del self.memory_resources[self.position]
                self.have_thought("Resource not found at remembered location")

        # Similarly handle mates from memory
        if mate_target is None and self.position == target_position:
            # The mate may no longer be there; remove from memory
            if self.position in self.memory_creatures:
                del self.memory_creatures[self.position]
            self.have_thought("Mate not found at remembered location")
            
        # Determine pathfinding method based on intelligence
        if self.traits['intelligence'] > 0.8:
            # Very high intelligence: Use bidirectional A* pathfinding
            path = self.bidirectional_a_star(target_position)
        elif self.traits['intelligence'] > 0.6:
            # High intelligence: Use A* pathfinding
            path = self.a_star_pathfinding(target_position)
        elif self.traits['intelligence'] > 0.4:
            # Medium intelligence: Use Dijkstra's pathfinding
            path = self.dijkstra_pathfinding(target_position)
        elif self.traits['intelligence'] > 0.2:
            # Low intelligence: Greedy movement
            path = self.greedy_move(target_position)
        else:
            # Very low intelligence: Random movement
            path = [self.random_adjacent_position()]
            
        if path:
            next_position = path[0]
            self.position = next_position
            # Consume energy when moving
            self.energy -= self.calculate_movement_energy()
            if self.energy < 0:
                self.energy = 0
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
            self.have_thought("Can't move towards target")

    def calculate_movement_energy(self):
        # Energy cost of movement
        base_cost = 1
        speed = self.traits.get('speed', 0.5)
        endurance = self.traits.get('endurance', 0.5)
        metabolism = self.traits.get('metabolism', 1.0)
        # Faster creatures consume more energy to move
        energy_cost = base_cost + speed * 2
        # Endurance reduces energy cost
        energy_cost /= (endurance + 0.1)
        # Metabolism affects energy cost
        energy_cost *= metabolism
        return energy_cost

    def calculate_compatibility(self, creature2: Creature):
        # Compatibility based on similarity of key traits
        key_traits = ['social_behavior', 'adaptability', 'stress_tolerance']
        compatibility = sum(
            1 - abs(self.traits[trait] - creature2.traits[trait])
            for trait in key_traits
        ) / len(key_traits)
        return compatibility

    def reproduce(self, parent2: Creature):
        # Sexual reproduction with crossover and mutation
        child_chromosomes = []
        for i in range(Config.NUM_CHROMOSOMES):
            chromosome1 = self.chromosomes[i]
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

    def attempt_reproduction(self, mate: Creature):
        # Ensure opposite sexes
        if self.sex == mate.sex:
            self.simulation.events.append(f"Reproduction failed between same sex creatures at {self.position} and {mate.position}.")
            return

        # Check if mate is adjacent
        x0, y0 = self.position
        x1, y1 = mate.position
        if abs(x0 - x1) <= 1 and abs(y0 - y1) <= 1:
            # Check if female is fertile
            female = self if self.sex == 'female' else mate
            if not (female.fertile_window[0] <= female.cycle_day <= female.fertile_window[1]):
                self.simulation.events.append(f"Female at {female.position} is not fertile.")
                return

            if self.calculate_compatibility(mate) > 0.4:
                # Reproduce
                child_chromosomes = self.reproduce(mate)
                child_position = self.simulation.find_adjacent_free_cell(self.position)
                if child_position:
                    child = Creature(self.simulation, chromosomes=child_chromosomes, position=child_position)
                    self.simulation.new_offspring.append(child)
                    self.simulation.events.append(f"{self.sex.capitalize()} ({self.phenotype}) at {self.position} and {mate.sex} ({mate.phenotype}) at {mate.position} reproduced -> {child.phenotype}.")
                    self.reproduction_counter += 1
                    mate.reproduction_counter += 1
                    # Set reproduction cooldown
                    cooldown_time = Config.REVOLUTIONS_PER_WAVE
                    self.reproduction_cooldown = cooldown_time
                    mate.reproduction_cooldown = cooldown_time
                    # Reset mate target
                    self.mate = None
                    mate.mate = None
                else:
                    # No space to place offspring
                    self.simulation.events.append(f"No space to place offspring for parents at {self.position} and {mate.position}.")
            else:
                # Incompatible mate
                self.simulation.events.append(f"Creatures at {self.position} and {mate.position} incompatible for reproduction.")

    def cooldown(self):
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

    def move(self):
        """
        Move the creature based on its traits and needs.
        
        High intelligence creatures analyze surroundings and choose best path.
        Medium intelligence creatures move towards resources or fewer creatures.
        Low intelligence creatures move randomly.
        """
        
        vision_range = self.vision_range
        
        # Enhanced general movement based on intelligence and vision
        if self.traits['intelligence'] > 0.8:
            self.have_thought("Deeply analyzing current surroundings.")
            # High intelligence: Analyze surroundings and choose best path
            target = self.find_best_tile(vision_range)
            if target:
                path = self.bidirectional_a_star(target)
                if path:
                    self.have_thought("Found best path, moving towards target.")
                    self.position = path[0]
                else:
                    self.position = self.random_adjacent_position()
            else:
                self.position = self.random_adjacent_position()
        elif self.traits['intelligence'] > 0.6:
            self.have_thought("Analyzing surroundings.")
            # High intelligence: Analyze surroundings and choose best path
            target = self.find_best_tile(vision_range)
            if target:
                path = self.a_star_pathfinding(target)
                if path:
                    self.have_thought("Found best path, moving towards target.")
                    self.position = path[0]
                else:
                    self.position = self.random_adjacent_position()
            else:
                self.position = self.random_adjacent_position()
        elif self.traits['intelligence'] > 0.4:
            self.have_thought("Moving towards resources or fewer creatures.")
            # Medium intelligence: Move towards tile with resources or fewer creatures
            target = self.find_best_tile(vision_range)
            if target:
                path = self.dijkstra_pathfinding(target)
                if path:
                    self.position = path[0]
                else:
                    self.position = self.random_adjacent_position()
            else:
                self.position = self.random_adjacent_position()
        elif self.traits['intelligence'] > 0.2:
            self.have_thought("Moving towards resources or fewer creatures.")
            # Medium intelligence: Move towards tile with resources or fewer creatures
            target = self.find_best_tile(vision_range)
            if target:
                path = self.greedy_move(target)
                if path:
                    self.position = path[0]
                else:
                    self.position = self.random_adjacent_position()
            else:
                self.position = self.random_adjacent_position()
        else:
            # Low intelligence: Random movement
            self.have_thought("Moving randomly.")
            self.position = self.random_adjacent_position()
        
        # Consume energy when moving
        self.energy -= self.calculate_movement_energy()
        if self.energy < 0:
            self.energy = 0

    def find_best_tile(self, vision_range):
        # Analyze surroundings to find the best tile to move towards
        best_score = -float('inf')
        best_tile = None
        x0, y0 = self.position
        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                x = (x0 + dx) % Config.GRID_SIZE
                y = (y0 + dy) % Config.GRID_SIZE
                if (x, y) == (x0, y0):
                    continue
                # Evaluate tile
                score = self.evaluate_tile(x, y)
                if score > best_score:
                    best_score = score
                    best_tile = (x, y)
        return best_tile

    def evaluate_tile(self, x, y):
        # Assign a score to the tile based on certain criteria
        score = 0
        cell = self.simulation.grid[y][x]
        if cell is None:
            score += 1  # Prefer empty cells
        elif isinstance(cell, Resource) and not cell.consumed and not cell.claimed:
            score += 5  # Prefer resource tiles
        elif isinstance(cell, Creature):
            # Avoid tiles with other creatures unless aggressive
            if self.traits['aggression'] > 0.7:
                score += 2  # May want to attack
            else:
                score -= 5  # Avoid confrontation
        if (x, y) in self.simulation.obstacles:
            score -= 10  # Avoid obstacles
        return score

    def attack(self, target: Creature):
        hit_probability = max(min(self.hit_chance - target.evade_chance, 0.95), 0.05)  # Ensure between 5% and 95%

        if random.random() < hit_probability: # Successful hit
            attack_power = self.attack_power
            
            is_critical = random.random() < self.critical_chance
        
            if is_critical:
                attack_power = attack_power * 1.5  # Critical hit deals 150% damage
            else:
                attack_power = attack_power
                
            damage = attack_power - target.defensive_power * 0.5  # Defense reduces damage by up to 50%
            damage = max(damage, 0)  # No negative damage

            # Apply damage to target
            target.health -= damage

            if is_critical:
                self.simulation.events.append(
                    f"Creature at {self.position} landed a critical hit on creature at {target.position} for {damage:.1f} damage."
                )
            else:
                self.simulation.events.append(
                    f"Creature at {self.position} attacked creature at {target.position} for {damage:.1f} damage."
                )

            if target.health <= 0:
                target.die(f"Killed by creature at {self.position}")
                self.simulation.events.append(
                    f"Creature at {self.position} killed creature at {target.position}"
                )
                # Consume the creature as food
                self.consume_creature(target)
            else:
                # Target is still alive
                self.simulation.events.append(
                    f"Creature at {target.position} has {target.health:.1f} health remaining."
                )
                # Target may counterattack if aggressive
                if target.traits.get('aggression', 0.5) > 0.4:
                    target.counterattack(self)
        else:
            self.simulation.events.append(
                f"Creature at {self.position} failed attack on creature at {target.position}"
            )
            
        self.simulation.attack_events.append({
            'attacker_pos': self.position,
            'target_pos': target.position,
            'start_time': time.time(),  # Record the current time
            'duration': 0.5  # Animation duration in seconds
        })
            
    def counterattack(self, attacker: Creature):
        # Same as attack, but without critical hits
        hit_probability = max(min(self.hit_chance - attacker.evade_chance, 0.95), 0.05)  # Ensure between 5% and 95%

        if random.random() < hit_probability: # Successful hit
            damage = self.attack_power - attacker.defensive_power * 0.5
            damage = max(damage, 0)  # No negative damage
            attacker.health -= damage
            
            if attacker.health <= 0:
                attacker.die(f"Killed in counterattack by creature at {self.position}")
                self.simulation.events.append(
                    f"Creature at {self.position} killed creature at {attacker.position} with a counterattack"
                )
                # Counterattacker consumes the attacker as food
                self.consume_creature(attacker)
            else:
                self.simulation.events.append(
                    f"Creature at {attacker.position} has {attacker.health:.1f} health remaining."
                )
        else:
            self.simulation.events.append(
                f"Creature at {self.position} missed counterattack on creature at {attacker.position}"
            )
            
        self.simulation.attack_events.append({
            'attacker_pos': self.position,
            'target_pos': attacker.position,
            'start_time': time.time(),
            'duration': 0.5  # Animation duration in seconds
        })
            
    def consume_resource(self, position: Tuple[int, int]):
        resource = self.simulation.resources.get(position)
        if resource and not resource.consumed:
            resource.consume()
            if resource.type == 'food':
                self.hunger = max(self.hunger - 15, 0)
                self.energy = min(self.energy + 10, 100)  # Regain energy
                self.health = min(self.health + 10, 100)  # Gain health from consuming
            elif resource.type == 'water':
                self.thirst = max(self.thirst - 15, 0)
                self.energy = min(self.energy + 5, 100)  # Regain some energy
                self.health = min(self.health + 5, 100)  # Gain health from consuming
            self.simulation.events.append(f"Creature at {self.position} consumed {resource.type}")
            self.target_resource = None
        else:
            # Resource no longer available
            self.target_resource = None
            self.have_thought("Resource no longer available")
            
        if position in self.known_resources:
            del self.known_resources[position]
            
        if position in self.memory_resources:
            del self.memory_resources[position]

    def consume_creature(self, target: Creature):
        # Consuming the defeated creature reduces hunger significantly
        self.hunger = max(self.hunger - 70, 0)
        self.energy = min(self.energy + 50, 100)  # Regain significant energy
        self.health = min(self.health + 30, 100)  # Gain health from consuming
        self.simulation.events.append(f"Creature at {self.position} consumed creature at {target.position}")

    def random_adjacent_position(self):
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
        max_iterations = 100  # Limit to prevent excessive computation
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
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

    def bidirectional_a_star(self, goal):
        start = self.position
        grid = self.simulation.grid

        # Open sets and closed sets for both directions
        open_set_start = []
        open_set_goal = []
        heapq.heappush(open_set_start, (0, start))
        heapq.heappush(open_set_goal, (0, goal))

        came_from_start = {}
        came_from_goal = {}

        g_score_start = {start: 0}
        g_score_goal = {goal: 0}

        f_score_start = {start: self.heuristic(start, goal)}
        f_score_goal = {goal: self.heuristic(goal, start)}

        closed_set_start = set()
        closed_set_goal = set()

        max_iterations = 200  # Limit to prevent excessive computation
        iterations = 0

        while open_set_start and open_set_goal and iterations < max_iterations:
            iterations += 1

            # Expand from start side
            _, current_start = heapq.heappop(open_set_start)
            closed_set_start.add(current_start)

            # Expand from goal side
            _, current_goal = heapq.heappop(open_set_goal)
            closed_set_goal.add(current_goal)

            # Check for intersection
            if current_start in closed_set_goal:
                meeting_point = current_start
                return self.construct_bidirectional_path(came_from_start, came_from_goal, meeting_point, start, goal)
            if current_goal in closed_set_start:
                meeting_point = current_goal
                return self.construct_bidirectional_path(came_from_start, came_from_goal, meeting_point, start, goal)

            # For neighbors from start
            for neighbor in self.simulation.get_adjacent_positions(*current_start):
                if neighbor in closed_set_start:
                    continue
                if not self.simulation.is_cell_free(*neighbor) and neighbor != goal:
                    continue
                tentative_g_score = g_score_start[current_start] + 1
                if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                    came_from_start[neighbor] = current_start
                    g_score_start[neighbor] = tentative_g_score
                    f_score_start[neighbor] = g_score_start[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set_start, (f_score_start[neighbor], neighbor))

            # For neighbors from goal
            for neighbor in self.simulation.get_adjacent_positions(*current_goal):
                if neighbor in closed_set_goal:
                    continue
                if not self.simulation.is_cell_free(*neighbor) and neighbor != start:
                    continue
                tentative_g_score = g_score_goal[current_goal] + 1
                if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                    came_from_goal[neighbor] = current_goal
                    g_score_goal[neighbor] = tentative_g_score
                    f_score_goal[neighbor] = g_score_goal[neighbor] + self.heuristic(neighbor, start)
                    heapq.heappush(open_set_goal, (f_score_goal[neighbor], neighbor))

        # If we reach here, no path was found within max_iterations
        return None

    def dijkstra_pathfinding(self, goal: Tuple[int, int]):
        """
        Dijkstra's algorithm for pathfinding. Applies to creatures with medium intelligence.
        
        *This is just a-star without the heuristic.
        """
        start = self.position
        grid = self.simulation.grid
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        max_iterations = 100  # Limit to prevent excessive computation
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current_cost, current = heapq.heappop(open_set)
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
                heapq.heappush(open_set, (g_score[neighbor], neighbor))
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
    
    def construct_bidirectional_path(
        self, 
        came_from_start: Dict[Tuple[int, int], Tuple[int, int]],
        came_from_goal: Dict[Tuple[int, int], Tuple[int, int]],
        meeting_point: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ):
        # Reconstruct path from start to meeting point
        path_from_start = [meeting_point]
        current = meeting_point
        while current != start:
            current = came_from_start[current]
            path_from_start.append(current)
        path_from_start.reverse()

        # Reconstruct path from meeting point to goal
        path_from_goal = []
        current = meeting_point
        while current != goal:
            current = came_from_goal[current]
            path_from_goal.append(current)

        # Combine paths, excluding the duplicate meeting point
        full_path = path_from_start + path_from_goal
        # Exclude the starting position for movement
        return full_path[1:]


class Resource:
    
    """
    Class representing a resource in the simulation.
    
    Resources can only exist within a simulation.
    """
    
    TYPES = ['food', 'water'] 
    
    COLOURS = {
        'food': 'yellow',
        'water': 'blue'
    }
    
    def __init__(
        self,
        simulation: Simulation,
        *,
        resource_type: Literal['food', 'water'],
        position: Tuple[int, int]
    ):
        self.simulation = simulation
        self.type = resource_type
        self.position = position
        self.claimed = False
        self.consumed = False
        self.last_seen = 0
        
    def __repr__(self):
        return f"Resource(type={self.type}, position={self.position}, claimed={self.claimed}, consumed={self.consumed})"
    
    @property
    def color(self):
        return self.COLOURS[self.type]
    
    def consume(self):
        """
        Consume the resource and then remove it from the simulation.
        """
        self.consumed = True
        # Remove resource from simulation
        del self.simulation.resources[self.position]
        
    def claim(self):
        self.claimed = True

class Simulation:
    def __init__(self):

        self.population_size = Config.POPULATION_SIZE
        self.population: List[Creature] = []
        self.new_offspring: List[Creature] = []  # Offspring to be added after revolution
        self.epoch = 0
        self.wave = 0
        self.revolution = 0
        self.environmental_factors = self.generate_environmental_factors()
        self.grid = self.create_grid()
        self.obstacles = set()
        self.resources: Dict[Tuple[int, int], Resource] = {}
        self.events: List[str] = []
        self.attack_events = []
        self.total_revolutions: int = 0
        self.gui: SimulationGUI = None  # Reference to GUI
        self.graveyard: List[Creature] = []  # List to store dead creatures

        self.place_obstacles()
        self.place_resources()
        
        # Diversify gene and trait mapping
        for n in split_n(self.population_size, DIVERSITY):
            self.generate_gene_names_and_traits()
            self.create_population(n)

        self.update_stats()

    @staticmethod
    def _show_constants():
        return gene_names, traits_list, gene_trait_mapping, trait_weights

    def generate_gene_names_and_traits(self):
        global gene_names, traits_list, gene_trait_mapping, trait_weights

        gene_names = [f'gene{i + 1}' for i in range(Config.TOTAL_GENES)]

        # Define traits
        traits_list = [
            'strength', 'speed', 'intelligence', 'endurance', 'aggression',
            'vision', 'camouflage', 'fertility', 'longevity', 'temperature_resistance',
            'hearing', 'immunity', 'metabolism', 'reproductive_rate', 'disease_resistance',
            'water_conservation', 'night_vision', 'social_behavior', 'adaptability', 'stress_tolerance',
            'diurnal_activity', 'nocturnal_activity', 'crepuscular_activity', 'rest_efficiency'
        ]

        # Ensure we have enough genes to affect traits
        if Config.TOTAL_GENES < len(traits_list):
            print("Not enough genes to cover all traits")
            raise Exception("Insufficient genes")

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
        trait_weights = {
            'strength': 1.0,
            'speed': 0.9,
            'intelligence': 1.2,
            'endurance': 1.0,
            'aggression': 0.8,
            'vision': 1.1,
            'camouflage': 1.0,
            'fertility': 0.9,
            'longevity': 2.0,
            'temperature_resistance': 1.0,
            'hearing': 0.9,
            'immunity': 1.2,
            'metabolism': 0.8,
            'reproductive_rate': 0.9,
            'disease_resistance': 1.2,
            'water_conservation': 1.0,
            'night_vision': 1.0,
            'social_behavior': 1.1,
            'adaptability': 1.2,
            'stress_tolerance': 1.0,
            'diurnal_activity': 1.0,
            'nocturnal_activity': 1.0,
            'crepuscular_activity': 1.0,
            'rest_efficiency': 1.0
        }

    def generate_environmental_factors(self):
        # Environmental factors can affect the fitness calculation
        factors = {
            'temperature': random.uniform(0, 1),  # 0: cold, 1: hot
            'predation': random.uniform(0, 1),    # Level of predation: 0: low, 1: high
            'resource_abundance': random.uniform(0, 1),  # Availability of resources: 0: scarce, 1: abundant
            'disease': random.uniform(0, 1),      # Disease prevalence: 0: low, 1: high
            'natural_disasters': random.uniform(0, 1)  # Frequency of natural disasters: 0: rare, 1: frequent
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
                resource_type = random.choice(resource_types)
                self.resources[(x, y)] = Resource(self, resource_type=resource_type, position=(x, y))

    def create_population(self, n: int):
        for _ in range(n):
            position = self.random_free_position() # Ensure creatures don't spawn on obstacles or resources
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

    def step(self):
        if self.is_simulation_complete():
            return
        if self.revolution >= Config.REVOLUTIONS_PER_WAVE:
            # End of wave
            self.increase_hunger_thirst()
            self.cull_population()
            self.wave += 1
            self.revolution = 0
            if self.wave >= Config.WAVES_PER_EPOCH:
                # End of epoch
                self.age_population()
                self.epoch += 1
                self.wave = 0
                self.events.append(f"===== Epoch {self.epoch} =====")
            else:
                self.events.append(f"--- Wave {self.wave} in Epoch {self.epoch} ---")
        else:
            # Run revolution
            self.run_revolution()

    def run_revolution(self):
        self.revolution += 1
        self.events.append(f"Revolution {self.revolution} in Wave {self.wave}, Epoch {self.epoch}")
        self.total_revolutions += 1
        self.update_stats()
        
        # Process each creature individually
        for creature in self.population:
            if creature.alive:
                creature.act()
                creature.cooldown()
        
        # Remove consumed resources
        self.resources = {pos: res for pos, res in self.resources.items() if not res.consumed}
             
        # Update grid once after all creatures have acted and resources have been consumed
        self.update_grid()
        
        # Add new offspring to the population
        if self.new_offspring:
            self.population.extend(self.new_offspring)
            self.new_offspring = []
            
        # Update grid after adding offspring
        self.update_grid()

    def get_time_of_day(self):
        # Return 'day', 'night', or 'twilight' based on revolution
        revolution_in_day = self.revolution % Config.REVOLUTIONS_PER_WAVE
        if (
            revolution_in_day < (Config.REVOLUTIONS_PER_WAVE * 0.25)
            or revolution_in_day >= (Config.REVOLUTIONS_PER_WAVE * 0.75)
        ):
            return 'night'
        elif (
            (Config.REVOLUTIONS_PER_WAVE * 0.25) <= revolution_in_day < (Config.REVOLUTIONS_PER_WAVE * 0.333333)
            or (Config.REVOLUTIONS_PER_WAVE * 0.666666) <= revolution_in_day < (Config.REVOLUTIONS_PER_WAVE * 0.75)
        ):
            return 'twilight'
        else:
            return 'day'

    def increase_hunger_thirst(self):
        # Increase hunger and thirst for all creatures at the end of a wave
        for creature in self.population[:]:  # Make a copy to avoid modification issues
            if creature.alive:
                if creature.increase_hunger_thirst():
                    try:
                        self.population.remove(creature)
                    except ValueError:
                        pass

    def add_to_graveyard(self, creature: Creature):
        self.graveyard.append(creature)
        x, y = creature.position
        self.grid[y][x] = None
        try:
            # Remove from population
            self.population.remove(creature)
        except ValueError:
            pass

    def is_simulation_complete(self):
        return self.epoch >= Config.NUM_EPOCHS or len(self.population) == 0

    def update_grid(self):
        self.grid = [[None for _ in range(Config.GRID_SIZE)] for _ in range(Config.GRID_SIZE)]
        for x, y in self.obstacles:
            self.grid[y][x] = 'obstacle'
        for (x, y), resource in self.resources.items():
            self.grid[y][x] = resource
        for creature in self.population:
            if creature.alive:
                x, y = creature.position
                self.grid[y][x] = creature

    def is_cell_free(self, x, y):
        x %= Config.GRID_SIZE
        y %= Config.GRID_SIZE
        cell = self.grid[y][x]
        return cell is None or isinstance(cell, Resource)

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
        # Age creatures and remove those that die of old age
        for creature in self.population[:]:  # Copy the list to avoid issues
            if creature.alive:
                if creature.age_creature():
                    try:
                        self.population.remove(creature)
                    except ValueError:
                        pass

    def update_stats(self):
        # Calculate stats for GUI display
        self.total_population = len(self.population)
        if self.population:
            self.average_fitness = sum(creature.fitness for creature in self.population) / self.total_population
        else:
            self.average_fitness = 0

    def set_gui(self, gui: SimulationGUI):
        self.gui = gui


class SimulationGUI:
    def __init__(self):
        self.simulation = None
        self.root = tk.Tk()
        self.root.title("Evolution Simulator")
        self.paused = False
        self.selected_creature = None
        self.last_event_index = 0  # For efficient event log updates
        self.animations = []
        self.create_configuration_interface()

    def create_configuration_interface(self):
        # Create GUI elements to set configuration
        self.config_frame = tk.Frame(self.root)
        self.config_frame.pack(pady=10)

        self.population_size_var = tk.IntVar(value=Config.POPULATION_SIZE)
        self.num_chromosomes_var = tk.IntVar(value=Config.NUM_CHROMOSOMES)
        self.genes_per_chromosome_var = tk.IntVar(value=Config.GENES_PER_CHROMOSOME)
        self.num_epochs_var = tk.IntVar(value=Config.NUM_EPOCHS)
        self.waves_per_epoch_var = tk.IntVar(value=Config.WAVES_PER_EPOCH)
        self.revolutions_per_wave_var = tk.IntVar(value=Config.REVOLUTIONS_PER_WAVE)
        self.mutation_rate_var = tk.DoubleVar(value=Config.MUTATION_RATE)
        self.grid_size_var = tk.IntVar(value=Config.GRID_SIZE)
        self.obstacles_percentage_var = tk.DoubleVar(value=Config.OBSTACLES_PERCENTAGE)
        self.resources_percentage_var = tk.DoubleVar(value=Config.RESOURCES_PERCENTAGE)

        self.create_config_field("Population Size:", self.population_size_var)
        self.create_config_field("Number of Chromosomes:", self.num_chromosomes_var)
        self.create_config_field("Genes per Chromosome:", self.genes_per_chromosome_var)
        self.create_config_field("Number of Epochs:", self.num_epochs_var)
        self.create_config_field("Waves per Epoch:", self.waves_per_epoch_var)
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
        Config.NUM_EPOCHS = self.num_epochs_var.get()
        Config.WAVES_PER_EPOCH = self.waves_per_epoch_var.get()
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
        self.canvas_size = 800

        # Create main GUI frames using notebook tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Simulation Tab
        self.simulation_frame = tk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text="Simulation")

        # Graveyard Tab
        self.graveyard_frame = tk.Frame(self.notebook)
        self.notebook.add(self.graveyard_frame, text="Graveyard")
        
        # Eventlog Tab
        self.event_log_frame = tk.Frame(self.notebook)
        self.notebook.add(self.event_log_frame, text="Event Log")

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
        self.epoch_label = tk.Label(self.stats_frame, text="")
        self.epoch_label.pack()
        self.wave_label = tk.Label(self.stats_frame, text="")
        self.wave_label.pack()
        self.revolution_label = tk.Label(self.stats_frame, text="")
        self.revolution_label.pack()
        self.time_of_day_label = tk.Label(self.stats_frame, text="")
        self.time_of_day_label.pack()

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
        
        # Eventlog listbox in the eventlog tab
        self.event_log_frame_listbox = tk.Listbox(self.event_log_frame, width=80)
        self.event_log_frame_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for the eventlog listbox
        self.event_log_frame_scrollbar = tk.Scrollbar(self.event_log_frame)
        self.event_log_frame_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.event_log_frame_listbox.config(yscrollcommand=self.event_log_frame_scrollbar.set)
        self.event_log_frame_scrollbar.config(command=self.event_log_frame_listbox.yview)

        self.draw_population()
        self.root.after(1000, self.run_simulation)  # Start simulation after 1 second

    def restart_simulation(self):
        # Reset the simulation and GUI
        self.simulation = None
        self.selected_creature = None
        self.paused = False
        self.last_event_index = 0
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
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=resource.color, outline="")

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
                    x0, y0, x1, y1, fill=creature.phenotype, outline=outline_color, width=outline_width
                )
                
        self.draw_animations(cell_size)
        
        # Update stats and event log
        self.update_stats_display()
        self.update_event_log()
        self.update_event_log_frame()
        self.update_selected_creature_info()  # Update selected creature info

    def draw_animations(self, cell_size: int):
        
        current_time = time.time()
        
        # Remove expired animations
        self.animations = [
            anim for anim in self.animations
            if current_time - anim['start_time'] < anim['duration']
        ]
        
        # Add new attack events to animations
        while self.simulation.attack_events:
            event = self.simulation.attack_events.pop(0)
            self.animations.append(event)

        for anim in self.animations:
            attacker_x, attacker_y = anim['attacker_pos']
            target_x, target_y = anim['target_pos']

            # Calculate pixel positions
            ax = attacker_x * cell_size + cell_size // 2
            ay = attacker_y * cell_size + cell_size // 2
            tx = target_x * cell_size + cell_size // 2
            ty = target_y * cell_size + cell_size // 2

            # Draw a line between attacker and target
            self.canvas.create_line(ax, ay, tx, ty, fill='red', width=2)

            # Draw a flashing circle at the target
            elapsed = current_time - anim['start_time']
            # Make the circle flash by changing its color based on time
            if int(elapsed * 10) % 2 == 0:
                circle_color = 'red'
            else:
                circle_color = 'yellow'

            self.canvas.create_oval(
                tx - cell_size // 4, ty - cell_size // 4,
                tx + cell_size // 4, ty + cell_size // 4,
                outline=circle_color, width=2
            )
        
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
            info += f"Sex: {c.sex.capitalize()}\n"
            info += f"Health: {c.health:.1f}\n"
            info += f"Current Thought: {c.current_thought}\n"
            for trait, value in c.traits.items():
                info += f"{trait.capitalize()}: {value:.2f}\n"
            info += f"Hunger: {c.hunger:.1f}\n"
            info += f"Thirst: {c.thirst:.1f}\n"
            info += f"Energy: {c.energy:.1f}\n"
            info += f"Age: {c.age}\n"
            info += f"Reproduction Count: {c.reproduction_counter}\n"
            if c.sex == 'female':
                info += f"Cycle Day: {c.cycle_day}/{c.cycle_length}\n"
            # Display memory capacity and current memory usage
            info += f"Memory Capacity: {c.memory_capacity}\n"
            info += f"Known Resources: {len(c.memory_resources)}\n"
            info += f"Known Creatures: {len(c.memory_creatures)}\n"
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
                info += f"Sex: {c.sex.capitalize()}\n"
                info += f"Current Thought: {c.current_thought}\n"
                for trait, value in c.traits.items():
                    info += f"{trait.capitalize()}: {value:.2f}\n"
                info += f"Hunger: {c.hunger:.1f}\n"
                info += f"Thirst: {c.thirst:.1f}\n"
                info += f"Energy: {c.energy:.1f}\n"
                info += f"Age: {c.age}\n"
                if c.sex == 'female':
                    info += f"Cycle Day: {c.cycle_day}/{c.cycle_length}\n"
                break
        else:
            if (x, y) in self.simulation.resources:
                resource = self.simulation.resources[(x, y)]
                info += f'Resource at ({x}, {y}): {resource.type.capitalize()}\n'
            elif (x, y) in self.simulation.obstacles:
                info += f"Obstacle at ({x}, {y})\n"
            else:
                info += f"Empty cell at ({x}, {y})\n"
        self.creature_info_label.config(text=info)

    def run_simulation(self):
        if not self.paused and self.simulation:
            if self.simulation.is_simulation_complete():
                print("Simulation ended.")
                self.paused = True
                self.update_graveyard()
                return
            else:
                # Process one simulation step per call
                self.simulation.step()
                self.draw_population()
                self.root.after(100, self.run_simulation)
        else:
            self.draw_population()
            self.root.after(100, self.run_simulation)

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
        self.epoch_label.config(text=f"Current Epoch: {self.simulation.epoch}")
        self.wave_label.config(text=f"Current Wave: {self.simulation.wave}")
        self.revolution_label.config(text=f"Current Revolution: {self.simulation.revolution}")
        time_of_day = self.simulation.get_time_of_day().capitalize()
        self.time_of_day_label.config(text=f"Time of Day: {time_of_day}")

    def update_event_log(self):
        if self.simulation is None:
            return
        new_events = self.simulation.events[self.last_event_index:]
        if new_events:
            self.event_log_text.config(state='normal')
            for event in new_events:
                self.event_log_text.insert(tk.END, event + "\n")
            self.event_log_text.see(tk.END)  # Scroll to the end
            self.event_log_text.config(state='disabled')
            self.last_event_index = len(self.simulation.events)
            
    def update_event_log_frame(self):
        if self.simulation is None:
            return
        new_events = self.simulation.events[self.last_event_index:]
        if new_events:
            self.event_log_frame_listbox.delete(0, tk.END)
            for event in new_events:
                self.event_log_frame_listbox.insert(tk.END, event)
            self.event_log_frame_listbox.see(tk.END)
    

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SimulationGUI()
    gui.run()
