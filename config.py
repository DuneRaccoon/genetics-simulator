
class Config:
    NUM_CHROMOSOMES = 5
    GENES_PER_CHROMOSOME = 10
    TOTAL_GENES = NUM_CHROMOSOMES * GENES_PER_CHROMOSOME
    POPULATION_SIZE = 25
    NUM_WAVES = 50
    REVOLUTIONS_PER_WAVE = 10
    MUTATION_RATE = 0.05  # 5%
    GRID_SIZE = 40  # Grid dimensions for creature positions
    OBSTACLES_PERCENTAGE = 0.1  # Percentage of grid cells that are obstacles
    RESOURCES_PERCENTAGE = 0.1  # Percentage of grid cells that have resources
