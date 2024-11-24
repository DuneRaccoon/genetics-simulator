
class Config:
    NUM_CHROMOSOMES = 5
    GENES_PER_CHROMOSOME = 10
    TOTAL_GENES = NUM_CHROMOSOMES * GENES_PER_CHROMOSOME
    POPULATION_SIZE = 25
    MIN_LIFESPAN = 20                                       # Minimum lifespan in epochs
    MAX_LIFESPAN = 100                                      # Maximum lifespan in epochs
    NUM_EPOCHS = 15                                         # Akin to a "year" in the simulation
    WAVES_PER_EPOCH = 24                                    # Akin to a "day" in the simulation
    REVOLUTIONS_PER_WAVE = 32                               # Akin to an "hour" in the simulation
    MUTATION_RATE = 0.05                                    # 5%
    GRID_SIZE = 40                                          # Grid dimensions for creature positions
    OBSTACLES_PERCENTAGE = 0.05                             # Percentage of grid cells that are obstacles
    RESOURCES_PERCENTAGE = 0.075                            # Percentage of grid cells that have resources
