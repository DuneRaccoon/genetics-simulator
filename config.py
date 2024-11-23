
class Config:
    NUM_CHROMOSOMES = 5
    GENES_PER_CHROMOSOME = 10
    TOTAL_GENES = NUM_CHROMOSOMES * GENES_PER_CHROMOSOME
    POPULATION_SIZE = 25
    MIN_LIFESPAN = 20                                       # Minimum lifespan in epochs
    MAX_LIFESPAN = 100                                      # Maximum lifespan in epochs
    NUM_EPOCHS = 5                                          # Akin to a "year" in the simulation
    WAVES_PER_EPOCH = 20                                    # Akin to a "day" in the simulation
    REVOLUTIONS_PER_WAVE = 24                               # Akin to an "hour" in the simulation
    MUTATION_RATE = 0.05                                    # 5%
    GRID_SIZE = 40                                          # Grid dimensions for creature positions
    OBSTACLES_PERCENTAGE = 0.1                              # Percentage of grid cells that are obstacles
    RESOURCES_PERCENTAGE = 0.1                              # Percentage of grid cells that have resources
