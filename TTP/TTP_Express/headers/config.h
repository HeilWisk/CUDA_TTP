// Kernel Config
#define BLOCKS 1
#define THREADS 100
#define ISLANDS BLOCKS * THREADS
#define BLOCK_SIZE 16 // Validate

// Genetic Algorithm Config
#define MUTATION_RATE 0.05
#define NUM_EVOLUTIONS 100
#define TOURNAMENT_SIZE 50
#define SELECTED_PARENTS 4
#define ELITISM true
#define MAX_COORD 250
#define LOCAL_SEARCH_PROBABILITY 0.2

// Travelling Thief Problem Config
#define POPULATION_SIZE 1
#define CITIES 5
#define ITEMS 4
#define TOURS 100
#define ITEMS_PER_CITY 1

// Other
#define DEBUG
#ifdef DEBUG
#define SHOW printf
#else
#define SHOW // macros
#endif // DEBUG

#define NO_GPU true
#define CUDA true
#define NO_CUDA false





