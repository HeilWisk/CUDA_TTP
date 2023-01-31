// Kernel Config
#define BLOCKS 1
#define THREADS 100
#define BLOCK_SIZE 16 // Validate

// Genetic Algorithm Config
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
#define NUMBER_EXECUTIONS 2

// Other
//#define DEBUG
#ifdef DEBUG
#define SHOW printf
#else
#define SHOW // macros
#endif // DEBUG

#define GPU true
#define CPU true
#define CUDA true
#define NO_CUDA false
#define WRITE_BUFFER 500
#define NAME_BUFFER 500

// OUTPUT VARIABLES
#define STATISTICS_FILE_NAME_CPU ".\\output\\STATISTICS_CPU_%s.txt"
#define STATISTICS_FILE_NAME_GPU ".\\output\\STATISTICS_GPU_%s.txt"
#define RESULTS_FILE_NAME_CPU ".\\output\\EXECUTION_%d_%s_CPU.txt"
#define RESULTS_FILE_NAME_GPU ".\\output\\EXECUTION_%d_%s_GPU.txt"