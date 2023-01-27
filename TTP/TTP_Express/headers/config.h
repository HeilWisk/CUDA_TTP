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
#define CITIES 52
#define ITEMS 51
#define TOURS 100
#define ITEMS_PER_CITY 1

// Other
//#define DEBUG
#ifdef DEBUG
#define SHOW printf
#define THREAD_TOURNAMENT 3
#define THREAD_EVOLUTION 57
#else
#define SHOW // macros
#define THREAD_TOURNAMENT // macros
#define THREAD_EVOLUTION // macros
#endif // DEBUG

#define GPU true
#define CPU true
#define CUDA true
#define NO_CUDA false
#define WRITE_BUFFER 500
#define NAME_BUFFER 500