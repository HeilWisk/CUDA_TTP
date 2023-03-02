// Kernel Config
#define BLOCKS 1
#define THREADS 128 // Must be a multiple of 32 AND EQUAL OR GREATER THAN TOURS
#define THREADS_X 8 // The result of multiply eith THREADS_Y must be a multiple of 32 
#define THREADS_Y 4 // The result of multiply eith THREADS_X must be a multiple of 32
#define BLOCK_SIZE 16 // Validate

// Genetic Algorithm Config
#define NUM_EVOLUTIONS 10000
#define TOURNAMENT_SIZE 50
#define SELECTED_PARENTS 4
#define ELITISM true
#define MAX_COORD 250
#define LOCAL_SEARCH_PROBABILITY 1
#define EXECUTION_TIME 600
#define TIME_RESTRICTED true

// Travelling Thief Problem Config
#define POPULATION_SIZE 1
#define CITIES 51
#define ITEMS 50
#define TOURS 128
#define ITEMS_PER_CITY 1
#define NUMBER_EXECUTIONS 30

// Other
//#define DEBUG
#ifdef DEBUG
#define SHOW printf
#else
#define SHOW // macros
#endif // DEBUG

#define GPU true
#define CPU false
#define CUDA true
#define NO_CUDA false
#define WRITE_BUFFER 6000
#define NAME_BUFFER 1000

// OUTPUT VARIABLES
#define STATISTICS_FILE_NAME_CPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\STATISTICS_CPU_%s_%d.txt"
#define STATISTICS_FILE_NAME_GPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\STATISTICS_GPU_%s_%d.txt"
#define RESULTS_FILE_NAME_CPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\EXECUTION_%d_%s_CPU.txt"
#define RESULTS_FILE_NAME_GPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\EXECUTION_%d_%s_GPU.txt"
#define GLOBALSTATS_FILE_NAME_CPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\GLOBAL_STATS_CPU_%s.txt"
#define GLOBALSTATS_FILE_NAME_GPU "D:\\Development\\CUDA_TTP\\TTP\\TTP_EXPRESS\\output\\GLOBAL_STATS_GPU_%s.txt"