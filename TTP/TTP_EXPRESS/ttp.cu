﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <time.h>

#include "headers/helper_functions.h"
#include "headers/helper_cuda.h"
#include "headers/config.h"
#include "headers/item.cuh"
#include "headers/node.cuh"
#include "headers/params.h"
#include "headers/distance.cuh"
#include "headers/greedy.h"
#include "headers/tour.cuh"
#include "headers/population.cuh"
#include "headers/genetic.cuh"
#include "headers/File.h"

#define NAME "PROBLEM NAME:"
#define DIMENSION "DIMENSION:"
#define ITEM_QTY "NUMBER OF ITEMS:"
#define KNAPSACK_CAPACITY "CAPACITY OF KNAPSACK:"
#define MIN_SPEED "MIN SPEED:"
#define MAX_SPEED "MAX SPEED:"
#define RENTING_RATIO "RENTING RATIO:"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE:"
#define NODE_COORD_SECTION "NODE_COORD_SECTION	(INDEX, X, Y):"
#define ITEMS_SECTION "ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):"

float mean(float array[], int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + array[i];
	}
	return (sum / n);
}

float mean(double array[], int n)
{
	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + array[i];
	}
	return (float)(sum / n);
}

float median(float array[], int n)
{
	float temp;
	for (int i = n - 1; i >= 0; --i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (array[j] >= array[j + 1])
			{
				temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
	if (n % 2 == 0)
	{
		return (array[n / 2] + array[n / 2 - 1]) / 2;
	}
	else
	{
		return array[n / 2];
	}
}

float median(double array[], int n)
{
	double temp;
	for (int i = n - 1; i >= 0; --i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (array[j] >= array[j + 1])
			{
				temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
	if (n % 2 == 0)
	{
		return (float)(array[n / 2] + array[n / 2 - 1]) / 2;
	}
	else
	{
		return (float)array[n / 2];
	}
}

float mode(float array[], int n)
{
	return (3 * median(array, n) - 2 * mean(array, n));
}

float mode(double array[], int n)
{
	return (3 * median(array, n) - 2 * mean(array, n));
}

double standardDeviation(float array[], int n)
{
	double* max = (double*)malloc(n * sizeof(double));
	if (max == NULL)
	{
		fprintf(stderr, "Out of memory");
	}
	double sum, variance, meanThis;
	meanThis = mean(array, n);
	sum = 0;
	int j;
	for (j = 0; j < n; j++)
	{
		max[j] = pow((array[j] - meanThis), 2);
		sum += max[j];
	}
	variance = sum / (j - 1);
	free(max);
	return sqrt(variance);
}

double standardDeviation(double array[], int n)
{
	double* max = (double*)malloc(n * sizeof(double));
	if (max == NULL)
	{
		fprintf(stderr, "Out of memory");
	}
	
	double sum, variance, meanThis;
	meanThis = mean(array, n);
	sum = 0;
	int j;
	for (j = 0; j < n; j++)
	{
		max[j] = pow((array[j] - meanThis), 2);
		sum += max[j];
	}
	variance = sum / (j - 1);
	free(max);
	return sqrt(variance);
}

double sumArray(double array[], int n)
{
	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + array[i];
	}
	return sum;
}

float sumArray(float array[], int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + array[i];
	}
	return sum;
}

/// <summary>
/// Function to count the amount of lines in a file
/// </summary>
/// <param name="fileName">- File path and name of the file to evaluate</param>
/// <returns>Amount of lines in the file</returns>
int countFileLines(char fileName[]) {

	FILE* filePtr;
	int lineCount = 0;
	char chr;

	filePtr = fopen(fileName, "r");
	chr = getc(filePtr);
	while (chr != EOF)
	{
		if (chr == '\n')
			lineCount++;
		chr = getc(filePtr);
	}
	fclose(filePtr);
	return lineCount;
}

/// <summary>
/// Function to find a character position in a string
/// </summary>
/// <param name="stringToSearch">- String to search</param>
/// <param name="characterToFind">- Character to find in the string</param>
/// <returns>Position in the string of the character</returns>
size_t findCharacterPosition(char stringToSearch[], char characterToFind)
{
	size_t stringLength = 0, i, characterPosition = 0;

	stringLength = strlen(stringToSearch);
	for (i = 0; i < stringLength; i++)
	{
		if (stringToSearch[i] == characterToFind)
			characterPosition = i + 1;
	}

	return (characterPosition);
}

/// <summary>
/// Extracts a string from another string
/// </summary>
/// <param name="originalString">- Original string</param>
/// <param name="subString">- Resulting Substring</param>
/// <param name="position">- Initial position where the substring is about to begin</param>
/// <param name="length">- Length of the desired substring</param>
void subString(char originalString[], char subString[], size_t position, size_t length)
{
	int c = 0, d = 0;
	char tempSubString[1000];

	while (c < length) {
		subString[c] = originalString[position + c - 1];
		c++;
	}

	subString[c] = '\0';
	c = 0;

	while (subString[c] != '\0')
	{
		if (subString[c] == ' ' || subString[c] == '	') {
			int temp = c + 1;
			if (subString[temp] != '\0') {
				while ((subString[temp] == ' ' || subString[c] == '	') && subString[temp] != '\0') {
					if (subString[temp] == ' ' || subString[c] == '	')
						c++;
					temp++;
				}
			}
		}
		tempSubString[d] = subString[c];
		c++;
		d++;
	}

	tempSubString[d] = '\0';
	strcpy(subString, tempSubString);
}

/// <summary>
/// Count the rows for a matrix in a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name of the file to evaluate</param>
/// <param name="sectionName">- Section name in the file where the matrix begins</param>
/// <returns>Amount of rows in the matrix</returns>
int countMatrixRows(const char fileName[], const char sectionName[])
{
	FILE* filePtr;
	char str[255], sub[255];
	int lineCount = 0, initialPosition = 0, rows = 0;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			rows++;
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}
	fclose(filePtr);
	return rows;
}

/// <summary>
/// Extracts matrix from a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name</param>
/// <param name="sectionName">- Section name in the file</param>
/// <param name="rows">- Amount of columns</param>
/// <param name="cols">- Amount of rows</param>
/// <returns>Double pointer matrix of ints</returns>
int** extractMatrixFromFile(const char fileName[], const char sectionName[], int rows, int cols)
{
	FILE* filePtr;
	char str[255], sub[255], * token;
	int lineCount = 0, initialPosition = 0, matrixRow, matrixCol;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	// Allocate memory for rows
	int** matrixResult = (int**)malloc(rows * sizeof(int*));
	if (matrixResult == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Allocate memory for columns
	for (int i = 0; i < rows; i++) {
		matrixResult[i] = (int*)malloc(cols * sizeof(int));
		if (matrixResult[i] == NULL) {
			fprintf(stderr, "Out of Memory");
			exit(0);
		}
	}

	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			token = strtok(str, "	");
			matrixCol = 0;
			matrixRow = atoi(token) - 1;
			while (token != NULL)
			{
				matrixResult[matrixRow][matrixCol] = atoi(token);
				token = strtok(NULL, "	");
				if (matrixCol < cols)
					matrixCol++;
			}
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}

	fclose(filePtr);

	return matrixResult;
}

/// <summary>
/// Calculates euclidean distance between a matrix of source points and a matrix of destination points
/// </summary>
/// <param name="srcPoint">- Matrix of source points</param>
/// <param name="dstPoint">- Matrix of destination points</param>
/// <param name="out">- Result matrix with distances</param>
/// <param name="rCount">- Row count</param>
/// <param name="size">- Total size of the result matrix</param>
void euclideanDistanceCPU(node* srcPoint, node* dstPoint, distance* out, int rCount, int size) {
	for (int s = 0; s < size; s++) {
		for (int xSrc = 0; xSrc < rCount; xSrc++) {
			for (int xDst = 0; xDst < rCount; xDst++) {
				out[s].source = srcPoint[xSrc].id;
				out[s].destiny = dstPoint[xDst].id;
				out[s].value = (float)sqrt(pow(dstPoint[xDst].x - srcPoint[xSrc].x, 2) + pow(dstPoint[xDst].y - srcPoint[xSrc].y, 2) * 1.0);
				s++;
			}
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="state"></param>
/// <param name="seed"></param>
/// <returns></returns>
__global__ void initCuRand(curandState* state, time_t seed)
{
	// Calculate global index of the threads for the 2D GRID
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index >= TOURS)
		return;

	curand_init(seed, thread_global_index, 0, &state[thread_global_index]);
}

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="tour_size"></param>
/// <returns></returns>
__global__ void tourTest(tour* tour, int tour_size)
{
	for (int t = 0; t < tour_size; ++t)
	{
		printf(" > tour[%d].fitness: %f\n", t, tour[t].fitness);
		printf(" > tour[%d].total_distance: %f\n", t, tour[t].total_distance);
		for (int n = 0; n < CITIES; ++n)
		{
			printf(" > tour[%d].nodes[%d].id: %d\n", t, n, tour[t].nodes[n].id);
			printf(" > tour[%d].nodes[%d].x: %lf\n", t, n, tour[t].nodes[n].x);
			printf(" > tour[%d].nodes[%d].y: %lf\n", t, n, tour[t].nodes[n].y);
			for (int i = 0; i < ITEMS; ++i)
			{
				printf(" > tour[%d].nodes[%d].items[%d].id: %d\n", t, n, i, tour[t].nodes[n].items[i].id);
				printf(" > tour[%d].nodes[%d].items[%d].node: %d\n", t, n, i, tour[t].nodes[n].items[i].node);
				printf(" > tour[%d].nodes[%d].items[%d].value: %d\n", t, n, i, tour[t].nodes[n].items[i].value);
				printf(" > tour[%d].nodes[%d].items[%d].weight: %d\n", t, n, i, tour[t].nodes[n].items[i].weight);
			}
		}
	}
	printf("\n\n");
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <returns></returns>
__global__ void populationTest(population* population)
{
	for (int p = 0; p < POPULATION_SIZE; ++p)
	{
		for (int t = 0; t < TOURS; ++t)
		{
			printf(" > population[%d].tours[%d].fitness: %f\n", p, t, population[p].tours[t].fitness);
			printf(" > population[%d].tours[%d].total_distance: %f\n", p, t, population[p].tours[t].total_distance);
			for (int n = 0; n < CITIES; ++n)
			{
				if (population[p].tours[t].nodes[n].id > 0)
				{
					printf(" > population[%d].tours[%d].nodes[%d].id: %d\n", p, t, n, population[p].tours[t].nodes[n].id);
					printf(" > population[%d].tours[%d].nodes[%d].x: %lf\n", p, t, n, population[p].tours[t].nodes[n].x);
					printf(" > population[%d].tours[%d].nodes[%d].y: %lf\n", p, t, n, population[p].tours[t].nodes[n].y);
					for (int i = 0; i < ITEMS; ++i)
					{
						if (population[p].tours[t].nodes[n].items[i].id >= 0)
						{
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].id: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].id);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].node: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].node);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].value: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].value);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].weight: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].weight);
						}
					}
				}
			}
		}
	}
	printf("\n\n");
}

/// <summary>
/// 
/// </summary>
/// <param name="initial_population"></param>
/// <param name="distances"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <param name="state"></param>
/// <returns></returns>
__global__ void initPopulationGPU(population* initial_population, tour* initial_tour, const int population_size, curandState* state)
{
	node temp;

	// Get thread ID
	// Global index of every block on the grid
	unsigned int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	unsigned int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	unsigned int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	unsigned int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	curandState local_state = state[thread_global_index];

	// Set the tours
	for (int p = 0; p < population_size; ++p)
	{
		for (int n = 0; n < CITIES; ++n)
		{
			initial_population->tours[thread_global_index].nodes[n] = initial_tour[p].nodes[n];
		}
	}

	for (int j = 1; j < CITIES; ++j)
	{
		int random_position = 1 + (curand(&local_state) % (CITIES - 1));

		temp = initial_population->tours[thread_global_index].nodes[j];
		for (int k = 0; k < ITEMS; ++k)
		{
			temp.items[k] = initial_population->tours[thread_global_index].nodes[j].items[k];
		}

		printf(" > thread_global_index: %d > %d cambia con %d\n", thread_global_index, j, random_position);
		printf(" > thread_global_index: %d > El id de %d es: %d\n", thread_global_index, j, initial_population->tours[thread_global_index].nodes[j].id);
		printf(" > thread_global_index: %d > El id de %d es: %d\n", thread_global_index, random_position, initial_population->tours[thread_global_index].nodes[random_position].id);

		initial_population->tours[thread_global_index].nodes[j] = initial_population->tours[thread_global_index].nodes[random_position];
		for (int l = 0; l < ITEMS; ++l)
		{
			initial_population->tours[thread_global_index].nodes[j].items[l] = initial_population->tours[thread_global_index].nodes[random_position].items[l];
		}

		initial_population->tours[thread_global_index].nodes[random_position] = temp;
		for (int m = 0; m < ITEMS; ++m)
		{
			initial_population->tours[thread_global_index].nodes[random_position].items[m] = temp.items[m];
		}

		printf(" > thread_global_index: %d > initial_population->tours[%d].nodes[%d]: %d\n", thread_global_index, thread_global_index, j, initial_population->tours[thread_global_index].nodes[j].id);
		printf(" > thread_global_index: %d > initial_population->tours[%d].nodes[%d]: %d\n", thread_global_index, thread_global_index, random_position, initial_population->tours[thread_global_index].nodes[random_position].id);
	}
}

/// <summary>
/// Basic implementation of matrix transpose
/// </summary>
/// <param name="m_dev">- Matrix to be transposed on device memory</param>
/// <param name="t_m_dev">- Matrix Transpose result on device memory</param>
/// <param name="width">- Width of the matrix</param>
/// <param name="height">- Height of the matrix</param>
/// <returns></returns>
__global__ void transpose(node* m_dev, node* t_m_dev, int width, int height) {

	/* Calculate global index for this thread */
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Copy m_dev[rowIdx][colIdx] to t_m_dev[rowIdx][colIdx] */
	if (colIdx < width && rowIdx < height)
	{
		unsigned int index_in = colIdx + width * rowIdx;
		unsigned int index_out = rowIdx + height * colIdx;
		t_m_dev[index_out] = m_dev[index_in];
		for (int i = 0; i > ITEMS; ++i)
		{
			t_m_dev[index_out].items[i] = m_dev[index_in].items[i];
		}
	}
}

/// <summary>
/// Kernel to calculate distances between point matrixes
/// </summary>
/// <param name="m_src_dev">- Matrix with source coodinates</param>
/// <param name="m_dst_dev">- Matrix with destination coordinates</param>
/// <param name="m_dist_dev">- Result Matrix with euclidean distances</param>
/// <param name="m_dist_dev_rows">- Result matrix row count</param>
/// <param name="m_dist_dev_cols">- Result matrix column count</param>
/// <returns></returns>
__global__ void matrixDistances(node* m_src_dev, node* m_dst_dev, distance* m_dist_dev, int m_dist_dev_rows, int m_dist_dev_cols) {

	// Define variables
	const unsigned int width = 1;

	// Calculate global indexes
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundary conditions
	if (rowIdx < m_dist_dev_rows && colIdx < m_dist_dev_cols)
	{
		// Execute distance calculation
		float value = 0;
		int sourceId = 0;
		int destinyId = 0;
		for (int k = 0; k < width; k++)
		{
			sourceId = m_src_dev[rowIdx * width + k].id;
			destinyId = m_dst_dev[k * m_dist_dev_cols + colIdx].id;
			value += pow(m_dst_dev[k * m_dist_dev_cols + colIdx].x - m_src_dev[rowIdx * width + k].x, 2) + pow(m_dst_dev[k * m_dist_dev_cols + colIdx].y - m_src_dev[rowIdx * width + k].y, 2);
		}
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].source = sourceId;
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].destiny = destinyId;
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].value = sqrt(value);
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="distanceTable"></param>
/// <returns></returns>
__global__ void evaluatePopulation(population* population, distance* distanceTable)
{
	// Get Thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	evaluateTour(population->tours[tid], distanceTable);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="distanceTable"></param>
/// <returns></returns>
__global__ void evaluatePopulation(population* population, parameters* problem_parameters)
{
	// Calculate global index of the threads for the 2D GRID
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index >= TOURS)
		return;

	evaluateTour(population->tours[thread_global_index], problem_parameters);
}

int main()
{

#pragma region VARIABLES
	/****************************************************************************************************
	* DECLARE VARIABLES
	****************************************************************************************************/

	// File variables
	char file_name[255];
	char str[255];
	char sub[255];
	FILE* file_open;
	size_t position;
	const char openMode[] = "r";

	// Problem params
	parameters problem;
	int** matrix;
	char edge_weight_type[255];

	unsigned int population_size = POPULATION_SIZE;
	unsigned int tour_size = TOURS;

	tour initial_tour;
	population initial_population_cpu;
	population initial_population_gpu;
	tour fittestOnEarth;

	tour host_parents[SELECTED_PARENTS];
	tour host_tournament[TOURNAMENT_SIZE];

	int deviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t deviceErr;
	cudaError_t err = cudaSuccess;

	// Device Variables
	population* device_population;
	tour* device_initial_tour;
	tour* device_parents;
	tour* device_offspring;
	parameters* device_parameters;
	curandState* device_states;
	//node* device_node_matrix;
	//node* device_node_t_matrix;
	//distance* device_distance;

	// Counters
	struct timespec startMethod;
	struct timespec stopMethod;

	struct timespec startCPU;
	struct timespec stopCPU;

	float gpuExecutionTime = 0.0;

	double elapsedTimeInitialPopulationCPU[NUMBER_EXECUTIONS];
	double elapsedTimeCPU[NUMBER_EXECUTIONS];
	double elapsedSelectionTotalCPU[NUMBER_EXECUTIONS];
	double elapsedCrossoverTotalCPU[NUMBER_EXECUTIONS];
	double elapsedLocalSearchTotalCPU[NUMBER_EXECUTIONS];

	double elapsedSelectionCPU[NUM_EVOLUTIONS];
	double elapsedCrossoverCPU[NUM_EVOLUTIONS];
	double elapsedLocalSearchCPU[NUM_EVOLUTIONS];

	double solutionQualityCPU[NUM_EVOLUTIONS + 1];
	double solutionQualityTotalCPU[NUMBER_EXECUTIONS];

	cudaEvent_t startKernel;
	cudaEvent_t stopKernel;

	cudaEvent_t startGPU;
	cudaEvent_t stopGPU;	

	float elapsedTimeInitialPopulationGPU[NUMBER_EXECUTIONS];
	float elapsedTimeGPU[NUMBER_EXECUTIONS];
	float elapsedSelectionTotalGPU[NUMBER_EXECUTIONS];
	float elapsedCrossoverTotalGPU[NUMBER_EXECUTIONS];
	float elapsedLocalSearchTotalGPU[NUMBER_EXECUTIONS];

	float elapsedSelectionGPU[NUM_EVOLUTIONS];
	float elapsedCrossoverGPU[NUM_EVOLUTIONS];
	float elapsedLocalSearchGPU[NUM_EVOLUTIONS];

	double meanSelection;
	double meanCrossover;
	double meanLocalSearch;
	double meanSolution;

	double medianSelection;
	double medianCrossover;
	double medianLocalSearch;
	double medianSolution;

	double modeSelection;
	double modeCrossover;
	double modeLocalSearch;
	double modeSolution;

	double sdSelection;
	double sdCrossover;
	double sdLocalSearch;
	double sdSolution;

	double meanGlobalSelection;
	double meanGlobalCrossover;
	double meanGlobalLocalSearch;
	double meanGlobalInitializePopulation;
	double meanGlobalExecutionTime;
	double meanGlobalSolution;

	double medianGlobalSelection;
	double medianGlobalCrossover;
	double medianGlobalLocalSearch;
	double medianGlobalInitializePopulation;
	double medianGlobalExecutionTime;
	double medianGlobalSolution;
	
	double modeGlobalSelection;
	double modeGlobalCrossover;
	double modeGlobalLocalSearch;
	double modeGlobalInitializePopulation;
	double modeGlobalExecutionTime;
	double modeGlobalSolution;
	
	double sdGlobalSelection;
	double sdGlobalCrossover;
	double sdGlobalLocalSearch;
	double sdGlobalInitializePopulation;
	double sdGlobalExecutionTime;
	double sdGlobalSolution;

#pragma endregion

#pragma region CAPTURE FILE PATH

	/****************************************************************************************************
	* CAPTURE FILE PATH AND LOAD HIS DATA
	****************************************************************************************************/
	// Ask for the filepath & name where the problem is defined
	printf("Enter name of a file you wish to see\n");
	gets_s(file_name);
	printf("\n");

	// Open the file in read mode
	file_open = fopen(file_name, openMode);

	// Validates for errors on file opening
	if (file_open == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}

	// Print headers
	printf("****************************************************************************************\n");
	printf("CONTENTS OF THE FILE:\n");
	printf("****************************************************************************************\n");
	printf("The quantity of lines in the file are:	%d\n", countFileLines(file_name));

	// Obtain general data from file
	while (fgets(str, 100, file_open) != NULL) {
		position = findCharacterPosition(str, ':');
		// Extract problem name
		if (strncmp(str, NAME, strlen(NAME)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position - 1);
			strcpy(problem.name, sub);
		}
		// Extract amount of nodes (cities)
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.cities_amount = atoi(sub);
			printf("Nodes (Cities):				%d\n", problem.cities_amount);
		}
		// Extract the amount of items
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.items_amount = atoi(sub);
			problem.items_per_city = problem.items_amount / (problem.cities_amount - 1);
			printf("Item:					%d\n", problem.items_amount);
			printf("Items Per City:				%f\n", problem.items_per_city);
		}
		// Extract the knapsack capacity
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.knapsack_capacity = atof(sub);
			printf("Knapsack Capacity:			%lf\n", problem.knapsack_capacity);
		}
		// Extract the minimal speed
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.min_speed = atof(sub);
			printf("Minimum Speed:				%lf\n", problem.min_speed);
		}
		// Extract the maximum speed
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.max_speed = atof(sub);
			printf("Maximum Speed:				%lf\n", problem.max_speed);
		}
		// Extract the renting ratio
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.renting_ratio = atof(sub);
			printf("Renting Ratio:				%lf\n", problem.renting_ratio);
		}
		// Extract the edge weight type
		else if (strncmp(str, EDGE_WEIGHT_TYPE, strlen(EDGE_WEIGHT_TYPE)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			strcpy(edge_weight_type, sub);
			printf("Edge Weight Type is			%s", edge_weight_type);
		}
	}

	// Close file
	fclose(file_open);
	printf("****************************************************************************************\n");
	printf("\n");

#pragma endregion

#pragma region PROGRAM SETUP

	/****************************************************************************************************
	* PRINT CUDA AND GENETIC VALUES
	****************************************************************************************************/
	printf("****************************************************************************************\n");
	printf("PRGRAM SETUP\n");
	printf("****************************************************************************************\n");
	printf("LOCAL SEARCH PROPABILITY:		%f\n", LOCAL_SEARCH_PROBABILITY);
	printf("AMOUNT OF PARENTS:			%d\n", SELECTED_PARENTS);
	printf("TOURNAMENT SIZE:			%d\n", TOURNAMENT_SIZE);
	printf("AMOUNT OF EVOLUTIONS:			%d\n", NUM_EVOLUTIONS);
	printf("AMOUNT OF SOLUTIONS PER EXECUTION:	%d\n", TOURS);
	printf("AMOUNT OF EXECUTIONS:			%d\n", NUMBER_EXECUTIONS);
	printf("****************************************************************************************\n");

#pragma endregion

#pragma region PRINT SIZE OF STRUCT AND VARIABLES
	/****************************************************************************************************
	* PRINT SIZE OF STRUCTS
	****************************************************************************************************/
	printf("****************************************************************************************\n");
	printf("SIZE OF STRUCTS AND VARIABLES\n");
	printf("****************************************************************************************\n");
	printf("ITEM:				%lld\n", sizeof(item));
	printf("NODE:				%lld\n", sizeof(node));
	printf("TOUR:				%lld\n", sizeof(tour));
	printf("POPULATION:			%lld\n", sizeof(population));
	printf("PARAMETERS:			%lld\n", sizeof(parameters));
	printf("****************************************************************************************\n");
#pragma endregion

#pragma region READ FILE VALUES AND DISPLAY DATA

	/*************************************************************************************************
	* POPULATION INITIALIZATION ON HOST (CPU)
	*************************************************************************************************/

	// Obtain the items
	// Calculate amount of rows
	unsigned int item_rows = countMatrixRows(file_name, ITEMS_SECTION);

	// Calculate amount of columns
	unsigned int item_columns = 4;

	// Validate file consistency
	if (item_rows != problem.items_amount)
	{
		perror("The file information is not consistent. Number of items Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Get matrix
	matrix = extractMatrixFromFile(file_name, ITEMS_SECTION, problem.items_amount, item_columns);

	// Allocate memory for the array of structs
	item* cpu_item = (item*)malloc(problem.items_amount * sizeof(item));
	if (cpu_item == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Convert to array of struct
	extractItems(matrix, problem.items_amount, cpu_item);

	// Visualize values for item matrix	
	displayItems(cpu_item, problem.items_amount);

	// Obtain nodes
	// Calculate amount of nodes
	unsigned int node_rows = countMatrixRows(file_name, NODE_COORD_SECTION);

	// Validate file consistency
	if (node_rows != problem.cities_amount)
	{
		perror("The file information is not consistent. Number of node Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Calculate amount of columns
	unsigned int node_columns = 3;

	// Get matrix
	matrix = extractMatrixFromFile(file_name, NODE_COORD_SECTION, problem.cities_amount, node_columns);

	// Allocate memory for the array of structs
	node* cpu_node = (node*)malloc(problem.cities_amount * sizeof(node));
	if (cpu_node == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	// Convert to array of struct
	extractNodes(matrix, problem.cities_amount, cpu_node);

	// Assign items to node
	assignItems(cpu_item, cpu_node);

	// Print node information
	displayNodes(cpu_node, problem.cities_amount);

	// Create File For Statistics
	createStatisticsFile(problem.name, GPU, CPU);

#pragma endregion

	for (int clockCounter = 0; clockCounter < NUMBER_EXECUTIONS; ++clockCounter)
	{
#pragma region CENTRAL PROCESSING UNIT
		if (CPU)
		{
			// Create Results File 
			createOutputFile(problem.name, false, CPU, clockCounter);

			if (timespec_get(&startCPU, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}

			// Assign nodes to tour		
			defineInitialTour(initial_tour, &problem, cpu_node, cpu_item);

			if (timespec_get(&startMethod, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}

			// Initialize population by generating POPULATION_SIZE number of permutations of the initial tour, all starting at the same city
			initializePopulation(initial_population_cpu, initial_tour, problem);
			
			if (timespec_get(&stopMethod, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}
			
			elapsedTimeInitialPopulationCPU[clockCounter] = (double)(stopMethod.tv_sec - startMethod.tv_sec) + ((double)(stopMethod.tv_nsec - startMethod.tv_nsec) * 1.e-6);

			// Save the initial population to fil
			// Optional
			// saveInitialPopulation(problem.name, initial_population_cpu, problem, NO_CUDA, clockCounter, elapsedTimeInitialPopulationCPU[clockCounter]);
			//printPopulation(initial_population_cpu);			

			fittestOnEarth = getFittestTour(initial_population_cpu.tours, TOURS);
			saveFittest(problem.name, fittestOnEarth, problem, 0, NO_CUDA, clockCounter);
			solutionQualityCPU[0] = fittestOnEarth.fitness;
			printf("FITTEST TOUR OF INITIAL POPULATION: \n");
			printf("TIME: %f - FITNESS: %f - PROFIT: %f\n", fittestOnEarth.time, fittestOnEarth.fitness, fittestOnEarth.profit);
			printf("ROUTE: %d", fittestOnEarth.nodes[0].id);
			for (int i = 1; i < CITIES + 1; ++i)
			{
				printf(" > %d", fittestOnEarth.nodes[i].id);
			}
			printf("\n");
			printf("ITEMS: %d[%d]", fittestOnEarth.item_picks[0].id, fittestOnEarth.item_picks[0].pickup);
			for (int i = 1; i < ITEMS; ++i)
			{
				if (fittestOnEarth.item_picks[i].id > 0)
				{
					printf(" > %d[%d]", fittestOnEarth.item_picks[i].id, fittestOnEarth.item_picks[i].pickup);
				}
			}
			printf("\n\n");

			for (int i = 0; i < NUM_EVOLUTIONS; ++i)
			{
				// Select the best parents of the current generation
				if (timespec_get(&startMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				selection(initial_population_cpu, host_parents);
				if (timespec_get(&stopMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				elapsedSelectionCPU[i] = (double)(stopMethod.tv_sec - startMethod.tv_sec) + ((double)(stopMethod.tv_nsec - startMethod.tv_nsec) * 1.e-6);
				
				// Optional: Save data of parents to file
				//saveParents(problem.name, host_parents, problem, i + 1, NO_CUDA, clockCounter, elapsedSelectionCPU[i]);

				// Breed the population performing crossover (Combination of Ordered Crossover 
				// for the TSP sub-problem and One Point Crossover for the KP sub-problem)
				if (timespec_get(&startMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				crossover(initial_population_cpu, host_parents, problem);
				if (timespec_get(&stopMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				elapsedCrossoverCPU[i] = (double)(stopMethod.tv_sec - startMethod.tv_sec) + ((double)(stopMethod.tv_nsec - startMethod.tv_nsec) * 1.e-6);

				if (timespec_get(&startMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				localSearch(initial_population_cpu, problem);
				if (timespec_get(&stopMethod, TIME_UTC) != TIME_UTC)
				{
					printf("Error in calling timespec_get\n");
					exit(EXIT_FAILURE);
				}
				elapsedLocalSearchCPU[i] = (double)(stopMethod.tv_sec - startMethod.tv_sec) + ((double)(stopMethod.tv_nsec - startMethod.tv_nsec) * 1.e-6);

				//Optional: Save offspring data to file
				//saveOffspring(problem.name, initial_population_cpu, problem, i + 1, NO_CUDA, clockCounter, elapsedCrossoverCPU[i], elapsedLocalSearchCPU[i]);

				// Get Fittest tour of the generation
				fittestOnEarth = getFittestTour(initial_population_cpu.tours, TOURS);
				saveFittest(problem.name, fittestOnEarth, problem, i + 1, NO_CUDA, clockCounter);
				solutionQualityCPU[i + 1] = fittestOnEarth.fitness;
				printf("FITTEST TOUR OF GENERATION %d: \n", i + 1);
				printf("TIME: %f - FITNESS: %f - PROFIT: %f\n", fittestOnEarth.time, fittestOnEarth.fitness, fittestOnEarth.profit);
				printf("ROUTE: %d", fittestOnEarth.nodes[0].id);
				for (int i = 1; i < CITIES + 1; ++i)
				{
					printf(" > %d", fittestOnEarth.nodes[i].id);
				}
				printf("\n");
				printf("ITEMS: %d[%d]", fittestOnEarth.item_picks[0].id, fittestOnEarth.item_picks[0].pickup);
				for (int i = 1; i < ITEMS; ++i)
				{
					if (fittestOnEarth.item_picks[i].id > 0)
					{
						printf(" > %d[%d]", fittestOnEarth.item_picks[i].id, fittestOnEarth.item_picks[i].pickup);
					}
				}
				printf("\n\n");
			}

			if (timespec_get(&stopCPU, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}

			elapsedTimeCPU[clockCounter] = (double)(stopCPU.tv_sec - startCPU.tv_sec) + ((double)(stopCPU.tv_nsec - startCPU.tv_nsec) * 1.e-6);

			meanSelection = mean(elapsedSelectionCPU, NUM_EVOLUTIONS);
			meanCrossover = mean(elapsedCrossoverCPU, NUM_EVOLUTIONS);
			meanLocalSearch = mean(elapsedLocalSearchCPU, NUM_EVOLUTIONS);
			meanSolution = mean(solutionQualityCPU, NUM_EVOLUTIONS + 1);

			medianSelection = median(elapsedSelectionCPU, NUM_EVOLUTIONS);
			medianCrossover = median(elapsedCrossoverCPU, NUM_EVOLUTIONS);
			medianLocalSearch = median(elapsedLocalSearchCPU, NUM_EVOLUTIONS);
			medianSolution = median(solutionQualityCPU, NUM_EVOLUTIONS + 1);

			modeSelection = mode(elapsedSelectionCPU, NUM_EVOLUTIONS);
			modeCrossover = mode(elapsedCrossoverCPU, NUM_EVOLUTIONS);
			modeLocalSearch = mode(elapsedLocalSearchCPU, NUM_EVOLUTIONS);
			modeSolution = mode(solutionQualityCPU, NUM_EVOLUTIONS + 1);

			sdSelection = standardDeviation(elapsedSelectionCPU, NUM_EVOLUTIONS);
			sdCrossover = standardDeviation(elapsedCrossoverCPU, NUM_EVOLUTIONS);
			sdLocalSearch = standardDeviation(elapsedLocalSearchCPU, NUM_EVOLUTIONS);
			sdSolution = standardDeviation(solutionQualityCPU, NUM_EVOLUTIONS + 1);

			saveStatistics(problem.name, NO_CUDA, clockCounter, elapsedTimeInitialPopulationCPU[clockCounter], meanSelection, meanCrossover, meanLocalSearch, medianSelection, medianCrossover, medianLocalSearch, modeSelection, modeCrossover, modeLocalSearch, sdSelection, sdCrossover, sdLocalSearch, meanSolution, medianSolution, modeSolution, sdSolution, elapsedTimeCPU[clockCounter]);

			elapsedSelectionTotalCPU[clockCounter] = sumArray(elapsedSelectionCPU, NUM_EVOLUTIONS);
			elapsedCrossoverTotalCPU[clockCounter] = sumArray(elapsedCrossoverCPU, NUM_EVOLUTIONS);
			elapsedLocalSearchTotalCPU[clockCounter] = sumArray(elapsedLocalSearchCPU, NUM_EVOLUTIONS);
			solutionQualityTotalCPU[clockCounter] = sumArray(solutionQualityCPU, NUM_EVOLUTIONS + 1);
		}
#pragma endregion
#pragma region GRAPHICAL PROCESS UNIT
		if (GPU)
		{
			/****************************************************************************************************
			* PRINT PROPERTIES OF THE CUDA DEVICE (IF ANY)
			****************************************************************************************************/
			deviceErr = cudaGetDeviceCount(&deviceCount);
			if (deviceCount > 0 && deviceErr == cudaSuccess && GPU)
			{
				printf("****************************************************************************************\n");
				printf("PROPERTIES OF THE GRAPHICAL PROCESSING UNIT\n");
				printf("****************************************************************************************\n");
				for (int i = 0; i < deviceCount; i++)
				{
					checkCudaErrors(cudaGetDeviceProperties(&properties, i));
					printf("GPU:					%s\n", properties.name);
					printf("Warp Size:				%d\n", properties.warpSize);
					printf("Total Global Memory:			%zd\n", properties.totalGlobalMem);
					printf("Total Constant Memory:			%zd\n", properties.totalConstMem);
					printf("Shared Memory Per Block:		%zd\n", properties.sharedMemPerBlock);
					printf("Multiprocessor:				%d\n", properties.multiProcessorCount);
					printf("Max Threads Per Multiprocessor:		%d\n", properties.maxThreadsPerMultiProcessor);
					printf("Max Blocks Per Multiprocessor:		%d\n", properties.maxBlocksPerMultiProcessor);
					printf("Max Threads Per Block:			%d\n", properties.maxThreadsPerBlock);
				}
				printf("****************************************************************************************\n");
			}

			// Create Results File 
			createOutputFile(problem.name, GPU, false, clockCounter);

			if (timespec_get(&startCPU, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}

			// Assign nodes to tour		
			defineInitialTour(initial_tour, &problem, cpu_node, cpu_item);

			if (timespec_get(&stopCPU, TIME_UTC) != TIME_UTC)
			{
				printf("Error in calling timespec_get\n");
				exit(EXIT_FAILURE);
			}

			elapsedTimeGPU[clockCounter] = (float)(stopCPU.tv_sec - startCPU.tv_sec) + ((float)(stopCPU.tv_nsec - startCPU.tv_nsec) * 1.e-6);

			if (deviceCount > 0 && deviceErr == cudaSuccess && GPU)
			{
				checkCudaErrors(cudaEventCreate(&startGPU));
				checkCudaErrors(cudaEventCreate(&stopGPU));
				checkCudaErrors(cudaEventRecord(startGPU, 0));
				/*************************************************************************************************
				* ALLOCATE MEMORY FOR STRUCTS ON DEVICE
				*************************************************************************************************/

				// Allocate device memory for population
				checkCudaErrors(cudaMalloc((void**)&device_population, sizeof(population) * size_t(population_size)));

				// Allocate device memory for initial tour
				checkCudaErrors(cudaMalloc((void**)&device_initial_tour, sizeof(tour)));

				// Allocate device memory for parents selected from tournament selection
				checkCudaErrors(cudaMalloc((void**)&device_parents, sizeof(tour) * size_t(tour_size) * SELECTED_PARENTS));

				// Allocate device memory for states
				checkCudaErrors(cudaMalloc((void**)&device_states, sizeof(curandState) * TOURS * size_t(problem.cities_amount)));

				// Allocate device memory for the descendants
				checkCudaErrors(cudaMalloc((void**)&device_offspring, sizeof(tour) * size_t(tour_size) * TOURS));

				// Allocate device memory for the parameters
				checkCudaErrors(cudaMalloc((void**)&device_parameters, sizeof(parameters)));

				// Allocate device memory for node matrix, node matrix transpose and distance matrix
				//checkCudaErrors(cudaMalloc(&device_node_matrix, size_t(problem.cities_amount) * sizeof(node)));
				//checkCudaErrors(cudaMalloc(&device_node_t_matrix, size_t(problem.cities_amount) * sizeof(node)));
				//checkCudaErrors(cudaMalloc(&device_distance, sizeof(distance) * CITIES * CITIES));

				/*************************************************************************************************
				* COPY HOST MEMORY TO DEVICE
				*************************************************************************************************/

				// Copy problem data
				checkCudaErrors(cudaMemcpy(device_parameters, &problem, sizeof(parameters), cudaMemcpyHostToDevice));

				// Copy initial tour data
				checkCudaErrors(cudaMemcpy(device_initial_tour, &initial_tour, sizeof(tour), cudaMemcpyHostToDevice));

				// Copy node data
				//checkCudaErrors(cudaMemcpy(device_node_matrix, cpu_node, size_t(problem.cities_amount) * sizeof(node), cudaMemcpyHostToDevice));

				/*************************************************************************************************
				* INITIALIZE RANDOM VALUES
				*************************************************************************************************/				
				initCuRand << <BLOCKS, THREADS >> > (device_states, time(NULL));				
				checkCudaErrors(cudaDeviceSynchronize());

				/*************************************************************************************************
				* POPULATION INITIALIZATION ON DEVICE (GPU)
				*************************************************************************************************/
				checkCudaErrors(cudaEventCreate(&startKernel));
				checkCudaErrors(cudaEventCreate(&stopKernel));
				checkCudaErrors(cudaEventRecord(startKernel, 0));
				initializePopulationCuda << <BLOCKS, THREADS >> > (device_population, device_initial_tour, device_parameters, device_states);
				checkCudaErrors(cudaEventRecord(stopKernel, 0));
				checkCudaErrors(cudaEventSynchronize(stopKernel));
				checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInitialPopulationGPU[clockCounter], startKernel, stopKernel));
				//checkCudaErrors(cudaDeviceSynchronize());

				/*************************************************************************************************
				* COPY RESULTS TO HOST
				*************************************************************************************************/
				checkCudaErrors(cudaMemcpy(&initial_population_gpu, device_population, sizeof(population), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaDeviceSynchronize());

				/*************************************************************************************************
				* OUTPUT
				*************************************************************************************************/
				//printPopulation(initial_population_gpu);
				saveInitialPopulation(problem.name, initial_population_gpu, problem, CUDA, clockCounter, elapsedTimeInitialPopulationGPU[clockCounter]);

				// Copy Device Information to Host
				checkCudaErrors(cudaMemcpy(&initial_population_gpu, device_population, sizeof(population), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaDeviceSynchronize());

				fittestOnEarth = getFittestTour(initial_population_gpu.tours, TOURS);
				saveFittest(problem.name, fittestOnEarth, problem, 0, CUDA, clockCounter);
				printf("FITTEST TOUR OF CUDA INITIAL POPULATION: \n");
				printf("TIME: %f - FITNESS: %f - PROFIT: %f\n", fittestOnEarth.time, fittestOnEarth.fitness, fittestOnEarth.profit);
				printf("ROUTE: %d", fittestOnEarth.nodes[0].id);
				for (int i = 1; i < CITIES + 1; ++i)
				{
					printf(" > %d", fittestOnEarth.nodes[i].id);
				}
				printf("\n");
				printf("ITEMS: %d[%d]", fittestOnEarth.item_picks[0].id, fittestOnEarth.item_picks[0].pickup);
				for (int i = 1; i < ITEMS; ++i)
				{
					if (fittestOnEarth.item_picks[i].id > 0)
					{
						printf(" > %d[%d]", fittestOnEarth.item_picks[i].id, fittestOnEarth.item_picks[i].pickup);
					}
				}
				printf("\n\n");

				for (int i = 0; i < NUM_EVOLUTIONS; ++i)
				{
					printf("ITERATION %d\n", i);

					// Select Parents For The Next Generation
					checkCudaErrors(cudaEventRecord(startKernel, 0));
					selectionKernel << <BLOCKS, THREADS >> > (device_population, device_parents, device_states);
					err = cudaGetLastError();
					if (err != cudaSuccess)
					{
						fprintf(stderr, "Selection Kernel: %s\n", cudaGetErrorString(err));
						exit(0);
					}
					checkCudaErrors(cudaEventRecord(stopKernel, 0));
					checkCudaErrors(cudaEventSynchronize(stopKernel));
					checkCudaErrors(cudaEventElapsedTime(&elapsedSelectionGPU[i], startKernel, stopKernel));
					//checkCudaErrors(cudaDeviceSynchronize());

					// Copy Device Information to Host
					checkCudaErrors(cudaMemcpy(&host_parents, device_parents, sizeof(tour) * SELECTED_PARENTS, cudaMemcpyDeviceToHost));
					checkCudaErrors(cudaDeviceSynchronize());

					// Save Parents Information To File
					saveParents(problem.name, host_parents, problem, i + 1, CUDA, clockCounter, elapsedSelectionGPU[i]);

					// Breed the population performing crossover (Combination of Ordered Crossover 
					// for the TSP sub-problem and One Point Crossover for the KP sub-problem)
					checkCudaErrors(cudaEventRecord(startKernel, 0));
					crossoverKernel << <BLOCKS, THREADS >> > (device_population, device_parents, device_offspring, device_parameters, device_states);
					err = cudaGetLastError();
					if (err != cudaSuccess)
					{
						fprintf(stderr, "Crossover Kernel: %s\n", cudaGetErrorString(err));
						exit(0);
					}
					checkCudaErrors(cudaEventRecord(stopKernel, 0));
					checkCudaErrors(cudaEventSynchronize(stopKernel));
					checkCudaErrors(cudaEventElapsedTime(&elapsedCrossoverGPU[i], startKernel, stopKernel));
					//checkCudaErrors(cudaDeviceSynchronize());

					// Perform local search (mutation)
					checkCudaErrors(cudaEventRecord(startKernel, 0));
					localSearchKernel << <BLOCKS, THREADS >> > (device_population, device_parameters, device_states);
						err = cudaGetLastError();
					if (err != cudaSuccess)
					{
						fprintf(stderr, "Local Search Kernel: %s\n", cudaGetErrorString(err));
						exit(0);
					}
					checkCudaErrors(cudaEventRecord(stopKernel, 0));
					checkCudaErrors(cudaEventSynchronize(stopKernel));
					checkCudaErrors(cudaEventElapsedTime(&elapsedLocalSearchGPU[i], startKernel, stopKernel));
					//checkCudaErrors(cudaDeviceSynchronize());

					// Copy Device Information to Host
					checkCudaErrors(cudaMemcpy(&initial_population_gpu, device_population, sizeof(population), cudaMemcpyDeviceToHost));
					checkCudaErrors(cudaDeviceSynchronize());

					saveOffspring(problem.name, initial_population_gpu, problem, i + 1, CUDA, clockCounter, elapsedCrossoverGPU[i], elapsedLocalSearchGPU[i]);

					// Get Fittest tour of the generation
					fittestOnEarth = getFittestTour(initial_population_gpu.tours, TOURS);
					saveFittest(problem.name, fittestOnEarth, problem, i + 1, CUDA, clockCounter);
					printf("FITTEST CUDA TOUR OF GENERATION %d: \n", i + 1);
					printf("TIME: %f - FITNESS: %f - PROFIT: %f\n", fittestOnEarth.time, fittestOnEarth.fitness, fittestOnEarth.profit);
					printf("ROUTE: %d", fittestOnEarth.nodes[0].id);
					for (int i = 1; i < CITIES + 1; ++i)
					{
						printf(" > %d", fittestOnEarth.nodes[i].id);
					}
					printf("\n");
					printf("ITEMS: %d[%d]", fittestOnEarth.item_picks[0].id, fittestOnEarth.item_picks[0].pickup);
					for (int i = 1; i < ITEMS; ++i)
					{
						if (fittestOnEarth.item_picks[i].id > 0)
						{
							printf(" > %d[%d]", fittestOnEarth.item_picks[i].id, fittestOnEarth.item_picks[i].pickup);
						}
					}
					printf("\n\n");
				}

				checkCudaErrors(cudaEventRecord(stopGPU, 0));
				checkCudaErrors(cudaEventSynchronize(stopGPU));
				checkCudaErrors(cudaEventElapsedTime(&gpuExecutionTime, startGPU, stopGPU));
				elapsedTimeGPU[clockCounter] += gpuExecutionTime;

				/*************************************************************************************************
				* RELEASE CUDA MEMORY
				*************************************************************************************************/
				checkCudaErrors(cudaFree(device_population));
				checkCudaErrors(cudaFree(device_initial_tour));
				checkCudaErrors(cudaFree(device_parents));
				checkCudaErrors(cudaFree(device_offspring));
				checkCudaErrors(cudaFree(device_parameters));
				checkCudaErrors(cudaFree(device_states));
				//cudaFree(device_node_matrix);
				//cudaFree(device_node_t_matrix);
				//cudaFree(device_distance);
				checkCudaErrors(cudaEventDestroy(startGPU));
				checkCudaErrors(cudaEventDestroy(startKernel));
				checkCudaErrors(cudaEventDestroy(stopKernel));
				checkCudaErrors(cudaEventDestroy(stopGPU));
			}

			meanSelection = mean(elapsedSelectionGPU, NUM_EVOLUTIONS);
			meanCrossover = mean(elapsedCrossoverGPU, NUM_EVOLUTIONS);
			meanLocalSearch = mean(elapsedLocalSearchGPU, NUM_EVOLUTIONS);

			medianSelection = median(elapsedSelectionGPU, NUM_EVOLUTIONS);
			medianCrossover = median(elapsedCrossoverGPU, NUM_EVOLUTIONS);
			medianLocalSearch = median(elapsedLocalSearchGPU, NUM_EVOLUTIONS);

			modeSelection = mode(elapsedSelectionGPU, NUM_EVOLUTIONS);
			modeCrossover = mode(elapsedCrossoverGPU, NUM_EVOLUTIONS);
			modeLocalSearch = mode(elapsedLocalSearchGPU, NUM_EVOLUTIONS);

			sdSelection = standardDeviation(elapsedSelectionGPU, NUM_EVOLUTIONS);
			sdCrossover = standardDeviation(elapsedCrossoverGPU, NUM_EVOLUTIONS);
			sdLocalSearch = standardDeviation(elapsedLocalSearchGPU, NUM_EVOLUTIONS);

			saveStatistics(problem.name, CUDA, clockCounter, elapsedTimeInitialPopulationGPU[clockCounter], meanSelection, meanCrossover, meanLocalSearch, medianSelection, medianCrossover, medianLocalSearch, modeSelection, modeCrossover, modeLocalSearch, sdSelection, sdCrossover, sdLocalSearch, 0, 0, 0, 0, elapsedTimeGPU[clockCounter]);

			elapsedSelectionTotalGPU[clockCounter] = sumArray(elapsedSelectionGPU, NUM_EVOLUTIONS);
			elapsedCrossoverTotalGPU[clockCounter] = sumArray(elapsedCrossoverGPU, NUM_EVOLUTIONS);
			elapsedLocalSearchTotalGPU[clockCounter] = sumArray(elapsedLocalSearchGPU, NUM_EVOLUTIONS);
		}
#pragma endregion

		// Calculate distance matrix in CPU
		//int distance_matrix_size = problem.cities_amount * problem.cities_amount;

		// Allocate memory for the distance matrix
		//distance* d = (distance*)malloc(distance_matrix_size * sizeof(distance));
		//if (d == NULL) {
		//	fprintf(stderr, "Out of Memory");
		//	exit(0);
		//}

		//euclideanDistanceCPU(cpu_node, cpu_node, d, problem.cities_amount, distance_matrix_size);
		//displayDistance(d, distance_matrix_size);

#pragma region DISTANCE MATRIX GRAPHICAL PROCESSING UNIT

		//if (deviceCount > 0 && deviceErr == cudaSuccess && GPU)
		//{
			/*************************************************************************************************
			* CALCULATE DISTANCE MATRIX IN CUDA
			*************************************************************************************************/

			// Execute CUDA Matrix Transposition
			//printf("Transponiendo la matrix de nodos de tamaño [%d][%d]\n", node_rows, 1);
			//transpose << <BLOCKS, THREADS >> > (device_node_matrix, device_node_t_matrix, node_rows, 1);
			//checkCudaErrors(cudaDeviceSynchronize());

			// Copy results from device to host
			//node* h_node_t_matrix = (node*)malloc(sizeof(node) * problem.cities_amount);
			//checkCudaErrors(cudaMemcpy(h_node_t_matrix, device_node_t_matrix, sizeof(node) * problem.cities_amount, cudaMemcpyDeviceToHost));

			// Show information on screen
			//displayNodes(h_node_t_matrix, problem.cities_amount);

			// TODO: FIX GRID AND THREADS AND MATRIXDISTANCES KERNEL
			//dim3 grid(8, 8, 1);
			//dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

			//printf("Calculando la matriz de distancias en GPU\n");
			//matrixDistances << <grid, threads >> > (device_node_matrix, device_node_t_matrix, device_distance, problem.cities_amount, problem.cities_amount);
			//checkCudaErrors(cudaDeviceSynchronize());

			//Copy results from device to host
			//distance* h_distance = (distance*)malloc(sizeof(distance) * CITIES * CITIES);
			//checkCudaErrors(cudaMemcpy(h_distance, device_distance, sizeof(distance) * CITIES * CITIES, cudaMemcpyDeviceToHost));

			// Show Data
			//displayDistance(h_distance, CITIES * CITIES);
		//}
#pragma endregion		
	}

	if (CPU)
	{
		meanGlobalInitializePopulation = mean(elapsedTimeInitialPopulationCPU, NUMBER_EXECUTIONS);
		meanGlobalSelection = mean(elapsedSelectionTotalCPU, NUMBER_EXECUTIONS);
		meanGlobalCrossover = mean(elapsedCrossoverTotalCPU, NUMBER_EXECUTIONS);
		meanGlobalLocalSearch = mean(elapsedLocalSearchTotalCPU, NUMBER_EXECUTIONS);
		meanGlobalExecutionTime = mean(elapsedTimeCPU, NUMBER_EXECUTIONS);
		meanGlobalSolution = mean(solutionQualityTotalCPU, NUMBER_EXECUTIONS);

		medianGlobalInitializePopulation = median(elapsedTimeInitialPopulationCPU, NUMBER_EXECUTIONS);
		medianGlobalSelection = median(elapsedSelectionTotalCPU, NUMBER_EXECUTIONS);
		medianGlobalCrossover = median(elapsedCrossoverTotalCPU, NUMBER_EXECUTIONS);
		medianGlobalLocalSearch = median(elapsedLocalSearchTotalCPU, NUMBER_EXECUTIONS);
		medianGlobalExecutionTime = median(elapsedTimeCPU, NUMBER_EXECUTIONS);
		medianGlobalSolution = median(solutionQualityTotalCPU, NUMBER_EXECUTIONS);

		modeGlobalInitializePopulation = mode(elapsedTimeInitialPopulationCPU, NUMBER_EXECUTIONS);
		modeGlobalSelection = mode(elapsedSelectionTotalCPU, NUMBER_EXECUTIONS);
		modeGlobalCrossover = mode(elapsedCrossoverTotalCPU, NUMBER_EXECUTIONS);
		modeGlobalLocalSearch = mode(elapsedLocalSearchTotalCPU, NUMBER_EXECUTIONS);
		modeGlobalExecutionTime = mode(elapsedTimeCPU, NUMBER_EXECUTIONS);
		modeGlobalSolution = mode(solutionQualityTotalCPU, NUMBER_EXECUTIONS);

		sdGlobalInitializePopulation = standardDeviation(elapsedTimeInitialPopulationCPU, NUMBER_EXECUTIONS);
		sdGlobalSelection = standardDeviation(elapsedSelectionTotalCPU, NUMBER_EXECUTIONS);
		sdGlobalCrossover = standardDeviation(elapsedCrossoverTotalCPU, NUMBER_EXECUTIONS);
		sdGlobalLocalSearch = standardDeviation(elapsedLocalSearchTotalCPU, NUMBER_EXECUTIONS);
		sdGlobalExecutionTime = standardDeviation(elapsedTimeCPU, NUMBER_EXECUTIONS);
		sdGlobalSolution = standardDeviation(solutionQualityTotalCPU, NUMBER_EXECUTIONS);

		saveGlobalStatistics(problem.name, NO_CUDA, meanGlobalInitializePopulation, meanGlobalSelection, meanGlobalCrossover, meanGlobalLocalSearch, meanGlobalExecutionTime, medianGlobalInitializePopulation, medianGlobalSelection, medianGlobalCrossover, medianGlobalLocalSearch, medianGlobalExecutionTime, modeGlobalInitializePopulation, modeGlobalSelection, modeGlobalCrossover, modeGlobalLocalSearch, modeGlobalExecutionTime, sdGlobalInitializePopulation, sdGlobalSelection, sdGlobalCrossover, sdGlobalLocalSearch, meanGlobalSolution, medianGlobalSolution, modeGlobalSolution, sdGlobalSolution, sdGlobalExecutionTime);
	}

	if (GPU)
	{
		meanGlobalInitializePopulation = mean(elapsedTimeInitialPopulationGPU, NUMBER_EXECUTIONS);
		meanGlobalSelection = mean(elapsedSelectionTotalGPU, NUMBER_EXECUTIONS);
		meanGlobalCrossover = mean(elapsedCrossoverTotalGPU, NUMBER_EXECUTIONS);
		meanGlobalLocalSearch = mean(elapsedLocalSearchTotalGPU, NUMBER_EXECUTIONS);
		meanGlobalExecutionTime = mean(elapsedTimeGPU, NUMBER_EXECUTIONS);

		medianGlobalInitializePopulation = median(elapsedTimeInitialPopulationGPU, NUMBER_EXECUTIONS);
		medianGlobalSelection = median(elapsedSelectionTotalGPU, NUMBER_EXECUTIONS);
		medianGlobalCrossover = median(elapsedCrossoverTotalGPU, NUMBER_EXECUTIONS);
		medianGlobalLocalSearch = median(elapsedLocalSearchTotalGPU, NUMBER_EXECUTIONS);
		medianGlobalExecutionTime = median(elapsedTimeGPU, NUMBER_EXECUTIONS);

		modeGlobalInitializePopulation = mode(elapsedTimeInitialPopulationGPU, NUMBER_EXECUTIONS);
		modeGlobalSelection = mode(elapsedSelectionTotalGPU, NUMBER_EXECUTIONS);
		modeGlobalCrossover = mode(elapsedCrossoverTotalGPU, NUMBER_EXECUTIONS);
		modeGlobalLocalSearch = mode(elapsedLocalSearchTotalGPU, NUMBER_EXECUTIONS);
		modeGlobalExecutionTime = mode(elapsedTimeGPU, NUMBER_EXECUTIONS);

		sdGlobalInitializePopulation = standardDeviation(elapsedTimeInitialPopulationGPU, NUMBER_EXECUTIONS);
		sdGlobalSelection = standardDeviation(elapsedSelectionTotalGPU, NUMBER_EXECUTIONS);
		sdGlobalCrossover = standardDeviation(elapsedCrossoverTotalGPU, NUMBER_EXECUTIONS);
		sdGlobalLocalSearch = standardDeviation(elapsedLocalSearchTotalGPU, NUMBER_EXECUTIONS);
		sdGlobalExecutionTime = standardDeviation(elapsedTimeGPU, NUMBER_EXECUTIONS);

		saveGlobalStatistics(problem.name, CUDA, meanGlobalInitializePopulation, meanGlobalSelection, meanGlobalCrossover, meanGlobalLocalSearch, meanGlobalExecutionTime, medianGlobalInitializePopulation, medianGlobalSelection, medianGlobalCrossover, medianGlobalLocalSearch, medianGlobalExecutionTime, modeGlobalInitializePopulation, modeGlobalSelection, modeGlobalCrossover, modeGlobalLocalSearch, modeGlobalExecutionTime, sdGlobalInitializePopulation, sdGlobalSelection, sdGlobalCrossover, sdGlobalLocalSearch, 0,0,0,0, sdGlobalExecutionTime);
	}
	free(cpu_item);
	free(cpu_node);
	return 0;
}