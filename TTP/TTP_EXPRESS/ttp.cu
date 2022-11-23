
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POPULATION CONTROL
#define MAX_COORD 250
#define POPULATION_SIZE 10 //blockPerGrid*blockPerGrid*BLOCK_SIZE*BLOCK_SIZE
#define BLOCK_SIZE 16
#define NUM_EVOLUTIONS 100
#define MUTATION_RATE 0.05
#define ELITISM true
#define TOURNAMENT_SIZE 128
//BLOCKS
//NUM_THREADS

const int blockPerGrid = 8;

#define DIMENSION "DIMENSION:"
#define ITEM_QTY "NUMBER OF ITEMS:"
#define KNAPSACK_CAPACITY "CAPACITY OF KNAPSACK:"
#define MIN_SPEED "MIN SPEED:"
#define MAX_SPEED "MAX SPEED:"
#define RENTING_RATIO "RENTING RATIO:"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE:"
#define NODE_COORD_SECTION "NODE_COORD_SECTION	(INDEX, X, Y):"
#define ITEMS_SECTION "ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):"

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



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


int main()
{
	/****************************************************************************************************
	* DECLARE VARIABLES
	****************************************************************************************************/

	// File variables
	char file_name[255], str[255], sub[255];
	FILE* fp;
	size_t position;
	const char openMode[] = "r";

	// Problem variables
	int** matrix;
	double knapsack_capacity;
	double minimal_speed;
	double maximun_speed;
	double renting_ratio;
	char edge_weight_type[1000];

	unsigned int node_size;
	unsigned int item_size;
	unsigned int population_size = 1;
	unsigned int tour_size = POPULATION_SIZE;

#pragma region PRINT GPU PROPERTIES

	/****************************************************************************************************
	* PRINT START OF THE PROGRAM
	****************************************************************************************************/
	int count;
	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	printf("****************************************************************************************\n");
	printf("PROPERTIES OF THE GRAPHICAL PROCESSING UNIT\n");
	printf("****************************************************************************************\n");
	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&properties, i));
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
	fp = fopen(file_name, openMode);

	// Validates for errors on file opening
	if (fp == NULL)
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
	while (fgets(str, 100, fp) != NULL) {
		position = findCharacterPosition(str, ':');
		// Extract amount of nodes (cities)
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			node_size = atoi(sub);
			printf("Nodes (Cities):				%d\n", node_size);
		}
		// Extract the amount of items
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			item_size = atoi(sub);
			printf("Item:					%d\n", item_size);
		}
		// Extract the knapsack capacity
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			knapsack_capacity = atof(sub);
			printf("Knapsack Capacity:			%lf\n", knapsack_capacity);
		}
		// Extract the minimal speed
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			minimal_speed = atof(sub);
			printf("Minimum Speed:				%lf\n", minimal_speed);
		}
		// Extract the maximum speed
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			maximun_speed = atof(sub);
			printf("Maximum Speed:				%lf\n", maximun_speed);
		}
		// Extract the renting ratio
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			renting_ratio = atof(sub);
			printf("Renting Ratio:				%lf\n", renting_ratio);
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
	fclose(fp);
	printf("****************************************************************************************\n");
	printf("\n");

#pragma endregion    

    return 0;
}