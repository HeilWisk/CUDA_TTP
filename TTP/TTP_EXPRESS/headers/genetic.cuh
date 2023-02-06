/***********************************************************************************************************************
* This file contains all helper functions needed for manipulation of population on the GPU
***********************************************************************************************************************/
#include <curand_kernel.h>
#include <crt/device_functions.h>
#pragma region DEVICE HOST UTILITIES

/// <summary>
/// Given an array of tours, choose fittest and return it
/// </summary>
/// <param name="tours">- Array of tours</param>
/// <param name="population_size">- Size of population</param>
/// <returns></returns>
__host__ __device__ tour getFittestTour(tour* tours, const int& population_size)
{
	// Set the default fittest tour
	tour fittest = tours[0];

	// Evaluates fitness of each tour in the array against the current fittest
	for (int i = 0; i < population_size; ++i)
	{
		if (tours[i].fitness >= fittest.fitness)
		{
			fittest = tours[i];
		}
	}

	// Return the fittest tour on the array
	return fittest;
}

#pragma endregion

#pragma region DEVICE ONLY UTILITIES

__device__ void tournamentSelectionDevice(population* population, tour* parents, curandState* d_state, int thread_id)
{		
	int random_number = curand(d_state) % (TOURS - 1);

	double tournamentFitness = population->tours[random_number].fitness;
	int fittestPosition = random_number;

	for (int t = 0; t < TOURNAMENT_SIZE; ++t)
	{
		// Gets random number from global random state on GPU
		random_number = curand(d_state) % (TOURS - 1);
		if (population->tours[random_number].fitness >= tournamentFitness)
		{
			tournamentFitness = population->tours[random_number].fitness;
			fittestPosition = random_number;
		}
	}

	// Evaluate the fittest tour on the tournament
	parents[thread_id] = population->tours[fittestPosition];
}

__device__ void orderedCrossoverDevice(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour, curandState* state, int thread)
{
	// Get the total size of the tours
	int size = CITIES + 1;

	// Choose two random numbers for the start and end indices of the slice
	// In TTP the origin and destiny are the same, then the random must be in between
	int randPosOne = (curand(state) % (size - 2)) + 1;
	int randPosTwo = (curand(state) % (size - 2)) + 1;

	// Make the smaller the start and the larger the end
	if (randPosOne > randPosTwo)
	{
		int tempNumber = randPosTwo;
		randPosTwo = randPosOne;
		randPosOne = tempNumber;
	}

	// Instanciate child tour
	int indexChild = randPosTwo % size;
	
	// Copy first and last position to child
	childTour.nodes[0] = parents[parentIndexOne].nodes[0];
	childTour.nodes[CITIES] = parents[parentIndexOne].nodes[CITIES];

	// Add the sublist in between the start and the end points to the children
	for (int i = randPosOne; i < randPosTwo; ++i)
	{
		childTour.nodes[i] = parents[parentIndexOne].nodes[i];
		for (int j = 0; j < ITEMS_PER_CITY; ++j)
		{
			if (childTour.nodes[i].items[j].id > 0)
			{
				childTour.item_picks[(childTour.nodes[i].items[j].id - 1)].pickup = childTour.nodes[i].items[j].pickup;
			}
		}
	}

	// Iterate over each city in the parents tour
	int currentCityIndex = 0;
	int currentCityInParentTwo = 0;

	for (int j = 0; j < size; ++j)
	{
		indexChild = (randPosTwo + j) % size;

		if (childTour.nodes[indexChild].id < 0)
		{
			for (int k = 0; k < size; ++k)
			{
				// Get the index of the current city
				currentCityIndex = (randPosTwo + k) % size;

				// Get the city at the current index in each of the two parent tours
				currentCityInParentTwo = parents[parentIndexTwo].nodes[currentCityIndex].id;

				// If child does not already contain the current city in parent two, add it
				bool isPresentInChild = false;
				for (int a = 0; a < size; ++a)
				{
					if (childTour.nodes[a].id == currentCityInParentTwo)
					{
						isPresentInChild = true;
						break;
					}
				}

				if (!isPresentInChild)
				{
					childTour.nodes[indexChild] = parents[parentIndexTwo].nodes[currentCityIndex];
					for (int j = 0; j < ITEMS_PER_CITY; ++j)
					{
						if (childTour.nodes[indexChild].items[j].id > 0)
						{
							childTour.item_picks[(childTour.nodes[indexChild].items[j].id - 1)].pickup = childTour.nodes[indexChild].items[j].pickup;
						}
					}
					break;
				}
			}
		}	
	}
}

__device__ void onePointCrossoverDevice(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour, curandState* state, int thread)
{
	// Choose a random position for cutting the picking plans of the parents
	int cuttingPosition = (curand(state) % (ITEMS));

	for (int i = 0; i < cuttingPosition; ++i)
	{
		childTour.item_picks[i] = parents[parentIndexOne].item_picks[i];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[i].node)
			{
				for (int ia = 0; ia < ITEMS_PER_CITY; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[i].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[i];
						break;
					}
				}
			}
		}
	}

	for (int j = cuttingPosition; j < ITEMS; ++j)
	{
		childTour.item_picks[j] = parents[parentIndexTwo].item_picks[j];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[j].node)
			{
				for (int ia = 0; ia < ITEMS_PER_CITY; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[j].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[j];
						break;
					}
				}
			}
		}
	}
}

__device__ void flipDevice(tour& pickingPlan, curandState* state)
{
	// Choose a random position for the flip
	int flipPosition = (curand(state) % (ITEMS));

	if (pickingPlan.item_picks[flipPosition].pickup == 1)
	{
		pickingPlan.item_picks[flipPosition].pickup = 0;
		for (int i = 1; i < CITIES; ++i)
		{
			if (pickingPlan.nodes[i].id == pickingPlan.item_picks[flipPosition].node)
			{
				for (int j = 0; j < ITEMS_PER_CITY; ++j)
				{
					if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[flipPosition].id)
					{
						pickingPlan.nodes[i].items[j].pickup = 0;
						break;
					}
				}
			}
		}
	}
	else
	{
		pickingPlan.item_picks[flipPosition].pickup = 1;
		for (int i = 1; i < CITIES; ++i)
		{
			if (pickingPlan.nodes[i].id == pickingPlan.item_picks[flipPosition].node)
			{
				for (int j = 0; j < ITEMS_PER_CITY; ++j)
				{
					if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[flipPosition].id)
					{
						pickingPlan.nodes[i].items[j].pickup = 1;
						break;
					}
				}
			}
		}
	}
}

__device__ void twoOptSwapDevice(tour* baseTour, tour& twoOptTour, curandState* state, int thread)
{
	int posOne = (curand(state) % (CITIES - 1)) + 1;
	int posTwo = (curand(state) % (CITIES - 1)) + 1;

	// Make sure the two random numbers are different
	do
	{
		posTwo = (curand(state) % (CITIES - 1)) + 1;
	} 
	while (posOne == posTwo);	

	// 1. Copy the segment of tour from tour[0] to tour[posOne- 1]
	for (int i = 0; i <= posOne - 1; ++i)
	{
		twoOptTour.nodes[i] = baseTour->nodes[i];
	}

	// 2. From tour[posOne] to tour[posTwo] add them to the optTour in reverse order
	int dec = 0;
	for (int c = posOne; c <= posTwo; ++c)
	{
		twoOptTour.nodes[c] = baseTour->nodes[posTwo - dec];
		dec = dec + 1;
	}

	// 3. Add the rest of the tour to optTour
	for (int z = posTwo + 1; z < CITIES + 1; ++z)
	{
		twoOptTour.nodes[z] = baseTour->nodes[z];
	}
}

__device__ void exchangeDevice(tour& pickingPlan, curandState* state)
{
	// Choose a random position for the flip
	int exPosOne = (curand(state) % (ITEMS));
	int exPosTwo = (curand(state) % (ITEMS));
	item tempItem;

	tempItem = pickingPlan.item_picks[exPosOne];
	pickingPlan.item_picks[exPosOne].pickup = pickingPlan.item_picks[exPosTwo].pickup;
	pickingPlan.item_picks[exPosTwo].pickup = tempItem.pickup;

	for (int i = 1; i < CITIES; ++i)
	{
		if ((pickingPlan.nodes[i].id == pickingPlan.item_picks[exPosOne].node) || (pickingPlan.nodes[i].id == pickingPlan.item_picks[exPosTwo].node))
		{
			for (int j = 0; j < ITEMS_PER_CITY; ++j)
			{
				if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[exPosOne].id)
				{
					pickingPlan.nodes[i].items[j].pickup = pickingPlan.item_picks[exPosOne].pickup;
					break;
				}

				if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[exPosTwo].id)
				{
					pickingPlan.nodes[i].items[j].pickup = pickingPlan.item_picks[exPosTwo].pickup;
					break;
				}
			}
		}
	}
}

#pragma endregion

#pragma region HOST ONLY FUNCTIONS

__host__ tour tournamentSelection(population& population)
{
	int random_number = (rand() % ((TOURS - 1) + 1));;

	tour fittest_on_tournament = population.tours[random_number];

	for (int t = 0; t < TOURNAMENT_SIZE; ++t)
	{
		// Gets random number using the following formula
		// FORMULA: random = (rand() % (upper - lower + 1)) + lower
		random_number = (rand() % ((TOURS-1) + 1));
		if (population.tours[random_number].fitness > fittest_on_tournament.fitness)
			fittest_on_tournament = population.tours[random_number];
	}

	// Return the best tour on the population
	return fittest_on_tournament;
}

__host__ void selection(population &population, tour* parents)
{
	for (int i = 0; i < SELECTED_PARENTS; ++i)
	{
		parents[i] = tournamentSelection(population);
	}
}

__host__ void orderedCrossover(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour)
{
	// Get the total size of the tours
	int size = CITIES + 1;

	// Choose two random numbers for the start and end indices of the slice
	// In TTP the origin and destiny are the same, then the random must be in between
	int randPosOne = (rand() % (size - 2)) + 1;
	int randPosTwo = (rand() % (size - 2)) + 1;

	// Make the smaller the start and the larger the end
	if (randPosOne > randPosTwo)
	{
		int tempNumber = randPosTwo;
		randPosTwo = randPosOne;
		randPosOne = tempNumber;
	}

	// Instanciate child tour
	int indexChild = randPosTwo % size;

	// Copy first and last position to child
	childTour.nodes[0] = parents[parentIndexOne].nodes[0];
	childTour.nodes[CITIES] = parents[parentIndexOne].nodes[CITIES];

	// Add the sublist in between the start and the end points to the children
	for (int i = randPosOne; i < randPosTwo; ++i)
	{
		childTour.nodes[i] = parents[parentIndexOne].nodes[i];
	}

	// Iterate over each city in the parents tour
	int currentCityIndex = 0;
	int currentCityInParentTwo = 0;

	for (int j = 0; j < size; ++j)
	{
		indexChild = (randPosTwo + j) % size;

		if (childTour.nodes[indexChild].id < 0)
		{
			for (int k = 0; k < size; ++k)
			{
				// Get the index of the current city
				currentCityIndex = (randPosTwo + k) % size;

				// Get the city at the current index in each of the two parent tours
				currentCityInParentTwo = parents[parentIndexTwo].nodes[currentCityIndex].id;

				// If child does not already contain the current city in parent two, add it
				bool isPresentInChild = false;
				for (int a = 0; a < size; ++a)
				{
					if (childTour.nodes[a].id == currentCityInParentTwo)
					{
						isPresentInChild = true;
						break;
					}
				}

				if (!isPresentInChild)
				{
					childTour.nodes[indexChild] = parents[parentIndexTwo].nodes[currentCityIndex];
					break;
				}
			}
		}
	}
}

__host__ void onePointCrossover(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour)
{
	// Choose a random position for cutting the picking plans of the parents
	int cuttingPosition = (rand() % (ITEMS));	
	
	for (int i = 0; i < cuttingPosition; ++i)
	{
		childTour.item_picks[i] = parents[parentIndexOne].item_picks[i];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[i].node)
			{
				for (int ia = 0; ia < ITEMS_PER_CITY; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[i].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[i];
						break;
					}
				}
			}
		}
	}

	for (int j = cuttingPosition; j < ITEMS; ++j)
	{
		childTour.item_picks[j] = parents[parentIndexTwo].item_picks[j];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[j].node)
			{
				for (int ia = 0; ia < ITEMS_PER_CITY; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[j].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[j];
						break;
					}
				}
			}
		}
	}
}

__host__ void flip(tour& pickingPlan)
{
	// Choose a random position for the flip
	int flipPosition = (rand() % (ITEMS));

	if (pickingPlan.item_picks[flipPosition].pickup == 1)
	{
		pickingPlan.item_picks[flipPosition].pickup = 0;
		for (int i = 1; i < CITIES; ++i)
		{
			if (pickingPlan.nodes[i].id == pickingPlan.item_picks[flipPosition].node)
			{
				for (int j = 0; j < ITEMS_PER_CITY; ++j)
				{
					if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[flipPosition].id)
					{
						pickingPlan.nodes[i].items[j].pickup = 0;
						break;
					}
				}
			}
		}
	}
	else
	{
		pickingPlan.item_picks[flipPosition].pickup = 1;
		for (int i = 1; i < CITIES; ++i)
		{
			if (pickingPlan.nodes[i].id == pickingPlan.item_picks[flipPosition].node)
			{
				for (int j = 0; j < ITEMS_PER_CITY; ++j)
				{
					if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[flipPosition].id)
					{
						pickingPlan.nodes[i].items[j].pickup = 1;
						break;
					}
				}
			}
		}
	}
}

__host__ void twoOptSwap(tour& tour)
{
	int posOne = (rand() % (CITIES -1)) + 1;
	int posTwo = (rand() % (CITIES -1)) + 1;

	// Instanciate new tour
	node optTour[CITIES + 1];

	// Make sure the two random numbers are different
	do
	{
		posTwo = (rand() % (CITIES - 1)) + 1;
	} 
	while (posOne == posTwo);

	// 1. Copy the segment of tour from tour[0] to tour[posOne- 1]
	for (int i = 0; i <= posOne - 1; ++i)
	{
		optTour[i] = tour.nodes[i];
	}

	// 2. From tour[posOne] to tour[posTwo] add them to the optTour in reverse order
	int dec = 0;
	for (int c = posOne; c <= posTwo; ++c)
	{
		optTour[c] = tour.nodes[posTwo - dec];
		dec = dec + 1;
	}

	// 3. Add the rest of the tour to optTour
	for (int z = posTwo + 1; z < CITIES + 1; ++z)
	{
		optTour[z] = tour.nodes[z];
	}

	for (int a = 0; a < CITIES + 1; ++a)
	{
		tour.nodes[a] = optTour[a];
	}
}

__host__ void exchange(tour& pickingPlan)
{
	// Choose a random position for the flip
	int exPosOne = (rand() % (ITEMS));
	int exPosTwo = (rand() % (ITEMS));
	item tempItem;

	tempItem = pickingPlan.item_picks[exPosOne];
	pickingPlan.item_picks[exPosOne].pickup = pickingPlan.item_picks[exPosTwo].pickup;
	pickingPlan.item_picks[exPosTwo].pickup = tempItem.pickup;

	for (int i = 1; i < CITIES; ++i)
	{
		if ((pickingPlan.nodes[i].id == pickingPlan.item_picks[exPosOne].node) || (pickingPlan.nodes[i].id == pickingPlan.item_picks[exPosTwo].node))
		{
			for (int j = 0; j < ITEMS_PER_CITY; ++j)
			{
				if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[exPosOne].id)
				{
					pickingPlan.nodes[i].items[j].pickup = pickingPlan.item_picks[exPosOne].pickup;
					break;
				}

				if (pickingPlan.nodes[i].items[j].id == pickingPlan.item_picks[exPosTwo].id)
				{
					pickingPlan.nodes[i].items[j].pickup = pickingPlan.item_picks[exPosTwo].pickup;
					break;
				}
			}
		}
	}
}

__host__ int getOffspringAmount(tour* solutions)
{
	// Set the default amount of offspring
	int offspringAmount = TOURS;

	// Evaluate every solution to determine the amount of offspring to generate
	for (int i = 0; i < TOURS; ++i)
	{
		if (solutions[i].fitness < 0)
		{
			offspringAmount--;
		}
	}

	// Return the amount of offspring to generate
	return TOURS - offspringAmount;
}

__host__ void crossover(population& population, tour* parents, parameters params)
{
	tour* childs = (tour*)malloc(TOURS * sizeof(tour));
	if (childs == NULL)
	{
		fprintf(stderr, "Out of Memory in function crossover");
		exit(0);
	}

	for (int o = 0; o < TOURS; ++o)
	{
		// Select parents from the parents array
		int parentIndexOne = rand() % SELECTED_PARENTS;
		int parentIndexTwo = rand() % SELECTED_PARENTS;
		childs[o] = tour();

		// Generate child for the TSP Sub-Problem using ordered crossover
		orderedCrossover(parents, parentIndexOne, parentIndexTwo, childs[o]);

		// Generate child for the KP Sub-Problem using one point crossover
		onePointCrossover(parents, parentIndexOne, parentIndexTwo, childs[o]);

		// Evaluate the new child
		evaluateTour(childs[o], &params);

		if (population.tours[o].fitness < 0)
		{
			population.tours[o] = childs[o];
		}
	}

	free(childs);
}

__host__ void localSearch(population& currentPopulation, parameters params)
{
	for (int i = 0; i < TOURS; ++i)
	{
		double probability = (double)rand() / (double)RAND_MAX;
		if (probability < LOCAL_SEARCH_PROBABILITY)
		{
			tour testTour2Opt = currentPopulation.tours[i];
			twoOptSwap(testTour2Opt);
			evaluateTour(testTour2Opt, &params);

			tour testTourFlip = currentPopulation.tours[i];
			flip(testTourFlip);
			evaluateTour(testTourFlip, &params);

			tour testTourEx = currentPopulation.tours[i];
			exchange(testTourEx);
			evaluateTour(testTourEx, &params);

			if (testTour2Opt.fitness >= currentPopulation.tours[i].fitness && testTour2Opt.fitness >= testTourFlip.fitness && testTour2Opt.fitness >= testTourEx.fitness)
			{
				currentPopulation.tours[i] = testTour2Opt;
			}

			if (testTourFlip.fitness >= currentPopulation.tours[i].fitness && testTourFlip.fitness >= testTour2Opt.fitness && testTourFlip.fitness >= testTourEx.fitness)
			{
				currentPopulation.tours[i] = testTourFlip;
			}

			if (testTourEx.fitness >= currentPopulation.tours[i].fitness && testTourEx.fitness >= testTour2Opt.fitness && testTourEx.fitness >= testTourFlip.fitness)
			{
				currentPopulation.tours[i] = testTourEx;
			}
		}
	}
}

#pragma endregion

#pragma region KERNELS

__global__ void selectionKernel(population* population, tour* parents, curandState* state)
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

	if (thread_global_index >= SELECTED_PARENTS)
	{
		return;
	}

	curandState local_state = state[thread_global_index];

	tournamentSelectionDevice(population, parents, &local_state, thread_global_index);

	state[thread_global_index] = local_state;
}

__global__ void crossoverKernel(population* population, tour* parents, tour* offspring, parameters* params, curandState* state)
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
	{
		return;
	}

	curandState local_state = state[thread_global_index];

	// Select parents from the parents array
	int parentIndexOne = curand(&local_state) % SELECTED_PARENTS;
	int parentIndexTwo = curand(&local_state) % SELECTED_PARENTS;

	offspring[thread_global_index] = tour();

	// Generate child for the TSP Sub-Problem using ordered crossover
	orderedCrossoverDevice(parents, parentIndexOne, parentIndexTwo, offspring[thread_global_index], &local_state, thread_global_index);

	// Generate child for the KP Sub-Problem using one point crossover
	onePointCrossoverDevice(parents, parentIndexOne, parentIndexTwo, offspring[thread_global_index], &local_state, thread_global_index);

	// Evaluate the new child
	evaluateTour(offspring[thread_global_index], params);
	
	if (population->tours[thread_global_index].fitness < 0)
	{
		population->tours[thread_global_index] = offspring[thread_global_index];
	}

	state[thread_global_index] = local_state;
}

__global__ void localSearchKernel(population* currentPopulation, parameters* params, curandState* state)
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
	{
		return;
	}

	curandState local_state = state[thread_global_index];

	// Generate random probability
	double probability = curand_uniform_double(&local_state);

	if (probability < LOCAL_SEARCH_PROBABILITY)
	{
		tour testTour2Opt = currentPopulation->tours[thread_global_index];
		twoOptSwapDevice(&currentPopulation->tours[thread_global_index], testTour2Opt, &local_state, thread_global_index);
		evaluateTour(testTour2Opt, params);

		tour testTourFlip = currentPopulation->tours[thread_global_index];
		flipDevice(testTourFlip, &local_state);
		evaluateTour(testTourFlip, params);

		tour testTourEx = currentPopulation->tours[thread_global_index];
		exchangeDevice(testTourEx, &local_state);
		evaluateTour(testTourEx, params);

		if (testTour2Opt.fitness >= currentPopulation->tours[thread_global_index].fitness && testTour2Opt.fitness >= testTourFlip.fitness && testTour2Opt.fitness >= testTourEx.fitness)
		{
			currentPopulation->tours[thread_global_index] = testTour2Opt;
		}

		if (testTourFlip.fitness >= currentPopulation->tours[thread_global_index].fitness && testTourFlip.fitness >= testTour2Opt.fitness && testTourFlip.fitness >= testTourEx.fitness)
		{
			currentPopulation->tours[thread_global_index] = testTourFlip;
		}

		if (testTourEx.fitness >= currentPopulation->tours[thread_global_index].fitness && testTourEx.fitness >= testTour2Opt.fitness && testTourEx.fitness >= testTourFlip.fitness)
		{
			currentPopulation->tours[thread_global_index] = testTourEx;
		}

		state[thread_global_index] = local_state;
	}
}
	
#pragma endregion