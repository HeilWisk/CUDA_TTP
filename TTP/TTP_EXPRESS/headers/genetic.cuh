/***********************************************************************************************************************
* This file contains all helper functions needed for manipulation of population on the GPU
***********************************************************************************************************************/
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

/// <summary>
/// Evaluates and returns the best tour on a tournament
/// </summary>
/// <param name="population">- Population to evaluate</param>
/// <param name="d_state">- State for random generation</param>
/// <param name="thread_id">- Thread Id of the executor thread</param>
/// <param name="node_quantity">- Node quantity</param>
/// <param name="item_quantity">- Item quantity</param>
/// <returns></returns>
__device__ tour tournamentSelection(population& population, curandState* d_state, const int& thread_id)
{
	tour tournament[TOURNAMENT_SIZE];

	int random_number;
	for (int t = 0; t < TOURNAMENT_SIZE; ++t)
	{
		// Gets random number from global random state on GPU
		random_number = curand_uniform(&d_state[thread_id]) * (TOURS - 1);
		tournament[t] = population.tours[random_number];
	}

	// Evaluate the fittest tour on the tournament
	tour fittest_on_tournament = getFittestTour(tournament, TOURNAMENT_SIZE);

	// Return the best tour on the population
	return fittest_on_tournament;
}

/// <summary>
/// Obtain the position of a node in a tour
/// </summary>
/// <param name="node">- Node to look for</param>
/// <param name="tour">- Tour where to look</param>
/// <param name="tour_size">- Tour Size</param>
/// <returns></returns>
__device__ int getIndexOfNode(node& node, tour& tour, const int& tour_size)
{
	for (int i = 0; i < tour_size; ++i)
	{
		if (node == tour.nodes[i])
			return i;
	}
	return -1;
}

/// <summary>
/// 
/// </summary>
/// <param name="n"></param>
/// <param name="tour"></param>
/// <returns></returns>
__device__ node getNode(int& n, tour& tour)
{
	for (int i = 0; i < CITIES; ++i)
	{
		if (tour.nodes[i].id == n)
			return tour.nodes[i];
	}

	printf("%d, %d", blockIdx.x, threadIdx.x);
	printf("Could not find node %d in this tour: ", n);
	printTour(tour, CITIES);
	return node();
}

/// <summary>
/// 
/// </summary>
/// <param name="parent"></param>
/// <param name="child"></param>
/// <param name="current_node"></param>
/// <param name="child_size"></param>
/// <returns></returns>
__device__ node getValidNextNode(tour& parent, tour& child, node& current_node, const int& child_size)
{
	node valid_node;
	int index_of_current_node = getIndexOfNode(current_node, parent, CITIES);

	// Search for first valid node (not already a child) occurring after current_node location in parent tour
	for (int i = index_of_current_node + 1; i < CITIES; ++i)
	{
		// If not in chlid already, select it
		if (getIndexOfNode(parent.nodes[i], child, child_size) == -1)
			return parent.nodes[i];
	}

	// Loop through node ids [1...Amount of Nodes] and find first valid node to choose as a next point in construction of child tour
	for (int i = 1; i < CITIES; ++i)
	{
		bool in_tour_already = false;
		for (int j = 1; j < child_size; ++j)
		{
			if (child.nodes[j].id == i)
			{
				in_tour_already = true;
				break;
			}
		}

		if (!in_tour_already)
			return getNode(i, parent);
	}

	// if there is an error
	printf("No valid city was found\n\n");
	return node();
}

__device__ void selectionKernel(population& population, tour* parents, curandState* state)
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
	
	parents[thread_global_index] = tournamentSelection(population, state, thread_global_index);
}

__device__ void orderedCrossoverKernel(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour, curandState* state)
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
	}

	// Iterate over each city in the parents tour
	int currentCityIndex = 0;
	int currentCityInParentTwo = 0;

	for (int j = 0; j < size; ++j)
	{
		// Get the index of the current city
		currentCityIndex = (randPosTwo + j) % size;

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
			if (indexChild == size - 1)
				indexChild = 0;
			else
				indexChild = indexChild + 1;
		}
	}
}

__device__ void onePointCrossoverKernel(tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour, curandState* state)
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
				for (int ia = 0; ia < ITEMS; ++ia)
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
				for (int ia = 0; ia < ITEMS; ++ia)
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

__device__ void crossoverKernel(population& population, tour* parents, tour* offspring, int offspringAmount, parameters params, curandState* state)
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

	if (thread_global_index >= offspringAmount)
	{
		return;
	}

	curandState local_state = state[thread_global_index];

	// Select parents from the parents array
	int parentIndexOne = curand(&local_state) % SELECTED_PARENTS;
	int parentIndexTwo = curand(&local_state) % SELECTED_PARENTS;

	//int* child = (int*)malloc(CITIES + 1 * sizeof(int));

	// Generate unique offspring not already in solution
	bool alreadyInPopulation = false;

	do
	{
		// Generate child for the TSP Sub-Problem using ordered crossover
		orderedCrossoverKernel(parents, parentIndexOne, parentIndexTwo, offspring[thread_global_index], &local_state);

		// Generate child for the KP Sub-Problem using one point crossover
		onePointCrossoverKernel(parents, parentIndexOne, parentIndexTwo, offspring[thread_global_index], &local_state);

		// Evaluate the new child
		evaluateTour(offspring[thread_global_index], params);

		for (int f = 0; f < TOURS; ++f)
		{
			if (population.tours[f] == offspring[thread_global_index])
			{
				alreadyInPopulation = true;
				break;
			}
		}
	} while (alreadyInPopulation);
}

__device__ void flipKernel(tour& pickingPlan, curandState* state)
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
				for (int j = 0; j < ITEMS; ++j)
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
				for (int j = 0; j < ITEMS; ++j)
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

__device__ void twoOptSwapKernel(tour& tour, curandState* state)
{
	int posOne = (curand(state) % (CITIES - 1)) + 1;
	int posTwo = (curand(state) % (CITIES - 1)) + 1;

	// Instanciate new tour
	node optTour[CITIES + 1];

	// Make sure the two random numbers are different
	do
	{
		posTwo = (curand(state) % (CITIES - 1)) + 1;
	} while (posOne == posTwo);

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

__device__ void exchangeKernel(tour& pickingPlan, curandState* state)
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
			for (int j = 0; j < ITEMS; ++j)
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

__device__ void localSearchKernel(population& currentPopulation, parameters params, curandState* state)
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
	double probability = (double)curand(&local_state) / (double)RAND_MAX;
	
	if (probability < LOCAL_SEARCH_PROBABILITY)
	{
		tour testTour2Opt = currentPopulation.tours[thread_global_index];
		twoOptSwapKernel(testTour2Opt, &local_state);
		evaluateTour(testTour2Opt, params);

		tour testTourFlip = currentPopulation.tours[thread_global_index];
		flipKernel(testTourFlip, &local_state);
		evaluateTour(testTourFlip, params);

		tour testTourEx = currentPopulation.tours[thread_global_index];
		exchangeKernel(testTourEx, &local_state);
		evaluateTour(testTourEx, params);

		if (testTour2Opt.fitness >= currentPopulation.tours[thread_global_index].fitness && testTour2Opt.fitness >= testTourFlip.fitness && testTour2Opt.fitness >= testTourEx.fitness)
		{
			currentPopulation.tours[thread_global_index] = testTour2Opt;
		}

		if (testTourFlip.fitness >= currentPopulation.tours[thread_global_index].fitness && testTourFlip.fitness >= testTour2Opt.fitness && testTourFlip.fitness >= testTourEx.fitness)
		{
			currentPopulation.tours[thread_global_index] = testTourFlip;
		}

		if (testTourEx.fitness >= currentPopulation.tours[thread_global_index].fitness && testTourEx.fitness >= testTour2Opt.fitness && testTourEx.fitness >= testTourFlip.fitness)
		{
			currentPopulation.tours[thread_global_index] = testTourEx;
		}
	}
}

#pragma endregion

#pragma region HOST ONLY FUNCTIONS

/// <summary>
/// Host Function for tournament selection
/// </summary>
/// <param name="population"></param>
/// <returns></returns>
__host__ tour tournamentSelection(population& population)
{
	tour fittest_on_tournament;

	int random_number;
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

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="parents"></param>
/// <returns></returns>
__host__ void selection(population &population, tour* parents)
{
	for (int i = 0; i < SELECTED_PARENTS; ++i)
	{
		parents[i] = tournamentSelection(population);
	}
}

__host__ void orderedCrossover(int* childNode, tour* parents, int parentIndexOne, int parentIndexTwo, tour& childTour)
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
	int child[CITIES + 1];
	int indexChild = randPosTwo % size;

	// Copy first and last position to child
	child[0] = parents[parentIndexOne].nodes[0].id;
	childTour.nodes[0] = parents[parentIndexOne].nodes[0]; //Test

	child[CITIES] = parents[parentIndexOne].nodes[CITIES].id;
	childTour.nodes[CITIES] = parents[parentIndexOne].nodes[CITIES]; //Test

	// Add the sublist in between the start and the end points to the children
	for (int i = randPosOne; i < randPosTwo; ++i)
	{
		child[i] = parents[parentIndexOne].nodes[i].id;
		childTour.nodes[i] = parents[parentIndexOne].nodes[i]; //Test
	}

	// Iterate over each city in the parents tour
	int currentCityIndex = 0;
	int currentCityInParentTwo = 0;

	for (int j = 0; j < size; ++j)
	{
		// Get the index of the current city
		currentCityIndex = (randPosTwo + j) % size;

		// Get the city at the current index in each of the two parent tours
		currentCityInParentTwo = parents[parentIndexTwo].nodes[currentCityIndex].id;

		// If child does not already contain the current city in parent two, add it
		bool isPresentInChild = false;
		for (int a = 0; a < size; ++a)
		{
			if (child[a] == currentCityInParentTwo && childTour.nodes[a].id == currentCityInParentTwo)
			{
				isPresentInChild = true;
				break;
			}
		}

		if (!isPresentInChild)
		{
			child[indexChild] = currentCityInParentTwo;
			childTour.nodes[indexChild] = parents[parentIndexTwo].nodes[currentCityIndex]; //Test
			if (indexChild == size - 1)
				indexChild = 0;
			else
				indexChild = indexChild + 1;
		}
	}

	// Assign the resulting child to the tour
	for (int f = 0; f < size; ++f)
	{
		childNode[f] = child[f];
	}
}

__host__ void onePointCrossover(tour* parents, int parentIndexOne, int parentIndexTwo, item* child, tour& childTour)
{
	// Choose a random position for cutting the picking plans of the parents
	int cuttingPosition = (rand() % (ITEMS));	
	
	for (int i = 0; i < cuttingPosition; ++i)
	{
		child[i] = parents[parentIndexOne].item_picks[i];

		//Test - BEGIN
		childTour.item_picks[i] = parents[parentIndexOne].item_picks[i];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[i].node)
			{
				for (int ia = 0; ia < ITEMS; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[i].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[i];
						break;
					}
				}
			}
		}
		//Test - END
	}

	for (int j = cuttingPosition; j < ITEMS; ++j)
	{
		child[j] = parents[parentIndexTwo].item_picks[j];

		//Test - BEGIN
		childTour.item_picks[j] = parents[parentIndexTwo].item_picks[j];
		for (int p = 1; p < CITIES; ++p)
		{
			if (childTour.nodes[p].id == childTour.item_picks[j].node)
			{
				for (int ia = 0; ia < ITEMS; ++ia)
				{
					if (childTour.nodes[p].items[ia].id == childTour.item_picks[j].id)
					{
						childTour.nodes[p].items[ia] = childTour.item_picks[j];
						break;
					}
				}
			}
		}
		//Test - END
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
				for (int j = 0; j < ITEMS; ++j)
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
				for (int j = 0; j < ITEMS; ++j)
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
			for (int j = 0; j < ITEMS; ++j)
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

__host__ void crossover(population& population, tour* parents, int offspringAmount, parameters params)
{
	tour* childs = (tour*)malloc(offspringAmount * sizeof(tour));
	if (childs == NULL)
	{
		fprintf(stderr, "Out of Memory in function crossover");
		exit(0);
	}

	for (int o = 0; o < offspringAmount; ++o)
	{
		// Select parents from the parents array
		int parentIndexOne = rand() % SELECTED_PARENTS;
		int parentIndexTwo = rand() % SELECTED_PARENTS;

		int* child = (int*)malloc(CITIES + 1 * sizeof(int));

		// Generate unique offspring not already in solution
		bool alreadyInPopulation = false;

		do
		{
			// Generate child for the TSP Sub-Problem using ordered crossover
			orderedCrossover(child, parents, parentIndexOne, parentIndexTwo, childs[o]);

			// Generate child for the KP Sub-Problem using one point crossover
			onePointCrossover(parents, parentIndexOne, parentIndexTwo, childs[o].item_picks, childs[o]);

			// Evaluate the new child
			evaluateTour(childs[o], params);

			for (int f = 0; f < TOURS; ++f)
			{
				if (population.tours[f] == childs[o])
				{
					alreadyInPopulation = true;
					break;
				}
			}
		} while (alreadyInPopulation);
	}

	for (int b = 0; b < offspringAmount; ++b)
	{
		for (int a = 0; a < TOURS; ++a)
		{
			if (population.tours[a].fitness < 0)
			{
				population.tours[a] = childs[b];
				break;
			}
		}
	}
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
			evaluateTour(testTour2Opt, params);

			tour testTourFlip = currentPopulation.tours[i];
			flip(testTourFlip);
			evaluateTour(testTourFlip, params);

			tour testTourEx = currentPopulation.tours[i];
			exchange(testTourEx);
			evaluateTour(testTourEx, params);

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