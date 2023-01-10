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

__host__ void crossover(population &population, tour* parents)
{
	int randPosOne = (rand() % (CITIES - 1)) + 1;
	SHOW("randPosOne = %d\n", randPosOne);
	int randPosTwo = randPosOne + (rand() % ((CITIES - randPosOne)));
	SHOW("randPosTwo = %d\n", randPosTwo);
	//population->tours[tid].nodes[0] = parents[2 * tid].nodes[0];

	//node nodeOne = getValidNextNode(parents[tid * 2], population->tours[tid], population->tours[tid].nodes[index - 1], index);
	//node nodeTwo = getValidNextNode(parents[tid * 2 + 1], population->tours[tid], population->tours[tid].nodes[index - 1], index);

	//// Compare the two nodes from parents to the last node that was chosen in the child
	//if (distanceTable[nodeOne.id * CITIES + population->tours[tid].nodes[index - 1].id].value <= distanceTable[nodeTwo.id * CITIES + population->tours[tid].nodes[index - 1].id].value)
	//	population->tours[tid].nodes[index] = nodeOne;
	//else
	//	population->tours[tid].nodes[index] = nodeTwo;
}

#pragma endregion