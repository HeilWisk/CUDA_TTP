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

__device__ node getValidNextNode(tour& parent, tour& child, node& current_node, const int& child_size, const int &node_quantity)
{
	node valid_node;
	//int index_of_current_node = getIndexOfNode(current_node, parent, node_quantity);
	return valid_node;
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
		//if (node == tour.nodes[i])
			//return i;
	}
	return -1;
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
__device__ tour tournamentSelection(population& population, curandState* d_state, const int& thread_id, const int node_quantity, const int item_quantity)
{
	const int size = sizeof(tour);
	char ptr[POPULATION_SIZE * size];
	tour* tournament = (tour*)ptr;
	for (int i = 0; i < POPULATION_SIZE; ++i)
		tournament[i] = tour(node_quantity, item_quantity, true);

	int random_number;
	for (int t = 0; t < TOURNAMENT_SIZE; ++t)
	{
		// Gets random number from global random state on GPU
		random_number = curand_uniform(&d_state[thread_id]) * (POPULATION_SIZE - 1);
		tournament[t] = population.tours[random_number];
	}

	// Evaluate the fittest tour on the tournament
	tour fittest_on_tournament = getFittestTour(tournament, TOURNAMENT_SIZE);

	// Return the best tour on the population
	return fittest_on_tournament;
}

#pragma endregion

