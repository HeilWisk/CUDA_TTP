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

/// <summary>
/// Obtain the position of a node in a tour
/// </summary>
/// <param name="node">- Node to look for</param>
/// <param name="tour">- Tour where to look</param>
/// <param name="tour_size">- Tour Size</param>
/// <returns></returns>
__device__ int getIndexOfNode(node& node, tour& tour, const int &tour_size)
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
	for (int i = 0; i < tour.node_qty; ++i)
	{
		if (tour.nodes[i].id == n)
			return tour.nodes[i];
	}

	printf("%d, %d", blockIdx.x, threadIdx.x);
	printf("Could not find node %d in this tour: ", n);
	printTour(tour);
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
	int index_of_current_node = getIndexOfNode(current_node, parent, parent.node_qty);

	// Search for first valid node (not already a child) occurring after current_node location in parent tour
	for (int i = index_of_current_node + 1; i < parent.node_qty; ++i)
	{
		// If not in chlid already, select it
		if (getIndexOfNode(parent.nodes[i], child, child_size) == -1)
			return parent.nodes[i];
	}

	// Loop through node ids [1...Amount of Nodes] and find first valid node to choose as a next point in construction of child tour
	for (int i = 1; i < parent.node_qty; ++i)
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

/// <summary>
/// Function to test memory allocation in CUDA
/// </summary>
/// <param name="initial_population"></param>
/// <param name="population_size"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <returns></returns>
__global__ void testMemoryAllocation(population* initial_population, const int population_size)
{
	// Get thread ID
	// Global index of every block on the grid
	unsigned int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	unsigned int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	unsigned int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	unsigned int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;


	// Set the tours
	if (thread_global_index < POPULATION_SIZE)
	{
		printf("Thread Id: %d", thread_global_index);
		printf(" > population->id: %d", initial_population->id);
		printf(" > population->tours: %d", initial_population->tours);
		for (int p = 0; p < population_size; ++p)
		{
			printf(" > population[%d].id: %d", p, initial_population[p].id);
			printf(" > population[%d].tours: %d", p, initial_population[p].tours);
			printf(" > population[%d].tours->fitness: %d", p, initial_population[p].tours->fitness);
			printf(" > population[%d].tours->total_distance: %d", p, initial_population[p].tours->total_distance);
			printf(" > population[%d].tours->node_qty: %d", p, initial_population[p].tours->node_qty);
			printf(" > population[%d].tours->nodes: %d", p, initial_population[p].tours->nodes);
			for (int t = 0; t < POPULATION_SIZE; ++t)
			{
				printf(" > population[%d].tours[%d]: %d", p, t, initial_population[p].tours[t]);
				printf(" > population[%d].tours[%d].fitness: %d", p, t, initial_population[p].tours[t].fitness);
				printf(" > population[%d].tours[%d].total_distance: %d", p, t, initial_population[p].tours[t].total_distance);
				printf(" > population[%d].tours[%d].node_qty: %d", p, t, initial_population[p].tours[t].node_qty);
				printf(" > population[%d].tours[%d].nodes: %d", p, t, initial_population[p].tours[t].nodes);
				printf(" > population[%d].tours[%d].nodes->id: %d", p, t, initial_population[p].tours[t].nodes->id);
				printf(" > population[%d].tours[%d].nodes->x: %d", p, t, initial_population[p].tours[t].nodes->x);
				printf(" > population[%d].tours[%d].nodes->y: %d", p, t, initial_population[p].tours[t].nodes->y);
				printf(" > population[%d].tours[%d].nodes->item_qty: %d", p, t, initial_population[p].tours[t].nodes->item_qty);
				printf(" > population[%d].tours[%d].nodes->items: %d", p, t, initial_population[p].tours[t].nodes->items);
				for (int n = 0; n < initial_population[p].tours[t].node_qty; ++n)
				{
					printf(" > population[%d].tours[%d].nodes[%d]: %d", p, t, n, initial_population[p].tours[t].nodes[n]);
					printf(" > population[%d].tours[%d].nodes[%d].id: %d", p, t, n, initial_population[p].tours[t].nodes[n].id);
					printf(" > population[%d].tours[%d].nodes[%d].x: %d", p, t, n, initial_population[p].tours[t].nodes[n].x);
					printf(" > population[%d].tours[%d].nodes[%d].y: %d", p, t, n, initial_population[p].tours[t].nodes[n].y);
					printf(" > population[%d].tours[%d].nodes[%d].item_qty: %d", p, t, n, initial_population[p].tours[t].nodes[n].item_qty);
					printf(" > population[%d].tours[%d].nodes[%d].items: %d", p, t, n, initial_population[p].tours[t].nodes[n].items);
					printf(" > population[%d].tours[%d].nodes[%d].items->id: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->id);
					printf(" > population[%d].tours[%d].nodes[%d].items->node: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->node);
					printf(" > population[%d].tours[%d].nodes[%d].items->taken: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->taken);
					printf(" > population[%d].tours[%d].nodes[%d].items->value: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->value);
					printf(" > population[%d].tours[%d].nodes[%d].items->weight: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->weight);
					for (int i = 0; i < initial_population[p].tours[t].nodes[n].item_qty; ++i)
					{
						printf(" > population[%d].tours[%d].nodes[%d].items[%d]: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i]);
						printf(" > population[%d].tours[%d].nodes[%d].items[%d].id: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].id);
						printf(" > population[%d].tours[%d].nodes[%d].items[%d].node: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].node);
						printf(" > population[%d].tours[%d].nodes[%d].items[%d].taken: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].taken);
						printf(" > population[%d].tours[%d].nodes[%d].items[%d].value: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].value);
						printf(" > population[%d].tours[%d].nodes[%d].items[%d].weight: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].weight);
					}
				}
			}
		}
		printf("\n\n");
	}
}