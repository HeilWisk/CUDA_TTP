// DEFINES: Population as an array of tours
struct population
{
	int id;
	int tour_qty;
	tour *tours;
};

void initializePopulationCPU(population& initial_population, tour& initial_tour, distance* distances, const int population_size, const int node_quantity)
{
	node temp_node;
	item temp_item;

	if(population_size > 0)
	{
		//Allocate memory for the TOURS
		initial_population.tours = (tour*)malloc(population_size * sizeof(tour));
		if (initial_population.tours == NULL) {
			printf("Unable to allocate memory for nodes");
			return;
		}

		// Set the tours
		for (int i = 0; i < population_size; ++i)
		{
			if (node_quantity > 0)
			{
				//Allocate memory for the nodes on tours
				initial_population.tours[i].nodes = (node*)malloc(node_quantity * sizeof(node));
				if (initial_population.tours[i].nodes == NULL) {
					printf("Unable to allocate memory for nodes");
					return;
				}

				initial_population.tours[i].node_qty = node_quantity;

				for (int r = 0; r < node_quantity; ++r)
				{
					initial_population.tours[i].nodes[r].id = initial_tour.nodes[r].id;
					initial_population.tours[i].nodes[r].x = initial_tour.nodes[r].x;
					initial_population.tours[i].nodes[r].y = initial_tour.nodes[r].y;
					initial_population.tours[i].nodes[r].item_qty = initial_tour.nodes[r].item_qty;
					for (int w = 0; w < initial_population.tours[i].nodes[r].item_qty; ++w)
					{
						//Allocate memory for the items on nodes
						initial_population.tours[i].nodes[r].items = (item*)malloc(initial_population.tours[i].nodes[r].item_qty * sizeof(item));
						if (initial_population.tours[i].nodes[r].items == NULL) {
							printf("Unable to allocate memory for nodes");
							return;
						}

						initial_population.tours[i].nodes[r].items[w].id = initial_tour.nodes[r].items[w].id;
						initial_population.tours[i].nodes[r].items[w].node = initial_tour.nodes[r].items[w].node;
						initial_population.tours[i].nodes[r].items[w].taken = initial_tour.nodes[r].items[w].taken;
						initial_population.tours[i].nodes[r].items[w].value = initial_tour.nodes[r].items[w].value;
						initial_population.tours[i].nodes[r].items[w].weight = initial_tour.nodes[r].items[w].weight;
					}
				}

				for (int j = 1; j < node_quantity; ++j)
				{
					int random_position = 1 + (rand() % (node_quantity - 1));
					temp_node.items = (item*)malloc(initial_population.tours[i].nodes[j].item_qty * sizeof(item));
					if (initial_population.tours[i].nodes == NULL) {
						printf("Unable to allocate memory for nodes");
						return;
					}
					temp_node = initial_population.tours[i].nodes[j];
					temp_node.items = initial_population.tours[i].nodes[j].items;

					initial_population.tours[i].nodes[j] = initial_population.tours[i].nodes[random_position];
					if (initial_population.tours[i].nodes[j].item_qty > 0)
					{
						initial_population.tours[i].nodes[j].items = initial_population.tours[i].nodes[random_position].items;
					}
					initial_population.tours[i].nodes[random_position] = temp_node;
					if (initial_population.tours[i].nodes[random_position].item_qty > 0)
					{
						initial_population.tours[i].nodes[random_position].items = temp_node.items;
					}

					for (int s = 0; s < initial_population.tours[i].nodes[j].item_qty; ++s)
					{
						initial_population.tours[i].nodes[j].items[s].taken = (int)round((rand() % 2));
					}
				}
				evaluateTour(initial_population.tours[i], distances, node_quantity);
			}
		}
	}
}

__global__ void initializePopulationGPU(population* initial_population, distance* distances, const int node_quantity, const int item_quantity, curandState* state)
{
	node temp;

	//unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	//unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int thread_global_index_x = colIdx + POPULATION_SIZE * rowIdx;

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
	if (thread_global_index < POPULATION_SIZE)
	{
		initial_population->tours[thread_global_index].node_qty = node_quantity;
		
		for (int j = 1; j < node_quantity; ++j)
		{
			int random_position = 1 + (curand(&local_state) % (node_quantity - 1));
			//cudaMalloc(&temp.items, sizeof(item) * initial_population->tours[thread_global_index].nodes[j].item_qty);
			temp = initial_population->tours[thread_global_index].nodes[j];
			temp.items = initial_population->tours[thread_global_index].nodes[j].items;
			initial_population->tours[thread_global_index].nodes[j] = initial_population->tours[thread_global_index].nodes[random_position];
			if (initial_population->tours[thread_global_index].nodes[j].item_qty > 0)
			{
				initial_population->tours[thread_global_index].nodes[j].items = initial_population->tours[thread_global_index].nodes[random_position].items;
			}
			initial_population->tours[thread_global_index].nodes[random_position] = temp;
			if (initial_population->tours[thread_global_index].nodes[random_position].item_qty > 0)
			{
				initial_population->tours[thread_global_index].nodes[random_position].items = temp.items;
			}

			for (int s = 0; s < initial_population->tours[thread_global_index].nodes[j].item_qty; ++s)
			{
				//printf("tour[%d]>nodes[%d]>items[%d]>id %d\n", thread_global_index, j, s, initial_population->tours[thread_global_index].nodes[j].items[s].id);//(int)((curand(&local_state)) + 0.5f);
				//printf("tour[%d]>nodes[%d]>items[%d]>node %d\n", thread_global_index, j, s, initial_population->tours[thread_global_index].nodes[j].items[s].node);//(int)((curand(&local_state)) + 0.5f);
				//printf("tour[%d]>nodes[%d]>items[%d]>value %f\n", thread_global_index, j, s, initial_population->tours[thread_global_index].nodes[j].items[s].value);//(int)((curand(&local_state)) + 0.5f);
				//printf("tour[%d]>nodes[%d]>items[%d]>weight %f\n", thread_global_index, j, s, initial_population->tours[thread_global_index].nodes[j].items[s].weight);//(int)((curand(&local_state)) + 0.5f);
				//printf("tour[%d]>nodes[%d]>items[%d]>taken %d\n", thread_global_index, j, s, initial_population->tours[thread_global_index].nodes[j].items[s].taken);//(int)((curand(&local_state)) + 0.5f);
			}
			//__syncthreads();
		}		
		
		initial_population->tours[thread_global_index].total_distance = 0;
		for (int i = 0; i < node_quantity; ++i)
		{
			for (int k = 0; k < node_quantity * node_quantity; ++k)
			{
				if (i < node_quantity - 1)
				{
					if ((distances[k].source == initial_population->tours[thread_global_index].nodes[i].id) && (distances[k].destiny == initial_population->tours[thread_global_index].nodes[i + 1].id))
					{
						initial_population->tours[thread_global_index].total_distance += distances[k].value;
					}
				}
				else
				{
					if ((distances[k].source == initial_population->tours[thread_global_index].nodes[i].id) && (distances[k].destiny == initial_population->tours[thread_global_index].nodes[0].id))
					{
						initial_population->tours[thread_global_index].total_distance += distances[k].value;
					}
				}
			}

			// Calculate the fitness
			if (initial_population->tours[thread_global_index].total_distance != 0)
				initial_population->tours[thread_global_index].fitness = 1 / initial_population->tours[thread_global_index].total_distance;
			else
				initial_population->tours[thread_global_index].fitness = 0;
			__syncthreads();
		}
	}
}

void printPopulation(population population, const int population_size)
{
	for (int i = 0; i < population_size; ++i)
	{
		printf("Individual %d\n", i);
		printf("> Fitness: %f\n", population.tours[i].fitness);
		printf("> Total Distance: %f\n", population.tours[i].total_distance);
		printf("> Node Quantity: %d\n", population.tours[i].node_qty);
		printf("> Nodes		>Item ID[Taken]\n");
		for (int j = 0; j < population.tours[i].node_qty; ++j)
		{
			printf("> %d", population.tours[i].nodes[j].id);
			if (population.tours[i].nodes[j].item_qty > 0)
			{
				for (int h = 0; h < population.tours[i].nodes[j].item_qty; h++)
				{
					printf("		> %d[%d]", population.tours[i].nodes[j].items[h].id, population.tours[i].nodes[j].items[h].taken);
				}
				printf("\n");
			}
			else
				printf("\n");
		}
		printf("\n\n");
	}
}