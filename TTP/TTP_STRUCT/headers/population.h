// DEFINES: Population as an array of tours
struct population
{
	int id;
	tour *tours;
};

void initializePopulationCPU(population& initial_population, tour& initial_tour, distance* distances, const int population_size, const int node_quantity)
{
	node temp;

	//Allocate memory for the TOURS
	initial_population.tours = (tour*)malloc(population_size * sizeof(tour));
	if (initial_population.tours == NULL) {
		printf("Unable to allocate memory for nodes");
		return;
	}	

	// Set the tours
	for (int i = 0; i < population_size; ++i)
	{
		//Allocate memory for the nodes on tours
		initial_population.tours[i].nodes = (node*)malloc(node_quantity * sizeof(node));
		if (initial_population.tours[i].nodes == NULL) {
			printf("Unable to allocate memory for nodes");
			return;
		}

		for(int r = 0; r<node_quantity; ++r)
		{
			initial_population.tours[i].nodes[r].id = initial_tour.nodes[r].id;
			initial_population.tours[i].nodes[r].x = initial_tour.nodes[r].x;
			initial_population.tours[i].nodes[r].y = initial_tour.nodes[r].y;
		}

		for (int j = 1; j < node_quantity; ++j)
		{
			int random_position = 1 + (rand() % (node_quantity - 1));			
			temp = initial_population.tours[i].nodes[j];
			initial_population.tours[i].nodes[j] = initial_population.tours[i].nodes[random_position];
			initial_population.tours[i].nodes[random_position] = temp;			
		}
		evaluateTour(initial_population.tours[i], distances, node_quantity);
	}
}

__global__ void initializePopulationGPU(population* initial_population, distance* distances, const int node_quantity, const int item_quantity, curandState* state)
{
	node temp;

	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int thread_global_index_x = colIdx + POPULATION_SIZE * rowIdx;

	curandState local_state = state[thread_global_index_x];

	// Set the tours
	if (thread_global_index_x < POPULATION_SIZE)
	{
		for (int j = 1; j < node_quantity; ++j)
		{
			int random_position = 1 + (curand(&local_state) % (node_quantity - 1));
			temp = initial_population->tours[thread_global_index_x].nodes[j];
			initial_population->tours[thread_global_index_x].nodes[j] = initial_population->tours[thread_global_index_x].nodes[random_position];
			initial_population->tours[thread_global_index_x].nodes[random_position] = temp;
			//__syncthreads();
		}		
		
		initial_population->tours[thread_global_index_x].total_distance = 0;
		for (int i = 0; i < node_quantity; ++i)
		{
			for (int k = 0; k < node_quantity * node_quantity; ++k)
			{
				if (i < node_quantity - 1)
				{
					if ((distances[k].source == initial_population->tours[thread_global_index_x].nodes[i].id) && (distances[k].destiny == initial_population->tours[thread_global_index_x].nodes[i + 1].id))
					{
						initial_population->tours[thread_global_index_x].total_distance += distances[k].value;
					}
				}
				else
				{
					if ((distances[k].source == initial_population->tours[thread_global_index_x].nodes[i].id) && (distances[k].destiny == initial_population->tours[thread_global_index_x].nodes[0].id))
					{
						initial_population->tours[thread_global_index_x].total_distance += distances[k].value;
					}
				}
			}
			//if (i < node_quantity - 1)
			//{
				//initial_population->tours[thread_global_index_x].total_distance += distances[(initial_population->tours[thread_global_index_x].nodes[i].id-1) * node_quantity + (initial_population->tours[thread_global_index_x].nodes[i + 1]).id-1].value;
			//}
			//else
			//{
				//initial_population->tours[thread_global_index_x].total_distance += distances[(initial_population->tours[thread_global_index_x].nodes[i].id-1) * node_quantity + (initial_population->tours[thread_global_index_x].nodes[0]).id-1].value;
			//}

			// Calculate the fitness
			if (initial_population->tours[thread_global_index_x].total_distance != 0)
				initial_population->tours[thread_global_index_x].fitness = 1 / initial_population->tours[thread_global_index_x].total_distance;
			else
				initial_population->tours[thread_global_index_x].fitness = 0;
			__syncthreads();
		}
	}
}

void printPopulation(population population, const int population_size, const int node_quantity)
{
	for (int i = 0; i < population_size; ++i)
	{
		printf("Individual %d\n", i);
		printf("> Fitness: %f\n", population.tours[i].fitness);
		printf("> Total Distance: %f\n", population.tours[i].total_distance);
		printf("> Nodes:\n");
		for (int j = 0; j < node_quantity; ++j)
		{
			printf("%d\n", population.tours[i].nodes[j].id);
		}
		printf("\n");
	}
}