// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	node *nodes;
	double fitness;
	double total_distance;
	int node_qty;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="node_quantity"></param>
	/// <param name="item_quantity"></param>
	/// <returns></returns>
	__host__ __device__ tour(const int node_quantity, const int item_quantity, const bool gpu )
	{
		//Allocate memory for the nodes (cities)
		if (gpu)
		{
			cudaMalloc(&nodes, sizeof(node) * node_quantity);
		}
		else
		{
			nodes = (node*)malloc(node_quantity * sizeof(node));
			if (nodes == NULL) {
				printf("Unable to allocate memory for nodes");
				return;
			}

			//Load data on nodes
			for (int n = 0; n < node_quantity; n++)
			{
				nodes[n] = node();
			}
		}		

		//Allocate memory for the items
		/*if (gpu)
		{
			cudaMalloc(&items, sizeof(item) * item_quantity);
		}
		else
		{
			items = (item*)malloc(item_quantity * sizeof(item));
			if (items == NULL) {
				printf("Unable to allocate memory for items");
				return;
			}

			//Load data on items
			for (int i = 0; i < item_quantity; i++)
			{
				items[i] = item();
			}
		}*/

		fitness = 0;
		total_distance = 0;
		node_qty = node_quantity;
	}

	/// <summary>
	/// Overload of the == operator
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	__host__ __device__ bool operator==(tour& t)
	{
		for (int i = 0; i < t.node_qty; ++i)
		{
			if (nodes[i].x != t.nodes[i].x || nodes[i].y != t.nodes[i].y)
			{
				return false;
			}
		}
		return true;
	}

	__host__ __device__ tour& operator=(const tour& t)
	{
		for (int i = 0; i < t.node_qty; ++i)
		{
			nodes[i] = t.nodes[i];
		}
		fitness = t.fitness;
		total_distance = t.total_distance;
		node_qty = t.node_qty;
		return *this;
	}
};

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="distance_table"></param>
/// <param name="node_quantity"></param>
__host__ __device__ void evaluateTour(tour& tour, const distance* distance_table, const int node_quantity)
{
	tour.total_distance = 0;
	for (int i = 0; i < node_quantity; ++i)
	{
		for (int k = 0; k < node_quantity * node_quantity; ++k)
		{
			if (i < node_quantity - 1)
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[i + 1].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
			else
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[0].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
		}

		// Calculate the fitness
		if (tour.total_distance != 0)
			tour.fitness = 1 / tour.total_distance;
		else
			tour.fitness = 0;
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="distance_table"></param>
/// <returns></returns>
__host__ __device__ void evaluateTour(tour& tour, const distance* distance_table)
{
	tour.total_distance = 0;
	for (int i = 0; i < tour.node_qty; ++i)
	{
		for (int k = 0; k < tour.node_qty * tour.node_qty; ++k)
		{
			if (i < tour.node_qty - 1)
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[i + 1].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
			else
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[0].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
		}

		// Calculate the fitness
		if (tour.total_distance != 0)
			tour.fitness = 1 / tour.total_distance;
		else
			tour.fitness = 0;
	}
}

void initializeRandomTour(tour &tour, const int node_quantity)
{
	// Only randomizes the tail of the tour
	// this is because every tour stars in the same node
	tour.nodes[0] = node(0, 0, 0);
	for (int i = 1; i < node_quantity; ++i)
	{
		double random_x = rand() % MAX_COORD;
		double random_y = rand() % MAX_COORD;
		tour.nodes[i] = node(i, random_x, random_y);
	}
}

void initializeRandomTour(tour& tour)
{
	// Only randomizes the tail of the tour
	// this is because every tour stars in the same node
	tour.nodes[0] = node(0, 0, 0);
	for (int i = 1; i < tour.node_qty; ++i)
	{
		double random_x = rand() % MAX_COORD;
		double random_y = rand() % MAX_COORD;
		tour.nodes[i] = node(i, random_x, random_y);
	}
}

__host__ __device__ void printTour(const tour& tour, const int node_quantity)
{
	printf("TOUR INFORMATION\n");
	printf("FITNESS: %f\n", tour.fitness);
	printf("NODES:\n");
	for (int i = 0; i < node_quantity; ++i)
	{
		printf("NODE[%d]	ID: %d\n", i, tour.nodes[i].id);
	}
	
	printf("TOTAL DISTANCE: %f\n", tour.total_distance);
	printf("\n");
}

__host__ __device__ void printTour(const tour& tour)
{
	printf("TOUR INFORMATION\n");
	printf("FITNESS: %f\n", tour.fitness);
	printf("NODES:\n");
	for (int i = 0; i < tour.node_qty; ++i)
	{
		printf("NODE[%d]	ID: %d\n", i, tour.nodes[i].id);
	}

	printf("TOTAL DISTANCE: %f\n", tour.total_distance);
	printf("\n");
}

/// <summary>
/// Function to convert the extracted matrix into an array of node structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted nodes</param>
void extractNodes(int** matrix, int rows, tour& tour) 
{
	for (int i = 0; i < rows; i++) {
		tour.nodes[i].id = matrix[i][0];
		tour.nodes[i].x = matrix[i][1];
		tour.nodes[i].y = matrix[i][2];
	}
}

void defineInitialTour(tour& initial_tour, const int node_quantity, node* nodes)
{
	//Allocate memory for the nodes (cities)
	initial_tour.nodes = (node*)malloc(node_quantity * sizeof(node));
	if (initial_tour.nodes == NULL) {
		printf("Unable to allocate memory for nodes");
		return;
	}

	// Define the amount of nodes
	initial_tour.node_qty = node_quantity;

	//Load data on nodes
	for (int n = 0; n < initial_tour.node_qty; n++)
	{
		initial_tour.nodes[n] = node();
		initial_tour.nodes[n].id = nodes[n].id;
		initial_tour.nodes[n].x = nodes[n].x;
		initial_tour.nodes[n].y = nodes[n].y;
		initial_tour.nodes[n].item_qty = nodes[n].item_qty;

		//Allocate memory for the items
		initial_tour.nodes[n].items = (item*)malloc(initial_tour.nodes[n].item_qty * sizeof(item));
		if (initial_tour.nodes[n].items == NULL) {
			printf("Unable to allocate memory for items");
			return;
		}

		//Load data on items
		for (int i = 0; i < initial_tour.nodes[n].item_qty; i++)
		{
			initial_tour.nodes[n].items[i] = item();
			initial_tour.nodes[n].items[i].id = nodes[n].items[i].id;
			initial_tour.nodes[n].items[i].value = nodes[n].items[i].value;
			initial_tour.nodes[n].items[i].weight = nodes[n].items[i].weight;
			initial_tour.nodes[n].items[i].node = nodes[n].items[i].node;
			initial_tour.nodes[n].items[i].taken = nodes[n].items[i].taken;
		}
	}

	initial_tour.fitness = 0;
	initial_tour.total_distance = 0;
}