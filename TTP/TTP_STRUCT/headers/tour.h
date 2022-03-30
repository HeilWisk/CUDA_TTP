// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	node *nodes;
	double fitness;
	//TODO: Evaluate how it works with float to try to change for the struct distance
	double total_distance;
	item *items;

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
		if (gpu)
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
		}	

		fitness = 0;
		total_distance = 0;
	}
};

__host__ __device__ void evaluateTour(tour& tour, const distance* cost_table, const int node_quantity)
{
	tour.total_distance = 0;
	for (int i = 0; i < node_quantity; ++i)
	{
		if (i < node_quantity - 1)
		{
			tour.total_distance += cost_table[(tour.nodes[i].id) * node_quantity + (tour.nodes[i + 1]).id].value;
		}
		else
		{
			tour.total_distance += cost_table[(tour.nodes[i].id) * node_quantity + (tour.nodes[0]).id].value;
		}

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

/// <summary>
/// Function to convert the extracted matrix into an array of node structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted nodes</param>
void extractNodes(int** matrix, int rows, tour& tour) {
	for (int i = 0; i < rows; i++) {
		tour.nodes[i].id = matrix[i][0];
		tour.nodes[i].x = matrix[i][1];
		tour.nodes[i].y = matrix[i][2];
	}
}

/// <summary>
/// Function to convert the extracted matrix into an array of item structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted item</param>
void extractItems(int** matrix, int rows, tour& tour) {
	for (int s = 0; s < rows; s++) {
		tour.items[s].id = matrix[s][0];
		tour.items[s].value = (float)matrix[s][1];
		tour.items[s].weight = (float)matrix[s][2];
		tour.items[s].node = matrix[s][3];
	}
}