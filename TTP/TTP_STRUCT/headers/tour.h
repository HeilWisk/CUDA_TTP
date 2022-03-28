// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	node *nodes;
	float fitness;
	//TODO: Evaluate how it works with float to try to change for the struct distance
	float distance;
	item *items;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="node_quantity"></param>
	/// <param name="item_quantity"></param>
	/// <returns></returns>
	__host__ __device__ tour(const int node_quantity, const int item_quantity)
	{
		for (int n = 0; n < node_quantity; n++)
		{
			nodes[n] = node();
		}

		for (int i = 0; i < item_quantity; i++)
		{
			items[i] = item();
		}

		fitness = 0;
		distance = 0;
	}
};

__host__ __device__ void evaluateTour(tour& tour, const float* cost_table, const int node_quantity)
{
	tour.distance = 0;
	for (int i = 0; i < node_quantity; ++i)
	{
		if (i < node_quantity - 1)
		{
			tour.distance += cost_table[(tour.nodes[i].id) * node_quantity + (tour.nodes[i + 1]).id];
		}
		else
		{
			tour.distance += cost_table[(tour.nodes[i].id) * node_quantity + (tour.nodes[0]).id];
		}

		if (tour.distance != 0)
			tour.fitness = 1 / tour.distance;
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
		int random_x = rand() % MAX_COORD;
		int random_y = rand() % MAX_COORD;
		tour.nodes[i] = node(i, random_x, random_y);
	}
}

__host__ __device__ void printTour(const tour& tour, const int node_quantity)
{
	for (int i = 0; i < node_quantity; i++)
	{
		printf("%d ", tour.nodes[i].id + 1);
	}
	printf("\n");
}

