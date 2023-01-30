// Definition for a node

// DEFINES: Node Data Type
struct node {
	int id;
	double x;
	double y;
	item items[ITEMS_PER_CITY];

	__host__ __device__ node()
	{
		id = -1;
		x = -1;
		y = -1;
		for (int i = 0; i < ITEMS_PER_CITY; ++i)
		{
			items[i] = item(-1, -1, -1, -1);
		}
	}

	__host__ __device__ node(int id_node, double x_coordinate, double y_coordinate)
	{
		id = id_node;
		x = x_coordinate;
		y = y_coordinate;
		
	}

	/// <summary>
	/// Override of the assign (=) operator
	/// </summary>
	/// <param name="var"></param>
	/// <returns></returns>
	__host__ __device__ node& operator=(const node& var)
	{
		id = var.id;
		x = var.x;
		y = var.y;
		for (int i = 0; i < ITEMS_PER_CITY; ++i)
		{
			items[i] = var.items[i];
		}
		return *this;
	}

	/// <summary>
	/// Override of the is equal (==) operator
	/// </summary>
	/// <param name="var"></param>
	/// <returns></returns>
	__host__ __device__ bool operator==(const node& var)
	{
		bool result = id == var.id && x == var.x && y == var.y;

		for (int i = 0; i < ITEMS_PER_CITY; ++i)
		{
			if (items[i].id != var.items[i].id || items[i].value != var.items[i].value || items[i].weight != var.items[i].weight || items[i].node != var.items[i].node)
				result = false;
		}
		return result;
	}
};

/// <summary>
/// Calculate Distance between nodes
/// </summary>
/// <param name="src_node"></param>
/// <param name="dst_node"></param>
/// <returns></returns>
__host__ __device__ double distanceBetweenNodes(const node& src_node, const node& dst_node)
{
	double x_distance = pow(dst_node.x - src_node.x, 2);
	double y_distance = pow(dst_node.y - src_node.y, 2);
	return sqrt(x_distance + y_distance);
}

/// <summary>
/// Display the node array
/// </summary>
/// <param name="c">- Node array</param>
/// <param name="size">- Size of the array</param>
void displayNodes(node* c, int size) {
	printf("****************************************************************************************\n");
	printf("NODES (CITIES):		%d\n", size);
	printf("****************************************************************************************\n");
	printf("ID	X		Y		ITEMS\n");
	for (int i = 0; i < size; ++i) {
		printf("%d	%f	%f", c[i].id, c[i].x, c[i].y);
		for (int j = 0; j < ITEMS_PER_CITY; ++j)
		{
			if(c[i].items[j].id > 0)
				printf("	> %d", c[i].items[j].id);
		}
		printf("\n");
	}
	printf("****************************************************************************************\n");
	printf("\n");
}

/// <summary>
/// Function to convert the extracted matrix into an array of node structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="c">- Pointer to array of nodes structs</param>
void extractNodes(int** matrix, int rows, node* c) {
	for (int i = 0; i < rows; ++i) {
		c[i].id = matrix[i][0];
		c[i].x = (double)matrix[i][1];
		c[i].y = (double)matrix[i][2];
	}
}

/// <summary>
/// Function to convert the extracted matrix into an array of item structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted item</param>
void extractItems(int** matrix, int rows, node& node) {
	for (int s = 0; s < rows; ++s) {
		node.items[s].id = matrix[s][0];
		node.items[s].value = matrix[s][1];
		node.items[s].weight = matrix[s][2];
		node.items[s].node = matrix[s][3];
	}
}

/// <summary>
/// Function to assign every item to his corresponding node
/// </summary>
/// <param name="items">- Array of items to asign</param>
/// <param name="nodes">- Nodes to assign the extracted item</param>
void assignItems(item* items, node* nodes)
{
	int node_index;
	int amount_items_per_node = ITEMS_PER_CITY;
	// Loop through the node array
	for (int n = 0; n < CITIES; n++)
	{
		// Validate if the given node has asigned items
		if (amount_items_per_node > 0)
		{
			node_index = 0;
			for (int s = 0; s < ITEMS; ++s)
			{
				if (items[s].node == nodes[n].id)
				{
					nodes[n].items[node_index].id = items[s].id;
					nodes[n].items[node_index].value = items[s].value;
					nodes[n].items[node_index].weight = items[s].weight;
					nodes[n].items[node_index].node = items[s].node;
					node_index++;
				}
			}
		}
	}
}