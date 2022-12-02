// Definition for a node

// DEFINES: Node Data Type
struct node {
	int id;
	double x;
	double y;
	item items[ITEMS];

	__host__ __device__ node()
	{
		id = -1;
		x = -1;
		y = -1;
		for (int i = 0; i < ITEMS; ++i)
		{
			items[i] = item(-1, -1, -1, -1);
		}
	}

	__host__ __device__ node(int id_node, double x_coordinate, double y_coordinate/*, item its*/)
	{
		id = id_node;
		x = x_coordinate;
		y = y_coordinate;
		/*for (int i = 0; i < ITEMS; i++)
		{
			items[i] = its.items[i];
		}*/
		
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
		for (int i = 0; i < ITEMS; ++i)
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

		for (int i = 0; i < ITEMS; ++i)
		{
			if (items[i].id != var.items[i].id || items[i].value != var.items[i].value || items[i].weight != var.items[i].weight || items[i].pw_ratio != var.items[i].pw_ratio || items[i].node != var.items[i].node)
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
		for (int j = 0; j < ITEMS; ++j)
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
		c[i].x = (float)matrix[i][1];
		c[i].y = (float)matrix[i][2];
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
		node.items[s].value = (float)matrix[s][1];
		node.items[s].weight = (float)matrix[s][2];
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
	int amount_items_per_node;
	// Loop through the node array
	for (int n = 0; n < CITIES; n++)
	{
		// Count the amount of items per node to calculate memory allocation
		amount_items_per_node = 0;
		for (int s = 0; s < ITEMS; ++s)
		{
			if (items[s].node == nodes[n].id)
			{
				amount_items_per_node += amount_items_per_node + 1;
			}
		}

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
					nodes[n].items[node_index].pw_ratio = items[s].value / items[s].weight;
					node_index++;
				}
			}
		}
	}
}