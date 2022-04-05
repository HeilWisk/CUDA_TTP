// Definition for a node

// DEFINES: Node Data Type
struct node {
	int id;
	double x;
	double y;

	__host__ __device__ node() 
	{ 
		id = -1; 
		x = -1; 
		y = -1; 
	}

	__host__ __device__ node(int id_node, double x_coordinate, double y_coordinate)
	{ 
		id = id_node; 
		x = x_coordinate; 
		y = y_coordinate; 
	}

	__host__ __device__ node& operator=(const node& var)
	{
		id = var.id;
		x = var.x;
		y = var.y;
		return *this;
	}

	__host__ __device__ bool operator==(const node& var) const 
	{
		return(id == var.id && x == var.x && y == var.y);
	}
	
};

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
	printf("ID	X		Y\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%f	%f\n", c[i].id, c[i].x, c[i].y);
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
	for (int i = 0; i < rows; i++) {
		c[i].id = matrix[i][0];
		c[i].x = (float)matrix[i][1];
		c[i].y = (float)matrix[i][2];
	}
}