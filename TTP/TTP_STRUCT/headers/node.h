// Definition for a node

// DEFINES: Node Data Type
struct node {
	int id;
	float x;
	float y;

	__host__ __device__ node() 
	{ 
		id = -1; 
		x = -1; 
		y = -1; 
	}

	__host__ __device__ node(int id_node, float x_coordinate, float y_coordinate)
	{ 
		id = id_node; 
		x = x_coordinate; 
		y = y_coordinate; 
	}

	/*__host__ __device__ node& operator=(const node& var)
	{
		id = var.id;
		x = var.x;
		y = var.y;
		return *this;
	}

	__host__ __device__ bool operator==(const node& var)
	const {
		return(id == var.id && x == var.x && y == var.y);
	}*/
};

__host__ __device__ float distanceBetweenNodes(const node& src_node, const node& dst_node)
{
	float x_distance = (float)pow(dst_node.x - src_node.x, 2);
	float y_distance = (float)pow(dst_node.y - src_node.y, 2);
	return sqrt(x_distance + y_distance);
}