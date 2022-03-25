// Definition for an item

//DEFINES: Item Data Type
struct item {
	int id;
	float weight;
	float value;
	int node;

	__host__ __device__ item()
	{
		id = -1;
		weight = -1;
		value = -1;
		node = -1;
	}

	__host__ __device__ item(int id_item, float w, float v, int node_id)
	{
		id = id_item;
		weight = w;
		value = v;
		node = node_id;
	}
};

