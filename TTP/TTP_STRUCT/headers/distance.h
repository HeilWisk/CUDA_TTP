// Definition for distance

//DEFINES: distance data type
struct distance {
	int source;
	int destiny;
	float value;

	__host__ __device__ distance()
	{
		source = -1;
		destiny = -1;
		value = -1;
	}

	__host__ __device__ distance(int source_id, int destiny_id, float distance_value)
	{
		source = source_id;
		destiny = destiny_id;
		value = distance_value;
	}
};
