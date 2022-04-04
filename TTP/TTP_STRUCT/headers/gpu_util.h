/***********************************************************************************************************************
* This file contains all helper functions needed for manipulation of population on the GPU
***********************************************************************************************************************/


__device__ tour tournamentSelection(population& population, curandState* d_state, const int& thread_id, const int node_quantity, const int item_quantity)
{
	return tour(node_quantity, item_quantity, true);
}
