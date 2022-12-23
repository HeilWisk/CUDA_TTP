#pragma once

/// <summary>
/// 
/// </summary>
/// <param name="node"></param>
/// <returns></returns>
item selectItemByPwRatio(node node)
{
	// Calculate how many items are assigned to the node
	int items_count = 0;
	for (int i = 0; i < ITEMS; ++i)
	{
		if (node.items[i].id > 0)
			++items_count;
	}

	// Check if there's more than one item on the evaluated node
	if (items_count > 1)
	{
		// In case that exist more than one item assigned to the node, then order by pw_ratio
		item temp_item;

		for (int j = 0; j < items_count; ++j)
		{
			for (int k = j + 1; k < items_count; ++k)
			{
				if (node.items[j].pw_ratio < node.items[k].pw_ratio)
				{
					temp_item = node.items[j];
					node.items[j] = node.items[k];
					node.items[k] = temp_item;
				}
			}
		}
	}

	// Return the item with best pw_ratio
	return node.items[0];
}

/// <summary>
/// 
/// </summary>
/// <param name="items"></param>
/// <param name="capacity"></param>
/// <returns></returns>
__host__ __device__ item* freeKnapsackCapacity(item* items, double capacity)
{
	// Obtain the total weight of the items and the quantity of items
	double total_weight = 0.0;
	int item_count = 0;
	for (int i = 0; i < ITEMS; ++i)
	{
		if (items[i].id > 0)
		{
			total_weight += items[i].weight;
			++item_count;
		}
	}

	if (item_count > 1)
	{
		// In case that exist more than one item assigned to the node, then order by pw_ratio
		item temp_item;

		for (int j = 0; j < item_count; ++j)
		{
			for (int k = j + 1; k < item_count; ++k)
			{
				if (items[j].weight < items[k].weight)
				{
					temp_item = items[j];
					items[j] = items[k];
					items[k] = temp_item;
				}
			}
		}

		while (total_weight > capacity)
		{
			total_weight -= items[0].weight;
			for (int l = 0; l < item_count - 1; ++l)
			{
				items[l] = items[l + 1];
			}
			--item_count;
		}
	}

	return items;
}
