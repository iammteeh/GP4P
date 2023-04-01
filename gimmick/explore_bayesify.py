from preprocessing import prepare_dataset

ds = prepare_dataset()

#print(ds.all_configs)
#print(ds.position_map)
print(ds.prototype_config)
print(f"redundant feature: {ds.redundant_ft}")
print(f"redundant feature names: {ds.redundant_ft_names}")
print(f"alternative feature: {ds.alternative_ft}")
print(f"alternative feature names: {ds.alternative_ft_names}")