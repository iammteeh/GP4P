# ToDo
- make ConfigSysProxy compatible to dataframes

## features
- binary
- x_X, y

## configs == performance_map
- dictionary
    - key: tuple of features, e.g. (True, False, True)
    - value: y

## position map
- dictionary
    - key: feature
    - value: position in feature string

## prototype_config
- list having all features valued zero / False

## get_all_config_df
- dataframe with all features (but without y)

## get_measurement_df
- dataframe with all features and y