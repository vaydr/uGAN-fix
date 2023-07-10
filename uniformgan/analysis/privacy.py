
def dcr(js, dataset, sample_dataset, logger, error_logger):
    import numpy as np
    import statistics
    import pandas as pd
    import torch
    discrete_columns = js['discrete_columns']
    columns = list(sample_dataset)
    quantile_count = 100
    normalizer = {}
    for column in columns:
        if (column in discrete_columns):
            continue
        data = dataset[column]
        pure = [i/quantile_count for i in range(quantile_count+1)]
        quantiles = np.quantile(data, pure)
        normalizer[column] = quantiles

    def convert_quantiles(value, quantiles):
        for i in range(quantile_count + 1):
            if (value < quantiles[i]):
                return i/quantile_count
        return 1
    
    continuous_columns = list(set(columns) - set(discrete_columns))
    continuous_columns
    for column in continuous_columns:
        quantiles = normalizer[column]
        dataset[column] = dataset[column].apply(convert_quantiles, args=(quantiles,))
        sample_dataset[column] = sample_dataset[column].apply(convert_quantiles, args=(quantiles,))
    
    def calculate_distance(record, dataset):
        dataset_plus = abs(record[continuous_columns] - dataset[continuous_columns])
        dataset_plus = pd.concat([dataset_plus, record[discrete_columns] != dataset[discrete_columns]], axis=1)
        dataset_plus = dataset_plus.apply(np.linalg.norm, axis=1)
        dataset_plus = dataset_plus.sort_values() # type: ignore
        return dataset_plus

    logger.write("[INFO]\tPRIVACY\n")
    vals = []
    valsnth1 = []
    valsnth5 = []
    print(f"Checking privacy of {min(5000, len(sample_dataset))} rows")
    for i in range(min(5000, len(sample_dataset))):
        record = sample_dataset.iloc[i]
        dataset_plus = calculate_distance(record, dataset)
        dist = dataset_plus.iloc[0]
        print("Record", i, dist, dataset_plus.iloc[1] - dist, dataset_plus.iloc[5] - dist)
        vals.append(dist)
        if (dist < 1):
            valsnth1.append(dataset_plus.iloc[1] - dist)
            valsnth5.append(dataset_plus.iloc[5] - dist)
    logger.write(f"[STAT]\tAverage DCR (mean, median)\t{sum(vals)/len(vals)}\t{statistics.median(vals)}\n")
    logger.write(f"[STAT]\tAverage DCR Delta 1 (mean, median)\t{sum(valsnth1)/len(valsnth1)}\t{statistics.median(valsnth1)}\n")
    logger.write(f"[STAT]\tAverage DCR Delta 5 (mean, median)\t{sum(valsnth5)/len(valsnth5)}\t{statistics.median(valsnth5)}\n")
    vals = torch.tensor(vals, dtype=torch.float)
    valsnth1 = torch.tensor(valsnth1, dtype=torch.float)
    valsnth5 = torch.tensor(valsnth5, dtype=torch.float)
    percentiles = [0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5]
    perc = torch.tensor([0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5], dtype=torch.float)
    actual_percentiles = list(torch.quantile(vals, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles\t{stats}\n")
    actual_percentiles = list(torch.quantile(valsnth1, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles Delta 1\n{stats}\n")
    
    
def dcr_modified(js, dataset, sample_dataset, logger, error_logger):
    import numpy as np
    import statistics
    import pandas as pd
    import torch
    
    import pprint
    import random
    
    # torch.manual_seed(2)
    # np.random.seed(2)
    # random.seed(2)
    
    logger.write("IN DCR_MODIFIED METHOD")
    discrete_columns = js['discrete_columns']
    columns = list(sample_dataset)
    quantile_count = 100
    normalizer = {}
    # logger.write(f"\t discrete_columns: {discrete_columns}\n")
    # logger.write(f"\t sample dataset columns: {columns}\n")
    for column in columns:
        if (column in discrete_columns):
            continue
        data = dataset[column]
        pure = [i/quantile_count for i in range(quantile_count+1)]
        quantiles = np.quantile(data, pure)
        normalizer[column] = quantiles

    logger.write("NORMALIZER")
    # logger.write(f"\t after first pass creating normalizer: {pprint.pformat(normalizer)} \n")
    def convert_quantiles(value, quantiles):
        for i in range(quantile_count + 1):
            if (value < quantiles[i]):
                return i/quantile_count
        return 1
    
    continuous_columns = list(set(columns) - set(discrete_columns))
    # continuous_columns # why was this line originally here?
    for column in continuous_columns:
        quantiles = normalizer[column]
        dataset[column] = dataset[column].apply(convert_quantiles, args=(quantiles,))
        sample_dataset[column] = sample_dataset[column].apply(convert_quantiles, args=(quantiles,))
    # logger.write(f"\t after second pass editing normalizer: {pprint.pformat(normalizer)} \n")
    
    def calculate_distance(record, dataset):
        dataset_plus = abs(record[continuous_columns] - dataset[continuous_columns])
        dataset_plus = pd.concat([dataset_plus, record[discrete_columns] != dataset[discrete_columns]], axis=1)
        dataset_plus = dataset_plus.apply(np.linalg.norm, axis=1)
        # want to view what the closest real row looks like
        new_dataset = pd.DataFrame()
        new_dataset["distance"] = dataset_plus
        new_dataset[dataset.columns] = dataset
        new_dataset = new_dataset.sort_values(by="distance") # type: ignore
        return new_dataset

    logger.write("[INFO]\tPRIVACY\n")
    NUM_SAMPLES = 2500
    d_vals = []
    d_valsnth1 = [] # if take distance of
    d_valsnth5 = []
    d_close_rows = []
    og_valsnth1 = []
    og_valsnth5 = []
    real_close = []
    print("Checking privacy of 5000 rows")
    # for i in range(min(5000, len(sample_dataset))):
    min_distance = float('inf')
    fake_row, real_row = None, None
    for i in range(NUM_SAMPLES):
        record = sample_dataset.iloc[i]
        new_dataset = calculate_distance(record, dataset)
        dist = new_dataset.iloc[0]
        print("Record", i, dist["distance"])
        d_vals.append(dist["distance"])
        d_valsnth1.append(new_dataset.iloc[1]["distance"] - dist["distance"])
        d_valsnth5.append(new_dataset.iloc[5]["distance"] - dist["distance"])
        if dist["distance"] < min_distance:
            min_distance = dist["distance"]
            real_row = dist.drop(columns="distance")
            fake_row = record
        if dist["distance"] < 1:
            d_close_rows.append((i, record, dist.drop(columns="distance")))
            og_valsnth1.append(new_dataset.iloc[1]["distance"] - dist["distance"])
            og_valsnth5.append(new_dataset.iloc[5]["distance"] - dist["distance"])
        if dist["distance"] < 0.02:
            real_close.append((i, record, dist))

    # record the closest synth/real row pair in logger
    # logger.write(f"[FIRST 2500] Closest synthetic-real row pair with distance {min_distance}\n")
    # logger.write(f"\t Synthetic row:\n {fake_row}\n")
    # logger.write(f"\t Real row:\n {real_row}\n")
    
    print("Transitioning to random samples")
    r_vals = []
    r_valsnth1 = []
    r_valsnth5 = []
    r_close_rows = []
    for j in random.sample(range(len(sample_dataset)), NUM_SAMPLES):
        record = sample_dataset.iloc[j]
        new_dataset = calculate_distance(record, dataset)
        dist = new_dataset.iloc[0]
        print("Record", j, dist["distance"])
        r_vals.append(dist["distance"])
        r_valsnth1.append(new_dataset.iloc[1]["distance"] - dist["distance"])
        r_valsnth5.append(new_dataset.iloc[5]["distance"] - dist["distance"])
        if dist["distance"] < min_distance:
            min_distance = dist["distance"]
            real_row = dist.drop(columns="distance")
            fake_row = record
        if dist["distance"] < 1:
            r_close_rows.append((j, record, dist.drop(columns="distance")))
        if dist["distance"] < 0.02:
            real_close.append((i, record, dist))
    
    # record the closest synth/real row pair overall in logger
    logger.write(f"[OVERALL] Closest synthetic-real row pair with distance {min_distance}\n")
    logger.write(f"\t Synthetic row:\n {fake_row}\n")
    logger.write(f"\t Real row:\n {real_row}\n")
    
    vals = d_vals + r_vals
    valsnth1 = d_valsnth1 + r_valsnth1
    valsnth5 = d_valsnth5 + r_valsnth5
    close_rows = d_close_rows + r_close_rows
    
    if len(vals) > 0:
        logger.write("[STAT] Average DCR Values")
        logger.write(f"[STAT]\tAverage DCR (mean, median)\t{sum(vals)/len(vals)}\t{statistics.median(vals)}\n")
        logger.write(f"[STAT]\tAverage DCR Delta 1 (mean, median)\t{sum(valsnth1)/len(valsnth1)}\t{statistics.median(valsnth1)}\n")
        logger.write(f"[STAT]\tAverage DCR Delta 5 (mean, median)\t{sum(valsnth5)/len(valsnth5)}\t{statistics.median(valsnth5)}\n")
        
    # calculate more interesting DCR data
    def get_describe(vals):
        val_df = pd.DataFrame(vals, columns=["distance"])
        return val_df["distance"].describe().to_string()
    
    def summarize_details(vals, valsnth1, valsnth5, msg):
        logger.write(f"[DETAILED] {msg}\n")
        logger.write(f"[DETAILED]\tDCR: {get_describe(vals)}\n")
        logger.write(f"[DETAILED]\tDCR Delta 1: {get_describe(valsnth1)}\n")
        logger.write(f"[DETAILED]\tDCR Delta 5: {get_describe(valsnth5)}\n")
    
    summarize_details(vals, valsnth1, valsnth5, "Combined Values")
    summarize_details(d_vals, d_valsnth1, d_valsnth5, "First 2500 Samples Only")
    summarize_details(r_vals, r_valsnth1, r_valsnth5, "Random Samples Only")
    
    vals_df = pd.DataFrame(vals, columns=["distance"])
    vals_df.to_csv("privacy_distances_many38_003.csv", index=False)
    
    # record what the og privacy metric did for DCR Deltas
    if len(og_valsnth1) > 0:
        summarize_details(d_vals, og_valsnth1, og_valsnth5, "Original Calculations for DCR")
    
    # write out close rows (distance < 1) data
    for (msg, ratio,) in [
        ("BOTH", len(close_rows)/len(vals)),
        ("FIRST 2500", len(d_close_rows)/len(d_vals)),
        ("RANDOM", len(r_close_rows)/len(r_vals)),
    ]:
        logger.write(f"[DIST<1 {msg}] {ratio} fraction of all samples had a distance < 1.\n")
        
    logger.write(f"[REAL CLOSE] {len(real_close)} fake samples had distance < 0.02 from a real sample.")
    
    def write_close_rows(close_rows, msg):
        if len(close_rows) > 0:
            logger.write(f"[Close Row Ex] {msg}\n")
            logger.write(f"\t Synthetic Row:\n {close_rows[0][1]}\n")
            logger.write(f"\t Real Row:\n {close_rows[0][2]}\n")
    
    # write_close_rows(d_close_rows, "From first 2500 samples")
    # write_close_rows(r_close_rows, "From random samples")
    
    vals = torch.tensor(vals, dtype=torch.float)
    valsnth1 = torch.tensor(valsnth1, dtype=torch.float)
    valsnth5 = torch.tensor(valsnth5, dtype=torch.float)
    percentiles = [0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5]
    perc = torch.tensor([0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5], dtype=torch.float)
    actual_percentiles = list(torch.quantile(vals, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles\t{stats}\n")
    actual_percentiles = list(torch.quantile(valsnth1, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles Delta 1\n{stats}\n")