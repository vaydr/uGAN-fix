LONGEST_KEYWORD = "_longest_RR"
def complete():
    import glob
    txt_files = glob.glob(f"datasets_analysis/*/*-SUMMARY.txt")
    logger = open("SUMMARY.txt", "w")
    dict_ = {} # attribute -> { [generator] : value/values }
    generators = set()
    for file in txt_files:
        lines = open(file, "r").readlines()
        # LINE 1 is always the name
        generator_name, lines = lines[0].strip(), lines[1:]
        generators.add(generator_name)
        # We want to aggregate all the stats
        for line in lines:
            line = [l.strip() for l in line.strip().split("\t")]
            # [stat name] \t [stat values...]
            stat_name, stats = line[0], line[1:]
            if (stat_name not in dict_):
                dict_[stat_name] = {LONGEST_KEYWORD: 0}
            dict_[stat_name][LONGEST_KEYWORD] = max(dict_[stat_name][LONGEST_KEYWORD], len(stats))
            dict_[stat_name][generator_name] = stats
    generators = list(generators)    
    column_names=["Statistic"]
    for generator in sorted(generators):
        column_names.append(generator)
    logger.write("\t".join(column_names) + "\n")
    for stat_name in sorted(dict_.keys()):
        length = dict_[stat_name][LONGEST_KEYWORD]
        for l in range(length):
            row = [stat_name]
            if (l > 0):
                row = [stat_name + "(" + str(l) + ")"]
            for i in range(1, len(column_names)):
                column = column_names[i]
                if len(column) > l and column in dict_[stat_name]:
                    row.append(dict_[stat_name][column][l])
                else:
                    row.append("")
            logger.write("\t".join(row) + "\n")
    logger.close()

if __name__ == "__main__":
    complete()
        
