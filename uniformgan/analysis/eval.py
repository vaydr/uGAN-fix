def simple_eval(js, dataset, test_dataset, sample_dataset, logger, error_logger, verbose=False):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.compose import ColumnTransformer
    import sklearn.metrics
    import numpy as np

    target = js['target_column']
    discrete_columns = js['discrete_columns']
    dataset = dataset.copy()
    label_encoder = LabelEncoder()
    test_dataset = test_dataset.copy()
    if (target not in discrete_columns):
        error_logger.write("Target column not in discrete columns, cannot evaluate")
        return
    X_real, y_real = dataset.drop(target, axis=1), dataset[target]
    y_real = label_encoder.fit_transform(y_real)
    X_test, y_test = test_dataset.drop(target, axis=1), test_dataset[target]
    y_test = label_encoder.transform(y_test)
    X_synth, y_synth = sample_dataset.drop(target, axis=1), sample_dataset[target]
    y_synth = label_encoder.transform(y_synth)
    def create_models():
        models, names = list(), list()
        models.append(DecisionTreeClassifier())
        names.append('CART')
        models.append(SVC(gamma='scale'))
        names.append('SVM')
        models.append(BaggingClassifier(n_estimators=200))
        names.append('BAG')
        models.append(RandomForestClassifier(n_estimators=200))
        names.append('RF')
        models.append(GradientBoostingClassifier(n_estimators=200))
        names.append('GBM')
        return models, names

    models, names = create_models()
    def avg_results(arr):
        keys = arr[0].keys()
        res = {}
        for key in keys:
            s = 0
            for a in arr:
                s += a[key]
            res[key] = s/len(arr)
        return res
    discrete_columns = list(set(js["discrete_columns"]) - set([target]))
    columns = list(dataset)
    continuous_columns = list(set(columns) - set(discrete_columns) - set([target]))
    for i in range(len(names)):
        # define steps
        print("FITTING", names[i], i)
        logger.write(f"[INFO]:\t{names[i]}\t{i}\n")
        for xx, yy, label in [[X_real,y_real, "REAL"], [X_synth, y_synth, "SYNTH"]]:
            res = []
            for _ in range(5):
                models, names = create_models()
                steps = [('c',OneHotEncoder(handle_unknown='ignore'), discrete_columns), ('n', MinMaxScaler(), continuous_columns)]
                # one hot encode categorical, normalize numerical
                ct = ColumnTransformer(steps)
                # wrap the model i a pipeline
                pipeline = Pipeline(steps=[('t',ct),('m', models[i])])
                pipeline.fit(xx, yy)
                classifications = len(np.unique(yy))
                average_method = "micro" if classifications > 2 else "binary"
                pred_yi = pipeline.predict(X_test)
                auc = 0
                if (classifications > 2):
                    max_class = max(pred_yi.max(), y_test.max()) + 1
                    y_test_one_hot = np.zeros((y_test.size, max_class)) 
                    pred_yi_one_hot = np.zeros((pred_yi.size, max_class))
                    pred_yi_one_hot[np.arange(pred_yi.size),pred_yi] = 1
                    y_test_one_hot[np.arange(y_test.size),y_test] = 1
                    auc = sklearn.metrics.roc_auc_score(y_test_one_hot, pred_yi_one_hot, multi_class="ovo")
                else:
                    auc = sklearn.metrics.roc_auc_score(y_test, pred_yi)
                res.append({
                    'f1': sklearn.metrics.f1_score(y_test, pred_yi, average=average_method), 
                    'accuracy': sklearn.metrics.accuracy_score(y_test, pred_yi), 
                    'auc': auc,
                })
                logger.write(f"[SUBINFO]\t{label} F1,Accuracy,AUC for {names[i]}\t{res[-1]['f1']}\t{res[-1]['accuracy']}\t{res[-1]['auc']}\n")
                if (verbose):
                    print(f"[SUBINFO]\t{label} F1,Accuracy,AUC for {names[i]}\t{res[-1]['f1']}\t{res[-1]['accuracy']}\t{res[-1]['auc']}\n")
            # 5 f1s, accuracies, auc...
            avgs = avg_results(res)
            logger.write(f"[STAT]\t{label} F1,Accuracy,AUC for {names[i]}\t{avgs['f1']}\t{avgs['accuracy']}\t{avgs['auc']}\n")
            logger.flush()
    print("Done")
    logger.write("[INFO]\tDone\n")
    logger.flush()
    