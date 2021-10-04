import pandas as pd
from sklearn.linear_model import LogisticRegression
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from operator import itemgetter
from collections import defaultdict
from statistics import mode


try:
    from pymagnitude import *
except ModuleNotFoundError:
    from pymagnitude import * # It worked only when trying twice...

'''
COMPOSITION: how we mix noun and adjective vectors
adj: just the adjective
sum
av: average
concat: concatenation
mul: component-wise multiplication
diff: difference


FEATURES:
cosine OR euclidean similarity
cosine AND euclidean  similarities
the BERT/w2v/fasttext representations themselves
'''


def compose(vec_n, vec_a, composition_type):
    if composition_type == "adj":
        return vec_a
    elif composition_type == "concatenation":
        feat = np.concatenate((vec_n, vec_a))
    elif composition_type == "average":
        feat = np.average((np.array(vec_n), np.array(vec_a)), axis=0)
    elif composition_type == "difference":
        feat = np.array(vec_a - vec_n)
    elif composition_type == "mul":
        feat = vec_n * vec_a
    elif composition_type == "sum":
        feat = vec_n + vec_a
    return feat


def cosine_similarity(vec_n, vec_a):
    sim = 1 - cosine(vec_n, vec_a)
    return sim

def euclidean_distance(vec_n, vec_a):
    dist = np.linalg.norm(vec_n - vec_a)
    return dist


class ANpair:

    def __init__(self, noun, adjective, dataset, subset=None):
        self.noun = noun
        self.adjective = adjective
        self.dataset = dataset
        self.subset = subset


def load_data():
    data = []
    fn = "HVD_labels.csv"
    with open(fn) as f:
        for l in f:
            n, a, label = l.strip().split("\t")
            instance = ANpair(n, a, "HVD")
            instance.typ = label
            data.append(instance)

    return data


def booltype(typ):
    if typ == "YES":
        return 1
    elif typ == 'NO':
        return 0


def get_arrays(data, composition_type, vector_type, vectors, ctxt_combi, layer):
    '''
    first make the composition, then if necessary the similarity, and the feature.
    '''
    data_arrays = dict()
    data_vecs = dict()  ######## This will contain vec_n, vec composed.
    for subset in data:  # train, dev, test
        X, y, Xvecs = [], [], []
        for instance in data[subset]:
            if vector_type in ["w2v", "fasttext"]:
                vec1 = vectors.query(instance.noun)
                vec2 = vectors.query(instance.adjective)
            elif vector_type == "bert":
                n1 = vectors[(instance.noun, instance.adjective)]["reps_noun1"][layer]
                n2 = vectors[(instance.noun, instance.adjective)]["reps_noun2"][layer]
                a2 = vectors[(instance.noun, instance.adjective)]["reps_adj2"][layer]
                if ctxt_combi == ("n1", "n2"):
                    vec1 = n1
                    vec2 = n2
                elif ctxt_combi == ("n2", "a2"):
                    vec1 = n2
                    vec2 = a2
                elif ctxt_combi == ("n1", "a2"):
                    vec1 = n1
                    vec2 = a2
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            composed = compose(vec1, vec2, composition_type)
            X.append(composed)
            y.append(booltype(instance.typ))
            Xvecs.append((vec1, composed))

        X = np.array(X)
        y = np.array(y)
        data_arrays[subset] = (X, y)
        data_vecs[subset] = (Xvecs, y)
    return data_arrays, data_vecs


def get_simfeature_from_arrays(data_vecs, feature_function):
    new_data_arrays = dict()
    for subset in data_vecs:
        X = []
        if feature_function != "all_similarities":
            Xvecs, y = data_vecs[subset]
            for v1, v2 in Xvecs:
                feat = feature_function(v1, v2)
                X.append(feat)
        new_data_arrays[subset] = (np.array(X).reshape(-1, 1), np.array(y))
    return new_data_arrays



def load_vectors(vector_type, path_to_vectors):
    vectors = []
    if vector_type == "w2v":
        vectors = Magnitude(path_to_vectors, normalized=False)
    elif vector_type == "fasttext":
        vectors = Magnitude(path_to_vectors, normalized=False)
    elif vector_type == "bert":
        vectors = pickle.load(open(path_to_vectors, "rb"))

    return vectors


def join_similarities(arrays_by_similarity):
    new_data_arrays = dict()
    similarity_names = list(arrays_by_similarity.keys())
    for subset in ["train", "dev", "test"]:
        X, y = [], []
        new_data_arrays[subset] = []
        for i in range(len(arrays_by_similarity[similarity_names[0]][subset][0])):
            one_instance = []
            y.append(arrays_by_similarity[similarity_names[0]][subset][1][i])
            for similarity in similarity_names:
                value = float(arrays_by_similarity[similarity][subset][0][i])
                one_instance.append(value)
            X.append(one_instance)
        new_data_arrays[subset] = (np.array(X), y)

    return new_data_arrays


def run_classification(data_arrays, classifier_type="logreg"):
    if classifier_type == "logreg":
        classifier = LogisticRegression(solver="lbfgs")
        classifier.fit(data_arrays['train'][0], data_arrays['train'][1])
    return classifier



def run_predictions(data_arrays, classifier, results, predictions, composition_type, combi, layer, feature_name, k):
    results[k][composition_type][combi][layer][feature_name] = dict()
    predictions[k][composition_type][combi][layer][feature_name] = dict()
    for subset in ['dev', 'test']:
        preds = classifier.predict(data_arrays[subset][0])
        predictions[k][composition_type][combi][layer][feature_name][subset] = preds
        ##### evaluate
        acc = accuracy_score(data_arrays[subset][1], preds)
        f1 = f1_score(data_arrays[subset][1], preds)
        prec = precision_score(data_arrays[subset][1], preds)
        rec = recall_score(data_arrays[subset][1], preds)
        results[k][composition_type][combi][layer][feature_name][subset] = dict()
        results[k][composition_type][combi][layer][feature_name][subset]['accuracy'] = acc
        results[k][composition_type][combi][layer][feature_name][subset]['f1'] = f1
        results[k][composition_type][combi][layer][feature_name][subset]['recall'] = rec
        results[k][composition_type][combi][layer][feature_name][subset]['precision'] = prec

    return results, predictions


def load_folds():
    fn = "HVD_split.csv"
    data = defaultdict()
    with open(fn) as f:
        for l in f:
            noun, adjective, subset = l.strip().split("\t")
            subset = int(subset[-1]) if subset != "dev" else subset
            data[(noun, adjective)] = subset
    return data



def organize_data_byfold(fold_info, data, test_fold):
    new_data = dict()
    for subset in ["train","test","dev"]:
        new_data[subset] = []
    for instance in data:
        fold = fold_info[(instance.noun, instance.adjective)]
        if fold == test_fold:
            new_subset = "test"
        elif fold == "dev":
            new_subset = "dev"
        else:
            new_subset = "train"
        new_data[new_subset].append(instance)
    return new_data


def pprint_configuration(config):
    composition, combination, layer, features = config
    to_print = "\n- composition type: " + composition
    to_print += "\n- words used: " + combination[0] + " and " + combination[1]
    to_print += "\n- layer (ignore if w2v or fasttext): " + str(layer)
    to_print += "\n- type of feature: " + features
    return to_print




def determine_best_configuration(results):
    configs = set()

    for ct in results[1]:
        for cmb in results[1][ct]:
            for l in results[1][ct][cmb]:
                for ft in results[1][ct][cmb][l]:
                    configs.add((ct, cmb, l, ft))

    metrics = ["accuracy","f1","precision","recall"]    
    av_metrics_by_config = dict()
    for m in metrics:
        av_metrics_by_config[m] = dict()
    for config in configs:
        ct, cmb, l, ft = config
        dev_ms = dict()
        test_ms = dict()
        for m in metrics:
            av_metrics_by_config[m][config] = dict()
            dev_ms[m] = []
            test_ms[m] = []
        for f in results:
            for m in metrics:
                dev_ms[m].append(results[f][ct][cmb][l][ft]['dev'][m])
                test_ms[m].append(results[f][ct][cmb][l][ft]['test'][m])

        av_dev_ms = dict()
        av_test_ms = dict()
        for m in metrics:
            av_dev_ms[m] = np.average(dev_ms[m])
            av_test_ms[m] = np.average(test_ms[m])
            av_metrics_by_config[m][config]['dev'] = av_dev_ms[m]
            av_metrics_by_config[m][config]['test'] = av_test_ms[m]

        
    # tell me the config with highest dev f1
    
    best_config = max([(config, av_metrics_by_config['accuracy'][config]['dev']) for config in av_metrics_by_config['accuracy']], key=itemgetter(1))[0]
    print("The best configuration was:", pprint_configuration(best_config))
    print("\nResults with the best configuration:")
    for m in av_metrics_by_config:
        print(m.upper())
        print("dev",m + ":", av_metrics_by_config[m][best_config]['dev'])
        print("test",m + ":", av_metrics_by_config[m][best_config]['test'])

    return av_metrics_by_config



def run_baseline_model(data, results, baseline_model):
    test_truth = [ins.typ for ins in data["test"]]
    if baseline_model == "majority":
        mostfreqlabel_train = mode([ins.typ for ins in data["train"]])
        test_predictions = [mostfreqlabel_train] * len(data["test"])
    elif baseline_model == "allproto":
        test_predictions = ['YES'] * len(data["test"])
    acc = accuracy_score(test_truth, test_predictions)
    f1 = f1_score(test_truth, test_predictions, pos_label="YES")
    prec = precision_score(test_truth, test_predictions, pos_label="YES")
    rec = recall_score(test_truth, test_predictions, pos_label="YES")
    results[k]['accuracy'] = acc
    results[k]['f1'] = f1
    results[k]['precision'] = prec
    results[k]['recall'] = rec
    return results


similarityfuncs = dict()
similarityfuncs["cosine"] = cosine_similarity
similarityfuncs["euclidean"] = euclidean_distance


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_type", type=str,
                        help="can be 'w2v','fasttext' or 'bert'. It can also be 'majority' for the majority baseline or 'allproto' for the all proto baseline.")
    parser.add_argument("--path_to_vectors", type=str)
    parser.add_argument("--baseline", default='', type=str, help="it can be 'majority' (for the majority baseline) or 'allproto' for the baseline that predicts everything to be prototypical")

    args = parser.parse_args()

    if args.vector_type:
        if args.vector_type in ["w2v", "fasttext"]:
            combis = [("n", "a")]
            layers = [0]
        elif args.vector_type == "bert":
            combis = [("n1", "n2"), ("n2", "a2"), ("n1","a2")] # n1 is N_s_n, n2 is N_s_an, a2 is A_s_an
            layers = range(1, 13)


    vectors = load_vectors(args.vector_type, args.path_to_vectors)
    ori_data = load_data()
    all_composition_types = ["adj", "average", "concatenation", "difference", "mul", "sum"]
    all_similarities = similarityfuncs.keys()

    results = dict()
    predictions = dict()

    # Run the training loop for each CV split
    fold_info = load_folds()
    different_folds = set(fold_info.values())
    for k in different_folds:
        if k == "dev":
            continue
        test_fold = k
        data = organize_data_byfold(fold_info, ori_data, test_fold)
        results[k] = dict()
        predictions[k] = dict()

        if args.baseline in ["majority", "allproto"]:
            results = run_baseline_model(data, results, args.baseline)
            continue

        for composition_type in all_composition_types:
            results[k][composition_type] = dict()
            predictions[k][composition_type] = dict()
            for combi in combis:  # this refers to a combination of words (noun1-noun2, etc)
                if combi not in results[k][composition_type]:
                    results[k][composition_type][combi] = dict()
                    predictions[k][composition_type][combi] = dict()
                for layer in layers:
                    if layer not in results[k][composition_type][combi]:
                        results[k][composition_type][combi][layer] = dict()
                        predictions[k][composition_type][combi][layer] = dict()

                    ##### 1. Classification with the composed representation as a feature
                    data_arrays, data_vecs = get_arrays(data, composition_type, args.vector_type, vectors, combi, layer)
                    # data_arrays are used for classification, data_vecs are the vectors that I extracted according to this composition type and combination.

                    # Run classification and prediction, store results
                    classifier = run_classification(data_arrays)
                    results, predictions = run_predictions(data_arrays, classifier, results, predictions, composition_type, combi, layer, feature_name="representation", k=k)

                    ##### 2. Classification using individual similarities as features.
                    # We calculate the similarity between the first vector in the pair and the composed vector.
                    # For example, if combi == ("n1", "n2") and composition_type == "sum", we calculate the similarity between n1 and n1+n2.
                    arrays_by_similarity = dict()
                    if composition_type == "concatenation":  # incompatible
                        continue

                    for similarity in all_similarities:
                        data_arrays = get_simfeature_from_arrays(data_vecs, similarityfuncs[similarity])
                        arrays_by_similarity[similarity] = data_arrays
                        # Run classification and prediction for this type of similarity, store results
                        classifier = run_classification(data_arrays)
                        results, predictions = run_predictions(data_arrays, classifier, results, predictions, composition_type, combi, layer, feature_name=similarity, k=k)


                    ###### 3. Classification using more than one similarity
                    data_arrays = join_similarities(arrays_by_similarity)
                    # Run classification and prediction for the similarities combination, store results
                    classifier = run_classification(data_arrays)
                    results, predictions = run_predictions(data_arrays, classifier, results, predictions, composition_type, combi, layer, feature_name="all_similarities", k=k)


    if args.baseline not in ["majority","allproto"]:
        best_metrics_by_config = determine_best_configuration(results)
    else: # no need to choose a configuration on the dev set, just output the averages
        for metric in ["accuracy", "f1","precision","recall"]:
            av_metric = np.average([results[k][metric] for k in results])
            print("average " + metric + ":", av_metric)

    model_type = args.baseline if args.baseline else args.vector_type
    pickle.dump(results, open("CVresults/results_"+ model_type+".pkl", "wb"))
    pickle.dump(predictions, open("CVresults/predictions_"+ model_type + ".pkl", "wb"))