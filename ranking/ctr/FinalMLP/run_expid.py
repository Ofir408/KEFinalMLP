import os

import numpy as np
from fuxictr.metrics import evaluate_metrics

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path
import json

def get_labels(feature_map, inputs):
    labels = feature_map.labels
    assert len(labels) == 1, "Please override get_labels(), add_loss(), evaluate() when using multiple labels!"
    y = inputs[:, feature_map.get_column_index(labels[0])].to("cuda:0")
    return y.float().view(-1, 1)


def get_y_true(data_generator, feature_map):
    y_true = []
    for batch_data in data_generator:
        y_true.extend(get_labels(feature_map, batch_data).data.cpu().numpy().reshape(-1))
    y_true = np.array(y_true, np.float64)
    return y_true


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    k_tests_results = dict()
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    all_results = list()
    for k in range(1, 10):
        params['k'] = k
        params['gpu'] = args['gpu']
        set_logger(params)
        logging.info("Params: " + print_to_json(params))
        seed_everything(seed=params['seed'])

        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")
        if params["data_format"] == "csv":
            # Build feature_map and transform h5 data
            feature_encoder = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(feature_encoder, **params)
        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        logging.info("Feature specs: " + print_to_json(feature_map.features))

        model_class = getattr(src, params['model'])
        model = model_class(feature_map, **params)
        model.count_parameters()  # print number of parameters used in model

        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        model.fit(train_gen, validation_data=valid_gen, **params)

        logging.info('****** Validation evaluation ******')
        valid_result = model.evaluate(valid_gen)
        del train_gen, valid_gen
        gc.collect()

        logging.info('******** Test evaluation ********')
        test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = {}
        if test_gen:
            test_predictions = model.predict(H5DataLoader(feature_map, stage='test', **params).make_iterator())
            test_result = model.evaluate(H5DataLoader(feature_map, stage='test', **params).make_iterator())
            all_results.append(test_predictions)  # [model_predictions_k_1, model_predictions_k_2, ..., ]
        result_filename = Path(args['config']).name.replace(".yaml", "") + "_k=" + str(params['k']) + '.csv'
        print(f"result_filename={result_filename}")
        k_tests_results[k] = test_result["AUC"]
        with open(f"/sise/nadav-group/nadavrap-group/ofir/RS/final_mlp_results/debug/ensemble_results/k_results/{result_filename}", 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                     .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                             ' '.join(sys.argv), experiment_id, params['dataset_id'],
                             "N.A.", print_to_list(valid_result), print_to_list(test_result)))
    all_results = np.array(all_results)

    ensemble_predictions = list()
    for col_idx in range(all_results.shape[1]):
        example_ensemble_prediction = (all_results[:, col_idx]).tolist()
        example_ensemble_final_prediction = np.mean(example_ensemble_prediction)
        ensemble_predictions.append(example_ensemble_final_prediction)

    # test_result = model.evaluate(test_gen)
    test_generator_tmp = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    y_true = get_y_true(data_generator=test_generator_tmp, feature_map=feature_map)
    ensemble_test_results = evaluate_metrics(y_true=y_true, y_pred=ensemble_predictions,
                                             metrics=['logloss', 'AUC'], group_id=None)

    print(f"ensemble_test_results={ensemble_test_results}")
    dataset_id = params['dataset_id']
    ensemble_test_results = ensemble_test_results | k_tests_results
    with open(f"/sise/nadav-group/nadavrap-group/ofir/RS/final_mlp_results/debug/ensemble_results/{dataset_id}.json", "w") as json_file:
        json.dump(ensemble_test_results, json_file, indent=4)

