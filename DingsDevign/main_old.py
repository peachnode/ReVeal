import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train, save_after_ggnn
from utils import tally_param, debug

# python main.py --dataset ImageMagick --input_dir 
# /space2/ding/dl-vulnerability-detection/data/ggnn_input/ImageMagick 
# --feature_size 169 --model_type ggnn

# Does Devign need a pretrained model? -- Yes 
# /space2/ding/ReVeal/data/full_experiment_real_data_processed/chrome_debian/full_graph/v1

if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--combined', action='store_true', help="Whether to combine both bugzilla_snykio and reveal data together.")
    parser.add_argument('--model_dataset', type=str, default=None, help="Load model from this dataset")
    args = parser.parse_args()
    assert args.model_dataset != args.dataset, "If set model dataset, then it should be different."
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    if not args.combined:
        model_dir = os.path.join('models', args.dataset)
    else:
        model_dir = os.path.join('models', 'combined')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        if args.combined:
            tmp_path = "/scr/dlvp_local_data/reveal_own/ggnn_input/chrome_debian/chrome_debian-original/"
            dataset.read_dataset(tmp_path+"test_GGNNinput.json", 
                                tmp_path+"train_GGNNinput.json",
                                tmp_path+"valid_GGNNinput.json")
            dataset.initialize_dataset()
            file_combined = open("combined/processed.bin", 'wb')
            pickle.dump(dataset, file_combined)
            file_combined.close()
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        # dataset = DataSet(train_src=os.path.join(input_dir, 'devign', 'train'),
        #                   valid_src=os.path.join(input_dir, 'devign', 'valid'),
        #                   test_src=os.path.join(input_dir, 'devign', 'test'),
        #                   batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
        #                   l_ident=args.label_tag)
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                          valid_src=os.path.join(input_dir, 'valid_GGNNinput.json'),
                          test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    if args.train:
        debug('Total Parameters : %d' % tally_param(model))
        debug('#' * 100)
        model.cuda()
        loss_function = BCELoss(reduction='sum')
        optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        train(model=model, dataset=dataset, max_steps=1000000, dev_every=1024,
            loss_function=loss_function, optimizer=optim,
            save_path=model_dir + '/GGNNSumModel', max_patience=25, log_every=None)
    else:
        # Below is for save_after_ggnn
        args.model_dataset = "chrome_debian"
        if not args.model_dataset:
            print(f"loading model from: models/{args.dataset}/GGNNSumModel-model.bin")
            model.load_state_dict(torch.load(f"models/{args.dataset}/GGNNSumModel-model.bin"))
            model.cuda()
            os.makedirs("output/" + args.dataset, exist_ok=True)
        else:
            print(f"loading model from: models/{args.model_dataset}/GGNNSumModel-model.bin")
            model.load_state_dict(torch.load(f"models/{args.model_dataset}/GGNNSumModel-model.bin"))
            model.cuda()
            os.makedirs("output/" + args.dataset + "/" + args.model_dataset, exist_ok=True)
        if not args.model_dataset:
            save_after_ggnn(model, dataset.initialize_train_batch(), dataset.get_next_train_batch, args.dataset + "/train_GGNNinput_graph")
            save_after_ggnn(model, dataset.initialize_valid_batch(), dataset.get_next_valid_batch, args.dataset + "/valid_GGNNinput_graph")
            save_after_ggnn(model, dataset.initialize_test_batch(), dataset.get_next_test_batch, args.dataset + "/test_GGNNinput_graph")
        else:
            save_after_ggnn(model, dataset.initialize_test_batch(), dataset.get_next_test_batch, args.dataset + "/" + args.model_dataset + "/test_GGNNinput_graph")
        
        args.model_dataset = "combined"    
        if not args.model_dataset:
            print(f"loading model from: models/{args.dataset}/GGNNSumModel-model.bin")
            model.load_state_dict(torch.load(f"models/{args.dataset}/GGNNSumModel-model.bin"))
            model.cuda()
            os.makedirs("output/" + args.dataset, exist_ok=True)
        else:
            print(f"loading model from: models/{args.model_dataset}/GGNNSumModel-model.bin")
            model.load_state_dict(torch.load(f"models/{args.model_dataset}/GGNNSumModel-model.bin"))
            model.cuda()
            os.makedirs("output/" + args.dataset + "/" + args.model_dataset, exist_ok=True)
        if not args.model_dataset:
            save_after_ggnn(model, dataset.initialize_train_batch(), dataset.get_next_train_batch, args.dataset + "/train_GGNNinput_graph")
            save_after_ggnn(model, dataset.initialize_valid_batch(), dataset.get_next_valid_batch, args.dataset + "/valid_GGNNinput_graph")
            save_after_ggnn(model, dataset.initialize_test_batch(), dataset.get_next_test_batch, args.dataset + "/test_GGNNinput_graph")
        else:
            save_after_ggnn(model, dataset.initialize_test_batch(), dataset.get_next_test_batch, args.dataset + "/" + args.model_dataset + "/test_GGNNinput_graph")