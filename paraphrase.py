import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=10, metavar='NS',
                        help='num samplings (default: 1)')
    parser.add_argument('--input-file', type=str, default='input.txt', metavar='IF',
                        help='input file with source phrases (default: "input.txt")')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: '')')
    args = parser.parse_args()

    assert os.path.exists('saved_models/trained_RVAE_' + args.model_name), \
        'trained model not found'

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)
    rvae = RVAE(parameters)
    rvae.load_state_dict(t.load('saved_models/trained_RVAE_' + args.model_name))
    if args.use_cuda:
        rvae = rvae.cuda()

    with open(args.input_file) as f:
        source_phrases = f.readlines()
    source_phrases = [x.strip() for x in source_phrases]

    for input_phrase in source_phrases:
        # embed
        print('input: ', input_phrase)
        print('sampled: ')
        for iteration in range(args.num_sample):
            print(rvae.conditioned_sample(input_phrase, batch_loader, args))
            print()
            