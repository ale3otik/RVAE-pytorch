import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == '__main__':

    assert os.path.exists('trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Paraphraser')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=10, metavar='NS',
                        help='num samplings (default: 1)')
    parser.add_argument('--input-file', type=str, default='input.txt', metavar='IF',
                        help='input file with source phrases (default: "input.txt")')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    rvae.load_state_dict(t.load('trained_RVAE'))
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
            
            encoder_word_input_np = np.array([[batch_loader.encode_word(
                batch_loader.word_to_idx[w]) for w in input_phrase.split()]])

            encoder_character_input_np = np.array([[batch_loader.encode_characters(
                [batch_loader.word_to_idx[c] for c in w]) for w in input_phrase.split()]])
            print('word shape ' , encoder_word_input_np.shape())
            print('char shape ' , encoder_character_input_np.shape())

            encoder_word_input = Variable(t.from_numpy(encoder_word_input_np).long())
            encoder_character_input = Variable(t.from_numpy(encoder_character_input_np).long())
            # encode input into distribution parameters
            mu, std = rvae.encode_to_mu_std(encoder_word_input, encoder_character_input)

            # sample N(0, 1)
            z = np.random.normal(size=[1, parameters.latent_variable_size])
            
            # transform into N(mu , std**2)
            z = mu + z * std

            result = rvae.sample(batch_loader, 50, z, args.use_cuda)

            print(result)
            print()