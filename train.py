import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE
from paraphrase import build_paraphrase

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=60000, metavar='NI',
                        help='num iterations (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--model-name', default='', metavar='MN',
                        help='name of model to save (default: '')')
    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    ce_result = []
    kld_result = []

    if args.use_trained:
        rvae.load_state_dict(t.load('saved_models/trained_RVAE_' + args.model_name))
        ce_result = list(np.load('saved_models/ce_result_{}.npy'.format(args.model_name)))
        kld_result = list(np.load('saved_models/kld_result_npy_{}.npy'.format(args.model_name)))

    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate, valid_sample = rvae.validater(batch_loader)

    for iteration in range(args.num_iterations):

        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 100 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy()[0])
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy()[0])
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        # validation
        if iteration % 300 == 0:
            target_sentence, predicted_sentence = valid_sample(args.use_cuda)
            cross_entropy, kld = validate(args.batch_size, args.use_cuda)

            cross_entropy = cross_entropy.data.cpu().numpy()[0]
            kld = kld.data.cpu().numpy()[0]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')
            print('target : ', target_sentence)
            print('sample : ', predicted_sentence)
            print('------------------------------')

            ce_result += [cross_entropy]
            kld_result += [kld]

        # generate sample
        if iteration % 300 == 0:
            source = 'i want to buy a book'
            result = build_paraphrase(source, batch_loader, rvae, args, parameters)
            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print('source : ', source)
            print('sample : ', result)
            print('------------------------------')

        # save model
        if iteration % 1000 == 0 or iteration == (args.num_iterations - 1):
            t.save(rvae.state_dict(), 'saved_models/trained_RVAE_' + args.model_name)
            np.save('saved_models/ce_result_{}.npy'.format(args.model_name), np.array(ce_result))
            np.save('saved_models/kld_result_npy_{}'.format(args.model_name), np.array(kld_result))