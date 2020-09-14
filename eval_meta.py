import numpy as np
import argparse
import torch
import glob
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import utils

def default_meta_dir(names):
    '''Returna the default meta directory name from the names for the experiments'''
    return os.path.join(os.path.commonpath(names), 'eval_meta')

def default_meta_name(names):
    '''Return the default meta name from the different experiment names'''
    return 'default'

def main(argv):

    names = argv.exp_names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64  # require float64 for the OT computations?
    if argv.meta_dir is not None:
        meta_dir = argv.meta_dir
    else:
        meta_dir = default_meta_dir(names)
    if argv.meta_name is not None:
        meta_name = argv.meta_name
    else:
        meta_name = default_meta_name(names)

    meta_dirs = set()
    if not argv.plot_only:
        for name in names:  # for the different experiments
            checkpoints_lst = [f for f in (glob.glob(os.path.join(name, "**", "checkpoint.pth"), recursive=True))]
            for f in checkpoints_lst:
                dirname, basename = os.path.split(f)

                log_fname = os.path.join(dirname, 'logs.txt')
                checkpoint = torch.load(f, map_location=device)
                args = checkpoint['args']
                stats = checkpoint['stats']
                if argv.auto:  # overrides the input parameters
                    meta_dir=os.path.join(os.path.dirname(dirname), 'eval_meta')
                    meta_name='auto'
                archi = utils.parse_archi(log_fname)
                model = utils.construct_FCN(archi)
                norm_weights = utils.get_norm_weights(model)
                #if not 'norm_weights' in stats.keys():
                    # legacy for when the norm of the weights was not tracked
                stats['norm_weights'] = [norm_weights]  # has to be a list

                meta_path = os.path.join(meta_dir, meta_name)
                os.makedirs(meta_path, exist_ok=True)
                meta_fname = os.path.join(meta_path, 'data.npz')
                label = args.name # todo: change to diff arguments name / value
                save_meta(meta_fname, stats, args, label)

                print('Saved meta data to {}'.format(meta_fname))

                meta_dirs.add(meta_fname)





    else:
        for name in names:  # for the different experiments
            meta_dirs = meta_dirs.union(set([f for f in (glob.glob(os.path.join(name, "**", "data.npz"), recursive=True))]))


    for f in meta_dirs:

        dirname, basename = os.path.split(f)

        print('Processing {}...'.format(dirname))
        output_path = dirname
        meta_data = np.load(f, allow_pickle=True)
        meta_train = meta_data['train']
        meta_test = meta_data['test']

        loss_train = meta_train['loss'].ravel()#[select]
        loss_test = meta_test['loss'].ravel()#[select]

        norm_weights = meta_train['norm_weights'].ravel()

        losses_train = defaultdict(list)  # will keep the un raveled elements for the different losses
        losses_test = defaultdict(list)
        for (a, b) in zip(loss_train, loss_test):
            # each element is a dict...
            for key in a.keys():
                losses_train[key].append(a[key])
                losses_test[key].append(b[key])

        for key in losses_train.keys():
            losses_train[key] = np.array(losses_train[key])
            losses_test[key] = np.array(losses_test[key])

        nepochs = meta_train['nepochs']


        if argv.abscisse == 'num_parameters': # if the abscisse is the number of model parameters
            x = meta_train['num_parameters'].ravel()
            order = np.argsort(x)
            x = x[order]
            for key in losses_train.keys():  # assumes a dictionnary
                losses_train[key] = losses_train[key][order]
                losses_test[key] = losses_test[key][order]
            norm_weights = norm_weights[order]
            nepochs = nepochs[order]

        else:
            x = meta_train['label'].ravel()  # should be the same for the test data
        #select = np.where([l.find('gpw') == 0 for l in labels])
        #labels = labels[select]
        common_train = get_common_args(meta_train['args'].ravel())
        common_test = get_common_args(meta_test['args'].ravel())
        diff_train = get_diff_args(meta_train.ravel(), common_train)


        save_dict(common_train, fname=os.path.join(output_path, 'info.txt'))

        for key,val in diff_train.items():  # save every experiments
            trantab = str.maketrans('/', '_')
            fname = '{}.txt'.format(key.translate(trantab))
            save_dict(val, fname=os.path.join(output_path, fname))



        ORDINATES = {
            'zo': 'Classification error',
            'mse': 'Mean squared loss',
            'ce': 'Cross-entropy loss'
        }

        ABSCISSES = {
            'num_parameters': 'Number of parameters',
            'label': 'Experiment'
        }

        title = '{}, {} layers'.format(args.dataset, len(model.main))
        if 'size_max' in common_train.keys():
            size_max = common_train['size_max']
            title += ', training size={}'.format(size_max if size_max is not None else 60000)

        for key in losses_train.keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, losses_train[key], marker='o', label='train')
            ax.plot(x, losses_test[key], marker='x', label='test')
            if argv.xscale == 'log':
                ax.set_xscale('log')
            ax.legend()
            ax.set_xlabel(ABSCISSES[argv.abscisse])
            ax.set_ylabel(ORDINATES[key])
            ax.set_title(title)
            plt.savefig(os.path.join(output_path, '{}.pdf'.format(key)), format='pdf')
            plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, norm_weights, marker='o')
        if argv.xscale == 'log':
            ax.set_xscale('log')
        ax.set_title('Scaled weight norm')
        plt.savefig(os.path.join(output_path, 'norm_weights.pdf'), format='pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, nepochs, marker='o')
        if argv.xscale == 'log':
            ax.set_xscale('log')
        ax.set_title('Training epochs')
        plt.savefig(os.path.join(output_path, 'epochs.pdf'), format='pdf')

        plt.close('all')
        print('Done!')




def save_dict(d, fname):

    with open(fname, 'w') as _file:

        for key, val in d.items():

                _file.write(f'{key}= {val}\n')




def get_common_args(args_arr):
    '''returns the common set of arguments from an array of dictionnaries'''

    keys = set(args_arr[0]).intersection(*[set(args) for args in args_arr])

    ret = dict()
    for k in keys:
        for args in args_arr:
            if args[k] != args_arr[0][k]:
                break
        else:
            ret[k] = args_arr[0][k]

    return ret

def get_diff_args(meta_data, common_args):
    '''return the set of all different arguments for each label'''

    ret = dict()  # will keep the pairs  label,  diff_args
    for label, arg in zip(meta_data['label'], meta_data['args']):

        diff_args = {k: v for k,v in arg.items() if k not in common_args}  # the set of different arguments
        ret[label] = diff_args

    return ret








def save_meta(meta_npz_fname, stats, args, label, loss='zo'):
    '''accumulate the obeservation in a meta file'''


    #N_train = train_array.shape[1]
    #N_test = test_array.shape[1]
    if isinstance(stats['loss_train'], dict):
        loss_train = dict()
        loss_test = dict()
        for key, val in stats['loss_train'].items():
            loss_train[key] = val[-1]
            loss_test[key] = stats['loss_test'][key][-1]
        loss_dtype = ('loss', dict)
    else:
        loss_train, loss_test = stats['loss_train'][-1], stats['loss_test'][-1]
        loss_dtype = ('loss', np.float32)

    norm_weights = stats['norm_weights'][-1]  # should be a list

    num_parameters = stats['num_parameters']

    nepochs = stats['epochs'][-1]

    new_entry_train = np.array([(label, args.__dict__, loss_train, num_parameters, norm_weights, nepochs)],
                               dtype=[('label', 'U25'),
                                      ('args', dict),
                                      loss_dtype,
                                      ('num_parameters', np.int32 ),
                                      ('norm_weights', np.float32),
                                      ('nepochs', np.int32),
                                      ])
    new_entry_test = np.array([(label, args.__dict__, loss_test) ],
                              dtype=[('label', 'U25'),
                                     ('args', dict),
                                     loss_dtype,
                                     ])
    if os.path.isfile(meta_npz_fname):
        meta_data = np.load(meta_npz_fname, allow_pickle=True)

        meta_train = meta_data['train']
        meta_test = meta_data['test']
        #meta_nparam = meta_train['num_parameters']
        labels=meta_train['label']

        #names = meta_train['label'].squeeze()
        if label in labels:
            # we found a duplicate with the same number of parameters
            idx = np.where(label == labels)[0]
            try:
                meta_train[idx] = new_entry_train
                meta_test[idx] = new_entry_test
            except ValueError:
                # the two objects are not compatible, erase the old one
                meta_train = new_entry_train
                meta_test = new_entry_test

        else:

            # the label is not there yet, add it at the end

            meta_train = np.vstack((meta_train, new_entry_train))
            meta_test = np.vstack((meta_test, new_entry_test))
    else:
        meta_train = new_entry_train
        meta_test =  new_entry_test
    #meta_train[exp_name] =
    np.savez_compressed(meta_npz_fname, train=meta_train, test=meta_test)

    return



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='script for generating and saving images')
    parser.add_argument('exp_names',  nargs='*', help='the meta data to plot')

    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-f', '--force', default=False, action='store_true', help='force the computation again')

    parser.add_argument('--auto', action='store_true', help='automatically figure out the different meta groups to evaluate (based on common experiment name)')
    parser.add_argument('--meta_dir', type=str, help='the directory for the meta comparison')
    parser.add_argument('--meta_name', type=str, help='the name for the meta group')
    parser.add_argument('--abscisse', default='num_parameters', choices=['label', 'num_parameters'], help='The choice for the abscisse of the plot')
    parser.add_argument('--xscale', default='log', choices=['log', 'lin'], help='Use a logarithmic scale for the abscinsse')
    parser.add_argument('--plot_only', action='store_true', help='compute the data to be plotted')

    #  parser.add_argument('--dry_run', action='store_true', help='dry run mode (do not call sbatch, only copy the files)')
    #  parser.add_argument('--start_idx', type=int, default=1, help='the number of index to start from')
    #  parser.add_argument('--config', help='the default configuration to start from')
    #  parser.add_argument('--batch_template',  default='slurm/scripts/template.sbatch', help='the template sbatch file')
    #  parser.add_argument('script', help='the training script to use')
    #  parser.add_argument('--force_resume', action='store_true', default=False, help='if true, we resume even runs already resumes')
    #  parser.add_argument('--no-touch_resume', action='store_false', dest='touch_resume',  help='if true, we resume even runs already resumes')
    #  iter_parser  = parser.add_mutually_exclusive_group(required=False)
    filter_list = parser.add_mutually_exclusive_group(required=False)
    filter_list.add_argument('--whitelist', help='whitelisting the suffix')
    filter_list.add_argument('--blacklist', help='blacklisting the suffix')
    argv = parser.parse_args()

    main(argv)


