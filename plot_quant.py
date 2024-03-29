import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from matplotlib import cm
import seaborn as sns
sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={
    'text.usetex' : False,
        'lines.linewidth': 3
    }
)
# sns.set_theme()
# sns.set_style('whitegrid')
import glob
import copy

import math

import models
import random

import torch.optim
import torch
import argparse
import utils

from sklearn.linear_model import LogisticRegression

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x

def process_df(quant, dirname, stats_ref=None, args=None, args_model=None, save=True, split=False):

    global table_format

    col_names = ["experiment", "stat", "set", "layer"]
    quant =  utils.assert_col_order(quant, col_names, id_vars="draw")
    keys = list(quant.columns.levels[0].sort_values())

    output_root = os.path.join(dirname, f"meta_" + "_".join(keys))
    os.makedirs(output_root, exist_ok=True)
    idx = pd.IndexSlice
    cols_error = idx[:, 'error', :, :]
    N_L = len(quant.columns.unique(level="layer"))  # number of hidden layers
    # errors = quant["error"]
    # losses = quant["loss"]
    quant.drop("val", axis=1,level="set", inplace=True, errors='ignore')
    quant.drop(("test", "loss"), axis=1, inplace=True, errors='ignore')

    if save:
        quant.to_csv(os.path.join(output_root, 'merge.csv'))
        if stats_ref is not None:
            stats_ref.to_csv(os.path.join(output_root, 'stats_ref.csv'))


    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    quant.groupby(level=["experiment", "stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))
    #csvlosses.to_csv(os.path.join(output_root, 'losses.csv'))
    #errors.to_csv(os.path.join(output_root, 'errors.csv'))

    #quant_describe = pd.DataFrame(group.describe().rename(columns={'value': name}).squeeze()
    #                              for (name, group) in quant.groupby(level=["stat", "set"], axis=1))
    #quant_describe.to_csv(os.path.join(output_root, 'describe.csv'))


    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='draw')#.query("layer>0")
    # df_plot_no_0 = df_plot.query('layer>0')
    # df_plot_0 = df_plot.query('layer==0')
    #relative quantities
    # quant_ref = quant.loc[1, Idx[:, :, 0, :]]  # for all the  widths
    # N_S = len(quant_ref)
    quant_ref = None
    if stats_ref is not None:
        N_S = len(stats_ref)
        quant_ref_merge = pd.DataFrame()
        stats_ref.loc["error"] = stats_ref["error"].values * 100
        for key in keys:
            quant_ref_merge = pd.concat([quant_ref_merge, quant_ref])
            # N_S_key = len(quant[key].columns.get_level_values("stat").unique())
            N_L_key = len(quant[key].columns.get_level_values("layer").unique())
            quant_ref_key = stats_ref.iloc[np.tile(np.arange(N_S).reshape(N_S, 1), (N_L_key)).ravel()].to_frame(name="value").droplevel("layer")
            quant_ref_merge = pd.concat([quant_ref_merge, quant_ref_key])

        quant_ref = stats_ref.iloc[np.repeat(np.arange(N_S), (N_L))].to_frame(name="value").droplevel("layer").value
        try:
            utils.to_latex(output_root, (quant-quant_ref_merge.value.values).abs(), table_format, key_err="error")
        except:
            pass
    # quant_rel = (quant.loc[:, Idx[:, :, 1:]] - quant_ref_val).abs()
    #quant_plus = quant.loc[:, Idx[:, :, 1:]] + quant_ref + 1e-10
    #quant_rel /= quant_plus
    #quant_rel *= 2

    # utils.to_latex(output_root, quant.loc[:, Idx[:, :, 1:]], table_format, key_err="error")
    # utils.to_latex(output_root, quant, table_format, key_err="error")

    # df_reset_rel = quant_rel.reset_index()
    # df_plot_rel = pd.melt(df_reset_rel, id_vars="draw")



    # rp = sns.relplot(
        # #data=rel_losses.min(axis=0).to_frame(name="loss"),
        # # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
        # data=df_plot.pivot(index="draw", columns=col_names).min(axis=0).to_frame(name="value"),
        # #hue="width",
        # hue="experiment",
        # hue_order=["A", "B"],
        # col="stat",
        # col_order=["loss", "error"],
        # # col_wrap=3,
        # row="set",
        # row_order=["train", "test"],
        # x="layer",
        # y="value",
        # kind='line',
        # legend="full",
        # # style='set',
        # ci='sd',
        # palette=palette,
        # #style='layer',
        # markers=False,
        # dashes=True,
        # #legend_out=True,
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True
        # }
        # #y="value",
    # )

    is_vgg = 'vgg' in dirname
    dataset = 'CIFAR10' if 'cifar' in dirname else 'MNIST'
    # if args_model is not None:

    # if is_vgg:
        # xlabels=["0", "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1", "fc2"]
    # else:
    xlabels=[str(i) for i in range(N_L)]

    logstr = "_log" if args.yscale == "log" else ""

    # if len(keys) <= 2:
        # palette=sns.color_palette(n_colors=2)[2-len(keys):] # the two experiments
    # else:
    if "dropout" in keys:
        keys = keys[::-1]

    palette=sns.color_palette(n_colors=len(keys))[::-1]

    if not split:
        fig, axes = plt.subplots(2, 1, figsize=(4, 8), sharex=False)
    # sns.set(font_scale=1,rc={"lines.linewidth":3})

    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["loss","error" ]):
        for j, setn in enumerate(["train","test"]):
            if stat == "loss" and setn=="test":
                continue
            if stat == "error" and setn=="train":
                continue
            # axes[k] = rp.axes[j,i]
            if split:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False)
            else:
                ax = axes.flatten()[k]

            df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="layer",
                y="value",
                legend=None,
                # style='set',
                ci='sd',
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                # linewidth=3.,
                #legend_out=True,
                #y="value",
            )
            lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            # else:
                # lp.set_xticklabels(len(xlabels)*[None])
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("layer index l")
            ax.set_ylabel(None)
            ax.set_yscale(args.yscale)
            # ax.tick_params(labelbottom=True)


            if quant_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                ax.axline((0,quant_ref[stat, setn][0]), (1, quant_ref[stat, setn][0]),  ls=":", zorder=2, c='g')
                # data_ref.index = pd.Index(range(len(data_ref)))
                    # ax=ax,
            if split:
                if k == 1:
                    labels=keys + ["ref."]
                    fig.legend(handles=ax.lines, labels=labels,
                              # title="Exp.",
                               loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.9))#, bbox_transform=fig.transFigure)

                # fig.tight_layout()
                plt.margins()

                plt.savefig(fname=os.path.join(output_root, f"{setn}_{stat}{logstr}.pdf"), bbox_inches='tight')

            k += 1

    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    if not split:
        labels=keys + ["ref."]
        fig.legend(handles=ax.lines, labels=labels,
                  # title="Exp.",
                   loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.9))#, bbox_transform=fig.transFigure)
        fig.tight_layout()
        # plt.margins()
        fig.savefig(fname=os.path.join(output_root, f"train_loss_test_error{logstr}.pdf"), bbox_inches='tight')
    k=0
    # sns.set(font_scale=1,rc={"lines.linewidth":3})

    fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharex=False)
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))

    for i, stat in enumerate(["error"]):
        for j, setn in enumerate(["train"]):
            if stat == "loss" and setn=="test":
                continue
            if stat=="error" and setn=="test":
                continue
            # axes[k] = rp.axes[j,i]
            ax = axes

            df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="experiment",
                hue_order=keys,
                x="layer",
                y="value",
                legend=None,
                # style='set',
                ci='sd',
                palette=palette,
                #style='layer',
                markers=False,
                ax=ax,
                dashes=True,
                #legend_out=True,
                #y="value",
            )
            lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*'ing', stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            ax.set_xlabel("layer index l")
            ax.set_ylabel(None)
            ax.set_yscale(args.yscale)

            if quant_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                ax.axline((0,quant_ref[stat, setn][0]), (1,quant_ref[stat, setn][0]), ls=":", zorder=2, c='g')
                # data_ref.index = pd.Index(range(len(data_ref)))
                # sns.lineplot(
                    # data=data_ref,  # repeat the datasaet N_L times
                    # ax=ax,
                    # # x=range(len(data_ref)),
                    # # y="value",
                    # # xc np.tile(np.linspace(1, N_L, num=N_L), 2),
                    # # x='',
                    # # hue='r',
                    # # color='red',
                    # palette=['red'],
                    # # style='set',
                    # # x='index',
                    # # dashes=True,
                    # legend=False,
                    # # y="value"
                # )

                # for ax in ax.lines[-1:]:  # the last two
                    # ax.set_linestyle('--')
            k += 1


    # fig.subplots_adjust(top=0.85)
    # if is_vgg:
    labels=keys + ["ref."]
    fig.legend(handles=ax.lines, labels=keys,
               #title="Exp.",
               loc="upper right", bbox_to_anchor=(0.9,0.9),borderaxespad=0)#, bbox_transform=fig.transFigure)
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, f"error_train{logstr}.pdf"), bbox_inches='tight')

    if "B" in keys:
        df_B = quant["B"]
    elif "B2" in keys:
        df_B = quant["B2"]
    else:
        return
    n_draws = len(df_B.index)
    # vary_draw=copy.deepcopy(df_B)
    df_B_plot = pd.melt(df_B.reset_index(), id_vars="draw")
    cp = sns.FacetGrid(
        data=df_B_plot,
        # hue="experiment",
        # hue_order=["A", "B"],
        col="stat",
        col_order=["loss", "error"],
        row="set",
        row_order=["train", "test"],
        # x="layer",
        # y="value",
        # kind='line',
        # legend="full",
        # style='set',
        # ci='sd',
        # palette=palette,
        #style='layer',
        # markers=False,
        # dashes=True,
        #legend_out=True,
        # facet_kws={
        sharey= False,
        sharex= True,
        #y="value",
    )
    styles=['dotted', 'dashed', 'dashdot',  'solid']
    # for i_k, k in enumerate([10, 50, 100, 200]):
    draws = len(df_B.index)
    df_bound = pd.DataFrame(columns=df_B.columns)
    # df_bound.columns = df_B.columns
    for k in range(1, draws+1):
        # df_cut = pd.melt(df_B[:k].reset_index(), id_vars="draw")
        df_bound.loc[k, :] = df_B[:k].min(axis=0)
        # idx_min = df_cut.query('stat=="loss"idxmin")
    fig, axes= plt.subplots(2,2,figsize=(12,12), sharex=True)
    for i, stat in enumerate(["loss", "error"]):
        for j, setn in enumerate(["train", "test"]):
            df_bound_plot = df_bound[stat,setn].max(axis=1)
            ax=axes[i,j]
            ax.set_title("{} {}".format(setn.title(), stat.title()))
            sns.lineplot(
                data=df_bound_plot,
                ax=ax,
            )
                # cp.axes[j,i].set_title("{} {}".format(setn.title(), stat.title()))
    plt.savefig(fname=os.path.join(output_root, "comp_draws.pdf"), bbox_inches='tight')

    plt.close('all')
                # ylabel = stat if stat == "loss" else "error (%)"
                # cp.axes[j,i].set_ylabel(ylabel)
                # cp.axes[j,i].set_xlabel("layer index l")

                # df_cut_plot = pd.melt(df_cut_min.query(f'stat=="{stat}" & set=="{setn}"'))
                # if quant_ref is not None:
                    # data_ref  = quant_ref[stat, setn].reset_index()

                    # data_ref.index = pd.Index(range(len(data_ref)))
                # sns.lineplot(
                    # data=df_cut_plot,  repeat the datasaet N_L times
                    # ax=cp.axes[j,i],
                    # x=range(len(data_ref)),
                    # y="value",
                    # xc np.tile(np.linspace(1, N_L, num=N_L), 2),
                    # x='layer',
                    # hue='r',
                    # color='red',
                    # palette=['red'],
                    # style='set',
                    # x='index',
                    # dashes=True,
                    # legend=False,
                    # y="value"
                # )

                # for ax in cp.axes[j,i].lines[-1:]:  the last two
                    # ax.set_linestyle(styles[i_k])




def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    process_df(quant, os.path.dirname(file_csv))
    return

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument('--experiments', nargs='*', default=['A', 'B'], help='whitelist for the experiments to cat')
    parser.add_argument('--yscale', choices=["linear", "log"], default='linear', help='the scale for the y axis')
    parser.add_argument('dirs', nargs='*', help='the directories to process')
    parser.add_argument('--split', action='store_true', help='split the err/loss figures in two')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))

    def get_grand_parent(path):
        return os.path.dirname(os.path.dirname(path.rstrip(os.sep)))

    for directory in args.dirs:

        lst_file = glob.glob(os.path.join(directory, "**", "quant.csv"), recursive=True)  # all the saved results
        roots = set(map(get_grand_parent, lst_file))

        for root in roots:
            id_lst_file = glob.glob(os.path.join(root, "**", "quant.csv"), recursive=True)
            df_bundle = pd.DataFrame()
            stats_ref = None
            args_model = None
            for f in id_lst_file:
                experiment = get_parent(f)
                if not experiment in args.experiments:
                    continue
                # try:
                    # f_model = torch.load(os.path.join(os.path.dirname(f), "checkpoint.pth"), map_location=device)
                # except IOError as e:
                    # print(f"error {e} for {f}")
                    # f_model = None
                # if f_model:
                    # chkpt_model =torch.load(f_model, map_location=device)
                    # # args_chkpt  = chkpt['args']
                    # args_model = chkpt_model['args']

                Idx = pd.IndexSlice

                # quant = chkpt['quant'].sort_index(axis=1)
                quant = pd.read_csv(f, header=[0,1,2], index_col=0)
                nlevels = quant.columns.nlevels
                layer_idx = quant.columns.names.index("layer")
                stat_idx = quant.columns.names.index("stat")
                if quant.columns.get_level_values(layer_idx).dtype != int:  # 0 are the layers
                    new_layer_lvl = list(map(int, quant.columns.get_level_values(layer_idx)))
                    levels = [quant.columns.get_level_values(i) if i != layer_idx else new_layer_lvl for i in range(nlevels)]
                    cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
                    quant.columns = cols
                if "err" in quant.columns.get_level_values("stat"):
                    new_stat_lvl = [s.replace("err", "error") for s in quant.columns.get_level_values(stat_idx)]
                    # new_stat.sort()
                    levels = [quant.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
                    cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
                    quant.columns = cols

                quant.index.rename("draw", inplace=True)
                quant.sort_index(axis=1, inplace=True)

                if 'B' in experiment or 'dropout' in experiment:
                    stats_ref = quant.loc[1, Idx[:, :, 0]]  #layer = 0 is the reference
                    quant = quant.loc[:, Idx[:, :, 1:]]  # only keep layers > 0

                ncol = len(quant.columns)
                levels = list([ncol*[experiment]] +[quant.columns.get_level_values(i) for i in range(quant.columns.nlevels)])
                # quant.columns = pd.MultiIndex.from_product(levels,
                                                        # names= ['experiment'] + quant.columns.names,
                                                        # )
                quant.columns = pd.MultiIndex.from_arrays(levels,
                                        names= ['experiment'] + quant.columns.names,
                                          )
                                                        # )

                df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)
                #df_bundle.loc[:, (log_mult, layers, 'loss')] = quant.xs('loss', level=1, axis=1)
                #df_bundle.loc[Idx[:, (log_mult, layers, 'error')]] = quant.xs('error', level=1, axis=1)
                #epochs[idx_entry] = epoch


            df_bundle.sort_index(axis=1, inplace=True)
            if not df_bundle.empty:
                process_df(df_bundle, root, stats_ref, args=args, args_model=args_model, split=args.split)

    sys.exit(0)





