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
sns.set(font_scale=1.5, style="whitegrid", rc={'text.usetex' : False, 'lines.linewidth': 3})
# sns.set_theme()
# sns.set_style('whitegrid')
import glob

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

def select_min(df):

    n_layers = len(quant.columns.levels[1])
    #columns = quant.columns.name
    indices = np.zeros(n_layers, dtype=int)
    Idx = pd.IndexSlice

    for idx in range(n_layers):
        # replace NaN with 0
        val_min = df.loc[:, Idx["error", :, idx]].min()  # for all widths
        mask = df.loc[:, Idx['error', :, idx]] == val_min
        indices[idx] = df.loc[mask, Idx["loss", :, idx]].idxmin()  # if several min, take the min of them
        # the indices for the try that has the minimum training
        # error at the epoch epoch

    # remove the column index 'try'
    cols = pd.MultiIndex.from_product(df.columns.levels, names=df.columns.names)  # all but the try
    df_min = pd.DataFrame(columns=cols, index=[1])
    df_min.index.name = 'step'

    for idx in range(n_layers):
        # select the try that has the minimum training error at the
        # last epoch (at index indices[idx])
        df_min.loc[1, Idx[idx, :]] = df.loc[indices[idx],Idx[idx, :]]#.droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('layer') == 'last'] = df.xs(('last', idx_last), axis=1, level=[2, 3], drop_level=False).droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('stat') == 'err'] *= 100
    #df_min = df_min.loc[pd.IndexSlice[:, df_min.columns.get_level_values('layer').isin(range(1, n_layers+1))]]

    #if not df.loc[epoch,  ('train', 'err', 1, indices[0] )] == 0:
    #    print('Not separated!', dirname)
    #else:
    return df_min

    #    print('Separated!', dirname)

def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    process_df(quant, args.path_output)
    return

def process_df(quant, dirname, args=None, args_model=None, save=True, quant_ref=None):

    global table_format
    Idx = pd.IndexSlice

    if quant.columns.names != ["stat", "set", "layer", "width"]:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="var").pivot(index="var", columns=["stat", "set", "layer", "width"], values="value")

    # output_root = os.path.join(dirname, f"merge", uid) if len(uid) > 0 else dirname
    output_root = dirname
    os.makedirs(output_root, exist_ok=True)
    idx = pd.IndexSlice
    cols_error = idx['error', :, :, :]
    N_L = len(quant.columns.unique(level="layer")) -1 # number of hidden layers
    errors = quant["error"]
    losses = quant["loss"]

    if save:
        quant.to_csv(os.path.join(output_root, 'min.csv'))


    quant.sort_index(axis=1, inplace=True)
    quant.loc[:, cols_error] *= 100  # in %
    quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe().to_csv(os.path.join(output_root, 'describe.csv'))
    #csvlosses.to_csv(os.path.join(output_root, 'losses.csv'))
    #errors.to_csv(os.path.join(output_root, 'errors.csv'))

    #quant_describe = pd.DataFrame(group.describe().rename(columns={'value': name}).squeeze()
    #                              for (name, group) in quant.groupby(level=["stat", "set"], axis=1))
    #quant_describe.to_csv(os.path.join(output_root, 'describe.csv'))

    #fig=plt.figure()
    #f, axes = plt.subplots(1, 2, figsize=[10., 5.])
    if args is not None and args.xlim is not None:
        idx_width = quant.columns.names.index("width")
        if len(args.xlim)==2:
            quant = quant.loc[:, args.xlim[0] <= quant.columns.get_level_values(idx_width) <= args.xlim[1]]
        elif len(args.xlim)==1:
            quant = quant.loc[:, quant.columns.get_level_values(idx_width) <= args.xlim[0]]



    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='var')#.query("layer>0")
    df_plot_no_0 = df_plot.query('layer>0')
    df_plot_0 = df_plot.query('layer==0')
    #relative quantities
    if quant_ref is None:
        quant_ref = quant.loc[:, Idx[:, :, 0, :]].droplevel("layer", axis=1)  # for all the  widths and all the vars
    # N_S = len(quant_ref.columns)  # should be the number of stats
    # quant_ref_val = quant_ref.iloc[np.repeat(np.arange(N_S), N_L)].values
    # quant_rel = (quant.loc[:, Idx[:, :, 1:]] - quant_ref_val).abs()
    #quant_plus = quant.loc[:, Idx[:, :, 1:]] + quant_ref + 1e-10
    #quant_rel /= quant_plus
    #quant_rel *= 2

    # utils.to_latex(output_root, quant.loc[:, Idx[:, :, 1:]], table_format, key_err="error")
    # utils.to_latex(output_root, quant_rel, table_format, key_err="error")

    # df_reset_rel = quant_rel.reset_index()
    # df_plot_rel = pd.melt(df_reset_rel, id_vars="run")

    # palette=sns.color_palette(n_colors=2)  # the two datasets
    palette=sns.color_palette(n_colors=N_L)  # the N_L layers
    # bp = sns.catplot(
        # data = df_plot.query('layer > 0'),
        # #col='log_mult',
        # hue='width',
        # dodge=False,
        # row='stat',
        # col='set',
        # #col='log_mult',
        # x='layer',
        # y='value',
        # #ax=axes[0],
        # #kind='line',
        # #ylabel='%',
        # #ci=100,
        # #col_wrap=2,
        # #facet_kws={
        # #    'sharey': False,
        # #    'sharex': True
        # #}
    # )
    # bp.axes[0,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["train"][0].iloc[0].values], color="red")

    # bp.axes[0,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["test"][0].iloc[0].values], color="red")

    # bp.axes[0,0].set_title("Error")
    # bp.axes[0,0].set_ylabel("error (%)")

    # # bp2 = sns.boxplot(
        # # data = df_plot.query('layer >0 & stat =="loss"'),
        # # x="layer",
        # # hue="width",
        # # doge=False,
        # # y="value",
        # # ax=axes[1]
    # # )



    # bp.axes[1,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["train"][0].iloc[0].values], color="red", label="full network")
    # bp.axes[1,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["test"][0].iloc[0].values], color="red", label="full network")
    # bp.axes[1,0].set_title("Loss")
    # bp.axes[1,0].set_ylabel("loss")
    # #plt.legend()
    # #f.legend()
    # bp.fig.subplots_adjust(top=0.85, left=0.10)
    # plt.savefig(fname=os.path.join(output_root, 'boxplot.pdf'))

    # rp = sns.relplot(
        # data = df_plot.query('layer > 0'),
        # #col='log_mult',
        # hue='width',
        # col='set',
        # row='stat',
        # # row='stat',
        # #col='log_mult',
        # x='layer',
        # y='value',
        # #style='event',
        # markers=True,
        # #ax=axes[0],
        # kind='line',
        # legend="auto",
        # #"full",
        # #ylabel='%',
        # #ci=100,
        # #col_wrap=2,
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True,
            # 'legend_out':True,
        # }
    # )


    # rp.axes[0,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["train"][0].iloc[0].values], color="red")

    # rp.axes[0,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[errors["test"][0].iloc[0].values], color="red")

    # rp.axes[0,0].set_title("Error")
    # rp.axes[0,0].set_ylabel("error (%)")

    # # rp.axes[1,0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["train"][0].iloc[0].values], color="red", label="full network")
    # # rp.axes[1,1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[losses["test"][0].iloc[0].values], color="red", label="full network")
    # rp.axes[1,0].set_title("Loss")
    # rp.axes[1,0].set_ylabel("loss")

    # #rp.axes[0,1].legend()
    # #plt.legend()
    # rp.fig.legend()
    # #rp.fig.subplots_adjust(top=0.9, left=1/rp.axes.shape[1] * 0.1)
    # rp.fig.subplots_adjust(top=0.85, left=0.10)
    # if args_model is not None and args is not None:
       # removed = "width / {}".format(args.fraction) if hasattr(args, 'fraction') and args.fraction is not None else args.remove
       # rp.fig.suptitle('ds = {}, width = {}, removed = {}, var = {}'.format(args_model.dataset, args_model.width, removed, args.ntry))
    # #rp.set(yscale='log')
    # #rp.set(ylabel='%')
    # plt.savefig(fname=os.path.join(output_root, 'relplot.pdf'))

    # rel_error = pd.DataFrame()
    # rel_losses = pd.DataFrame()
    # for W in quant.columns.levels[2]:  # for each width
        # idx_col = (errors.columns.get_level_values("layer") > 0) & (errors.columns.get_level_values("width") == W)
        # rel_error = pd.concat([rel_error, abs(errors.loc[:, idx_col] - errors[0][W][1]) / errors[0][W][1]], axis=1, ignore_index=False)
        # rel_losses = pd.concat([rel_losses,  abs(losses.loc[:, idx_col] - losses[0][W][1]) / losses[0][W][1]], axis=1, ignore_index=False)

    # #rel_error_plot = pd.melt(rel_error.reset_index(), id_vars="var")#, id_vars="steps")
    # #rel_losses_plot = pd.melt(rel_losses.min(axis=0).reset_index(), id_vars="layer")#, id_vars="var")

    df_plot = pd.melt(df_reset, id_vars='var')#.query("layer>0")
    #errors_plot = pd.melt(errors.reset_index(), id_vars="var").query("layer>0")
    #losses_plot = pd.melt(losses.reset_index(), id_vars="var").query("layer>0")
    cols = ["stat", "set", "layer", "width"]
    # plt.figure()

    # if N_L == 1:
        # col = "stat"
        # col_order = ["loss", "error"]
        # row="layer"
        # row_order =[1]
    # else:
        # col = "layer"
        # col_order=range(1, N_L+1)
        # row ="stat"
        # row_order = ["loss", "error"]

    # #lp = rel_losses.min(axis=0).plot(kind='line', hue='width', x='layer')
    # mp = sns.relplot(
        # #data=rel_losses.min(axis=0).to_frame(name="loss"),
        # # data=df_plot_rel, #df_plot.pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
        # data=df_plot.pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
        # # style="layer",
        # row=row,
        # row_order = row_order,
        # #row="stat",
        # #col_order=["train", "test"],
        # col=col,
        # col_order=col_order,
        # x="width",
        # y="value",
        # kind='line',
        # legend="full",
        # # legend_out=True,
        # palette=palette,
        # hue='set',
        # hue_order=["train", "test"],
        # # style_order=["],
        # markers=True,
        # facet_kws={
            # 'legend_out': True,
            # 'sharey': 'row' if (N_L>1) else False ,
            # 'sharex': True
        # }
        # #y="value",
    # )

    # # mp.fig.set_size_inches(10, 10)
    # if args_model is not None:
        # mp.fig.suptitle("(B) FCN {}".format(args_model.dataset.upper()))

    # mp.legend.set_title("Datasets")
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False)

    # xlabels=[str(i) for i in range(N_W)]
    is_vgg=False
    dataset="MNIST"
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["loss","error" ]):
        for j, setn in enumerate(["train", "test"]):
            if stat == "loss" and setn=="test":
                continue
            if stat == "error" and setn=="train":
                continue
            # axes[k] = rp.axes[j,i]
            # ax = axes[k]
            plt.figure()
            fig,ax = plt.subplots(1,1,figsize=(4,4))

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot= pd.melt(quant.loc[:, Idx[stat, setn, 1:, :]].reset_index(), id_vars="var")#.min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="layer",
                # hue_order=["A", "B"],
                x="width",
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
            # widths = quant.columns.get_level_values("width").unique()
            # b = widths[-1]
            # p = int(math.log(b, 10))
            # k = int(math.floor(b / (10**(math.floor(math.log(b, 10))))))

            # xticks=[widths[0]] + [i * 10**p for i in range(1,k)] + [widths[-1]]
            # lp.set(xticks=xticks)
            # lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # lp.set_xticklabels(xlabels, rotation=30*(is_vgg))
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"


            # ax.set_xlabel("width")
            ax.tick_params(labelbottom=True)
            df_ref = quant_ref[stat,setn]

            sns.lineplot(data=pd.melt(df_ref.reset_index(), id_vars="var"),
                         ax=ax,
                         # hue='layer',
                         # hue_order=["train", "test"],
                         # alpha=0.5,
                         x="width",
                         y="value",
                         legend=False,
                         )
            # ax.plot(df_ref, c='g', ls=':')
            ax.set_ylabel(None)

            for l in ax.lines[-1:]:
                l.set_linestyle(':')
                l.set_color('g')

            if k == 1:
                fig.legend(handles=ax.lines, labels=["1", "2", "Ref."], title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)
            k+=1

            plt.savefig(fname=os.path.join(output_root, f"{setn}_{stat}.pdf"), bbox_inches='tight')
    # fig.subplots_adjust(top=0.85)
    # fig.legend(ax.lines, labels=["A", "B", "Reference"], title="Experiment", loc="center right")

    # palette=sns.color_palette(n_colors=2)  # the two experiments
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))

    # xlabels=[str(i) for i in range(N_W)]
    is_vgg=False
    dataset="MNIST"
    # fig.suptitle("{} {}".format('VGG' if is_vgg else 'FCN', dataset.upper()))
    k = 0

    #rp.set_axis_labels("layer", "Loss", labelpad=10)
    #quant.loc[1, Idx["loss", :, 0]].lineplot(x="layer_ids", y="value", hue="")
    for i, stat in enumerate(["error"]):
        for j, setn in enumerate(["train"]):#, "test"]):
            if stat == "loss" and setn=="test":
                continue
            # axes[k] = rp.axes[j,i]
            # ax = axes[k]
            ax = axes

            # df_plot = quant.loc[:, Idx[:, stat, setn, :]].min(axis=0).to_frame(name="value")
            df_plot= pd.melt(quant.loc[:, Idx[stat, setn, 1:, :]].reset_index(), id_vars="var")
            # df_plot= quant.loc[:, Idx[stat, setn, 1:, :]].min(axis=0).to_frame(name="value")
            lp = sns.lineplot(
                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="var", columns=col_order).min(axis=0).to_frame(name="value"),
                data=df_plot,
                #hue="width",
                hue="layer",
                # hue_order=["A", "B"],
                x="width",
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
            widths = quant.columns.get_level_values("width").unique()
            b = widths[-1]
            p = int(math.log(b, 10))
            k = int(math.floor(b / (10**(math.floor(math.log(b, 10))))))

            xticks=[widths[0]] + [i * 10**p for i in range(1,k)] + [widths[-1]]
            lp.set(xticks=xticks)
            # lp.set_xticklabels(xticks)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # rp.axes[0,1].locator_params(axis='x', nbins=len(xlabels))

            # lp.set_xticklabels(xlabels, rotation=30*(is_vgg))
            ax.set_title("{} {}{}".format(setn.title(), stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"


            ax.set_xlabel("width")
            df_ref = quant_ref[stat,setn]

            sns.lineplot(data=pd.melt(df_ref.reset_index(), id_vars="var"),
                         ax=ax,
                         # hue='layer',
                         # hue_order=["train", "test"],
                         # alpha=0.5,
                         x="width",
                         y="value",
                         legend=False,
                         )
            # ax.plot(df_ref, c='g', ls=':')
            ax.set_ylabel(None)

            for l in ax.lines[-1:]:
                l.set_linestyle(':')
                l.set_color('g')

            k+=1

    # fig.subplots_adjust(top=0.85)
    fig.legend(handles=ax.lines, labels=["1", "2", "Ref."], title="Layer", bbox_to_anchor=(0.9,0.8), borderaxespad=0.)#, bbox_transform=fig.transFigure)
    # fig.legend(ax.lines, labels=[], title="Experiment", loc="center right")
    plt.margins()
    plt.savefig(fname=os.path.join(output_root, "error_train.pdf"), bbox_inches='tight')

    # for i in range(N_L):

    # or i, stat in enumerate(["loss", "error"]):
        # for j, setn in enumerate(["train", "test"]):
            # if stat == "loss" and setn=="test":
                # continue
            # axes[k] = rp.axes[j,i]

            # ax.plot((0,quant_ref[stat, i][0]), slope=0, ls=":", zorder=2, c='g')

            # sns.lineplot(
                # #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel.query(f"stat=='{stat}' & layer=={i+1}").pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
                # #hue="width",
                # hue="set",
                # hue_order=["train", "test"],
                # #col="stat",
                # #col_order=["loss", "error"],
                # x="width",
                # y="value",
                # #kind='line',
                # #legend="full",
                # # style='layer',
                # legend=False,#'brief',
                # ax=ax_loss,
                # alpha=0.5,
                # #style='layer',
                # #markers=['*', '+'],
                # # dashes=[(2,2),(2,2)],
                # # legend_out=True,
            # )
            # mp.axes[0,1].set_ylabel("error (%)")
            # mp.axes[0,0].set_title("Train Error")
            # mp.axes[0,1].set_title("Test Error")

            # ax_err = mp.axes[i, 1] if N_L == 1 else mp.axes[1, i]
            # ax_err.set_title(f"Error" + (N_L>1)* f", Layer {i+1}")
            # ax_err.set_ylabel("absolute delta error (%)")
            # # mp.axes[1,1].set_title("Test Loss")

        # for ax in ax_loss.lines[-2:]:  # the last two
            # ax.set_linestyle('--')
        # leg_loss = mp_loss.get_legend()



        # sns.lineplot(
            # #data=rel_losses.min(axis=0).to_frame(name="loss"),
            # data=df_plot_rel.query(f"stat=='error' & layer=={i+1}").pivot(index="var", columns=cols).min(axis=0).to_frame(name="value"),
            # #hue="width",
            # hue="set",
            # hue_order=["train", "test"],
            # #col="stat",
            # #col_order=["loss", "error"],
            # x="width",
            # y="value",
            # #kind='line',
            # #legend="full",
            # # style='layer',
            # # legend='brief',
            # legend=False,
            # ax=ax_err,
            # alpha=0.5,
            # #palette=sns.color_palette(n_colors=N_L),
            # #style='layer',
            # markers=True,
            # # dashes=[(2,2),(2,2)],
        # )
        # # rp.axes[0,1].lines[-1].set_linestyle('--')

        # for ax in ax_err.lines[-2:]:  # the last two + legend
            # ax.set_linestyle('--')

    # mp.add_legend(plt.legend(mp.axes[0,1].lines[-2:], ("min",)))
    # mp_err.legend().set_title("min")
    # # mp.axes[1,1].set_ylabel("loss")
    plt.margins()
    plt.savefig(fname=os.path.join(output_root,f"min_quant_{stat}.pdf"), bbox_inches="tight")

    # plt.figure()

    # lp = sns.lineplot(
        # data=rel_error.min(axis=0).to_frame(name="error"),
        # hue="layer",
        # x="width",
        # y="error",
        # legend="full",
        # palette=sns.color_palette(n_colors=N_L),
        # style='layer',
        # markers=True,
        # #y="value",
    # )
    # lp.axes.set_ylabel("relative difference")
    # lp.axes.set_title("Error")
    # rp.axes[0,1].set_ylabel("error")
    # plt.savefig(fname=os.path.join(output_root, "rel_error.pdf"))



    # only_min = select_min(quant)  # select the draw with minimum error
    # only_min_plot = pd.melt(only_min.reset_index(), id_vars='step')



    # m = sns.relplot(data=only_min_plot.query('layer > 0'),
    # #m = only_min_plot.plot(x='layer', kind='line', y='value')
        # col='stat',
        # x='layer',
        # y='value',
        # kind='scatter',
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True
        # }
    # )
    # plt.savefig(fname=os.path.join(output_root, 'plot_min.pdf'))
    plt.close('all')
    return

def process_csv(file_csv, args=None):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1,2,3], index_col=0)
    file_ref = os.path.join(os.path.dirname(file_csv), "ref.csv")
    quant_ref = pd.read_csv(file_ref, header=[0,1,2], index_col=0)
    for df in [quant, quant_ref]:
        int_idx_lst = [] # the list for int fields
        if "layer" in df.columns.names:
            int_idx_lst += [df.columns.names.index("layer")]
        width_idx = df.columns.names.index("width")
        int_idx_lst += [width_idx]
        stat_idx = df.columns.names.index("stat")
        nlevels = df.columns.nlevels
        # stat_idx = df.columns.names.index("stat")
# dirname = os.path.dirname(file_csv)
        for idx in int_idx_lst:  # parse to int
            if df.columns.get_level_values(idx).dtype != int:  # 0 are the layers
                new_lvl = list(map(int, df.columns.get_level_values(idx)))
                levels = [df.columns.get_level_values(i) if i != idx else new_lvl for i in range(nlevels)]
                cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
                df.columns = cols
        if "err" in df.columns.get_level_values("stat"):
            new_stat_lvl = [s.replace("err", "error") for s in df.columns.get_level_values(stat_idx)]
            # new_stat.sort()
            levels = [df.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
            cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
            df.columns = cols
        df.index.name = "var"
    # if quant.columns.get_level_values(width_idx).dtype != int:  # 0 are the layers
        # new_layer_lvl = list(map(int, quant.columns.get_level_values(width_idx)))
        # levels = [quant.columns.get_level_values(i) if i != layer_idx else new_layer_lvl for i in range(nlevels)]
        # cols = pd.MultiIndex.from_arrays(levels, names=quant.columns.names)
        # quant.columns = cols
    # uid = ''
    process_df(quant, os.path.dirname(file_csv),  args, save=False, quant_ref=quant_ref)
    return

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-o', type=str, help='output root for the results')
    parser.add_argument('--name', default='eval-copy', type=str, help='the name of the experiment')
    parser.add_argument('--vary_name', nargs='*', default=['depth', 'width'], help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--lr_update', type=int, default=0, help='update for the learning rate in epochs')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('-R', '--remove', type=int, default=100, help='the number of neurons to remove at each layer')
    # parser_model = parser.add_mutually_exclusive_group(required=True)
    # parser_model.add_argument('--model', help='path of the model to separate')
    # parser_model.add_argument('--directory', help='path of the directory where results are')
    # parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    # parser_model.add_argument('--csv', help='path of the previous saved csv file')
    # parser_model.add_argument('--width', type=int, help='width for a random network')
    parser.add_argument('--reset_random', action='store_true', help='randomly reset the model')
    parser.add_argument('--xlim', nargs='*', type=int, help='the bounds of the width')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth', type=int, help='the depth for init network')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    # parser.add_argument('--var', type=int, default=20, help='The number of steps to take')
   # parser.add_argument('--log_mult', type=int, default=1, help='The log2 of the mulltiplicative factor')
    parser.add_argument('dirs', nargs='*', help='the directories to process')
    parser.add_argument('--file', help='the csv data file')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))

    if args.file is not None and args.file.endswith(".csv"):
        process_csv(args.file, args)
        exit(0)


    common_dir = os.path.commonpath(args.dirs)
    path_merge = os.path.join(common_dir, 'merge')
    lst_file = glob.glob(os.path.join(common_dir, "**", "eval_copy.pth"), recursive=True)  # all the saved results
    unique_ids = set(list(map(get_parent, lst_file)))

    for uid in unique_ids:
        # id_lst_file = glob.glob(os.path.join(directory, "**", uid, "eval_copy.pth"), recursive=True)
        df_merge = pd.DataFrame()
        path_output = os.path.join(path_merge, uid)
        os.makedirs(path_output, exist_ok=True)

        for directory in args.dirs:

            lst_file = glob.glob(os.path.join(directory, "**", uid, "eval_copy.pth"), recursive=True)  # all the saved results
            # unique_ids = set(list(map(get_parent, lst_file)))


            for f in lst_file:

                chkpt = torch.load(f, map_location=device)
                # rid = int(''.join([c for c in os.path.basename(directory.rstrip('/')) if c.isdigit()]))

                if f.find('run') != -1:
                    fstr = "run"
                elif f.find('var') != -1:
                    fstr = "var"
                else:
                    raise ValueError("Neither 'var' nor 'run' in the name")
                rid = int(''.join([c for c in f[f.find(fstr):].split(os.sep)[0] if c.isdigit()]))
                f_model = os.path.join(os.path.dirname(os.path.dirname(f)), "checkpoint.pth")  # the original model
                chkpt_model =torch.load(f_model, map_location=device)

                #args_model = chkpt_model['args']
                #N_L = args_model.depth
                args_chkpt  = chkpt['args']
                args_model = chkpt_model['args']
                width = args_model.width
                N_L = args_model.depth
                # N_T = args_chkpt.var
                Idx = pd.IndexSlice

                    #index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')

                quant = chkpt['quant'].sort_index(axis=1)
                # quant.index.rename("var", inplace=True)
                # C = len(quant.columns)
                #level_width = C*[width]
                #levels = [[width]] + list(map(list, quant.columns.levels))
                levels = list([[width]] +quant.columns.levels)
                quant.columns = pd.MultiIndex.from_product(levels,
                                                        names= ['width'] + quant.columns.names,
                                                        )
                # quant.columns = pd.MultiIndex.from_arrays([quant.columns.get_level_values(1), quant.columns.get_level_values(0), level_width],
                                                        # quant.columns.names[::-1] + ['width'],
                                                        # )

                quant_min=quant.min(axis=0).to_frame(name=rid).transpose()
                quant_min.index.name = "var"
                if not rid in df_merge.index:  # id already in the index
                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=0)
                elif width not in df_merge.columns.get_level_values("width"):
                    df_merge = pd.concat([df_merge, quant_min], ignore_index=False, axis=1)  # concatenate along the columns
                else:
                    df_merge.update(quant_min)  # update the df
                    # df_merge.update(quant_min)
                # df_merge = pd.concat([df_merge, quant], ignore_index=False, axis=1)
                #df_merge.loc[:, (log_mult, layers, 'loss')] = quant.xs('loss', level=1, axis=1)
                #df_merge.loc[Idx[:, (log_mult, layers, 'error')]] = quant.xs('error', level=1, axis=1)
                #epochs[idx_entry] = epoch


        df_merge.sort_index(axis=1, inplace=True)
        df_merge.to_csv(os.path.join(path_output, 'min.csv'))
        df_merge = df_merge.sort_index(level=0, axis=1)
        dirname = os.path.join(path_merge, uid)
        process_df(df_merge, dirname, args_model=args_model, args=args)

    sys.exit(0)





