import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={
    'text.usetex' : False,
        'lines.linewidth': 3
    }
)


if __name__ == "__main__":

    filename  = "results/cifar10/210331/path/path.csv"
    df = pd.read_csv(filename, header=[0,1,2], index_col=[0,1])
    split = False
    palette=sns.color_palette(n_colors=1)[::-1]
    Idx = pd.IndexSlice
    yscale = "linear"
    logstr = "log" if yscale == "log" else ""
    df_ref = None
    # xlabels =
    output_root = "results/cifar10/210331/path"
    stat_idx = df.columns.names.index("stat")
    nlevels = df.columns.nlevels
    if "err" in df.columns.get_level_values("stat"):
        new_stat_lvl = [s.replace("err", "error") for s in df.columns.get_level_values(stat_idx)]
        # new_stat.sort()
        levels = [df.columns.get_level_values(i) if i != stat_idx else new_stat_lvl for i in range(nlevels)]
        cols = pd.MultiIndex.from_arrays(levels, names=df.columns.names)
        df.columns = cols

    if not split:
        fig, axes = plt.subplots(2, 1, figsize=(4, 8), sharex=False)

    k = 0
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

            df_plot = df.loc[:, (setn,stat)]
            # lp = sns.lineplot(
            df.plot(kind="line",

                #data=rel_losses.min(axis=0).to_frame(name="loss"),
                # data=df_plot_rel if not is_vgg else df_plot_rel.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
                # data=pd.melt(df_plot),
                #hue="width",
                # hue="try",
                # hue_order=keys,
                # x="layer",
                y=(setn,stat),
                legend=None,
                # style='set',
                # ci='sd',
                # palette=palette,
                #style='layer',
                # markers=False,
                ax=ax,
                # dashes=True,
                # linewidth=3.,
                #legend_out=True,
                #y="value",
            )
            # lp.set(xticks=range(0, len(xlabels)))
            # rp.set_xticklabels(xlabels)
            # rp.axes[0,0].locator_params(axis='x', nbins=len(xlabels))
            # lp.set_xticklabels(xlabels)#, rotation=40*(is_vgg))
            # else:
                # lp.set_xticklabels(len(xlabels)*[None])
            ax.set_title("{} {}{}".format(setn.title()+(setn=="train")*"ing", stat.title(), " (%)" if stat=="error" else ''))
            # ylabel = stat if stat == "loss" else "error (%)"
            # ax.set_xlabel("layer index l")
            ax.set_ylabel(None)
            ax.set_yscale(yscale)
            # ax.tick_params(labelbottom=True)


            if df_ref is not None:
                # data_ref  = quant_ref[stat, setn].reset_index()

                ax.axline((0,df_ref[stat, setn][0]), (1, df_ref[stat, setn][0]),  ls=":", zorder=2, c='g')
                # data_ref.index = pd.Index(range(len(data_ref)))
                    # ax=ax,

            if split:
                if k == 1:
                    labels= ["path"]
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
        labels=["path"]
        fig.legend(handles=ax.lines, labels=labels,
                  # title="Exp.",
                   loc="upper right", borderaxespad=0, bbox_to_anchor=(0.9,0.9))#, bbox_transform=fig.transFigure)
        fig.tight_layout()
        # plt.margins()
        fig.savefig(fname=os.path.join(output_root, f"train_loss_test_error{logstr}.pdf"), bbox_inches='tight')
