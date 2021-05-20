import os
import re
import glob
import argparse




if __name__ == '__main__':


    parser = argparse.ArgumentParser('Removing previous unecessary checkpoints')
    #parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    #parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='eval-copy', type=str, help='the name of the experiment')
    #parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser_model = parser.add_mutually_exclusive_group(required=False)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_model.add_argument('--csv', help='path of the previous saved csv file')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth_max', type=int, help='the maximum depth to which operate')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument("--yscale", choices=["log", "linear"], default="linear", help="the choice of the scale for y")
    parser.add_argument('--xlim', nargs='*', type=int, help='the bounds of the width')
    parser.add_argument('--dry_run', action="store_true", default=False, help="Does not delete the file but list them")
    parser.add_argument('--vlim', nargs=1, type=int, help='the bounds of the variations')
    parser.set_defaults(cpu=False)
    parser.add_argument('dirs', nargs='*', help='the directory to process')



    args = parser.parse_args()
    fnames = []
    for dname in args.dirs:
        fnames.extend(glob.glob(os.path.join(dname, "**", "checkpoint_draw_*.pth"), recursive=True))

    uid_dirs = set(list(map(os.path.dirname, fnames)))

    for dname in uid_dirs:
        chkpt_lst = glob.glob(os.path.join(dname, "checkpoint_draw_*.pth"))

        chkpt_id = map(os.path.basename, chkpt_lst)
        chkpt_id = [int(x.split('.')[0].split('_')[-1]) for x in  chkpt_id]
        chkpt_id.sort(reverse=True)
        if chkpt_id:
            for cid in chkpt_id[1:]:
                if args.dry_run:
                    print(os.path.join(dname, f"checkpoint_draw_{cid}.pth"))
                else:
                    os.remove(os.path.join(dname, f"checkpoint_draw_{cid}.pth"))




