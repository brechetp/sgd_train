import subprocess

import argparse


def main(argv):


    for dataset in argv['dataset']:


        subprocess.Popen(['sbatch', file_batch])





if __name__ == '__main__':

    parser = argparse.ArgumentParser

    parser.add_argument('')


    argv = parser.parse_args()

    main(argv)
