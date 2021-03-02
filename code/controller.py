from tfl_manager import TFLMan
import argparse


def init():
    args = get_args()
    pkl = None
    frame = []
    with open(args.pls) as f:
        for i, j in enumerate(f):
            if i == 0:
                assert j[-4:] == 'pkl\n'
                pkl = j[:-1]
            elif i == 1:
                assert j[:-1].isdigit()
                offset = int(j[:-1])
            else:
                assert j[-4:] == 'png\n'
                frame.append(j[:-1])

    tfl_manager = TFLMan(pkl)

    return args, tfl_manager, frame, offset


def run():
    _, tfl_manager, frame, offset = init()

    for index in range(len(frame)):
        tfl_manager.on_frame(frame[index], offset + index)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--pls", required=True, help="Path to pls file", type=str)
    
    return parser.parse_args()


