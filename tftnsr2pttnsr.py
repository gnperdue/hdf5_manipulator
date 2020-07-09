#!/usr/bin/env python
"""
Move the tensor axis channels position from -1 to 1 (TF -> PT convention)
"""
import argparse
import msg
import hdf5
import numpy as np


def get_args_tnsr():
    """parse arguments"""
    parser = argparse.ArgumentParser(
        description="HDF5 MANIPULATOR (tftnsr2pttnsr)",
        usage="./tftnsr2pttnsr.py <options>"
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input", action="store", dest="input",
        metavar="[path/to/filename]", required=True,
        help="path to input hdf5 file"
    )
    required.add_argument(
        "--output", action="store", dest="output",
        metavar="[path/to/filename]", required=True,
        help="path to output hdf5 file"
    )
    required.add_argument(
        "--tensor", action="store", dest="tensor",
        metavar="[imgs tensor]", required=True,
        help="name of tensor to be changed"
    )

    return parser.parse_args()


if __name__ == '__main__':
    msg.box("HDF5 MANIPULATOR: TFTNSR2PTTNSR")
    args = get_args_tnsr()
    data = hdf5.load(args.input)
    data[args.tensor] = np.moveaxis(data[args.tensor], -1, 1)
    hdf5.save(args.output, data)
