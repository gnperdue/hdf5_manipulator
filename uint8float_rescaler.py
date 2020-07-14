#!/usr/bin/env python
"""
Rescale uint8 images to 0->1 floats, and restrict the actual range from 0.05 to
0.95 (or some other bound) in order to preserve space for image perturbations.
"""
import argparse
import msg
import hdf5


def get_args_conv_rescale():
    """parse arguments"""
    parser = argparse.ArgumentParser(
        description="HDF5 MANIPULATOR (uint8float_rescaler)",
        usage="./uint8float_rescaler.py <options>"
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
        "--imgnames", action="store", dest="imgnames",
        metavar="[imgs tensor]", required=True,
        help="name of images tensor to be rescaled"
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--low", action="store", dest="low", metavar="[lower bound]",
        required=False, help="lower float bound", default=0.05
    )
    optional.add_argument(
        "--high", action="store", dest="high", metavar="[upper bound]",
        required=False, help="upper float bound", default=0.95
    )

    return parser.parse_args()


if __name__ == '__main__':
    msg.box("HDF5 MANIPULATOR: UINT8FLOAT_RESCALER")
    args = get_args_conv_rescale()
    print(args)
    data = hdf5.load(args.input)
    data[args.imgnames] = data[args.imgnames] * (args.high - args.low) + \
        args.low
    hdf5.save(args.output, data)
