from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_arguments():
    parser = ArgumentParser(description="TNE: A Latent Model for Representation Learning on Networks",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='deepwalk',
                        help='')
    parser.add_argument('--n', type=int, default=80,
                        help='')
    parser.add_argument('--l', type=int, default=40,
                        help='')
    parser.add_argument('--w', type=int, default=10,
                        help='')
    parser.add_argument('--d', type=int, default=128,
                        help='')
    parser.add_argument('--k', type=int, default=50,
                        help='')
    parser.set_defaults(directed=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(args.directed)