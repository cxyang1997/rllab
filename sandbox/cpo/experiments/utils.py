import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CPO_version', default=None,
                        help='give the CPO algorithm version') 
    return parser

def get_args():
    return get_parser().parse_args()

