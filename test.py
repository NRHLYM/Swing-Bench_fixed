from argparse import ArgumentParser
def main(apply_patch):
    print(apply_patch)
parser = ArgumentParser(
    description="Run evaluation harness for the given dataset and predictions.",
)

parser.add_argument(
    "--apply_patch",
    help="Whether to apply patch during evaluation",
    action="store_true",
)
args = parser.parse_args()
main(**vars(args))