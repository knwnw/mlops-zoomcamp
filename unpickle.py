from argparse import ArgumentParser
import pickle


parser = ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

with open(args.path, "rb") as f:
    data = pickle.load(f)

print(data)
