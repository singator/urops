import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--test', default=10, type=int, help='not much')

args = parser.parse_args()

if args.test:
  print(args.test)
