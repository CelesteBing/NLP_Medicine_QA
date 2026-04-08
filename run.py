import argparse
from src.ner.run_ner import run_ner
from src.qa.run_qa import run_qa

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["ner", "qa"], required=True)
    parser.add_argument("--question", type=str, default=None)

    args = parser.parse_args()

    if args.task == "ner":
        result = run_ner(args.input)

    elif args.task == "qa":
        result = run_qa(args.input, args.question)

    print(result)

if __name__ == "__main__":
    main()