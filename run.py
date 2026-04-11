import argparse
import json
import os
from src.ner.run_ner import run_ner
from src.qa.run_qa import run_qa


def save_result(result: dict, output_path: str) -> None:
    """
    Save result dictionary to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Medical NLP CLI for NER and QA tasks"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input text file"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["ner", "qa"],
        required=True,
        help="Task type: ner or qa"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for QA task"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Path to save the output JSON file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    try:
        if args.task == "ner":
            result = run_ner(args.input)

        elif args.task == "qa":
            if not args.question:
                raise ValueError("QA task requires --question")
            result = run_qa(args.input, args.question)

        else:
            raise ValueError(f"Unsupported task: {args.task}")

        print("\n=== Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        save_result(result, args.output)
        print(f"\nResult saved to: {args.output}")

    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e)
        }
        print("\n=== Error ===")
        print(json.dumps(error_result, ensure_ascii=False, indent=2))

        save_result(error_result, args.output)
        print(f"\nError log saved to: {args.output}")


if __name__ == "__main__":
    main()