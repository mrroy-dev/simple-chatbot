import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_gpt.datasets.conversation_builder import ConversationBuilder
from simple_gpt.datasets.cleaners import TextCleaner
from simple_gpt.datasets.filters import DataFilter
from simple_gpt.datasets.preprocess import Preprocessor


def main():
    parser = argparse.ArgumentParser(description="Prepare conversation data for training")
    parser.add_argument("--input", required=True, help="Path to raw dataset (official format: Messages/OpenAI ChatML with 'messages' key)")
    parser.add_argument("--output", default="data/pairs.json", help="Output path")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "json", "jsonl", "txt", "messages", "sharegpt", "alpaca", "csv"],
                        help="Input format")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--sliding-window", type=int, default=4, help="Context turns")
    parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--no-clean", action="store_true", help="Skip text cleaning")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    if args.format == "auto":
        ext = os.path.splitext(args.input)[1].lower()
        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".txt": "txt",
            ".csv": "csv",
            ".parquet": "parquet",
        }
        args.format = format_map.get(ext, "json")

    preprocessor = Preprocessor(seq_len=args.seq_len)

    if not args.no_clean:
        preprocessor.process(
            input_path=args.input,
            output_path=args.output,
            format=args.format,
            val_split=args.val_split,
            sliding_window=args.sliding_window,
        )
    else:
        builder = ConversationBuilder(seq_len=args.seq_len)
        examples = builder.load_and_build(
            args.input,
            format=args.format,
            sliding_window=args.sliding_window,
        )
        print(f"Loaded {len(examples)} raw examples")

        import json, random
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - args.val_split))
        train = examples[:split_idx]
        val = examples[split_idx:]

        for name, data in [("train", train), ("val", val)]:
            out_path = args.output.replace(".json", f"_{name}.json")
            if not args.output.endswith(".json"):
                out_dir = os.path.dirname(args.output) or "data"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} {name} examples to {out_path}")

    if args.stats:
        import json
        with open(args.output.replace(".json", "_train.json"), "r") as f:
            train_data = json.load(f)
        df = DataFilter()
        stats = df.dataset_stats(train_data)
        print(f"\nDataset stats:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
