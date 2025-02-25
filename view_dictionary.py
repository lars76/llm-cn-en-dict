import json
import gzip
import argparse
from typing import List, Dict


def print_entry(entry: Dict, detailed: bool = False):
    """Pretty print a dictionary entry."""
    print(f"\nEntry for: {entry['traditional']} ({entry['simplified']})")
    print("-" * 40)

    for reading in entry["readings"]:
        print(f"\nPinyin: {reading['pinyin']}")
        for i, meaning in enumerate(reading["meanings"], 1):
            print(f"  {i}. {meaning}")

    if detailed and entry.get("notes"):
        print("\nNotes:")
        for note in entry["notes"]:
            print(f"  • {note}")

    if detailed and entry.get("related_terms"):
        print("\nRelated terms:")
        for term in entry["related_terms"]:
            print(f"  • {term}")

    if detailed and entry.get("sentences"):
        print("\nExample sentences:")
        for sentence in entry["sentences"]:
            print(f"  • {sentence}")

    print("\n" + "=" * 40)


def read_jsonl(filepath: str, num_entries: int = 5, detailed: bool = True):
    """Read and display entries from a gzipped JSONL file."""
    try:
        count = 0
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if count >= num_entries:
                    break

                try:
                    entry = json.loads(line.strip())
                    print_entry(entry, detailed)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        print(f"\nDisplayed {count} entries from {filepath}")

    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Read and display entries from a gzipped JSONL dictionary file."
    )
    parser.add_argument("file", help="Input JSONL.gz file")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=5,
        help="Number of entries to display (default: 5)",
    )

    args = parser.parse_args()
    read_jsonl(args.file, args.num)


if __name__ == "__main__":
    main()
