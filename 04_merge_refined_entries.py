import json
import os
import re
import gzip
import argparse
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm


def extract_json_from_text(text: str) -> Dict:
    """Extract JSON from model output."""
    try:
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index == -1 or end_index == -1:
            return None
        json_str = text[start_index : end_index + 1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def load_comparison_results(filepath: str) -> Dict[str, Dict]:
    """Load the comparison results from the validated entries file."""
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            results[entry["traditional"]] = entry
    return results


def create_prompt(comparison_entry: Dict) -> str:
    # Extract fields
    traditional = comparison_entry["traditional"]
    cedict_entries = comparison_entry["cedict_entries"]
    llm_entries = comparison_entry["llm_entries"]
    llm_notes = comparison_entry.get("llm_notes", "")
    wikipedia_entry = comparison_entry.get("wikipedia_entry", "")

    prompt_parts = [
        f'Create an improved Chinese->English dictionary entry for "{traditional}" by combining and refining the following source information.',
        "",
        "Source 1 (CEDICT) Definitions:",
        "\n".join(
            f"   {entry['pinyin']} - {entry['meaning']}" for entry in cedict_entries
        ),
        "Source 2 (LLM) Definitions:",
        "\n".join(
            f"   {entry['pinyin']} - {', '.join(entry['meanings'])}"
            for entry in llm_entries
        ),
    ]

    if llm_notes:
        prompt_parts.extend([f"   {llm_notes}", ""])

    if wikipedia_entry:
        wikipedia_text = (
            wikipedia_entry[:500] + "..."
            if len(wikipedia_entry) > 500
            else wikipedia_entry
        )
        prompt_parts.extend(["Source 3 (Wikipedia Entry):", f"   {wikipedia_text}", ""])

    prompt_parts.extend(
        [
            "Requirements:",
            "1. All definitions MUST be in English only",
            "2. Function as a Chinese->English dictionary entry (not an encyclopedia):",
            "   - Focus on word meanings and usage",
            "   - Exclude encyclopedic/biographical information",
            "   - Split compound meanings into separate entries",
            "3. Combine and organize information:",
            "   - Merge similar definitions from different sources",
            "   - List meanings from most common to specialized",
            "   - Include source numbers for each meaning [1,2,3]",
            "4. Notes should focus on:",
            "   - Word usage and context",
            "   - Regional variations",
            "   - Register (formal/informal/colloquial)",
            "   - Etymology when relevant to learners",
            "",
            "Output Format (JSON):",
            """{
    "readings": [
        {
            "pinyin": "...",
            "meanings": ["English definition [source numbers]", ...],
        }
    ],
    "notes": [
        "usage note [source numbers]",
        ...
    ]
}""",
        ]
    )

    return "\n".join(prompt_parts)


def validate_merged_entry(entry: Dict, original: Dict) -> bool:
    """Validate the structure and content of merged entries."""
    try:
        if "readings" not in entry or not entry["readings"]:
            print("Reading not found")
            return False

        # Get original pronunciations from either source (they should be identical)
        source_readings = {e["pinyin"] for e in original["cedict_entries"]}

        # Verify each reading
        for reading in entry["readings"]:
            if not all(field in reading for field in ["pinyin", "meanings"]):
                print("pinyin or meanings not in reading")
                return False

            if reading["pinyin"] not in source_readings:
                print("Invalid pinyin found")
                return False

            if not reading["meanings"] or not all(
                isinstance(m, str) and m.strip() for m in reading["meanings"]
            ):
                print("Empty meaning found")
                return False

        # Check for duplicate pronunciations
        if len([r["pinyin"] for r in entry["readings"]]) != len(
            {r["pinyin"] for r in entry["readings"]}
        ):
            print("Duplicate pronunciation")
            return False

        # Verify all pronunciations are included
        generated_readings = {r["pinyin"] for r in entry["readings"]}
        if generated_readings != source_readings:
            print("Not all pronunciations are included")
            return False

        return True

    except Exception:
        return False


def process_dictionary(
    input_comparison: str, output_file: str, model_name: str, debug: bool = False
):
    """Process and merge validated dictionary entries, skipping already processed entries."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load previously processed entries if output file exists
    processed_entries = set()
    if os.path.exists(output_file):
        try:
            with gzip.open(output_file, "rt", encoding="utf-8", errors="ignore") as f:
                while True:
                    try:
                        line = f.readline()
                        if not line:  # EOF
                            break
                        entry = json.loads(line.strip())
                        processed_entries.add(entry["traditional"])
                    except:  # Skip any problematic entries
                        continue
            print(f"Found {len(processed_entries):,} previously processed entries")
        except:  # If file is completely unreadable, start fresh
            print("Warning: Could not read existing file, starting fresh")
            processed_entries = set()

    print("Loading comparison results...")
    comparison_results = load_comparison_results(input_comparison)
    print(f"Loaded {len(comparison_results):,} validated entries")

    # Calculate entries that need processing
    entries_to_process = {
        k: v for k, v in comparison_results.items() if k not in processed_entries
    }
    print(f"Found {len(entries_to_process):,} entries to process")

    if debug:
        print("\n=== DEBUG MODE: Printing sample prompts ===\n")
        sample_entries = list(entries_to_process.items())[:10]
        for _, entry in sample_entries:
            prompt = create_prompt(entry)
            print(f"\n--- Prompt for: {entry['traditional']} ---")
            print(prompt)
            print("\n" + "=" * 50 + "\n")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Initialize LLM components
    if entries_to_process:
        print(f"Initializing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=0.3, top_p=0.9, top_k=20, max_tokens=1024
        )
        llm = LLM(
            model=model_name,
            quantization="awq_marlin",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=4,
            enforce_eager=False,
        )

    processed_count = len(processed_entries)
    error_count = 0

    # Process entries
    for entry in tqdm(entries_to_process.values(), desc="Processing entries"):
        try:
            prompt = create_prompt(entry)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outputs = llm.generate([text], sampling_params)
            generated_text = outputs[0].outputs[0].text

            merged_entry = extract_json_from_text(generated_text)
            if merged_entry and validate_merged_entry(merged_entry, entry):
                # Check for consistent simplified character mapping
                simplified_chars = {e["simplified"] for e in entry["cedict_entries"]}
                if len(simplified_chars) > 1:
                    print(
                        f"Warning: Multiple simplified mappings found for {entry['traditional']}: {simplified_chars}"
                    )
                    error_count += 1
                    continue

                # Add characters to the merged entry
                merged_entry["traditional"] = entry["traditional"]
                merged_entry["simplified"] = entry["cedict_entries"][0]["simplified"]
                merged_entry["related_terms"] = entry["related_terms"]
                merged_entry["sentences"] = entry["sentences"]

                # Open new file handle for each write, write, and flush
                with gzip.open(output_file, "at", encoding="utf-8") as out_f:
                    json.dump(merged_entry, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    out_f.flush()
                processed_count += 1
            else:
                print(f"Invalid entry generated for: {entry['traditional']}")
                error_count += 1

        except Exception as e:
            print(f"Error processing {entry['traditional']}: {str(e)}")
            print(generated_text)
            error_count += 1

    print(f"\nProcessing complete:")
    print(f"Total processed entries: {processed_count:,}")
    print(f"New entries processed: {len(entries_to_process):,}")
    print(f"Errors encountered: {error_count:,} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Merge validated dictionary entries into a comprehensive dictionary."
    )
    parser.add_argument(
        "--input", required=True, help="Input comparison results file (JSON)"
    )
    parser.add_argument(
        "--output", required=True, help="Output merged dictionary file (gzipped JSONL)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-72B-Instruct-AWQ", help="Model name to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only print prompts without loading LLM",
    )

    args = parser.parse_args()

    try:
        process_dictionary(
            input_comparison=args.input,
            output_file=args.output,
            model_name=args.model,
            debug=args.debug,
        )
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
