import json
import os
import re
import gzip
import argparse
from typing import List, Set, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def create_prompt(
    traditional: str,
    cedict_entries: List[Tuple[str, str, str]],
    llm_readings: List[dict],
    notes: str,
) -> str:
    """Create prompt for comparing dictionary entries."""
    simplified = cedict_entries[0][0] if cedict_entries else "?"

    # Process CEDICT entries
    cedict_info = []
    for _, pinyin, meaning in cedict_entries:
        cedict_info.append(f"{pinyin} - {meaning}")

    # Process LLM entries
    llm_info = []
    for reading in llm_readings:
        if reading.get("pinyin") and reading.get("meanings"):
            llm_info.append(f"{reading['pinyin']} - {', '.join(reading['meanings'])}")

    prompt_parts = [
        f'Compare these two dictionary entries for "{traditional}" (simplified: {simplified}):',
        "",
        "CEDICT:",
        "\n".join(cedict_info),
        "",
        "LLM:",
        "\n".join(llm_info),
        f"Context: {notes}",
        "",
        "Score two independent aspects:",
        "1. Semantic similarity (0-100%):",
        "   How much do the meanings overlap? Score partial matches proportionally.",
        "   0%: Completely different meanings",
        "   100%: All meanings align",
        "",
        "2. Correctness of LLM entry (0-100%):",
        "   How accurate is the LLM entry based on modern usage and context?",
        "   Consider pinyin accuracy and contemporary meaning.",
        "   High scores possible even with low similarity if LLM is accurate.",
        "",
        "Output JSON format:",
        '{"similarity": X, "correctness": Y, "analysis": "Key difference or verification source"}',
        "",
    ]

    return "\n".join(prompt_parts)


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


def load_cedict_entries(input_file: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Load entries from CEDICT format file."""
    entries = defaultdict(list)
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            reg = re.match(r"(.+?) (.+?) (\[.*\]+?) (\/.*?\/)", line)
            if reg:
                traditional = reg.group(1)
                simplified = reg.group(2)
                pinyin = reg.group(3).strip("[]").replace("][", " ")
                translation = reg.group(4).strip("/")
                entries[traditional].append((simplified, pinyin, translation))
    return entries


def process_dictionary(
    input_cedict: str,
    input_llm: str,
    output_file: str,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
    debug: bool = False,
):
    """Process dictionary entries using vLLM."""
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load dictionaries
    cedict_entries = load_cedict_entries(input_cedict)

    # Load LLM dictionary
    llm_entries = {}
    with gzip.open(input_llm, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                llm_entries[entry["traditional"]] = entry
            except json.JSONDecodeError:
                continue

    # Process all terms
    all_terms = sorted(set(cedict_entries.keys()) & set(llm_entries.keys()))

    if debug:
        print("\n=== DEBUG MODE: Printing prompts only ===\n")
        for term in tqdm(all_terms[:100], desc="Generating prompts", unit="terms"):
            term_data = llm_entries.get(term, {})
            notes = term_data.get("notes", "") if isinstance(term_data, dict) else ""

            prompt = create_prompt(
                term,
                cedict_entries.get(term, []),
                term_data.get("readings", []) if isinstance(term_data, dict) else [],
                notes,
            )
            print(f"\n--- Prompt for term: {term} ---")
            print(prompt)
            print("\n" + "=" * 50 + "\n")
        return

    # Initialize tokenizer and model only if not in debug mode
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

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

    # Load existing processed terms
    processed_terms = set()
    if os.path.exists(output_file):
        with gzip.open(output_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_terms.add(entry["traditional"])
                except json.JSONDecodeError:
                    continue

    with tqdm(total=len(all_terms), desc="Processing terms", unit="terms") as pbar:
        for term in all_terms:
            if term in processed_terms:
                pbar.update(1)
                continue

            try:
                term_data = llm_entries.get(term, {})
                notes = (
                    term_data.get("notes", "") if isinstance(term_data, dict) else ""
                )
                ce_entries = cedict_entries.get(term, [])

                # Get simplified character from CEDICT entries
                simplified = ce_entries[0][0] if ce_entries else "?"

                prompt = create_prompt(
                    term,
                    ce_entries,
                    term_data.get("readings", [])
                    if isinstance(term_data, dict)
                    else [],
                    notes,
                )

                # Prepare for model
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate response
                outputs = llm.generate([text], sampling_params)
                generated_text = outputs[0].outputs[0].text

                # Parse response
                result = extract_json_from_text(generated_text)
                if result:
                    entry = {
                        "traditional": term,
                        "analysis": result.get("analysis", ""),
                        "similarity": result.get("similarity", 0),
                        "correctness": result.get("correctness", 0),
                    }

                    with gzip.open(output_file, "at", encoding="utf-8") as f:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write("\n")

                    processed_terms.add(term)
                else:
                    print(f"Failed to parse output for term {term}")

            except Exception as e:
                print(f"Error processing term {term}: {str(e)}")

            finally:
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge CEDICT and LLM dictionary entries."
    )
    parser.add_argument(
        "--cedict", "-c", required=True, help="Input CEDICT file (gzipped)"
    )
    parser.add_argument(
        "--llm-dict",
        "-l",
        required=True,
        help="Input LLM dictionary file (gzipped JSONL)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output similarity dictionary file (gzipped JSONL)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Model name to use",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode: only print prompts without loading LLM",
    )

    args = parser.parse_args()

    try:
        process_dictionary(
            input_cedict=args.cedict,
            input_llm=args.llm_dict,
            output_file=args.output,
            model_name=args.model,
            debug=args.debug,
        )
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
