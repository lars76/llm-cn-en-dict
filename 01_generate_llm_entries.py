import json
import os
import re
import gzip
import argparse
from typing import List, Set
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def create_prompt(traditional: str) -> str:
    return f'''Create dictionary entry for "{traditional}" in traditional Chinese. Requirements:

1. READINGS:
- Use tone-marked pinyin (ā é ǐ ò ǖ)
- Each reading includes:
  * Part of speech [n,v,mod,conj,prep,part,measure,phr]
  * Register [formal,neutral,colloquial]
    - 'formal' for literary/classical/technical terms
    - 'neutral' for standard usage
    - 'colloquial' for informal/spoken/internet language
  * Meanings MUST be in English only, ordered from most to least common
2. NOTES (English only):
- Include only for: cultural context, usage restrictions, significant etymology
- Omit if: meaning is clear, note would repeat definition
- Be concise, avoid "This term is used for..."
3. RELATED_TERMS:
- Include up to 2 most relevant terms in traditional characters
- Must help explain meaning/usage
- Omit if no natural related terms exist
- Do not include pinyin or translations
4. SENTENCES:
- Up to 2 natural examples showing different uses in traditional characters
- Omit for rare/obsolete terms
- Do not include pinyin or translations

For rare characters: note uncertainty, minimize claims, omit sentences.
For internet slang: explain current usage, use contemporary examples.

{{
    "traditional": "{traditional}",
    "readings": [
        {{
            "pinyin": "",
            "meanings": [""],
            "part_of_speech": [""],
            "register": [""]
        }}
    ],
    "notes": "",
    "related_terms": [""],
    "sentences": [""]
}}'''


def extract_terms(input_file: str, processed_terms: Set[str]) -> List[str]:
    """Extract unprocessed terms from input file."""
    terms = []
    with gzip.open(input_file, "rb") as f:
        for line in f:
            line = line.decode("utf-8")
            if line.startswith("#"):
                continue

            reg = re.match(r"(.+?) (.+?) (\[.*\]+?) (\/.*?\/)", line)
            if reg and reg.group(1) not in processed_terms:
                terms.append(reg.group(1))
    return terms


def extract_json_from_text(text: str):
    try:
        # Find the first '{' and the last '}'
        start_index = text.find("{")
        end_index = text.rfind("}")

        if start_index == -1 or end_index == -1:
            return None

        json_str = text[start_index : end_index + 1]

        # Remove comments (anything after # on a line)
        json_str = re.sub(r"#.*$", "", json_str, flags=re.MULTILINE)

        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def process_dictionary(
    input_file: str, output_file: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"
):
    """Process the dictionary file using vLLM."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.9, top_k=20, max_tokens=2048
    )
    llm = LLM(
        model=model_name,
        quantization="awq_marlin",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4,
        enforce_eager=False,
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    processed_terms = set()
    if os.path.exists(output_file):
        with gzip.open(output_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_terms.add(entry["traditional"])
                except json.JSONDecodeError:
                    continue
    terms = extract_terms(input_file, processed_terms)
    total_terms = len(terms)

    with tqdm(total=total_terms, desc="Processing terms", unit="terms") as pbar:
        for term in terms:
            if term in processed_terms:
                pbar.update(1)
                continue

            try:
                # Create the prompt for the current term
                messages = [{"role": "user", "content": create_prompt(term)}]

                # Tokenize the prompt
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate the response from the model
                outputs = llm.generate([text], sampling_params)
                generated_text = outputs[0].outputs[0].text
                entry = extract_json_from_text(generated_text)

                if entry:
                    print(entry)
                    with gzip.open(output_file, "at", encoding="utf-8") as f:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write("\n")

                    processed_terms.add(entry["traditional"])
                else:
                    print("Unable to parse term", generated_text)

            except Exception as e:
                # Print the exception details
                print(f"Error processing term {term}: {str(e)}")

            finally:
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process Chinese dictionary entries using vLLM."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input gzipped CEDICT file path"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output JSONL.gz file path"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Model name to use",
    )

    args = parser.parse_args()

    try:
        process_dictionary(
            input_file=args.input, output_file=args.output, model_name=args.model
        )
    except Exception as e:
        # Print the exception details
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
