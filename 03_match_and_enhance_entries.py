import gzip
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm


def is_standalone_er(chinese_char: str) -> bool:
    """Check if er syllable should be standalone based on the character.

    Args:
        chinese_char: The Chinese character (兒/儿 or other)

    Returns:
        False if it's 兒/儿 (should be erhua suffix), True otherwise
    """
    return chinese_char not in {"兒", "儿"}


def fix_erhua(pinyin: str, chinese: str) -> str:
    """Fix incorrect erhua patterns in pinyin using Chinese character context."""
    pinyin_words = pinyin.split()
    chinese_chars = list(chinese.replace(" ", ""))

    result = []
    i = 0
    char_index = 0

    while i < len(pinyin_words):
        if (
            i + 1 < len(pinyin_words)
            and pinyin_words[i + 1] == "ér"
            and char_index + 1 < len(chinese_chars)
        ):
            current_char = chinese_chars[char_index + 1]

            if is_standalone_er(current_char):
                result.extend([pinyin_words[i], "ér"])
            else:
                result.append(pinyin_words[i] + "r")

            i += 2
            char_index += 2
        else:
            result.append(pinyin_words[i])
            i += 1
            if pinyin_words[i - 1] not in {",", ".", "，", "。"}:
                char_index += 1

    return " ".join(result)


def convert_numbered_to_diacritic_pinyin(pinyin: str) -> str:
    """Convert numbered pinyin to diacritic format."""
    tone_marks = {
        "a": {"1": "ā", "2": "á", "3": "ǎ", "4": "à"},
        "e": {"1": "ē", "2": "é", "3": "ě", "4": "è"},
        "i": {"1": "ī", "2": "í", "3": "ǐ", "4": "ì"},
        "o": {"1": "ō", "2": "ó", "3": "ǒ", "4": "ò"},
        "u": {"1": "ū", "2": "ú", "3": "ǔ", "4": "ù"},
        "ü": {"1": "ǖ", "2": "ǘ", "3": "ǚ", "4": "ǜ"},
    }

    def find_vowel_for_tone_mark(syllable: str) -> str:
        """Determine which vowel should receive the tone mark."""
        vowels = "aeoiuü"
        for v in "aeoiuü":
            if v in syllable:
                if v == "i" and "u" in syllable[syllable.index("i") + 1 :]:
                    return "u"
                if v == "u" and "i" in syllable[syllable.index("u") + 1 :]:
                    return "i"
                return v
        return ""

    def convert_syllable(syllable: str) -> str:
        """Convert a single syllable with tone number to diacritic format."""
        if syllable.endswith("5"):
            return syllable[:-1]

        match = re.search(r"([1-4])$", syllable)
        if not match:
            return syllable

        tone = match.group(1)
        base = syllable[:-1]

        base = base.replace("u:", "ü")
        base = base.replace("v", "ü")

        vowel = find_vowel_for_tone_mark(base)
        if vowel and vowel in tone_marks and tone in tone_marks[vowel]:
            return base.replace(vowel, tone_marks[vowel][tone])
        return base

    pinyin = re.sub(r"(\w+)\s*r5", r"\1r5", pinyin)

    result = []
    current = ""
    for part in re.findall(r"[a-zA-Z:üv]+r?[1-5]?|[,\s]", pinyin):
        if part.isspace() or part == ",":
            if current:
                result.append(current)
                current = ""
            result.append(part)
        else:
            if "r5" in part:
                base = part.replace("r5", "")
                result.append(convert_syllable(base) + "r")
            else:
                result.append(convert_syllable(part))

    text = "".join(result)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = text.strip()

    return text


def load_cedict_entries(filepath: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Load entries from CEDICT format file."""
    entries = defaultdict(list)
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
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


def load_llm_entries(filepath: str) -> Dict[str, Dict]:
    """Load entries from LLM dictionary file."""
    entries = {}
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries[entry["traditional"]] = entry
            except json.JSONDecodeError:
                continue
    return entries


def load_similarity_data(filepath: str) -> Dict[str, float]:
    """Load and process the similarity scores file."""
    scores = {}
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entry["final_score"] = (
                    entry["similarity"] * entry["correctness"]
                ) / 100
                scores[entry["traditional"]] = entry["final_score"]
            except json.JSONDecodeError:
                continue
    return scores


def compare_pronunciations(
    cedict_data: Dict, llm_data: Dict, similarity_data: Dict, threshold: float = 0.8
) -> List[Dict]:
    """Compare pronunciations between CEDICT and LLM entries."""
    comparison_results = []

    # First, collect all traditional characters we need to look up
    needed_chars = set()
    for traditional, cedict_entries in cedict_data.items():
        if traditional in llm_data and traditional in similarity_data:
            if similarity_data[traditional] >= threshold:
                needed_chars.add(traditional)

    print(f"Looking up {len(needed_chars)} characters in Wikipedia...")

    # Create a dictionary to store only the Wikipedia entries we need
    wiki_entries = {}
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", streaming=False)

    for article in tqdm(ds["train"]):
        if article["title"] in needed_chars:
            wiki_entries[article["title"]] = article["text"]

    print(f"Found {len(wiki_entries)} matching Wikipedia entries")

    # Now proceed with the main comparison
    for traditional, cedict_entries in cedict_data.items():
        if traditional not in llm_data or traditional not in similarity_data:
            continue

        similarity_score = similarity_data[traditional]
        if similarity_score < threshold:
            continue

        # skip variant characters like 囓 (variant of 啮)
        m = ""
        for _, _, s in cedict_entries:
            m += s
        if "variant of" in m:
            continue

        llm_entry = llm_data[traditional]
        llm_readings = llm_entry.get("readings", [])
        llm_notes = llm_entry.get("notes", "")
        llm_related_terms = llm_entry.get("related_terms", [])
        llm_sentences = llm_entry.get("sentences", [])

        cedict_pinyins = set()
        for _, pinyin, _ in cedict_entries:
            converted_pinyin = convert_numbered_to_diacritic_pinyin(pinyin)
            cedict_pinyins.add(converted_pinyin.replace(" ", "").lower())

        llm_pinyins = set()
        for reading in llm_readings:
            if reading.get("pinyin"):
                llm_pinyins.add(
                    fix_erhua(reading["pinyin"], traditional).replace(" ", "").lower()
                )

        matching = cedict_pinyins & llm_pinyins
        only_in_cedict = cedict_pinyins - llm_pinyins
        only_in_llm = llm_pinyins - cedict_pinyins

        if not only_in_cedict and not only_in_llm:
            comparison_results.append(
                {
                    "traditional": traditional,
                    "matching_readings": sorted(matching),
                    "similarity_score": similarity_score,
                    "cedict_entries": [
                        {
                            "simplified": simplified,
                            "pinyin": convert_numbered_to_diacritic_pinyin(pinyin),
                            "meaning": meaning,
                        }
                        for simplified, pinyin, meaning in cedict_entries
                    ],
                    "llm_entries": [
                        {
                            "pinyin": fix_erhua(reading["pinyin"], traditional),
                            "meanings": reading["meanings"],
                        }
                        for reading in llm_readings
                        if reading.get("pinyin") and reading.get("meanings")
                    ],
                    "llm_notes": llm_notes,
                    "wikipedia_entry": wiki_entries.get(traditional),
                    "related_terms": llm_related_terms,
                    "sentences": llm_sentences,
                }
            )

    return comparison_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare CEDICT and LLM pronunciations"
    )
    parser.add_argument("--cedict", required=True, help="Path to CEDICT gzipped file")
    parser.add_argument(
        "--llm-dict", required=True, help="Path to LLM dictionary gzipped file"
    )
    parser.add_argument(
        "--sim-data", required=True, help="Path to similarity data gzipped file"
    )
    parser.add_argument(
        "--threshold", type=float, default=80.0, help="Similarity score threshold"
    )
    parser.add_argument(
        "--output", required=True, help="Output file for filtered results"
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of sample entries to display"
    )

    args = parser.parse_args()

    print("Loading CEDICT entries...")
    cedict_data = load_cedict_entries(args.cedict)
    print(f"Loaded {len(cedict_data):,} CEDICT entries")

    print("Loading LLM dictionary entries...")
    llm_data = load_llm_entries(args.llm_dict)
    print(f"Loaded {len(llm_data):,} LLM entries")

    print("Loading similarity data...")
    similarity_data = load_similarity_data(args.sim_data)
    print(f"Loaded {len(similarity_data):,} similarity scores")

    print("Comparing pronunciations...")
    results = compare_pronunciations(
        cedict_data, llm_data, similarity_data, args.threshold
    )

    print("\nResults Summary:")
    print(f"Total matching entries above threshold: {len(results):,}")
    avg_similarity = sum(r["similarity_score"] for r in results) / len(results)
    print(f"Average similarity score: {avg_similarity:.3f}")

    print(f"\nShowing {min(args.sample, len(results))} sample entries:")
    for result in sorted(results, key=lambda x: x["similarity_score"], reverse=True)[
        : args.sample
    ]:
        print(f"\nCharacter: {result['traditional']}")
        print(f"Similarity Score: {result['similarity_score']:.3f}")
        print("Readings:", ", ".join(result["matching_readings"]))
        print("CEDICT Meanings:")
        for entry in result["cedict_entries"]:
            print(f"  {entry['pinyin']} - {entry['meaning']}")
        print("LLM Meanings:")
        for entry in result["llm_entries"]:
            print(f"  {entry['pinyin']} - {', '.join(entry['meanings'])}")
        if result.get("llm_notes"):
            print(f"  {result['llm_notes']}")

    for result in sorted(results, key=lambda x: x["similarity_score"], reverse=True):
        if len(result["matching_readings"]) > 1:
            print(f"\nCharacter: {result['traditional']}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print("Readings:", ", ".join(result["matching_readings"]))
            print("CEDICT Meanings:")
            for entry in result["cedict_entries"]:
                print(f"  {entry['pinyin']} - {entry['meaning']}")
            print("LLM Meanings:")
            for entry in result["llm_entries"]:
                print(f"  {entry['pinyin']} - {', '.join(entry['meanings'])}")
            if result.get("llm_notes"):
                print(f"  {result['llm_notes']}")

    for result in sorted(results, key=lambda x: x["similarity_score"], reverse=True):
        m = ""
        for entry in result["cedict_entries"]:
            m += entry["meaning"]
        if "variant" in m:
            print(f"\nCharacter: {result['traditional']}")
            print(f"Similarity Score: {result['similarity_score']:.3f}")
            print("Readings:", ", ".join(result["matching_readings"]))
            print("CEDICT Meanings:")
            for entry in result["cedict_entries"]:
                print(f"  {entry['pinyin']} - {entry['meaning']}")
            print("LLM Meanings:")
            for entry in result["llm_entries"]:
                print(f"  {entry['pinyin']} - {', '.join(entry['meanings'])}")
            if result.get("llm_notes"):
                print(f"  {result['llm_notes']}")

    print(f"\nSaving {len(results)} entries to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
