# Synthetic Chinese-English Dictionary

This project generates a Chinese-English dictionary from scratch using Large Language Models (LLMs). It then reduces hallucinations and improves accuracy through Retrieval-Augmented Generation (RAG) by cross-referencing entries with the standard [CEDICT dictionary](https://cc-cedict.org/editor/editor.php) and Wikipedia. The final dictionary contains 75519 entries and has more detailed definitions and example sentences. The dictionary can be downloaded [here]().

## Overview

The dictionary generation process follows these steps:

1. **Generate AI-Enhanced Entries**: Create detailed dictionary entries using an LLM
2. **Filter & Validate**: Compare LLM-generated entries against CEDICT for accuracy
3. **Verification**: Matches pronunciations and enhances with Wikipedia data
4. **Final Refinement**: Merge the best elements into comprehensive entries

## Pipeline Components

### 1. LLM Entry Generation (`01_generate_llm_entries.py`)

Generates detailed Chinese dictionary entries using the Qwen2.5-72B-Instruct model:

```bash
python 01_generate_llm_entries.py --input cedict_1_0_ts_utf-8_mdbg_20250120_160440.txt.gz --output output.JSONL.gz
```

Each entry includes:
- Traditional and simplified characters
- Pinyin readings with tone marks
- Part of speech classification
- Register information (formal/neutral/colloquial)
- Detailed meanings in English
- Usage notes
- Related terms
- Example sentences

### 2. Quality Evaluation (`02_evaluate_entry_quality.py`)

Compares LLM-generated entries with CEDICT to assess quality:

```bash
python 02_evaluate_entry_quality.py --cedict cedict_1_0_ts_utf-8_mdbg_20250120_160440.txt.gz --llm-dict output.JSONL.gz --output filtered.JSONL.gz
```

Evaluates each entry on:
- Semantic similarity (0-100%)
- Correctness (0-100%)

### 3. Entry Matching and Enhancement (`03_match_and_enhance_entries.py`)

Compares pronunciations and enhances high-quality matches with Wikipedia data:

```bash
python 03_match_and_enhance_entries.py --cedict cedict_1_0_ts_utf-8_mdbg_20250120_160440.txt.gz --llm-dict output.JSONL.gz --output test.JSONL.gz --sim-data filtered.JSONL.gz
```

This step:
- Standardizes pinyin representations (converting numbered to diacritic format)
- Handles special cases like erhua (儿化音)
- Filters entries based on similarity threshold (default 80%)
- Loads relevant Wikipedia articles for matching entries
- Creates comprehensive records combining pronunciation and meaning data
- Filters out variant characters to avoid pronunciation conflicts

### 4. Final Processing (`04_merge_refined_entries.py`)

Creates the final dictionary by merging and refining the validated entries:

```bash
python 04_merge_refined_entries.py --input test.JSONL.gz --output merged_dict.jsonl.gz
```

The final entries include:
- Organized meanings from most common to specialized
- Source attribution for definitions
- Usage notes and register information
- Related terms and example sentences

### 5. Dictionary Viewer (`view_dictionary.py`)

A utility to browse the generated dictionary:

```bash
python view_dictionary.py merged_dict.jsonl.gz -n 5
```

## Data Format

The final dictionary is stored in gzipped JSONL format with this structure:

```json
{
  "readings": [
    {
      "pinyin": "yī lái",
      "meanings": [
        "firstly [1, 2]",
        "to begin with [2]",
        "one reason is [2]"
      ]
    }
  ],
  "notes": [
    "Used to introduce the first in a series of reasons or points in a conversation or argument [2].",
    "Commonly used in both formal and informal contexts [2].",
    "Can be followed by '二来' (èr lái) to introduce the second reason or point [2]."
  ],
  "traditional": "一來",
  "simplified": "一来",
  "related_terms": [
    "二來",
    "三來"
  ],
  "sentences": [
    "一來天氣不好，二來我還有工作要完成。",
    "一來他不喜歡運動，二來他也很忙。"
  ]
}
```

## Future Work

Despite the quality control steps in the pipeline, there are still some accuracy issues with the dictionary entries, as some LLM-generated content contains hallucinations. Some ideas for fixing these are:

1. Using Baike as an additional knowledge source to ground definitions, as they contain word definitions and cultural context
2. Generating meanings based on word context by asking the LLM to explain terms as they appear in authentic examples from a corpus of Chinese texts
3. Creating entries by combining and summarizing multiple sources (CEDICT, Baike, Wiktionary), and skipping completely the first step in the pipeline to avoid introducing hallucinations

## Acknowledgments

- CEDICT for the original Chinese-English dictionary data
- Wikipedia for contextual information
- Qwen for the language model used in generation