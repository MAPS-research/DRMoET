#!/usr/bin/env python3
"""
Download and prepare domain-specific datasets for expert specialization analysis.

Domains:
- code: CodeParrot (Python code)
- math: GSM8K (math word problems)
- wiki: Wikipedia
- books: PG19 (Project Gutenberg books)
- web: C4 (web crawl)

Usage:
    python prepare_domain_data.py --output-dir /path/to/domain_data --samples-per-domain 10000
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def prepare_code(output_path: Path, num_samples: int):
    """Download code samples from CodeParrot."""
    print("Loading code dataset (codeparrot/github-code)...")
    try:
        ds = load_dataset("codeparrot/github-code", "Python", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("code", "")
            if len(text) > 100:  # Skip very short samples
                samples.append({"text": text[:4096], "domain": "code"})  # Truncate long samples

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} code samples to {output_path}")
    except Exception as e:
        print(f"  Error loading codeparrot: {e}")
        print("  Trying bigcode/the-stack instead...")
        ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("content", "")
            if len(text) > 100:
                samples.append({"text": text[:4096], "domain": "code"})

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} code samples to {output_path}")


def prepare_math(output_path: Path, num_samples: int):
    """Download math samples from GSM8K and MATH."""
    print("Loading math dataset (gsm8k)...")
    samples = []

    # GSM8K
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        for item in ds:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            samples.append({"text": text, "domain": "math"})
            if len(samples) >= num_samples // 2:
                break
    except Exception as e:
        print(f"  Error loading gsm8k: {e}")

    # MATH dataset
    try:
        ds = load_dataset("hendrycks/competition_math", split="train")
        for item in ds:
            text = f"Problem: {item['problem']}\nSolution: {item['solution']}"
            samples.append({"text": text, "domain": "math"})
            if len(samples) >= num_samples:
                break
    except Exception as e:
        print(f"  Error loading competition_math: {e}")

    with open(output_path, "w") as f:
        for s in samples[:num_samples]:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(samples[:num_samples])} math samples to {output_path}")


def prepare_wiki(output_path: Path, num_samples: int):
    """Download Wikipedia samples."""
    print("Loading wikipedia dataset...")
    try:
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("text", "")
            if len(text) > 200:
                samples.append({"text": text[:4096], "domain": "wiki"})

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} wiki samples to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")


def prepare_books(output_path: Path, num_samples: int):
    """Download book samples from PG19."""
    print("Loading books dataset (pg19)...")
    try:
        ds = load_dataset("pg19", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("text", "")
            if len(text) > 500:
                # Take a random chunk from the book
                start = len(text) // 4
                samples.append({"text": text[start:start+4096], "domain": "books"})

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} book samples to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")


def prepare_web(output_path: Path, num_samples: int):
    """Download web samples from C4."""
    print("Loading web dataset (c4)...")
    try:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("text", "")
            if len(text) > 100:
                samples.append({"text": text[:4096], "domain": "web"})

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} web samples to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")


def prepare_scientific(output_path: Path, num_samples: int):
    """Download scientific paper samples."""
    print("Loading scientific dataset (scientific_papers)...")
    try:
        ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get("article", "")
            if len(text) > 200:
                samples.append({"text": text[:4096], "domain": "scientific"})

        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {len(samples)} scientific samples to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/domain_eval")
    parser.add_argument("--samples-per-domain", type=int, default=5000)
    parser.add_argument("--domains", type=str, nargs="+",
                        default=["code", "math", "wiki", "books", "web", "scientific"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing domain data in {output_dir}")
    print(f"Samples per domain: {args.samples_per_domain}")
    print(f"Domains: {args.domains}")
    print()

    domain_funcs = {
        "code": prepare_code,
        "math": prepare_math,
        "wiki": prepare_wiki,
        "books": prepare_books,
        "web": prepare_web,
        "scientific": prepare_scientific,
    }

    for domain in args.domains:
        if domain in domain_funcs:
            output_path = output_dir / f"{domain}.jsonl"
            domain_funcs[domain](output_path, args.samples_per_domain)
        else:
            print(f"Unknown domain: {domain}")

    print(f"\nDone! Data saved to {output_dir}/")
    print("Next: Run capture on each domain with capture_domain.sh")


if __name__ == "__main__":
    main()
