"""Base user-prompt distributions.

Vectors are fit on prompts from a broad instruction distribution and
evaluated on HELD-OUT prompts (strict split, per Stolfo et al.). Seed
sets below make the harness self-contained for smoke tests; real runs
pull from HF datasets cached on the cluster (alpaca / dolly), plus
keyword-filtered trigger sets for conditional instructions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEED_PROMPTS: dict[str, list[str]] = {
    "general": [
        "What causes the seasons to change?",
        "Explain how a refrigerator works.",
        "Give me three tips for improving my morning routine.",
        "What is the difference between a virus and a bacterium?",
        "Summarize the plot of Romeo and Juliet.",
        "How do airplanes stay in the air?",
        "What should I consider when adopting a dog?",
        "Describe the water cycle.",
        "Why is the sky blue?",
        "What are some good strategies for learning a new language?",
        "Explain the concept of compound interest.",
        "What happened during the French Revolution?",
        "How does a search engine decide what results to show?",
        "Recommend a good way to start jogging as a beginner.",
        "What is photosynthesis and why does it matter?",
        "Explain the rules of chess in simple terms.",
        "What makes a good cover letter?",
        "How do vaccines work at a high level?",
        "Describe the difference between weather and climate.",
        "What are black holes?",
        "Give me ideas for a vegetarian dinner party menu.",
        "How is chocolate made?",
        "What is inflation and what causes it?",
        "Explain how tides work.",
        "What are the main differences between classical and jazz music?",
        "How should I prepare for a job interview?",
        "What is the history of the Olympic Games?",
        "Explain what DNA is to a ten-year-old.",
        "What are good habits for saving money?",
        "How does GPS know where I am?",
        "Describe the life cycle of a butterfly.",
        "What is the tallest mountain in the world and how was it formed?",
        "Give me tips for taking better photographs with a phone.",
        "How do bees make honey?",
        "What is the difference between an alligator and a crocodile?",
        "Explain the basics of supply and demand.",
        "What should I pack for a week-long hiking trip?",
        "How do noise-cancelling headphones work?",
        "What was the Silk Road and why was it important?",
        "Suggest ways to make a small apartment feel bigger.",
    ],
    "medical": [
        "I've had a headache for three days. What could be causing it?",
        "What are the symptoms of the flu versus a common cold?",
        "How much sleep does an adult actually need?",
        "Is intermittent fasting safe?",
        "What does high blood pressure do to the body?",
        "My knee hurts after running. What should I do?",
        "What are the early signs of diabetes?",
        "How can I tell if a mole is dangerous?",
        "What vitamins should I take in winter?",
        "Why do I feel dizzy when I stand up quickly?",
        "What is the best way to treat a minor burn at home?",
        "How do I know if I have a food allergy?",
        "What causes migraines and how can I prevent them?",
        "Is it normal for my heart to race after coffee?",
        "What are the side effects of ibuprofen?",
        "How can I improve my posture to reduce back pain?",
        "My child has had a fever since yesterday. When should I worry?",
        "I get heartburn after most meals. What could help?",
        "What are the warning signs of a concussion?",
        "My hands tingle when I wake up. What might cause that?",
        "How do I know if a cut needs stitches?",
        "I've been feeling unusually tired for weeks. What could it be?",
        "What should I do about recurring lower back pain?",
        "Is a resting heart rate of 90 something to be concerned about?",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Show me how to read a CSV file in Python.",
        "Write a JavaScript function to debounce another function.",
        "How do I sort a list of dictionaries by a key in Python?",
        "Write a SQL query to find duplicate rows in a table.",
        "Implement binary search in Python.",
        "Write a bash script that renames all .txt files to .md.",
        "Show me a regular expression that matches email addresses.",
        "Write a Python class representing a bank account.",
        "How do I make an HTTP GET request in Python?",
        "Write a function to compute Fibonacci numbers with memoization.",
        "Show me how to parse JSON in JavaScript.",
        "Write a Python decorator that times function execution.",
        "Implement a simple linked list in Python.",
        "Write a one-liner to count word frequencies in a file.",
        "How do I create a virtual environment in Python?",
    ],
}


def seed_prompts(set_name: str) -> list[str]:
    return list(SEED_PROMPTS[set_name])


def split_prompts(
    prompts: list[str], holdout_frac: float = 0.5, salt: str = "sisteer"
) -> tuple[list[str], list[str]]:
    """Deterministic hash-based split → (fit, heldout). Stable across runs
    and across prompt-list orderings."""
    fit, heldout = [], []
    for p in prompts:
        h = hashlib.sha256((salt + p).encode()).digest()[0] / 255.0
        (heldout if h < holdout_frac else fit).append(p)
    return fit, heldout


def load_hf_prompts(
    dataset: str = "tatsu-lab/alpaca",
    n: int = 2000,
    seed: int = 0,
    max_chars: int = 300,
    revision: str | None = None,
) -> list[str]:
    """Instruction-only prompts from a cached HF dataset (cluster runs).

    ``revision`` pins the dataset to a full commit sha (content-disjoint splits
    for the benign teacher require the SAME pinned Alpaca every run)."""
    from datasets import load_dataset

    ds = (load_dataset(dataset, split="train", revision=revision)
          if revision is not None else load_dataset(dataset, split="train"))
    ds = ds.shuffle(seed=seed)
    prompts = []
    for row in ds:
        instr = row.get("instruction") or row.get("prompt") or ""
        if row.get("input"):
            continue  # keep single-turn, context-free prompts
        if 10 <= len(instr) <= max_chars:
            prompts.append(instr.strip())
        if len(prompts) >= n:
            break
    return prompts


def filter_by_keywords(
    prompts: list[str], keywords: list[str], include: bool = True
) -> list[str]:
    def hit(p: str) -> bool:
        low = p.lower()
        return any(k in low for k in keywords)

    return [p for p in prompts if hit(p) == include]


def save_prompts(prompts: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(prompts, indent=1))


def load_prompts(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text())
