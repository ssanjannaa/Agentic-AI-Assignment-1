"""
Task 2: Prompt Chaining (5 Marks)
===================================
A multi-step LLM workflow where the output of each step feeds the next:

    STEP 1: Generate a summary of the given topic.
    STEP 2: Extract key points FROM that summary.
    STEP 3: Produce three questions FROM the key points.

This demonstrates prompt chaining -- breaking one big task into smaller,
sequential LLM calls where each step builds on the previous one's output,
rather than trying to do everything in a single prompt.

Usage
-----
    python prompt_chaining.py --topic "The impact of climate change on agriculture"
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.llm_client import LLMClient  # noqa: E402


class PromptChain:
    def __init__(self):
        self.llm = LLMClient()

    # STEP 1 ------------------------------------------------------------
    def generate_summary(self, topic: str) -> str:
        system_prompt = "You are a knowledgeable writer. Write a clear, factual summary."
        user_prompt = f"Write a concise summary (4-6 sentences) about the following topic:\n\n{topic}"
        return self.llm.generate(system_prompt, user_prompt, max_tokens=400)

    # STEP 2 ------------------------------------------------------------
    def extract_key_points(self, summary: str) -> str:
        system_prompt = "You are an analyst who extracts the most important points from text."
        user_prompt = (
            f"From the following summary, extract the 4-6 most important key points "
            f"as a bullet list:\n\n{summary}"
        )
        return self.llm.generate(system_prompt, user_prompt, max_tokens=400)

    # STEP 3 ------------------------------------------------------------
    def generate_questions(self, key_points: str) -> str:
        system_prompt = "You are a teacher who writes thoughtful comprehension questions."
        user_prompt = (
            f"Based on these key points, write exactly THREE thought-provoking "
            f"questions a reader could use to test their understanding:\n\n{key_points}"
        )
        return self.llm.generate(system_prompt, user_prompt, max_tokens=300)

    # Full chain ----------------------------------------------------------
    def run(self, topic: str) -> dict:
        print(f"[PromptChain] STEP 1/3 -- Generating summary for: {topic}")
        summary = self.generate_summary(topic)

        print("[PromptChain] STEP 2/3 -- Extracting key points from the summary...")
        key_points = self.extract_key_points(summary)

        print("[PromptChain] STEP 3/3 -- Generating questions from the key points...")
        questions = self.generate_questions(key_points)

        return {"topic": topic, "summary": summary, "key_points": key_points, "questions": questions}


def main():
    parser = argparse.ArgumentParser(description="Task 2: Prompt Chaining Workflow")
    parser.add_argument("--topic", required=True, help="The topic to process through the chain")
    args = parser.parse_args()

    chain = PromptChain()
    result = chain.run(args.topic)

    print("\n" + "=" * 70)
    print(f"TOPIC: {result['topic']}")
    print("=" * 70)
    print("\n--- STEP 1: SUMMARY ---")
    print(result["summary"])
    print("\n--- STEP 2: KEY POINTS ---")
    print(result["key_points"])
    print("\n--- STEP 3: THREE QUESTIONS ---")
    print(result["questions"])
    print("=" * 70)


if __name__ == "__main__":
    main()
