"""
Task 1: LLM Workflow (5 Marks)
===============================
The simplest possible agentic building block: accept user input, send it
to an LLM, print the response.

Usage
-----
    # single question from the command line
    python llm_workflow.py --query "Explain quantum computing in 2 sentences"

    # interactive chat loop
    python llm_workflow.py --interactive
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.llm_client import LLMClient  # noqa: E402


SYSTEM_PROMPT = "You are a helpful, concise AI assistant."


def get_response(llm: LLMClient, user_input: str) -> str:
    return llm.generate(SYSTEM_PROMPT, user_input)


def main():
    parser = argparse.ArgumentParser(description="Task 1: Basic LLM Workflow")
    parser.add_argument("--query", help="A single prompt to send to the LLM")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop")
    args = parser.parse_args()

    llm = LLMClient()
    print(f"[llm_workflow] Using backend: {llm.backend}\n")

    if args.interactive:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            response = get_response(llm, user_input)
            print(f"\nAssistant: {response}\n")
    elif args.query:
        response = get_response(llm, args.query)
        print(f"User query: {args.query}\n")
        print(f"LLM response:\n{response}")
    else:
        # No args given -- prompt the user once, interactively.
        user_input = input("Enter your question for the LLM: ").strip()
        response = get_response(llm, user_input)
        print(f"\nLLM response:\n{response}")


if __name__ == "__main__":
    main()
