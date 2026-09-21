"""
Task 3: Agentic AI (5 Marks)
=============================
A simple AI agent that:
    1. Accepts a task (in plain English).
    2. PLANS the steps required to complete it (asks the LLM to break the
       task into a numbered list of sub-steps).
    3. EXECUTES each planned step (asks the LLM to carry out that specific
       step, given the running context of previous steps' outputs).
    4. Displays the final output.

This is the classic "plan -> act -> observe" agent loop, kept intentionally
simple (no external tools) so the planning/execution structure is clear.

Usage
-----
    python simple_agent.py --task "Write a short marketing email for a new eco-friendly water bottle"
"""

import os
import re
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.llm_client import LLMClient  # noqa: E402


class SimpleAgent:
    def __init__(self, max_steps: int = 6):
        self.llm = LLMClient()
        self.max_steps = max_steps

    # --- PLAN ------------------------------------------------------------
    def plan(self, task: str) -> list:
        # The offline fallback echoes/extracts prompt text rather than
        # producing a clean numbered list, so use a sensible generic plan
        # when no real LLM backend is available. With a real backend
        # (Ollama/OpenAI/Gemini), the LLM's own plan is used instead.
        if self.llm.backend == "offline":
            return self._generic_plan(task)

        system_prompt = (
            "You are a planning assistant. Break the given task into a short, "
            "numbered list of 3-5 concrete, sequential steps needed to complete "
            "it. Output ONLY the numbered list, one step per line, no extra text."
        )
        user_prompt = f"TASK: {task}\n\nList the steps:"
        raw_plan = self.llm.generate(system_prompt, user_prompt, max_tokens=300)
        steps = self._parse_steps(raw_plan)
        return steps[: self.max_steps] if steps else self._generic_plan(task)

    def _generic_plan(self, task: str) -> list:
        return [
            f"Understand and restate the goal of the task: {task}",
            "Gather or outline the key information/content needed",
            "Produce a draft addressing the task",
            "Review and refine the draft into the final output",
        ]

    def _parse_steps(self, raw_plan: str) -> list:
        lines = [l.strip() for l in raw_plan.split("\n") if l.strip()]
        steps = []
        for line in lines:
            # strip leading numbering like "1.", "1)", "- ", "Step 1:"
            cleaned = re.sub(r'^\s*(step\s*)?\d+[\.\)]\s*', '', line, flags=re.IGNORECASE)
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)
            if cleaned and "[OFFLINE" not in cleaned:
                steps.append(cleaned.strip())
        return steps

    # --- EXECUTE ----------------------------------------------------------
    def execute_step(self, task: str, step: str, step_number: int, prior_outputs: list) -> str:
        system_prompt = (
            "You are an execution assistant carrying out one specific step of "
            "a larger task. Produce the concrete output for THIS step only -- "
            "do not repeat earlier steps' output, do not describe the plan."
        )
        context = ""
        if prior_outputs:
            context = "PRIOR STEP OUTPUTS:\n" + "\n".join(
                f"Step {i+1} result: {o}" for i, o in enumerate(prior_outputs)
            ) + "\n\n"

        user_prompt = (
            f"OVERALL TASK: {task}\n\n{context}"
            f"CURRENT STEP ({step_number}): {step}\n\n"
            f"Produce the output for this step:"
        )
        return self.llm.generate(system_prompt, user_prompt, max_tokens=400)

    # --- Full agent loop ---------------------------------------------------
    def run(self, task: str) -> dict:
        print(f"[SimpleAgent] Received task: {task}")
        print("[SimpleAgent] Planning steps...")
        steps = self.plan(task)
        print(f"[SimpleAgent] Plan has {len(steps)} step(s):")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}")

        outputs = []
        print("\n[SimpleAgent] Executing steps...")
        for i, step in enumerate(steps, 1):
            print(f"    -> Executing step {i}: {step}")
            result = self.execute_step(task, step, i, outputs)
            outputs.append(result)

        final_output = self._compile_final_output(task, steps, outputs)
        return {"task": task, "plan": steps, "step_outputs": outputs, "final_output": final_output}

    def _compile_final_output(self, task: str, steps: list, outputs: list) -> str:
        # The last step's output is usually the deliverable; but we also
        # surface all intermediate outputs for transparency.
        if outputs:
            return outputs[-1]
        return "No output produced."


def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"TASK: {result['task']}")
    print("=" * 70)
    print("\n--- PLAN ---")
    for i, s in enumerate(result["plan"], 1):
        print(f"{i}. {s}")

    print("\n--- STEP-BY-STEP EXECUTION ---")
    for i, (step, output) in enumerate(zip(result["plan"], result["step_outputs"]), 1):
        print(f"\n[Step {i}] {step}")
        print(f"Output: {output}")

    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Task 3: Simple Agentic AI (Plan + Execute)")
    parser.add_argument("--task", required=True, help="The task for the agent to complete")
    parser.add_argument("--max_steps", type=int, default=6)
    args = parser.parse_args()

    agent = SimpleAgent(max_steps=args.max_steps)
    result = agent.run(args.task)
    print_result(result)


if __name__ == "__main__":
    main()
