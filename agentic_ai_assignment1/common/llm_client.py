"""
common/llm_client.py
---------------------
A single, unified LLM wrapper shared by all four tasks in this assignment.

The assignment asks for OpenAI / Gemini / Ollama support. This wrapper
auto-detects which backend to use, in this priority order:

    1. Ollama   -> if a local Ollama server is reachable (OLLAMA_MODEL env
                   var set, or the default 'llama3' model, and
                   http://localhost:11434 responds). No API key needed --
                   great for running this assignment fully offline/free.
    2. OpenAI   -> if OPENAI_API_KEY is set.
    3. Gemini   -> if GEMINI_API_KEY is set.
    4. Offline fallback -> if none of the above are available/reachable,
       so every script in this project still runs end-to-end for grading
       without any API key, model download, or internet access. Offline
       output is always clearly labeled as such.

You can also force a specific backend:
    LLMClient(backend="openai")
    LLMClient(backend="gemini")
    LLMClient(backend="ollama")
"""

import os
import re
import textwrap


class LLMClient:
    def __init__(self, backend: str = None, model: str = None, temperature: float = 0.3):
        self.temperature = temperature
        self.model = model
        self.backend = backend or self._auto_detect_backend()

    # ------------------------------------------------------------------
    def _auto_detect_backend(self) -> str:
        # 1. Try Ollama (local, free, no key)
        if self._ollama_reachable():
            return "ollama"
        # 2. OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                return "openai"
            except ImportError:
                print("[LLMClient] openai package not installed (pip install openai).")
        # 3. Gemini
        if os.environ.get("GEMINI_API_KEY"):
            try:
                import google.generativeai  # noqa: F401
                return "gemini"
            except ImportError:
                print("[LLMClient] google-generativeai package not installed "
                      "(pip install google-generativeai).")
        # 4. Offline fallback
        return "offline"

    def _ollama_reachable(self) -> bool:
        try:
            import requests
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            resp = requests.get(f"{base_url}/api/tags", timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Single entry point every task calls. Returns plain text."""
        try:
            if self.backend == "ollama":
                return self._call_ollama(system_prompt, user_prompt)
            elif self.backend == "openai":
                return self._call_openai(system_prompt, user_prompt, max_tokens)
            elif self.backend == "gemini":
                return self._call_gemini(system_prompt, user_prompt, max_tokens)
            else:
                return self._call_offline(system_prompt, user_prompt)
        except Exception as e:
            print(f"[LLMClient] {self.backend} call failed ({e}); using offline fallback.")
            return self._call_offline(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    def _call_ollama(self, system_prompt, user_prompt):
        import requests
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = self.model or os.environ.get("OLLAMA_MODEL", "llama3")
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openai(self, system_prompt, user_prompt, max_tokens):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    def _call_gemini(self, system_prompt, user_prompt, max_tokens):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            self.model or "gemini-1.5-flash",
            system_instruction=system_prompt,
        )
        resp = model.generate_content(
            user_prompt,
            generation_config={"temperature": self.temperature, "max_output_tokens": max_tokens},
        )
        return resp.text

    # ------------------------------------------------------------------
    def _call_offline(self, system_prompt, user_prompt):
        """
        Deterministic, dependency-free fallback so the whole assignment is
        runnable and demoable with zero API keys / zero internet / no local
        Ollama install.

        Strategy: naive extractive text processing -- score sentences in
        the prompt by keyword overlap and return the highest-scoring ones.
        Not a real LLM, but keeps every task's pipeline fully functional.
        """
        text = user_prompt
        sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return ("[OFFLINE MODE - no LLM backend available] "
                    "No content available to process.")

        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop = {"this", "that", "with", "from", "have", "will", "your",
                "about", "into", "which", "these", "those", "there", "please"}
        freq = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1

        def score(s):
            ws = re.findall(r"[a-zA-Z]{4,}", s.lower())
            return sum(freq.get(w, 0) for w in ws)

        ranked = sorted(sentences, key=score, reverse=True)
        top = ranked[: min(4, len(ranked))]
        top_in_order = [s for s in sentences if s in top]

        result = " ".join(top_in_order)
        banner = ("[OFFLINE FALLBACK MODE -- no Ollama server reachable and no "
                   "OPENAI_API_KEY / GEMINI_API_KEY found. This is a simple "
                   "extractive result, NOT a real LLM response. Configure a "
                   "backend to get real generative answers -- see README.]\n\n")
        return banner + textwrap.fill(result, width=100)
