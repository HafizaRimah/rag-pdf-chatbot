"""
llm.py — Generate answers using a local LLM via Ollama.

This module sends the retrieved context + user question to a
local LLM running through Ollama, and returns the generated answer.

How it works:
  1. Receive the user's question and relevant context chunks
  2. Build a prompt that instructs the LLM to answer using the context
  3. Send the prompt to Ollama's HTTP API
  4. Return the generated answer

Ollama runs locally — no API key needed, no data leaves your machine.

Usage:
    from src.llm import LLM

    llm = LLM()
    answer = llm.generate(question="What is Python?", context="Python is...")
"""

import json
import urllib.request
import urllib.error


class LLM:
    """
    Wrapper around Ollama's local API for text generation.

    Ollama exposes a simple HTTP API at http://localhost:11434.
    We use the /api/generate endpoint to send prompts and get responses.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the LLM client.

        Args:
            model:    Name of the Ollama model to use.
            base_url: Ollama server URL (default: localhost:11434).
        """
        self.model = model
        self.base_url = base_url

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using the LLM, grounded in the given context.

        Args:
            question: The user's question.
            context:  The relevant text chunks (from the retriever).

        Returns:
            The LLM's answer as a string.
        """
        prompt = self._build_prompt(question, context)
        response = self._call_ollama(prompt)
        return response

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the RAG prompt.

        Instructs the LLM to:
          - Respond warmly and naturally to conversational greetings, compliments, and thanks.
          - For document questions, ground answers strictly in the provided context and cite sources/pages.
          - If a factual question is genuinely not in the context, state that the documents do not contain that information.
        """
        prompt = f"""You are a helpful, friendly, and intelligent AI assistant specialized in analyzing uploaded PDF documents.

GUIDELINES:
1. CASUAL CHAT & COMPLIMENTS: If the user says hello, thanks, compliments your performance (e.g. "good you are doing good", "nice job"), or chats casually, respond naturally, warmly, and politely as an assistant. You do NOT need document context for pleasantries.
2. QUESTIONS ABOUT DOCUMENTS: When the user asks a question about facts, concepts, or topics:
   - Base your answer primarily on the CONTEXT below.
   - Mention the source document and page number when citing facts.
   - If the user asks a specific factual question that is genuinely not mentioned in the context, politely state that the provided documents do not contain that information.
3. Keep your answers clear, helpful, and concise.

CONTEXT FROM DOCUMENTS:
{context}

USER MESSAGE:
{question}

ANSWER:"""
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """
        Send a prompt to Ollama's API and return the response.

        Uses the /api/generate endpoint with stream=False
        for a simple request/response flow.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The generated text from the LLM.

        Raises:
            ConnectionError: If Ollama is not running.
            RuntimeError: If the API returns an error.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get the full response at once
            "options": {
                "temperature": 0.3,  # Lower = more focused/factual
                "num_predict": 512,  # Max tokens in response
            },
        }

        # Encode the request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()

        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if Ollama is reachable, False otherwise.
        """
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False


# --- Quick demo when run directly ---
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Module Demo (Ollama)")
    print("=" * 60)

    llm = LLM()

    # Check availability
    print(f"\nModel: {llm.model}")
    print(f"Server: {llm.base_url}")

    if not llm.is_available():
        print("\n[!] Ollama is not running or model not found.")
        print("    Start Ollama with: ollama serve")
        print(f"    Pull model with:   ollama pull {llm.model}")
        exit(1)

    print("[OK] Ollama is running and model is available")

    # Test with sample context
    context = (
        "--- Result 1 (score: 0.85) ---\n"
        "Source: python_guide.pdf, Page: 1\n"
        "Python is a high-level programming language created by "
        "Guido van Rossum in 1991. It emphasizes code readability "
        "and supports multiple programming paradigms.\n"
    )

    question = "Who created Python and when?"

    print(f"\nQuestion: {question}")
    print(f"\nGenerating answer...")
    answer = llm.generate(question=question, context=context)

    print(f"\nAnswer:")
    print(answer)
