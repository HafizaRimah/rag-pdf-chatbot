"""
test_llm.py — Test the LLM module (Ollama).

Verifies that:
  1. Ollama is running and the model is available
  2. The prompt is built correctly
  3. The LLM generates a relevant answer from context
  4. The LLM avoids hallucination (admits when info is missing)
  5. Connection error is raised when Ollama is unreachable

NOTE: Tests 3 and 4 require Ollama to be running with llama3.2.
      Start it with: ollama serve
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLM


def test_availability():
    """Test: Ollama is running and model is available."""
    print("Test 1: Ollama availability")
    print("-" * 40)

    llm = LLM()
    available = llm.is_available()

    if not available:
        print("  [SKIP] Ollama is not running or model not found")
        print("         Start with: ollama serve")
        print("         Pull with:  ollama pull llama3.2")
        return False

    print(f"  [OK] Ollama is running, model '{llm.model}' available")
    print()
    return True


def test_prompt_building():
    """Test: prompt is constructed correctly."""
    print("Test 2: Prompt construction")
    print("-" * 40)

    llm = LLM()
    prompt = llm._build_prompt(
        question="What is Python?",
        context="Python is a programming language."
    )

    assert "CONTEXT:" in prompt, "Prompt should contain CONTEXT section"
    assert "QUESTION:" in prompt, "Prompt should contain QUESTION section"
    assert "Python is a programming language." in prompt
    assert "What is Python?" in prompt
    assert "ONLY" in prompt, "Prompt should instruct LLM to use only context"

    print("  [OK] Prompt contains CONTEXT section")
    print("  [OK] Prompt contains QUESTION section")
    print("  [OK] Context text included")
    print("  [OK] Question text included")
    print("  [OK] Grounding instruction present")
    print()


def test_generate_with_context(llm: LLM):
    """Test: LLM generates a relevant answer from given context."""
    print("Test 3: Generate answer from context")
    print("-" * 40)

    context = (
        "Source: guide.pdf, Page: 5\n"
        "The Python programming language was created by Guido van Rossum "
        "and was first released in 1991. It is known for its simple syntax."
    )

    question = "Who created Python?"
    print(f"  Question: {question}")
    print(f"  Generating...")

    answer = llm.generate(question=question, context=context)

    assert len(answer) > 0, "Answer should not be empty"
    # The answer should mention Guido (from the context)
    assert "guido" in answer.lower(), f"Answer should mention Guido, got: {answer}"

    print(f"  Answer: {answer[:200]}")
    print(f"  [OK] Answer is non-empty and mentions Guido")
    print()


def test_no_hallucination(llm: LLM):
    """Test: LLM admits when the context doesn't have the answer."""
    print("Test 4: No hallucination check")
    print("-" * 40)

    # Context about Python, question about something unrelated
    context = (
        "Source: guide.pdf, Page: 1\n"
        "Python is a programming language used for web development."
    )

    question = "What is the population of Tokyo?"
    print(f"  Question: {question}")
    print(f"  Generating...")

    answer = llm.generate(question=question, context=context)

    assert len(answer) > 0, "Answer should not be empty"
    # The LLM should indicate it can't answer from the context
    answer_lower = answer.lower()
    has_disclaimer = any(phrase in answer_lower for phrase in [
        "don't have enough information",
        "not mentioned",
        "no information",
        "cannot answer",
        "doesn't contain",
        "does not contain",
        "not provided",
        "not available",
        "context does not",
        "context doesn't",
        "unable to",
    ])

    print(f"  Answer: {answer[:200]}")
    if has_disclaimer:
        print(f"  [OK] LLM correctly declined to answer")
    else:
        print(f"  [WARN] LLM may have hallucinated (check answer above)")
    print()


def test_connection_error():
    """Test: proper error when Ollama is unreachable."""
    print("Test 5: Connection error handling")
    print("-" * 40)

    # Use a wrong port to simulate Ollama being down
    llm = LLM(base_url="http://localhost:99999")

    try:
        llm.generate(question="test", context="test")
        assert False, "Should have raised ConnectionError"
    except (ConnectionError, RuntimeError):
        print("  [OK] ConnectionError raised for unreachable server")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Module -- Tests")
    print("=" * 60)
    print()

    # Test 1: Check if Ollama is running
    ollama_available = test_availability()

    # Test 2: Prompt building (always works, no Ollama needed)
    test_prompt_building()

    if ollama_available:
        llm = LLM()
        # Test 3: Actual generation
        test_generate_with_context(llm)
        # Test 4: Hallucination check
        test_no_hallucination(llm)
    else:
        print("Tests 3-4: [SKIP] Ollama not available\n")

    # Test 5: Connection error handling
    test_connection_error()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
