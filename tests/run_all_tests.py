"""
run_all_tests.py — Comprehensive test runner for all project modules.

Runs each test suite in sequence and displays a final summary.
"""

import os
import subprocess
import sys


TEST_SCRIPTS = [
    "test_pdf_loader.py",
    "test_chunker.py",
    "test_embedder.py",
    "test_vector_store.py",
    "test_retriever.py",
    "test_rag_pipeline.py",
]


def run_tests():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    python_executable = sys.executable

    print("=" * 70)
    print("RUNNING ALL RAG PDF CHATBOT TESTS")
    print("=" * 70)

    results = {}

    for script in TEST_SCRIPTS:
        script_path = os.path.join(tests_dir, script)
        print(f"\n--> Running {script}...")
        
        proc = subprocess.run(
            [python_executable, script_path],
            cwd=os.path.dirname(tests_dir),
            capture_output=False,
            text=True,
        )

        results[script] = proc.returncode == 0

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)

    all_passed = True
    for script, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {script:<25} : {status}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TEST SUITES PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED! Check outputs above.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_tests()
