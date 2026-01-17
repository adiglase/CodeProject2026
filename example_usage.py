"""
Example: Basic MLC LLM Usage
============================

This demonstrates how to use MLCEngine for chat completions.
Use this as a reference when building your solution.

Note: The first run will download the model, which may take a few minutes.
"""

from mlc_llm import MLCEngine


def main():
    # Choose a small model that works on CPU
    # Other options:
    #   - HF://mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC
    #   - HF://mlc-ai/phi-1_5-q4f16_1-MLC
    model = "HF://mlc-ai/Qwen3-0.6B-q4f16_1-MLC"

    print(f"Initializing MLCEngine with model: {model}")
    print("(First run will download the model...)")
    print()

    # Initialize the engine
    engine = MLCEngine(model, device="cpu")

    try:
        # Example 1: Simple chat completion (non-streaming)
        print("=== Example 1: Simple Completion ===")
        response = engine.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model=model,
            stream=False,
        )
        print(f"Response: {response.choices[0].message.content}")
        print()

        # Example 2: Streaming response
        print("=== Example 2: Streaming Completion ===")
        print("Response: ", end="", flush=True)
        for chunk in engine.chat.completions.create(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            model=model,
            stream=True,
        ):
            for choice in chunk.choices:
                if choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
        print("\n")

        # Example 3: Multi-turn conversation
        print("=== Example 3: Multi-turn Conversation ===")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        response = engine.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
        )
        print(f"User: What is 2 + 2?")
        print(f"Assistant: {response.choices[0].message.content}")
        print()

    finally:
        # Always clean up
        print("Terminating engine...")
        engine.terminate()
        print("Done!")


if __name__ == "__main__":
    main()
