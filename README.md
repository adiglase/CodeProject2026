# Semantic Sentence API - Coding Challenge

## Time Limit
**90 minutes recommended** (honor system)

Please record your actual time spent. We value honesty about time investment.

## Overview
Build a Python REST API with two endpoints using MLC LLM (a local LLM framework).

### Endpoint 1: Semantic Grouping
**POST /group-sentences**

Takes a list of sentences and returns them grouped by semantic similarity.

Example Input:
```json
{
  "sentences": [
    "The cat sat on the mat",
    "Dogs love to play fetch",
    "My feline friend enjoys napping",
    "The puppy ran across the yard",
    "Machine learning is fascinating",
    "AI models can process text"
  ]
}
```

Example Output:
```json
{
  "groups": [
    ["The cat sat on the mat", "My feline friend enjoys napping"],
    ["Dogs love to play fetch", "The puppy ran across the yard"],
    ["Machine learning is fascinating", "AI models can process text"]
  ]
}
```

### Endpoint 2: Paragraph Synthesis
**POST /synthesize**

Takes multiple sentences and returns a single coherent paragraph.

Example Input:
```json
{
  "sentences": [
    "The weather is sunny today",
    "I plan to go to the beach",
    "Swimming is my favorite activity"
  ]
}
```

Example Output:
```json
{
  "paragraph": "It's a beautiful sunny day, perfect for heading to the beach. Swimming has always been my favorite activity, and today's weather makes it an ideal opportunity to enjoy the water."
}
```

---

## Requirements

### Must Use
- **MLC LLM** for language model inference (https://llm.mlc.ai/)
- **Python 3.10+**

### Your Choice
- Web framework (Flask, FastAPI, etc.)
- Additional libraries for embeddings/similarity (if needed)
- Model selection from MLC-compatible models

### Hardware & Installation

**Recommended small models** (work on both CPU and GPU):
- `HF://mlc-ai/Qwen3-0.6B-q4f16_1-MLC` (~600M params)
- `HF://mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC`
- `HF://mlc-ai/phi-1_5-q4f16_1-MLC`

**GPU Installation** (NVIDIA CUDA):
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

**CPU Installation** (no GPU required):
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
```

**macOS with Apple Silicon** (M1/M2/M3):
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

---

## Setup

```bash
# 1. Create virtual environment (conda recommended)
conda create -n semantic-api python=3.11
conda activate semantic-api

# 2. Install MLC LLM (see "Hardware & Installation" above for the right command)

# 3. Install git-lfs (required for model downloads)
# macOS: brew install git-lfs
# Ubuntu: sudo apt install git-lfs
git lfs install

# 4. Install your additional dependencies
pip install -r requirements.txt
```

Or run the provided setup script for guidance:
```bash
./setup.sh
```

---

## Submission

### How to Submit

1. **Fork this repository** to your own GitHub account
2. Complete the challenge in your forked repo
3. **Submit the URL** of your forked GitHub repository

### Your repo must include:

1. **All source code** for your solution
2. **requirements.txt** with your dependencies
3. **Updated README** (the "Your Solution" section below) explaining:
   - How to run your solution
   - Your approach and design decisions
   - Any tradeoffs or limitations
   - Actual time spent (be honest!)
   - Whether you used AI coding assistants (allowed, just disclose it)

**Important:** Make sure your repository is **public** or that you've granted us access before submitting.

---

## Evaluation Criteria

- **Functionality**: Do both endpoints work correctly?
- **Code Quality**: Clean, readable, maintainable code
- **Technical Decisions**: Did you make sensible library/approach choices?
- **API Design**: Proper REST conventions, error handling
- **Documentation**: Clear instructions and explanations

---

## Hints

- MLC LLM provides an OpenAI-compatible chat API
- Semantic similarity might require creative problem-solving
- Start simple, then iterate if time permits
- Working code beats perfect code
- See `example_usage.py` for basic MLC LLM patterns

---

## Helpful Resources

- MLC LLM Documentation: https://llm.mlc.ai/docs/
- MLC LLM Models on HuggingFace: https://huggingface.co/mlc-ai

---

## Your Solution

**Please fill out this section before submitting:**

### How to Run

**1. Create and activate conda environment**
```bash
conda create -n semantic-api python=3.11
conda activate semantic-api
```

**2. Install MLC LLM (choose based on your hardware)**

GPU (NVIDIA CUDA):
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

CPU only:
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
```

macOS Apple Silicon:
```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

**3. Install git-lfs (required for model downloads)**
- Ubuntu: `sudo apt install git-lfs`
- macOS: `brew install git-lfs`

```bash
git lfs install
```

**4. Install project dependencies**
```bash
pip install -r requirements.txt
```

**5. Run the API server**
```bash
fastapi dev app.py
```

The API will be available at http://127.0.0.1:8000. Interactive docs at http://127.0.0.1:8000/docs.

---
### Approach

#### Endpoint 1: Semantic Grouping

My first intuition is just using the llm with structured output. But this is not working well because the model is too small so it does not following the instruction properly. And I am also not sure whether the mlc-llm support structured output as I can't find it in the docs. So I guess this is not the best solution.

This problem actually similar with how the semantic search work. To find a similarity between text, we can just compare their embeddings. So how if we use similar approach, I compare the sentence based on their embeddings and see if they are related.

After doing some research, I found out that we can use sentence transformer. Basically we convert each sentence into a single embeddings, and then we compare them with other text and group them if they are closely related with each other.

We can use several pretrained model for this. First I try using all-MiniLM-L6-v2 , but seems like it always fail to return the expected response. After checking the dendograms (you can find it inside experiments/dendograms/ folder) the model clearly failed to group the cat and feline. After switch to a better model (all-mpnet-base-v2), the dendogram showed that the model know well that cat and feline is a similar thing. After doing several iterations to find the best threshold, the endpoint managed to return the expected response.

#### Endpoint 2: Paragraph Synthesis

My first approach is to write a detailed rules and several examples. But I found out the model seems like take the example as a constraints. If I have example using most of past tenses, the returned response also mostly will be in past tenses.

So I decide to remove the example and make the prompt much more simpler. I still haven't found a perfect prompt for it, because of the time limitations.

The model sometimes keep adding the <thinking> section in the returned response. So I add some additional cleanup before returning the response.

### Tradeoffs/Limitations
- The distance_threshold=0.75 is hardcoded. Different types of sentences may need different thresholds for optimal grouping.
- Qwen3-0.6B has limited instruction-following capability, which is why I needed the </think> tag cleanup hack.
- My laptop does not have external GPU, results in really slow inference times.
- No proper error handling.


### Time Spent
2 hours 45 minutes. Slow model inference take so much time.


### AI Assistant Usage
I use Claude Code with Opus 4.5 for brainstorming, project scaffolding, fix error, writing the pydantic types and doing the clustering experiments to find a perfect model and treshold value.
