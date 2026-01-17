from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from mlc_llm import MLCEngine


MODEL_ID = "HF://mlc-ai/Qwen3-0.6B-q4f16_1-MLC"


# ============== Request/Response Models ==============

class GroupSentencesRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "sentences": [
                    "The cat sat on the mat",
                    "Dogs love to play fetch",
                    "My feline friend enjoys napping",
                    "The puppy ran across the yard",
                    "Machine learning is fascinating",
                    "AI models can process text"
                ]
            }
        },
    )

    sentences: list[str] = Field(
        ...,
        min_length=2,
        max_length=100,
        description="List of sentences to group by semantic similarity",
    )


class GroupSentencesResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "groups": [
                    ["The cat sat on the mat", "My feline friend enjoys napping"],
                    ["Dogs love to play fetch"],
                ]
            }
        },
    )

    groups: list[list[str]] = Field(
        ...,
        description="Sentences grouped by semantic similarity",
    )


class SynthesizeRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "sentences": [
                    "The weather is sunny today",
                    "I plan to go to the beach",
                    "Swimming is my favorite activity",
                ]
            }
        },
    )

    sentences: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of sentences to synthesize into a paragraph",
    )


class SynthesizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "paragraph": "It's a beautiful sunny day, perfect for heading to the beach. Swimming has always been my favorite activity."
            }
        },
    )

    paragraph: str = Field(
        ...,
        min_length=1,
        description="Synthesized coherent paragraph",
    )


# ============== App Lifespan (LLM init/cleanup) ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models (expensive, done once)
    print("Initializing SentenceTransformer...")
    app.state.sentence_model = SentenceTransformer('all-mpnet-base-v2')
    print("SentenceTransformer initialized successfully")

    print(f"Initializing MLCEngine with model: {MODEL_ID}")
    app.state.llm_engine = MLCEngine(MODEL_ID, device="cpu")
    app.state.model_id = MODEL_ID
    print("MLCEngine initialized successfully")

    yield

    # Shutdown: Cleanup LLM engine
    print("Terminating MLCEngine...")
    app.state.llm_engine.terminate()
    print("MLCEngine terminated")


# ============== FastAPI App ==============

app = FastAPI(
    title="Semantic Sentence API",
    description="API for semantic grouping and paragraph synthesis",
    version="1.0.0",
    lifespan=lifespan,
)


# ============== Endpoints ==============

@app.post("/group-sentences", response_model=GroupSentencesResponse)
async def group_sentences(request: GroupSentencesRequest):
    """
    Group sentences by semantic similarity.
    """
    sentences = request.sentences
    embeddings = app.state.sentence_model.encode(sentences)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.75, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    
    groups = {}
    for sentence, label in zip(sentences, labels):
        groups.setdefault(label, []).append(sentence)
    
    return GroupSentencesResponse(groups=list(groups.values()))                                            


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize multiple sentences into a coherent paragraph.
    """
    sentences = request.sentences
    engine: MLCEngine = app.state.llm_engine

    sentences_text = " ".join(sentences)
    
    prompt = f"""Rewrite this into a single flowing paragraph that sounds natural and engaging:

    {sentences_text}

    Rewritten paragraph:"""

    response = engine.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a skilled writer. Rewrite text to sound natural and engaging. Output only the rewritten text."},
            {"role": "user", "content": prompt},
        ],
        model=app.state.model_id,
        stream=False,
    )

    paragraph = response.choices[0].message.content.strip()
    
    # Clean up any thinking tags (safety net)
    if "</think>" in paragraph:
        paragraph = paragraph.split("</think>")[-1].strip()
    
    return SynthesizeResponse(paragraph=paragraph)