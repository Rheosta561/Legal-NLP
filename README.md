# Legal NLP Classification API

FastAPI application for LEDGAR legal provision classification. It combines the existing BGE-M3-trained classifier, its fitted scaler, local Ollama BGE-M3 embeddings, a LangChain `ChatGroq` tool agent, and in-process NVIDIA NeMo Guardrails.

## Architecture

```text
Client → FastAPI → (direct /predict or guarded /chat agent) → Ollama BGE-M3
                                                        → classifier tool
BGE-M3 embedding → existing StandardScaler → existing TensorFlow classifier
```

NeMo Guardrails runs in the API process as LangChain `GuardrailsMiddleware`; it is not a separate Compose service. `config/guardrails/config.yml` and `prompts.yml` define input/output checks. The classifier tool is available inside the guarded agent.

## Requirements

- Docker Desktop with Compose v2, at least 8 GB memory allocated (12 GB recommended for a comfortable local build/run).
- Ollama installed and running on the Mac host with `bge-m3` pulled.
- A Groq API key to use `/chat`.
- Existing model files in `models/bge_m3/` (the directory is included in the image).

Prepare Ollama and the environment:

```sh
ollama pull bge-m3
cp .env.example .env
# Set GROQ_API_KEY in .env; leave OLLAMA_BASE_URL as shown for Compose.
```

## Build and run

```sh
docker compose build
docker compose up -d
docker compose ps
```

Compose maps `host.docker.internal:11434` to the Mac host's Ollama server. It overrides `OLLAMA_BASE_URL` for container networking while your local `.env` can continue to use `http://localhost:11434` outside Docker.

Check liveness and readiness:

```sh
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Classify without an LLM:

```sh
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The agreement shall remain in effect until terminated by either party."}'
```

Chat with the guarded Groq agent:

```sh
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Classify this provision: The agreement shall remain in effect until terminated by either party."}'
```

View logs or stop the service:

```sh
docker compose logs -f legal-nlp-api
docker compose down
```

## Included libraries

Runtime dependencies are locked in `uv.lock` and installed in the image:

- **FastAPI, Uvicorn, Pydantic Settings** — API, serving, and configuration.
- **httpx** — asynchronous Ollama API calls.
- **TensorFlow / Keras** — load and run the pre-trained classifier.
- **scikit-learn and joblib** — apply the existing fitted scaler and load it.
- **LangChain, LangGraph, langchain-groq** — ChatGroq agent and classification tool.
- **NeMo Guardrails** — input/output rails around the agent loop.
- **NumPy** — validated embedding and prediction arrays.

Gemini libraries are not used. The classifier, scaler, and class mapping are loaded from `models/bge_m3`; they are not retrained or refitted.

## Image size and resource planning

These are planning estimates; exact image size depends on the selected platform, wheel builds, and Docker compression. Measure the produced image with `docker image ls legal-nlp-api:local` and `docker image inspect legal-nlp-api:local --format '{{.Size}}'`.

| Item | Expected size / allocation | Notes |
|---|---:|---|
| App source + classifier artifacts | Under 10 MB | Current model package is about 3.6 MB. |
| Python base + runtime libraries | Roughly 2–4 GB unpacked | TensorFlow, NumPy/SciPy, scikit-learn, LangChain and NeMo dominate; compressed image transfer is smaller and platform dependent. |
| API image | Plan for 2–4 GB | Build once, then reuse the image. `.venv`, `.env`, tests, and caches are excluded from build context. |
| API container memory | 4 GB starting point; 6 GB configured ceiling | TensorFlow + NeMo + model loading may peak above idle use. Watch actual RSS before lowering limits. |
| API CPU | 2 vCPU baseline; Compose allows up to 4 | Concurrent LLM/embedding calls and TensorFlow inference affect throughput. |
| Ollama BGE-M3 model | About 1.2 GB on this Mac | Managed by Ollama on the host and not copied into this image. Ollama runtime memory is additional to the API container. |
| Host memory | 8 GB minimum; 12–16 GB recommended | Docker Desktop, Ollama, model weights, and API container share Mac memory. |

The Docker image uses `python:3.11-slim` and locked packages to limit unrelated base layers. First build downloads the dependencies and can take several minutes; the uv lockfile layer is cached until dependency metadata changes.

## Resource tuning

The Compose service starts with `mem_limit: 6g` and `cpus: 4.0`. These are upper bounds, not guaranteed usage. For a small single-user deployment, try 4 GB and 2 CPUs after measuring startup and inference. Increase API memory if the container is OOM-killed. Ollama runs outside Compose here, so its model memory is separate from the API container limit but still consumes host RAM. Avoid running multiple API replicas on one Mac unless you have measured memory headroom.

## Configuration

Important environment variables are in `.env.example`:

- `GROQ_API_KEY`, `GROQ_MODEL` — Groq chat access.
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` — local Ollama endpoint/model; Compose overrides the URL for host routing.
- `MODEL_DIR` — defaults to `models/bge_m3` locally and `/app/models/bge_m3` in the container.
- `EMBEDDING_DIMENSION=1024`, `NUM_CLASSES=100` — validated against model config and artifact shapes.
- `REQUEST_TIMEOUT` — timeout for the Ollama HTTP request.

Never commit `.env` or API keys. The app listens on port 8000 and Compose publishes it on the host; restrict network exposure if running beyond a trusted development machine.
