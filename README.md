## RAG Library console application

### Features
- Loads a book dataset from JSON (auto-creates a small default if missing)
- Keyword retriever with synonym expansion, bigram (phrase) bonus, fuzzy fallback
- Optional tags in each book entry (weighted higher for recall)
- Two answer modes:
  • Identify mode → returns exact book title + snippet citation (e.g., Title [1])
  • Explain mode → one-sentence factual answer + title + citation (e.g., Fact — Title [1])
- Answer generation with Hugging Face (FLAN-T5 small/base)
- Safer generation defaults: beam search, no-repeat n-gram, temp=0
- Auto device selection: Apple MPS (Mac), CUDA (if available), or CPU
- Choose a Hub model (e.g., google/flan-t5-small) or a local on-disk folder

### Quickstart:
  - python -m venv .venv 
  - Linux/MacOS: source .venv/bin/activate 
  - Windows: .venv\Scripts\activate
  - pip install --upgrade pip
  - pip install torch==2.3.1 transformers==4.43.3
  - python rag_library.py --help

### Running the application:
##### BASIC RUNS
##### Default: uses auto-created books.json + flan-t5-small (hub)
- python rag_library.py

##### Use your own dataset file
- python rag_library.py --data books.json

##### MODEL SELECTION
##### Small hub model (fast, lightweight, default)
- python rag_library.py --model google/flan-t5-small


##### Load from a local model folder you saved with save_pretrained
- python rag_library.py --local-model-path ./flan-t5-small-local


##### RETRIEVAL SETTINGS
##### Use only the single best-matching snippet
- python rag_library.py --topk 1

##### Default: include top 3 snippets
- python rag_library.py --topk 3

##### Include top 5 snippets (more coverage, more noise)
- python rag_library.py --topk 5


##### DEMO + DEBUGGING
##### Run predefined demo questions then go interactive
- python rag_library.py --demo

##### Show retrieved snippets (context) before answering
- python rag_library.py --show-context

##### Combine demo + show-context
- python rag_library.py --demo --show-context

##### OFFLINE MODE
##### Force offline mode (no internet, only local cache/folders)
- python rag_library.py --offline

##### Offline + local model folder
- python rag_library.py --local-model-path ./flan-t5-small-local --offline

##### FULL COMBO EXAMPLES
##### Demo with hub base model, show retrieved context
- python rag_library.py --model google/flan-t5-small --demo --show-context

##### Custom dataset, top-5 retrieval, with local model folder
- python rag_library.py --local-model-path ./flan-t5-small-local --data books.json --topk 5

##### Custom dataset + offline (useful on air-gapped machines)
- python rag_library.py --local-model-path ./flan-t5-small-local --data books.json --offline

  