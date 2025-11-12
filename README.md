# Aurora - Applied AI/ML Engineer Assignment

**🌐 Live URL:** https://aurora-applied-aiml-engineer.onrender.com 

**🔑 Primary endpoint:** `GET /ask?q=...` → returns `{ "answer": "..." }`

**🎥 Video recording:** https://drive.google.com/file/d/1RTp3RcGn72gTF_cRWFd34TVl7B8zBfpH/view?usp=sharing

> A minimal, production-ready web service that reads the assignment’s `/messages` API, retrieves relevant snippets, and extracts answers for date / small-count / favorites-list questions. If the fact isn’t in the data, it responds with a clear fallback—no guessing.

## 🔗 Endpoints
- Health: `/health`
- Docs (Swagger): `/docs`
- Ask (example): `/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F`
- Refresh index: `POST /refresh`
- Insights: `/insights`
- Debug tools: `/members`, `/debug?q=...`, `/probe/cars[?member=...]`


### Debug/Explainability
- `GET /members` → member counts seen in data (if present)
- `GET /debug?q=...` → show parsed question + top retrieved docs
- `GET /probe/cars[?member=...]` → locate numbers near “car(s)/vehicle(s)”
- `GET /insights` → quick dataset checks


## ✨ What this service does
A tiny web service that answers natural-language questions about member messages fetched from the provided public API (`GET /messages`).

- Retrieves all messages at startup/on first call
- Builds a lightweight **TF-IDF** index (1–2-grams)
- Detects intent (date / small count / simple list)
- Extracts an answer with transparent heuristics
- Returns JSON: `{ "answer": "..." }`

### Supported intents
- **Date** (e.g., *When is Layla planning her trip to London?*)
- **Count** (e.g., *How many cars does Vikram Desai have?*)
- **List** (e.g., *What are Amira's favorite restaurants?*)

**Fallback:** If the dataset doesn’t contain the fact, returns a clear message (see examples below).

### Example queries (actual outputs)
```bash
# Date
curl -G 'https://aurora-applied-aiml-engineer.onrender.com/ask' --data-urlencode 'q=When is Layla planning her trip to London?'
# → {"answer":"2025-11-07"}

# List (favorites)
curl -G 'https://aurora-applied-aiml-engineer.onrender.com/ask' --data-urlencode "q=What are Amira's favorite restaurants?"
# → {"answer":"Eleven Madison Park"}

# Count
curl -G 'https://aurora-applied-aiml-engineer.onrender.com/ask' --data-urlencode 'q=How many cars does Vikram Desai have?'
# → {"answer":"Sorry, I couldn’t find a specific count."}
```

> If a fact isn’t present in `/messages`, the service returns a transparent fallback.  
> Use `/debug` (evidence) and `/probe/cars` (targeted numeric probe) to verify coverage.


## 🧰 Tech Stack
- **FastAPI** + **Uvicorn** (HTTP API)
- **httpx** (robust HTTP client with redirect-following)
- **scikit-learn** (TF-IDF + cosine similarity)
- **rapidfuzz** (fuzzy name matching when names appear)
- **dateparser** (normalizing natural-language dates)


## 🚀 Run locally

> Requires **Python 3.10+** (uses modern type unions like `str | None`).

```bash
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Health Check:
```bash
http://127.0.0.1:8000/health
```

Ask a question:
```bash
curl -G 'http://127.0.0.1:8000/ask' --data-urlencode 'q=When is Layla planning her trip to London?'
```
## ✅ Deployment (Render)
- Dockerfile provided; service binds to port 8000.
- Health Check Path: /health.
- No env vars required (optional: DATA_API_BASE to override the upstream API).

**Smoke tests**
```bash
curl 'https://aurora-applied-aiml-engineer.onrender.com/health'
curl -G 'https://aurora-applied-aiml-engineer.onrender.com/ask' --data-urlencode 'q=When is Layla planning her trip to London?'
```
> Note: On the free Render tier, instances may sleep after ~15 minutes of inactivity.
> The first request can briefly return `503` with header `x-render-routing: dynamic-hibernate-error-503`.
> Retry once; it wakes and responds.



## 🧠 Design Notes — Bonus 1

### 🎯 Objectives & Guardrails
- **Deterministic**: same input ⇒ same output (no external LLM calls).
- **Explainable**: reviewers can see *why* an answer was produced (or not).
- **Keyless & lightweight**: no cloud search, no embedding services.
- **Robust to data shape**: tolerate different field names and missing members.

### 🧱 Architecture (at a glance)
1. **Ingest**: fetch from `/messages/` with `httpx` (`follow_redirects=True`, trailing slash `/messages/`).
2. **Index**: build a **TF-IDF** matrix over message text  
   `TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)`.
3. **Parse intent** from the question: `date | count | list` (regex cues).
4. **Retrieve** top-K snippets via cosine similarity (`linear_kernel`) and optional **member** filter using RapidFuzz (threshold **70**).
5. **Extract** the answer with transparent rules per intent.
6. **Return** `{ "answer": "..." }` or a clear fallback if evidence is missing.

### 🔎 Retrieval Choice: Why TF-IDF (and not embeddings/LLMs)?
**Why TF-IDF works well here**
- **Zero dependencies** beyond scikit-learn; instant cold-starts on Render.
- **Deterministic & debuggable**: you can see top-K texts in `/debug` and reason about cosine scores.
- **Right-sized** for this dataset: few thousand short messages can be indexed in milliseconds.

**What we gain vs. alternatives**
- **BM25 (Elasticsearch/OpenSearch)**: likely better lexical ranking, but heavy infra for a take-home and slower to spin up.
- **Embeddings + ANN (FAISS/Qdrant)**: better at semantics and paraphrase, but adds memory/ops and needs careful tuning; overkill here.
- **LLM extraction**: highest recall but introduces latency, cost, and non-determinism (hallucinations). This project prioritizes *trust* and *traceability*.

**Trade-offs (and mitigations)**
- TF-IDF is lexical; may miss paraphrases → use bigrams `(1,2)` and intent-specific keywords to bias retrieval.
- Short queries with rare tokens → still handled because we include the whole question (and, for dates, extra topical tokens like “trip London”).

### 🧩 Intent Detection & Extraction Rules (grounded, not magical)

**Dates**
- Tooling: `dateparser.search.search_dates`.
- Strategy: scan top candidates, return the earliest normalized **`YYYY-MM-DD`** that matches the query context.
- Fallback: look for classic “on/by/around <Month Day(, Year)>” patterns.

**Counts**
- Numeric tokens: digits and number-words `zero..twenty` via `NUM_RE`.
- **Proximity window**: ±80 chars around the number; accept if target noun is nearby  
  (plural/singular + synonyms, e.g., `{car, cars, vehicle, vehicles}`).
- Prevents phone numbers / reservation codes from being misread as counts.

**Lists (favorites)**
- Triggers: `favorite|favourite|love|likes|go-to`.
- Context bias: `restaurant|cafe|bistro|bar|grill|kitchen|pizzeria|steakhouse|...`.
- Proper-noun harvesting: capitalized spans; prefer **multi-word** names or **restaurant-like suffixes**  
  (`Bistro|Bar|Grill|Kitchen|Pizza|Pizzeria|BBQ|Deli|Trattoria|Ristorante|Cantina|Taqueria|Pub`).
- Noise filter: drop articles/months/weekdays like `The`, `November`, `Friday` so we don’t return junk.
- Output: de-dupe, preserve order, cap to **top 3** concise names.

**Names**
- If member names appear in the payload, fuzzy-match them (`WRatio ≥ 70`).
- If not present (as in this dataset), retrieval falls back to text-only ranking—no fabrication.

### 🛡️ Robustness & Failure Handling
- **Data shape tolerance**: look for text across multiple keys (`text|message|content|body`) and names across (`member_name|member|name|from|sender|author`). `pick_key` only accepts **scalar** values to avoid indexing nested dicts/lists.
- **Networking**: `httpx` with `follow_redirects=True` and trailing slash on `/messages/`.
- **Errors**:  
  - Upstream/API issues → **HTTP 502** with reason.  
  - Internal exceptions → **HTTP 500** with a concise message.
- **Lazy init**: first `/ask` builds the index; `/refresh` forces a re-fetch/reindex.

---

## 📊 Data Insights — Bonus 2
  
> The `/insights` endpoint summarizes potential issues and shows how I verified them.

### 🔍 What `/insights` checks (and why)

1) **Duplicate name variants**  
   - **What:** same member appearing with different casing/spelling (e.g., `Amira`, `AMIRA`).  
   - **Why it matters:** fragments evidence and weakens fuzzy matching for name-scoped questions.

2) **Conflicting small-integer facts**  
   - **What:** the same `(member, attribute)` has multiple small numeric values (≤ 20) across messages, for attributes like `cars/kids/pets`.  
   - **Why it matters:** prevents returning a single misleading number when the source is inconsistent.

3) **Suspicious timestamps**  
   - **What:** timestamps that lack a 4-digit year (e.g., `"Nov 7"`).  
   - **Why it matters:** date normalization can be wrong/ambiguous if year context is missing.


### 🧪 How to run it

**Live:**
```bash
curl -s 'https://aurora-applied-aiml-engineer.onrender.com/insights' | python -m json.tool
```

### Example output (your run may differ):
```json
{
  "summary": [
    {
      "duplicate_name_variants": {
        "amira": ["Amira", "AMIRA"]
      }
    },
    {
      "conflicting_small_int_facts": {
        "vikram:car": [1, 2]
      }
    },
    {
      "invalid_timestamps": [
        {"member": "Layla", "timestamp": "Nov 7"}
      ]
    }
  ]
}
```

### 🧭 How to interpret (decision guide)

- **Empty `summary`** → No issues detected by these checks; a **“Sorry, I couldn’t find …”** result is likely because the fact truly isn’t in the dataset.  
- **Duplicate name variants** → Normalize casing or map aliases to keep evidence unified.  
- **Conflicting small-integer facts** → Don’t pick a single value; keep the honest fallback or (future enhancement) answer with a **range + citation**.  
- **Suspicious timestamps** → Prefer messages that include a **year**; otherwise treat as ambiguous or return a best-effort normalized date with caution.


### 🔬 Drill-down workflow

**See the summary**
```bash
curl -s 'https://aurora-applied-aiml-engineer.onrender.com/insights' | python -m json.tool
```
**Locate raw evidence**
```bash
# Keyword/theme-only retrieval (uses /debug to surface top evidence)
curl 'https://aurora-applied-aiml-engineer.onrender.com/debug?q=restaurants'

# Member + theme (bias retrieval toward the right context)
curl 'https://aurora-applied-aiml-engineer.onrender.com/debug?q=Amira%20restaurants'
```
**Reproduce the retrieval context (evidence)**
```bash
curl 'https://aurora-applied-aiml-engineer.onrender.com/debug?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F'
```
**Targeted probe for counts**
```bash
curl 'https://aurora-applied-aiml-engineer.onrender.com/probe/cars?member=Vikram'
```
**Explain the final answer**
```bash
# Use the exact question with /debug to see parsed intent + top_docs evidence
curl 'https://aurora-applied-aiml-engineer.onrender.com/debug?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F'
```
> This sequence demonstrates that a “no answer” result is a **data reality**, not a bug.

### 🧱 How it’s implemented (brief code notes)

- **Duplicate names:** lower-case bucket of `DOC_MEMBERS` → collect distinct cased variants per bucket; report buckets with **> 1** variant.  
- **Conflicting facts:** scan `DOCS` for attribute tokens (`car(s)`, `kid(s)`, `pet(s)`, …). Extract numbers via `NUM_RE` (digits + number words), keep values `≤ 20`, normalize attribute to **singular**, group by `(member_lower, attr)` and report groups with **> 1** distinct value.  
- **Invalid timestamps:** run `extract_time` across candidate fields; flag values missing a `\d{4}` year.

These are read-only, **O(N)** passes over the messages and **do not** affect `/ask` latency.

### ⚙️ Extensions

- **Citations in `/ask?explain=1`** — include message IDs/offsets so answers quote exact spans.
- **Semantic rerank after TF-IDF** — re-score top-K with a tiny in-process encoder (e.g., MiniLM) to catch paraphrases.
- **Alias/normalization table** — unify nicknames/initials (e.g., Vik ↔︎ Vikram) for stronger member matching.
- **Temporal disambiguation** — prefer dates that co-occur with a year; otherwise bias toward the most recent plausible year.
- **Outlier & schema checks** — flag unlikely numbers (e.g., `>10` cars) and monitor missing/nullable fields over time.





