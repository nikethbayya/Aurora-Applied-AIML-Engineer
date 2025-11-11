import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import httpx
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dateparser.search import search_dates

API_BASE = os.getenv("DATA_API_BASE", "https://november7-730026606190.europe-west1.run.app")
MESSAGES_URL = f"{API_BASE.rstrip('/')}/messages/"

app = FastAPI(title="Aurora QA Service", version="1.0")

# -------- In-memory state --------
RAW_MESSAGES: List[Dict[str, Any]] = []
DOCS: List[str] = []
DOC_MEMBERS: List[str] = []
MEMBERS: List[str] = []
VECTORIZER: Optional[TfidfVectorizer] = None
TFIDF = None

# -------- Utilities --------
TEXT_KEYS = ("text", "message", "content", "body")
NAME_KEYS = ("member_name", "member", "name", "from", "sender", "author")
TIME_KEYS = ("timestamp", "created_at", "ts", "time")

NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20
}
NUM_RE = re.compile(r"\b(?:(?:\d{1,3}(?:,\d{3})*|\d+)|" + "|".join(NUM_WORDS.keys()) + r")\b", re.I)

def pick_key(d: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in d and isinstance(d[k], (str, int, float)):
            return k
    return None


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def num_from_token(tok: str) -> Optional[int]:
    tok = tok.lower().strip()
    if tok in NUM_WORDS: return NUM_WORDS[tok]
    try:
        return int(tok.replace(",", ""))
    except:
        return None

def extract_text(d: Dict[str,Any]) -> str:
    k = pick_key(d, TEXT_KEYS)
    return normalize_whitespace(str(d.get(k,""))) if k else ""

def extract_member(d: Dict[str,Any]) -> str:
    k = pick_key(d, NAME_KEYS)
    return normalize_whitespace(str(d.get(k,""))) if k else ""

def extract_time(d: Dict[str,Any]) -> Optional[str]:
    k = pick_key(d, TIME_KEYS)
    if not k: return None
    return str(d.get(k))

def fetch_messages() -> List[Dict[str,Any]]:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        r = client.get(MESSAGES_URL)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "messages" in data and isinstance(data["messages"], list):
                return data["messages"]
            
            for v in data.values():
                if isinstance(v, list) and all(isinstance(it, dict) for it in v):
                    return v
        raise ValueError("Unexpected /messages payload shape")

def build_index(rows: List[Dict[str,Any]]):
    global RAW_MESSAGES, DOCS, DOC_MEMBERS, MEMBERS, VECTORIZER, TFIDF
    RAW_MESSAGES = rows
    DOCS = [extract_text(r) for r in rows]
    DOC_MEMBERS = [extract_member(r) for r in rows]
    MEMBERS = sorted(list({m for m in DOC_MEMBERS if m}))
    VECTORIZER = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    TFIDF = VECTORIZER.fit_transform(DOCS) if DOCS else None

def fuzzy_best_member(name_hint: str) -> Optional[str]:
    if not name_hint or not MEMBERS:
        return None
    match = process.extractOne(name_hint, MEMBERS, scorer=fuzz.WRatio)
    if not match: return None
    cand, score, _ = match
    return cand if score >= 70 else None 


def topk_docs(query: str, member: Optional[str], k: int=20) -> List[int]:
    if TFIDF is None or VECTORIZER is None or not DOCS:
        return []
    qv = VECTORIZER.transform([query])
    sims = linear_kernel(qv, TFIDF).ravel()
    idxs = np.argsort(-sims)
    if member:
        idxs = [i for i in idxs if DOC_MEMBERS[i] == member]
    return idxs[:k]

# -------- Question understanding --------
class Intent:
    DATE = "date"
    COUNT = "count"
    LIST = "list"
    UNKNOWN = "unknown"

def parse_question(q: str) -> Dict[str,Any]:
    qn = q.strip()
    name_guess = None
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", qn):
        span = m.group(1)
        if len(span.split()) >= 1:
            name_guess = span 
    intent = Intent.UNKNOWN
    if re.search(r"\bwhen\b", qn, re.I): intent = Intent.DATE
    elif re.search(r"\bhow\s+many\b", qn, re.I): intent = Intent.COUNT
    elif re.search(r"\bwhat\s+are\b", qn, re.I) and re.search(r"\bfavorite\b", qn, re.I): intent = Intent.LIST

    target = None
    if intent == Intent.COUNT:
        m = re.search(r"how\s+many\s+([a-zA-Z\-]+)", qn, re.I)
        if m: target = m.group(1).lower()
    elif intent == Intent.LIST:
        m = re.search(r"favorite\s+([a-zA-Z\- ]+?)\??$", qn.strip(), re.I)
        if m: target = m.group(1).strip().lower()
    elif intent == Intent.DATE:
        # A few topical anchors from the question helps for retrieval (e.g., trip, London)
        target = " ".join(w for w in re.findall(r"[A-Za-z]+", qn) if w.lower() not in {"when","is","are","the","a","an","to","of","for","does"})

    return {"name_guess": name_guess, "intent": intent, "target": target}

# -------- Answer extraction --------
def answer_date(candidates: List[str]) -> Optional[str]:
    for txt in candidates:
        found = search_dates(txt, languages=["en"])
        if found:
            # return the first normalized date string with its snippet
            dt_str = found[0][1].strftime("%Y-%m-%d")
            return f"{dt_str}"
    
    m = re.search(r"\b(on|by|around)\s+([A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)", " ".join(candidates))
    if m:
        return m.group(2)
    return None


def answer_count(candidates: List[str], target: str) -> Optional[str]:
    tg = (target or "").lower()
    tg_variants = {tg, tg.rstrip("s"), "car", "cars", "vehicle", "vehicles"}
    best = None
    for txt in candidates:
        for m in NUM_RE.finditer(txt):
            num = num_from_token(m.group(0))
            if num is None: 
                continue
            window = txt[max(0, m.start()-80): m.end()+80].lower()  
            if any(tv for tv in tg_variants if tv and tv in window):
                best = num
                break
        if best is not None:
            break
    return str(best) if best is not None else None


def answer_list(candidates: List[str], target: str) -> Optional[str]:
    trigger_re = re.compile(r"\b(favorite|favourite|love|loves|like|likes|go-?to)\b", re.I)
    food_ctx_re = re.compile(
        r"\b(restaurant|cafe|caf\u00e9|café|bistro|diner|bar|grill|kitchen|pizzeria|pizza|steakhouse|deli|canteen|trattoria|ristorante|taqueria|bbq|pub|eatery|cuisine|food)\b",
        re.I,
    )
    name_re = re.compile(r"\b([A-Z][\w'&]+(?:\s+[A-Z][\w'&]+)*)\b")
    suffix_re = re.compile(r"\b(Cafe|Caf\u00e9|Café|Bistro|Grill|Kitchen|Pizza|Pizzeria|BBQ|Bar|Deli|Steakhouse|Trattoria|Ristorante|Cantina|Taqueria|Pub)\b")
    drop = {
        "The","A","An","And","Of","In","On","At","For","To","From","By","With","My","Our",
        "January","February","March","April","May","June","July","August","September","October","November","December",
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    }

    picks = []
    for txt in candidates:
        sentences = re.split(r"(?<=[\.\!\?])\s+", txt)
        for s in sentences:
            s_low = s.lower()
            if not (trigger_re.search(s_low) or (target and target.lower() in s_low)):
                continue
            has_food_ctx = bool(food_ctx_re.search(s_low))

            cands = []
            for m in name_re.finditer(s):
                cand = m.group(1).strip()
                if cand in drop or len(cand) < 3:
                    continue
                cands.append(cand)

            if not cands:
                continue

            filtered = []
            if has_food_ctx:
                for c in cands:
                    if c not in drop:
                        filtered.append(c)
            else:
                for c in cands:
                    multiword = " " in c
                    has_suffix = bool(suffix_re.search(c))
                    if has_suffix or multiword:
                        filtered.append(c)

            m = re.search(r"\b(?:are|is)\s+([^\.]+)", s_low)
            if m:
                seg = s[m.start(1):m.end(1)]
                for piece in re.split(r",| and ", seg):
                    piece = piece.strip()
                    if name_re.match(piece):
                        if (has_food_ctx or suffix_re.search(piece) or " " in piece) and piece not in filtered:
                            filtered.append(piece)

            for f in filtered:
                if f not in picks:
                    picks.append(f)

    return ", ".join(picks[:3]) if picks else None



def synthesize_answer(q: str) -> str:
    if not DOCS:
        raise HTTPException(503, "Index is not ready")
    parsed = parse_question(q)
    member = fuzzy_best_member(parsed["name_guess"]) if parsed["name_guess"] else None

    # retrieval query is the question plus target for extra signal
    retrieval_query = q
    if parsed["target"]:
        retrieval_query += " " + parsed["target"]

    idxs = topk_docs(retrieval_query, member, k=30)
    candidates = [DOCS[i] for i in idxs]

    # if name was missing or fuzzy match failed, try to infer by message hits
    if not member:
        hit_members = [DOC_MEMBERS[i] for i in idxs if DOC_MEMBERS[i]]
        if hit_members:
            # most frequent member among top hits
            member = max(set(hit_members), key=hit_members.count)

    intent = parsed["intent"]
    target = parsed["target"]

    if intent == Intent.DATE:
        ans = answer_date(candidates)
        return ans or "Sorry, I couldn’t find a specific date."
    elif intent == Intent.COUNT:
        ans = answer_count(candidates, target)
        return ans or "Sorry, I couldn’t find a specific count."
    elif intent == Intent.LIST:
        ans = answer_list(candidates, target)
        return ans or "Sorry, I couldn’t find a clear list."
    else:
        # return the best snippet
        return (candidates[0][:180] + "…") if candidates else "Sorry, I couldn’t find that."

# -------- FastAPI endpoints --------
@app.get("/health")
def health():
    return {"status": "ok", "docs": len(DOCS)}

@app.post("/refresh")
def refresh():
    rows = fetch_messages()
    build_index(rows)
    return {"ok": True, "docs": len(DOCS), "members": len(MEMBERS)}

@app.get("/ask")
def ask(q: str = Query(..., description="Natural-language question")):
    try:
        if not DOCS:
            rows = fetch_messages()
            build_index(rows)
        answer = synthesize_answer(q)
        return JSONResponse({"answer": answer})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Upstream error calling /messages: {e}")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")

# -------- quick data insights--------
@app.get("/insights")
def insights():
    if not RAW_MESSAGES:
        rows = fetch_messages()
        build_index(rows)

    issues = []
    # 1) duplicate member names with conflicting cased variants
    name_counts = {}
    for m in DOC_MEMBERS:
        if not m: continue
        k = m.lower()
        name_counts.setdefault(k, set()).add(m)
    dup_variants = {k:v for k,v in name_counts.items() if len(v) > 1}
    if dup_variants:
        issues.append({"duplicate_name_variants": {k:list(v) for k,v in dup_variants.items()}})

    # 2) conflicting small integer facts (“cars”, “children”, etc.)
    facts = {} 
    small_attrs = ["car", "cars", "children", "kid", "kids", "pet", "pets", "dog", "dogs"]
    for r, txt in zip(RAW_MESSAGES, DOCS):
        member = extract_member(r).lower()
        for attr in small_attrs:
            if attr in txt.lower():
                for m in NUM_RE.finditer(txt):
                    n = num_from_token(m.group(0))
                    if n is not None and n <= 20:
                        facts.setdefault((member, attr.rstrip("s")), set()).add(n)
    conflicts = {f"{m}:{a}": sorted(list(vals)) for (m,a),vals in facts.items() if len(vals) > 1}
    if conflicts:
        issues.append({"conflicting_small_int_facts": conflicts})

    # 3) obviously invalid timestamps
    bad_ts = []
    for r in RAW_MESSAGES:
        t = extract_time(r)
        if t and not re.search(r"\d{4}", t):
            bad_ts.append({"member": extract_member(r), "timestamp": t})
    if bad_ts:
        issues.append({"invalid_timestamps": bad_ts[:10]})

    return {"summary": issues}

@app.get("/members")
def members():
    if not DOCS:
        rows = fetch_messages(); build_index(rows)
    counts = {}
    for m in DOC_MEMBERS:
        if m:
            counts[m] = counts.get(m, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return {"members": [{"name": k, "messages": v} for k, v in ordered]}

@app.get("/debug")
def debug(q: str):
    if not DOCS:
        rows = fetch_messages(); build_index(rows)
    parsed = parse_question(q)
    member = fuzzy_best_member(parsed.get("name_guess") or "") if parsed.get("name_guess") else None
    idxs = topk_docs(q, member, k=5)
    tops = [{"member": DOC_MEMBERS[i], "text": DOCS[i][:240]} for i in idxs]
    return {"parsed": parsed, "member": member, "top_docs": tops}

@app.get("/probe/cars")
def probe_cars(member: str | None = None, limit: int = 20):
    """
    Find messages that look like a car count:
    a number (digit or word) within ~60 chars of 'car(s)' / 'vehicle(s)'.
    Optionally require that the message mentions a member name (e.g., 'Vikram').
    """
    if not DOCS:
        rows = fetch_messages(); build_index(rows)

    hits = []
    car_words = re.compile(r"\b(car|cars|vehicle|vehicles)\b", re.I)

    for i, txt in enumerate(DOCS):
        low = txt.lower()
        if member and member.lower() not in low:
            continue
        found = None
        for m in NUM_RE.finditer(txt):
            window = low[max(0, m.start()-60): m.end()+60]
            if car_words.search(window):
                found = {
                    "idx": i,
                    "number_token": m.group(0),
                    "snippet": txt[max(0, m.start()-90): m.end()+90]
                }
                break
        if found:
            hits.append(found)
            if len(hits) >= limit:
                break

    return {"member_filter": member, "hits": hits, "total_hits": len(hits)}


