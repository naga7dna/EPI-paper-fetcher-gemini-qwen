
import os
import re
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Optional

import requests
import pandas as pd
from pydantic import BaseModel, Field
from Bio import Entrez
from openai import OpenAI

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
try:
    import GEOparse
    HAS_GEOPARSE = True
except ImportError:
    HAS_GEOPARSE = False


GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",
                            "AIzaSyAjvtxGK8Tv4Pz_0JflH1fDizYzAQx8vCM")  
GEMINI_MODEL    = "gemini-3.6-flash"     
GEMINI_MIN_INTERVAL = 7.0                
GEMINI_COOLDOWN = 180                     
# PROVIDER        = "gemini"    
PROVIDER = "ollama"            
OLLAMA_MODEL    = "qwen2.5:7b-instruct"  

EMAIL           = "naga.email@example.com"   
NCBI_API_KEY    = None                      

MAX_RESULTS     = 10        
SCORE_THRESHOLD = 6          
TOP_K_EXTRACT   = 2         
FETCH_BATCH     = 2       
LLM_TEMP        = 0.0      
SCORE_BATCH     = 1        
USE_PUBTATOR    = True     
USE_GEO         = True     
GEO_MAX_SERIES_PER_PAPER = 1
GEO_DATASET_HITS = 1

PT_API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

# ---------------------------------------------------------------- LOGGING ---
def setup_logging() -> str:
    log_file = f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);               sh.setFormatter(fmt)
    root.addHandler(fh); root.addHandler(sh)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # hide per-request INFO
    return log_file

log = logging.getLogger("pipeline")

# ------------------------------------------------- LLM OUTPUT SCHEMAS -------
class SearchPlan(BaseModel):
    intent_summary: str = ""
    detected_pmids: List[str] = Field(default_factory=list)
    disease_focus: str = ""
    comparison_groups: str = ""
    omics_methods: str = ""
    year_from: Optional[int] = None
    pubmed_query: str = ""

class Verdict(BaseModel):
    pmid: str
    score: int
    reason: str = ""

class VerdictList(BaseModel):
    verdicts: List[Verdict] = Field(default_factory=list)

class Extraction(BaseModel):
    study_type: str = ""
    datasets_mentioned: List[str] = Field(default_factory=list)
    accessions: List[str] = Field(default_factory=list)
    experimental_conditions: List[str] = Field(default_factory=list)
    n_control: Optional[int] = None
    n_metastasis: Optional[int] = None
    n_total: Optional[int] = None
    evidence_quote: str = ""
    extraction_confidence: str = ""

class CharLabel(BaseModel):
    text: str = ""
    group: str = ""

class CharLabelList(BaseModel):
    labels: List[CharLabel] = Field(default_factory=list)

# ------------------------------------------------------------ LLM CORE ------
_gemini_client = None
_last_gemini_call = 0.0
_gemini_cooldown_until = 0.0                      # rest Gemini until this time
_gemini_candidates: List[str] = [GEMINI_MODEL]    # fallback models, set at startup

def _schema_json(schema):
    try:
        return schema.model_json_schema()          # pydantic v2
    except AttributeError:
        return schema.schema()                     # pydantic v1

def _validate(schema, text):
    try:
        return schema.model_validate_json(text)
    except AttributeError:
        return schema.parse_raw(text)

def _parse_json(text, schema):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return _validate(schema, text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)  # salvage embedded JSON
        if m:
            return _validate(schema, m.group(0))
        raise

def _transient(err_text: str) -> bool:
    """Temporary error worth waiting out?"""
    t = str(err_text).lower()
    return any(k in t for k in ("503", "429", "high demand", "overloaded",
                                "rate limit", "timeout", "temporarily",
                                "service unavailable", "connection"))

def list_gemini_models() -> List[str]:
    """Ask Google which chat models this API key can actually use."""
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": GEMINI_API_KEY}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/")[-1]
                 for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        log.info(f"[gemini] usable models: {', '.join(names) or 'NONE'}")
        return names
    except Exception as e:
        log.warning(f"[gemini] could not list models: {e}")
        return []

def pick_gemini_model() -> str:
    """Choose best available model; build the fallback candidate list."""
    global _gemini_candidates
    prefer = [GEMINI_MODEL, "gemini-3.6-flash", "gemini-flash-latest",
              "gemini-2.5-flash", "gemini-2.0-flash"]
    avail = list_gemini_models()
    if not avail:
        _gemini_candidates = [GEMINI_MODEL]
        return GEMINI_MODEL
    chosen = next((p for p in dict.fromkeys(prefer) if p in avail), GEMINI_MODEL)
    if chosen not in avail:
        flash = sorted(m for m in avail if "flash" in m)
        chosen = flash[-1] if flash else GEMINI_MODEL
    _gemini_candidates = ([chosen]
                          + [m for m in avail if m != chosen and "flash" in m][:2])
    log.info(f"[gemini] model candidates (tried in order): {_gemini_candidates}")
    return chosen

def _switch_model_on_deprecation(err_text: str) -> None:
    """If Google's 404 names a replacement model, switch to it."""
    global GEMINI_MODEL
    m = re.search(r"models/([\w.-]+) for the latest", err_text)
    if m and m.group(1) != GEMINI_MODEL:
        log.warning(f"[gemini] model deprecated -> switching to '{m.group(1)}'")
        GEMINI_MODEL = m.group(1)
        if GEMINI_MODEL not in _gemini_candidates:
            _gemini_candidates.insert(0, GEMINI_MODEL)

def _rotate_gemini_model() -> bool:
    """After repeated failures, try the next model on the candidate list."""
    global GEMINI_MODEL
    if len(_gemini_candidates) < 2:
        return False
    try:
        i = _gemini_candidates.index(GEMINI_MODEL)
    except ValueError:
        i = -1
    nxt = _gemini_candidates[(i + 1) % len(_gemini_candidates)]
    if nxt == GEMINI_MODEL:
        return False
    GEMINI_MODEL = nxt
    log.warning(f"[gemini] overloaded -> trying model '{nxt}' instead")
    return True

def _call_gemini(messages, schema):
    global _last_gemini_call, _gemini_client
    wait = GEMINI_MIN_INTERVAL - (time.time() - _last_gemini_call)
    if wait > 0:
        time.sleep(wait)
    _last_gemini_call = time.time()

    if _gemini_client is None:
        # max_retries=0: WE control retries (library's fast retries are
        # useless against 'high demand'; we wait longer instead)
        _gemini_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY, max_retries=0)

    msgs = [dict(m) for m in messages]
    msgs[0] = {"role": "system",
               "content": msgs[0]["content"] +
               "\n\nOUTPUT FORMAT: reply with ONE JSON object only (no markdown, "
               "no commentary) that validates against this JSON schema:\n"
               + json.dumps(_schema_json(schema))}
    r = _gemini_client.chat.completions.create(
        model=GEMINI_MODEL, messages=msgs, temperature=LLM_TEMP,
        response_format={"type": "json_object"})
    return r.choices[0].message.content

def _call_ollama(messages, schema):
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           format=_schema_json(schema),
                           options={"temperature": LLM_TEMP})
    except Exception:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           format="json", options={"temperature": LLM_TEMP})
    return resp["message"]["content"]

def llm(messages, schema):
    """Primary provider; on overload: wait, rotate model, cool down, and keep
    working on the fallback provider; Gemini returns automatically after cooldown."""
    global _gemini_cooldown_until

    order = []
    gemini_ready = time.time() >= _gemini_cooldown_until
    if PROVIDER == "gemini":
        if gemini_ready:
            order.append("gemini")
        else:
            log.info("[gemini] cooling down — fallback provider goes first")
        if HAS_OLLAMA:
            order.append("ollama")
    else:
        if PROVIDER == "ollama" and not HAS_OLLAMA:
            raise RuntimeError("PROVIDER='ollama' but ollama is not installed")
        order.append(PROVIDER)
        if PROVIDER != "gemini":
            order.append("gemini")
    if not order:
        raise RuntimeError("No LLM provider available (no Gemini key, no Ollama)")

    errors = []
    for prov in order:
        for attempt in range(1, 4):
            t = time.time()
            try:
                content = (_call_gemini if prov == "gemini" else _call_ollama)(
                    messages, schema)
                obj = _parse_json(content, schema)
                log.info(f"[{prov}] {schema.__name__} OK ({time.time()-t:.1f}s)")
                return obj
            except Exception as e:
                et = str(e)
                if prov == "gemini":
                    if "no longer available" in et:
                        _switch_model_on_deprecation(et)
                    if "401" in et or "403" in et or "API key" in et:
                        log.error("[gemini] API key problem -> using fallback "
                                  "provider for the next hour")
                        _gemini_cooldown_until = time.time() + 3600
                        break
                if _transient(et):
                    wait = 20 * attempt        # 20s / 40s / 60s for 503 etc.
                    log.warning(f"[{prov}] transient error, attempt {attempt}/3, "
                                f"waiting {wait}s: {et[:150]}")
                else:
                    wait = 3 * attempt
                    log.warning(f"[{prov}] attempt {attempt}/3 failed "
                                f"(retry in {wait}s): {et[:150]}")
                errors.append(f"{prov}#{attempt}: {et[:150]}")
                time.sleep(wait)

        # provider exhausted -> one last shot on a DIFFERENT Gemini model
        if prov == "gemini" and _rotate_gemini_model():
            try:
                content = _call_gemini(messages, schema)
                obj = _parse_json(content, schema)
                log.info(f"[gemini] OK after switching to '{GEMINI_MODEL}'")
                return obj
            except Exception as e:
                errors.append(f"gemini-rotate: {e}")

        if prov == "gemini":
            _gemini_cooldown_until = time.time() + GEMINI_COOLDOWN
            if len(order) > 1:
                log.warning(f"[gemini] still failing -> {GEMINI_COOLDOWN}s "
                            f"cooldown; continuing with fallback provider")

    raise RuntimeError("LLM failed on all providers. Last errors: "
                       + " | ".join(errors[-3:]))

# --------------------------------------------------------- PUBTATOR3 --------
def pubtator_search(text: str, size: int = 100) -> List[str]:
    """Full-text search over PubMed (incl. PMC full text). No key needed."""
    try:
        r = requests.get(f"{PT_API}/search/",
                         params={"text": text, "size": size}, timeout=60)
        r.raise_for_status()
        js = r.json()
        pmids = js.get("pmids") if isinstance(js, dict) else []
        if not pmids and isinstance(js, dict) and isinstance(js.get("results"), list):
            pmids = [x.get("pmid") for x in js["results"] if x.get("pmid")]
        pmids = [str(p) for p in (pmids or [])][:size]
        log.info(f"[PubTator3] search '{text[:50]}' -> {len(pmids)} PMIDs")
        return pmids
    except Exception as e:
        log.warning(f"[PubTator3] search failed (continuing without it): {e}")
        return []

def pubtator_annotate(pmids: List[str]) -> dict:
    """NCBI's own gene/disease/cell-line annotations per PMID (TSV format)."""
    out = {}
    for i in range(0, len(pmids), 100):
        chunk = ",".join(p for p in pmids[i:i+100] if p)
        if not chunk:
            continue
        try:
            # pmids belong in the QUERY STRING, not the URL path (v3 fix)
            url = (f"{PT_API}/publications/export/pubtator"
                   f"?pmids={chunk}&concepts=gene,disease,cellline")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            for line in r.text.splitlines():
                f = line.split("\t")
                if len(f) != 7:
                    continue      # skips 'PMID|t|title' / 'PMID|a|abstract' lines
                pmid, _sec, etype, _id, _s, _e, text = f
                d = out.setdefault(pmid, {"genes": set(), "diseases": set(),
                                          "celllines": set()})
                if etype == "Gene":       d["genes"].add(text)
                elif etype == "Disease":  d["diseases"].add(text)
                elif etype == "CellLine": d["celllines"].add(text)
            time.sleep(1)
        except Exception as e:
            log.warning(f"[PubTator3] annotation chunk {i//100} failed: {e}")
    log.info(f"[PubTator3] got annotations for {len(out)}/{len(pmids)} papers")
    return out

def add_pubtator_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ann = pubtator_annotate(df["PMID"].tolist())
    df["Genes"]    = df["PMID"].map(lambda p: "; ".join(
        sorted(ann.get(p, {}).get("genes", []))[:15]))
    df["Diseases"] = df["PMID"].map(lambda p: "; ".join(
        sorted(ann.get(p, {}).get("diseases", []))[:10]))
    return df

# ------------------------------------------------------------- GEO ----------
_geo_cache = {}

def _kw_classify(ch: str) -> str:
    c = ch.lower()
    if "metast" in c:
        return "metastasis"
    if any(k in c for k in ("normal", "adjacent", "control", "non-tumor",
                            "non-tumour", "healthy", "non-neoplastic")):
        return "control"
    if any(k in c for k in ("tumor", "tumour", "carcinoma", "cancer")):
        return "tumor"
    return "other"

GEO_CLASS_SYSTEM = """You classify sample descriptions from a genomic dataset.
For EACH numbered description decide its group:
- "control": normal / healthy / adjacent non-tumor / control tissue
- "metastasis": metastatic lesion or metastatic sample (any site)
- "tumor": primary tumor / cancer sample (not metastatic, not control)
- "other": cell line, unknown, technical replicate, anything else
Return JSON {"labels":[{"text":"<exact original text>","group":"..."}]} with one
entry per input, echoing each text EXACTLY as given."""

def llm_classify_characteristics(unique_chars: List[str]) -> dict:
    u = [re.sub(r"\s+", " ", c).strip() for c in unique_chars]
    listing = "\n".join(f"{i+1}. {c}" for i, c in enumerate(u))
    out = llm([{"role": "system", "content": GEO_CLASS_SYSTEM},
               {"role": "user", "content": f"SAMPLE DESCRIPTIONS:\n{listing}"}],
              CharLabelList)
    norm = {c.lower(): c for c in u}
    mapping = {}
    for lab in out.labels:
        key = re.sub(r"\s+", " ", lab.text or "").strip().lower()
        grp = (lab.group or "other").strip().lower()
        if grp not in ("control", "metastasis", "tumor", "other"):
            grp = "other"
        if key in norm:
            mapping[norm[key]] = grp
    return mapping

def geo_group_counts(gse_id: str):
    """Download one GEO series; count control vs metastasis vs tumor samples."""
    if not HAS_GEOPARSE:
        return None
    gse_id = gse_id.upper().strip()
    if gse_id in _geo_cache:
        return _geo_cache[gse_id]
    try:
        g = GEOparse.get_GEO(geo=gse_id, destdir="geo_cache", silent=True)
    except Exception as e:
        log.warning(f"[GEO] download failed for {gse_id}: {e}")
        _geo_cache[gse_id] = None
        return None

    sample_chars = {}
    for name, gsm in g.gsms.items():
        raw = gsm.metadata.get("characteristics_ch1") or gsm.metadata.get("title") or []
        chars = re.sub(r"\s+", " ", " | ".join(str(x) for x in raw)).strip()
        sample_chars[name] = chars
    unique = sorted({c for c in sample_chars.values() if c})
    if not unique:
        log.warning(f"[GEO] {gse_id}: no sample characteristics found")
        _geo_cache[gse_id] = None
        return None

    labels = {c: _kw_classify(c) for c in unique}          # keyword baseline
    try:
        labels.update(llm_classify_characteristics(unique))  # LLM refinement
    except Exception as e:
        log.warning(f"[GEO] LLM classification failed for {gse_id} ({e}); "
                    f"using keywords only")

    counts = {"control": 0, "metastasis": 0, "tumor": 0, "other": 0}
    for c in sample_chars.values():
        counts[labels.get(c, "other")] += 1
    res = {"n_total": len(sample_chars), "n_control": counts["control"],
           "n_metastasis": counts["metastasis"], "n_tumor": counts["tumor"],
           "n_other": counts["other"]}
    log.info(f"[GEO] {gse_id}: total={res['n_total']} control={res['n_control']} "
             f"metastasis={res['n_metastasis']} tumor={res['n_tumor']} "
             f"other={res['n_other']}")
    _geo_cache[gse_id] = res
    return res

def geo_search_datasets(disease: str, extra: str = "", n: int = 15) -> pd.DataFrame:
    """Search NCBI GEO DataSets (db=gds) for series about the disease."""
    if not disease:
        return pd.DataFrame()
    try:
        term = f'"{disease}"[All Fields]'
        if extra:
            term += f' AND ({extra})'
        h = Entrez.esearch(db="gds", term=term, retmax=n * 3)
        ids = Entrez.read(h)["IdList"]; h.close()
        if not ids:
            return pd.DataFrame()
        h = Entrez.esummary(db="gds", id=",".join(ids))
        rows = []
        for s in Entrez.read(h):
            if str(s.get("entrytype", "")).upper() != "GSE":
                continue
            pms = s.get("PubMedIds") or s.get("PubmedIds") or []
            rows.append({
                "Accession": str(s.get("Accession", "")),
                "Title": str(s.get("title", "")),
                "Summary": str(s.get("summary", ""))[:200],
                "N_Samples": s.get("n_samples", ""),
                "Organism": str(s.get("taxon", "")),
                "Type": str(s.get("gdstype", s.get("gdsType", ""))),
                "PMIDs": ";".join(str(x) for x in pms),
            })
            if len(rows) >= n:
                break
        h.close()
        log.info(f"[GEO] dataset search '{disease}' -> {len(rows)} series")
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"[GEO] dataset search failed: {e}")
        return pd.DataFrame()

def geo_cross_check(result: pd.DataFrame) -> pd.DataFrame:
    """For each paper, check its GSE accessions in GEO for exact sample counts."""
    if not HAS_GEOPARSE:
        log.warning("[GEO] GEOparse not installed (pip install GEOparse) — "
                    "skipping GEO sample cross-check")
        return result
    log.info("[GEO] Cross-checking extracted accessions against GEO ...")
    result = result.copy()
    tot, ctl, met, src = [], [], [], []
    for _, row in result.iterrows():
        accs = [a.strip().upper() for a in
                re.split(r"[;,]", str(row.get("Accessions") or "")) if a.strip()]
        gses = [a for a in accs if a.startswith("GSE")][:GEO_MAX_SERIES_PER_PAPER]
        got, used = None, ""
        for gse in gses:
            got = geo_group_counts(gse)
            if got:
                used = gse
                break
        if got:
            tot.append(got["n_total"]); ctl.append(got["n_control"])
            met.append(got["n_metastasis"])
        else:
            tot.append(None); ctl.append(None); met.append(None)
        src.append(used or "-")
    result["GEO_Series"] = src
    result["GEO_N_Total"] = tot
    result["GEO_N_Control"] = ctl
    result["GEO_N_Metastasis"] = met
    return result

def finalize_sample_counts(result: pd.DataFrame) -> pd.DataFrame:
    """GEO numbers win over abstract numbers when both exist."""
    result = result.copy()
    for col in ("N_Control", "N_Metastasis", "N_Total",
                "GEO_N_Total", "GEO_N_Control", "GEO_N_Metastasis"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    for base, fin in (("N_Control", "N_Control_Final"),
                      ("N_Metastasis", "N_Metastasis_Final"),
                      ("N_Total", "N_Total_Final")):
        geo_col = "GEO_" + base
        if geo_col in result.columns:
            result[fin] = result[geo_col].combine_first(result[base])
        else:
            result[fin] = result[base]
    return result

# ------------------------------------------------------------ STAGE 1 -------
STAGE1_SYSTEM = """You convert a researcher's free-form request into parameters
for a PubMed search. Return JSON with:
- intent_summary: one sentence describing what the researcher wants.
- detected_pmids: any PubMed IDs (6-9 digit numbers) the user explicitly gave.
- disease_focus, comparison_groups, omics_methods: short strings.
- year_from: integer earliest publication year, or null.
- pubmed_query: ONE valid PubMed query following these rules:
    * Boolean operators UPPERCASE: AND, OR, NOT
    * Scope free-text terms with [tiab]; multi-word phrases in double quotes
    * If the user pasted an article title, search it in quotes with [Title]
    * Exclude reviews with NOT "Review"[Publication Type]
    * If year_from is set, append AND ("YYYY"[PDAT] : "3000"[PDAT])
    * Keep under 60 terms."""

def stage1_plan(user_input: str) -> SearchPlan:
    log.info("[Stage 1] Interpreting request with LLM ...")
    t0 = time.time()
    plan = llm([{"role": "system", "content": STAGE1_SYSTEM},
                {"role": "user", "content": user_input}], SearchPlan)
    log.info(f"[Stage 1] done in {time.time()-t0:.1f}s")
    log.info(f"  intent    : {plan.intent_summary}")
    log.info(f"  disease   : {plan.disease_focus}")
    log.info(f"  groups    : {plan.comparison_groups}")
    log.info(f"  omics     : {plan.omics_methods}")
    log.info(f"  year_from : {plan.year_from}")
    if plan.detected_pmids:
        log.info(f"  pmids     : {', '.join(plan.detected_pmids)}")
    log.info(f"  query     : {plan.pubmed_query}")
    return plan

def confirm_query(plan: SearchPlan) -> str:
    try:
        choice = input("\n  Press ENTER to run this search, "
                       "or type your own query: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    q = choice or plan.pubmed_query
    log.info(f"[Stage 1] FINAL QUERY: {q}")
    return q

# ------------------------------------------------------------ STAGE 2 -------
def stage2_search(query: str, max_results: int) -> List[str]:
    log.info(f"[Stage 2] PubMed search (top {max_results}, Best Match) ...")
    h = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(h); h.close()
    ids = [str(i) for i in record["IdList"]]
    log.info(f"[Stage 2] PubMed -> {len(ids)} PMIDs")
    return ids

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _parse_article(pa) -> Optional[dict]:
    try:
        cit, art = pa["MedlineCitation"], pa["MedlineCitation"]["Article"]
        pmid = str(cit["PMID"])
        title = str(art.get("ArticleTitle", "")).strip()
        journal = str(art.get("Journal", {}).get("Title", ""))
        pubdate = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pubdate.get("Year", pubdate.get("MedlineDate", "Unknown"))

        # AbstractText is a plain string for unstructured abstracts
        at = art.get("Abstract", {}).get("AbstractText", [])
        if isinstance(at, str):
            at = [at]
        parts = []
        for p in at:
            label = getattr(p, "attributes", {}).get("Label")
            parts.append(f"{label}: {p}" if label else str(p))
        abstract = " ".join(parts) or "No abstract available"

        mesh = "; ".join(str(m["DescriptorName"]) for m in cit.get("MeshHeadingList", []))

        # KeywordList can be list-of-strings or list-of-lists
        kwl = art.get("KeywordList", [])
        if kwl and isinstance(kwl[0], list):
            kws = "; ".join(str(k) for kl in kwl for k in kl)
        else:
            kws = "; ".join(str(k) for k in kwl)

        doi = ""
        for el in art.get("ELocationID", []):
            if str(getattr(el, "attributes", {}).get("EIdType", "")) == "doi":
                doi = str(el); break
        return {"PMID": pmid, "Title": title, "Year": year, "Journal": journal,
                "Abstract": abstract, "MeSH": mesh, "Keywords": kws, "DOI": doi,
                "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
    except Exception as e:
        log.warning(f"[Stage 2] could not parse one article: {e}")
        return None

def stage2_fetch(id_list: List[str]) -> pd.DataFrame:
    log.info(f"[Stage 2] Fetching metadata for {len(id_list)} articles ...")
    rows, delay = [], 0.4 if not NCBI_API_KEY else 0.12
    for chunk in _chunks(id_list, FETCH_BATCH):
        xml = None
        for attempt in range(3):
            try:
                h = Entrez.efetch(db="pubmed", id=",".join(chunk), retmode="xml")
                xml = Entrez.read(h); h.close(); break
            except Exception as e:
                log.warning(f"  efetch failed ({e}); retrying ...")
                time.sleep(2 ** (attempt + 1))
        if xml is None:
            continue
        for pa in xml.get("PubmedArticle", []):
            row = _parse_article(pa)
            if row:
                rows.append(row)
        time.sleep(delay)
    log.info(f"[Stage 2] parsed {len(rows)} articles")
    return pd.DataFrame(rows)

# --------------------------------------------------------- STAGE 3a ---------
STAGE3A_SYSTEM = """You screen scientific papers for a researcher.
You receive papers as: PMID | TITLE | ABSTRACT.
Score each 0-10 for relevance to the RESEARCH NEED:
  9-10 exactly the population, comparison and data type requested
  7-8  strongly relevant
  4-6  partially relevant (right disease, wrong design, etc.)
  0-3  not relevant
Return a verdict for EVERY paper using the exact PMIDs given."""

def stage3a_filter(df: pd.DataFrame, user_need: str,
                   batch_size: int = SCORE_BATCH) -> pd.DataFrame:
    log.info(f"[Stage 3a] Scoring {len(df)} papers (batches of {batch_size}) ...")
    verdicts, records = {}, df.to_dict("records")
    valid = set(df["PMID"])
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        listing = "\n\n".join(
            f"PMID {p['PMID']}\nTITLE: {p['Title']}\nABSTRACT: {p['Abstract'][:1500]}"
            for p in batch)
        msg = [{"role": "system", "content": STAGE3A_SYSTEM},
               {"role": "user", "content":
                f"RESEARCH NEED: {user_need}\n\nPAPERS:\n{listing}\n\n"
                f"Return one verdict per paper ({len(batch)} verdicts total)."}]
        got = []
        try:
            got = llm(msg, VerdictList).verdicts
        except Exception as e:
            log.error(f"[Stage 3a] batch at offset {i} failed entirely: {e} "
                      f"-> those papers get score 0")
        for v in got:
            if str(v.pmid) in valid:                 # ignore hallucinated PMIDs
                verdicts[str(v.pmid)] = v
        log.info(f"[Stage 3a] scored {min(i + batch_size, len(records))}/{len(records)}")
    df = df.copy()
    df["Score"]  = df["PMID"].map(lambda x: verdicts[x].score  if x in verdicts else 0)
    df["Reason"] = df["PMID"].map(lambda x: verdicts[x].reason if x in verdicts else "scoring failed")
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    log.info("[Stage 3a] score distribution:\n"
             + df["Score"].value_counts().sort_index(ascending=False).to_string())
    return df

# --------------------------------------------------------- STAGE 3b ---------
ACCESSION_RE = re.compile(
    r"\b(?:GSE\d+|GDS\d+|SRP\d+|ERP\d+|PRJNA\d+|PRJEB\d+|SCP\d+|"
    r"E-MTAB-\d+|E-GEOD-\d+|phs\d{6}|TCGA|CPTAC|GTEx)\b", re.IGNORECASE)

STAGE3B_SYSTEM = """You extract structured metadata from ONE biomedical paper
(title + abstract). STRICT RULES:
- Only extract facts EXPLICITLY present in the text. NEVER invent numbers or IDs.
- If something is not stated, use null / empty list.
- datasets_mentioned: named datasets/cohorts (TCGA, CPTAC, GTEx, named cohorts).
- accessions: database accession IDs in the text (GSE..., SRP..., PRJNA...).
- experimental_conditions: treatments, tissues, disease stages, platforms compared.
- n_control / n_metastasis: sample counts of the control (normal/non-tumor) group
  and the metastasis group. n_total: total samples if stated.
- evidence_quote: copy the exact sentence(s) justifying any numbers you report.
- extraction_confidence: "high" if explicit, "medium" if inferred, "low" if absent."""

def stage3b_extract(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"[Stage 3b] Deep extraction on top {len(df)} papers ...")
    rows = []
    for _, p in df.iterrows():
        try:
            ex = llm([{"role": "system", "content": STAGE3B_SYSTEM},
                      {"role": "user", "content":
                       f"TITLE: {p['Title']}\nABSTRACT: {p['Abstract'][:3500]}"}],
                     Extraction)
        except Exception as e:
            log.error(f"[Stage 3b] extraction failed for PMID {p['PMID']}: {e}")
            ex = Extraction()
        regex_hits = [a.upper() for a in
                      ACCESSION_RE.findall(f"{p['Abstract']} {p['Title']}")]
        rows.append({
            "PMID": p["PMID"], "Title": p["Title"], "Year": p["Year"],
            "Journal": p["Journal"], "Score": p["Score"], "Reason": p["Reason"],
            "Study_Type": ex.study_type,
            "Datasets": "; ".join(ex.datasets_mentioned),
            "Accessions": "; ".join(sorted(set(regex_hits) |
                                            {a.upper() for a in ex.accessions})),
            "Conditions": "; ".join(ex.experimental_conditions),
            "N_Control": ex.n_control, "N_Metastasis": ex.n_metastasis,
            "N_Total": ex.n_total,
            "Evidence_Quote": ex.evidence_quote,
            "Confidence": ex.extraction_confidence,
            "Genes": p.get("Genes", ""), "Diseases": p.get("Diseases", ""),
            "Abstract": p["Abstract"], "URL": p["URL"],
        })
        log.info(f"[Stage 3b] PMID {p['PMID']} | type={ex.study_type!r} | "
                 f"datasets={ex.datasets_mentioned} | n_ctl={ex.n_control} "
                 f"n_met={ex.n_metastasis} n_tot={ex.n_total} "
                 f"conf={ex.extraction_confidence}")
    return pd.DataFrame(rows)

# -------------------------------------------------------- VERIFICATION ------
def verify_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Is the LLM's 'evidence quote' actually in the abstract?"""
    df = df.copy()
    def ok(row):
        q = re.sub(r"\s+", " ", str(row.get("Evidence_Quote") or "")).strip()
        t = re.sub(r"\s+", " ", str(row.get("Abstract") or ""))
        return bool(q) and q[:120] in t
    df["Quote_Verified"] = df.apply(ok, axis=1)
    log.info(f"[Check] evidence quote verified in abstract: "
             f"{int(df['Quote_Verified'].sum())}/{len(df)} papers")
    return df

# ------------------------------------------------------------- PIPELINE -----
def run_pipeline(user_input: str):
    t0 = time.time()
    log.info("=" * 70)
    log.info(f"RUN START | input: {user_input}")

    # ---- Stage 1
    plan = stage1_plan(user_input)
    query = confirm_query(plan)

    # ---- Stage 2 (merge PMIDs from: user text, PubMed, PubTator3 full-text)
    pmids = [str(p) for p in plan.detected_pmids if re.fullmatch(r"\d{6,9}", str(p))]
    if query:
        pmids += stage2_search(query, MAX_RESULTS)
    if USE_PUBTATOR:
        pt_text = " ".join(x for x in
                           [plan.disease_focus, plan.comparison_groups,
                            plan.omics_methods] if x) or user_input[:200]
        pmids += pubtator_search(pt_text, MAX_RESULTS)
    pmids = list(dict.fromkeys(pmids))
    log.info(f"[Stage 2] {len(pmids)} unique PMIDs after merging all sources")
    if not pmids:
        log.error("No papers found. Try rephrasing your request.")
        return

    df = stage2_fetch(pmids)
    if df.empty:
        log.error("Nothing could be fetched from PubMed.")
        return

    if USE_PUBTATOR:
        df = add_pubtator_columns(df)

    # ---- Stage 3a
    df = stage3a_filter(df, user_input)
    df.to_csv("all_papers_scored.csv", index=False)
    log.info("saved all_papers_scored.csv")

    relevant = df[df["Score"] >= SCORE_THRESHOLD]
    log.info(f"[Stage 3a] {len(relevant)} papers passed threshold {SCORE_THRESHOLD}")
    if relevant.empty:
        log.error("No paper passed the threshold. Lower SCORE_THRESHOLD or rephrase.")
        return

    # ---- Stage 3b
    result = stage3b_extract(relevant.head(TOP_K_EXTRACT))

    if USE_GEO:
        result = geo_cross_check(result)
        geo_df = geo_search_datasets(plan.disease_focus, plan.omics_methods,
                                     GEO_DATASET_HITS)
        if not geo_df.empty:
            geo_df.to_csv("geo_datasets.csv", index=False)
            log.info(f"[GEO] saved geo_datasets.csv ({len(geo_df)} datasets)")

    result = finalize_sample_counts(result)
    result = verify_quotes(result)
    result.to_csv("extracted_papers.csv", index=False)
    log.info("saved extracted_papers.csv")

    # ---- preview
    cols = ["PMID", "Year", "Score", "Title", "Datasets", "Accessions",
            "N_Control_Final", "N_Metastasis_Final", "N_Total_Final", "Confidence"]
    cols = [c for c in cols if c in result.columns]
    pv = result[cols].copy()
    if "Title" in pv:
        pv["Title"] = pv["Title"].astype(str).str.slice(0, 55) + "..."
    print("\n" + "=" * 25 + f" TOP {len(result)} PAPERS " + "=" * 25)
    print(pv.to_string(index=False, max_colwidth=25))
    log.info(f"RUN FINISHED in {time.time()-t0:.0f}s -> "
             f"extracted_papers.csv ({len(result)} rows) | "
             f"all_papers_scored.csv ({len(df)} rows)")

# --------------------------------------------------------------- MAIN -------
def main():
    global GEMINI_MODEL
    log_file = setup_logging()
    Entrez.email = EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY

    if PROVIDER == "gemini":
        GEMINI_MODEL = pick_gemini_model()   # verify + build fallback list

    log.info("=" * 70)
    log.info(" LITERATURE & DATASET EXTRACTION TOOL (v3)")
    log.info(f" provider={PROVIDER} | gemini={GEMINI_MODEL} | "
             f"ollama={'yes' if HAS_OLLAMA else 'not installed'} | "
             f"geoparse={'yes' if HAS_GEOPARSE else 'not installed'}")
    log.info(f" max_results={MAX_RESULTS} | threshold={SCORE_THRESHOLD} | "
             f"top_k={TOP_K_EXTRACT}")
    log.info(f" logging to: {log_file}")
    if PROVIDER == "gemini":
        log.info(" NOTE: Gemini free tier ~= 10 req/min & ~250/day. One full "
                 "run uses ~35 requests; the script auto-paces itself and "
                 "falls back to Ollama if Gemini is overloaded.")
    log.info(" Type a research need in plain English, paste a paper title, "
             "or PMIDs. Type 'quit' to exit.")
    log.info("=" * 70)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        try:
            run_pipeline(user_input)
        except Exception as e:
            log.exception(f"[error] {e}")
            log.error("(Check internet / API key. If Gemini is down and Ollama "
                      "is installed it falls back automatically.)")
    log.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()