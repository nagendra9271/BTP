"""
MOCHEG Synthetic Misinformer Dataset Generator
================================================
Generates ~3k samples per type (Truthful, OOC, NEI, Hybrid) = ~12k total
Based on: "Synthetic Misinformers: Generating and Combating Multimodal Misinformation"

Setup (run once on your machine):
    pip install spacy
    python -m spacy download en_core_web_trf    # best accuracy (transformer-based)
    # OR for faster/lighter:
    python -m spacy download en_core_web_lg

Usage:
    python generate_dataset.py \
        --corpus2     /path/to/train/Corpus2.csv \
        --img_qrels   /path/to/train/img_evidence_qrels.csv \
        --images_dir  /path/to/train/images \
        --output      mocheg_synthetic_12k.csv \
        --target      3000 \
        --spacy_model en_core_web_trf

Output columns per row:
    claim_id | original_claim | image_path | grouped_text_evidence |
    misinformation_label | generated_misleading_claim | technique |
    original_truthfulness | swapped_entity_type | swapped_from | swapped_to
"""

import argparse
import csv
import os
import random
import re
import sys
import collections

# ─────────────────────────────────────────────
# 1.  NER BACKEND  (spaCy preferred, regex fallback)
# ─────────────────────────────────────────────

_nlp = None   # lazy-loaded spaCy model

def load_spacy(model_name):
    global _nlp
    try:
        import spacy
        _nlp = spacy.load(model_name)
        print(f"      spaCy model '{model_name}' loaded.")
    except Exception as e:
        print(f"      WARNING: Could not load spaCy model '{model_name}': {e}")
        print("      Falling back to regex NER.")
        _nlp = None


def extract_entities_spacy(text):
    """
    Use spaCy NER. Returns dict: {ent_type: [text_strings]}
    We keep: PERSON, DATE, GPE (geo-political), LOC, ORG, EVENT, NORP
    All mapped to canonical keys: PERSON, DATE, LOC, ORG, EVENT
    """
    doc = _nlp(text)
    ents = collections.defaultdict(list)
    type_map = {
        "PERSON": "PERSON",
        "DATE":   "DATE",
        "TIME":   "DATE",
        "GPE":    "LOC",
        "LOC":    "LOC",
        "FAC":    "LOC",
        "ORG":    "ORG",
        "EVENT":  "EVENT",
        "NORP":   "ORG",
        "ORDINAL":"DATE",
        "CARDINAL":"DATE",
    }
    for ent in doc.ents:
        mapped = type_map.get(ent.label_)
        if mapped:
            val = ent.text.strip()
            if len(val) > 1:
                ents[mapped].append(val)
    # deduplicate preserving order
    return {k: list(dict.fromkeys(v)) for k, v in ents.items()}


# ── regex fallback ──
_PERSON_RE  = re.compile(
    r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Sen\.|Rep\.|Gov\.|President|Vice President)?\s*'
    r'[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b')
_DATE_RE    = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
    r'\.?\s+\d{1,2}(?:,\s*\d{4})?\b'
    r'|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{4}\b'
    r'|\b(?:19|20)\d{2}\b')
_LOC_RE     = re.compile(
    r'\b(?:Michigan|California|Texas|New York|Florida|Washington|Alabama|Alaska|'
    r'Arizona|Arkansas|Colorado|Connecticut|Delaware|Georgia|Hawaii|Idaho|Illinois|'
    r'Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Minnesota|'
    r'Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|'
    r'North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|'
    r'South Carolina|South Dakota|Tennessee|Utah|Vermont|Virginia|Wisconsin|Wyoming|'
    r'United States|U\.S\.|Canada|Mexico|UK|United Kingdom|France|Germany|China|'
    r'Russia|Afghanistan|Iraq|Iran|Australia|India|Pakistan|'
    r'Los Angeles|New York City|Chicago|Houston|Phoenix|Philadelphia|San Francisco|'
    r'Washington D\.C\.|Boston|Seattle|Nashville|Miami|Atlanta)\b')

def extract_entities_regex(text):
    return {
        "PERSON": list(dict.fromkeys(_PERSON_RE.findall(text))),
        "DATE":   list(dict.fromkeys(_DATE_RE.findall(text))),
        "LOC":    list(dict.fromkeys(_LOC_RE.findall(text))),
        "ORG":    [],
        "EVENT":  [],
    }

def extract_entities(text):
    if _nlp is not None:
        return extract_entities_spacy(text)
    return extract_entities_regex(text)


# ─────────────────────────────────────────────
# 2.  ARGUMENT PARSING
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus2",     required=True)
    p.add_argument("--img_qrels",   required=True)
    p.add_argument("--images_dir",  required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--target",      type=int, default=3000,
                   help="Samples per type (default 3000 → 12k total)")
    p.add_argument("--spacy_model", default="en_core_web_trf",
                   help="spaCy model name (default: en_core_web_trf)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
# 3.  DATA LOADING
# ─────────────────────────────────────────────
def load_corpus2(path):
    claims = collections.defaultdict(lambda: {
        "claim": "", "truthfulness": "", "ruling_outline": "",
        "evidences": [], "snopes_url": ""
    })
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("claim_id", "")).strip()
            if not cid:
                continue
            rec = claims[cid]
            if not rec["claim"]:
                rec["claim"]          = row.get("Claim", "").strip()
                rec["truthfulness"]   = row.get("cleaned_truthfulness", "").strip()
                rec["ruling_outline"] = row.get("ruling_outline", "").strip()
                rec["snopes_url"]     = row.get("Snopes URL", "").strip()
            ev = row.get("Evidence", "").strip()
            if ev:
                ev_clean = re.sub(r"<[^>]+>", " ", ev)
                ev_clean = re.sub(r"\s+", " ", ev_clean).strip()
                if ev_clean:
                    rec["evidences"].append(ev_clean)
    return dict(claims)


def load_img_qrels(path):
    img_map     = collections.defaultdict(list)
    img_map_neg = collections.defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("TOPIC", "")).strip()
            doc = row.get("DOCUMENT#", "").strip()
            rel = str(row.get("RELEVANCY", "0")).strip()
            if cid and doc:
                (img_map if rel == "1" else img_map_neg)[cid].append(doc)
    for cid, imgs in img_map_neg.items():
        if cid not in img_map:
            img_map[cid] = imgs
    return dict(img_map)


def resolve_image(claim_id, img_map, images_dir):
    for fname in img_map.get(str(claim_id), []):
        if os.path.isfile(os.path.join(images_dir, fname)):
            return fname
    prefix = str(claim_id) + "-"
    try:
        for f in sorted(os.listdir(images_dir)):
            if f.startswith(prefix):
                return f
    except Exception:
        pass
    return ""


def group_evidence(evidences):
    return " ||| ".join(evidences) if evidences else ""


# ─────────────────────────────────────────────
# 4.  BUILD ENTITY DONOR POOLS
#     Pre-extract entities from ALL claims once,
#     then build per-type pools for fast lookup.
# ─────────────────────────────────────────────
def build_donor_pools(claims):
    """
    Returns:
      ent_cache  : dict  claim_id -> {type: [ents]}
      type_pools : dict  ent_type -> list of (claim_id, ent_text)
    """
    print("      Extracting entities from all claims (this may take a moment)...")
    ent_cache  = {}
    type_pools = collections.defaultdict(list)   # type -> [(cid, ent)]

    total = len(claims)
    for i, (cid, rec) in enumerate(claims.items()):
        if i % 500 == 0:
            print(f"        {i}/{total} processed...", end="\r", flush=True)
        ents = extract_entities(rec["claim"])
        ent_cache[cid] = ents
        for etype, vals in ents.items():
            for v in vals:
                type_pools[etype].append((cid, v))

    print(f"\n      Entity pool sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in type_pools.items()))
    return ent_cache, dict(type_pools)


# ─────────────────────────────────────────────
# 5.  MISINFORMER TECHNIQUES
# ─────────────────────────────────────────────

# --- 5a. TRUTHFUL ---
def make_truthful(claim_id, rec, img_path):
    return {
        "claim_id":                 claim_id,
        "original_claim":           rec["claim"],
        "image_path":               img_path,
        "grouped_text_evidence":    group_evidence(rec["evidences"]),
        "misinformation_label":     "Truthful",
        "generated_misleading_claim": rec["claim"],
        "technique":                "Truthful",
        "original_truthfulness":    rec["truthfulness"],
        "swapped_entity_type":      "",
        "swapped_from":             "",
        "swapped_to":               "",
    }


# --- 5b. OOC ---
def make_ooc(claim_id, rec, img_path, donor_img, donor_cid):
    return {
        "claim_id":                 claim_id,
        "original_claim":           rec["claim"],
        "image_path":               donor_img,          # swapped image
        "grouped_text_evidence":    group_evidence(rec["evidences"]),
        "misinformation_label":     "Misleading-OOC",
        "generated_misleading_claim": rec["claim"],     # text unchanged
        "technique":                "OOC-ImageSwap",
        "original_truthfulness":    rec["truthfulness"],
        "swapped_entity_type":      "IMAGE",
        "swapped_from":             img_path,
        "swapped_to":               donor_img,
    }


# --- 5c. NEI (spaCy-powered) ---
def make_nei(claim_id, rec, img_path, ent_cache, type_pools, rng):
    original  = rec["claim"]
    src_ents  = ent_cache.get(claim_id, {})
    modified  = original
    swap_type = swap_from = swap_to = ""

    # Try each entity type in priority order
    priority = ["PERSON", "LOC", "ORG", "DATE", "EVENT"]
    for etype in priority:
        src_list = src_ents.get(etype, [])
        if not src_list:
            continue

        # Candidate replacements: same type, different claim, not already in text
        candidates = [
            ent for (cid, ent) in type_pools.get(etype, [])
            if cid != claim_id and ent not in original
        ]
        if not candidates:
            continue

        # Pick the entity to swap (first one found in text)
        src_ent = src_list[0]
        if src_ent not in modified:
            continue

        replacement = rng.choice(candidates)
        new_modified = modified.replace(src_ent, replacement, 1)
        if new_modified != modified:
            swap_type = etype
            swap_from = src_ent
            swap_to   = replacement
            modified  = new_modified
            break   # one swap per claim (like CLIP-NESt strategy)

    if not swap_type:
        # Fallback: no swappable entity found — append context tag
        modified  = original + " [Note: context may be incomplete or misrepresented]"
        swap_type = "FALLBACK"

    return {
        "claim_id":                 claim_id,
        "original_claim":           original,
        "image_path":               img_path,
        "grouped_text_evidence":    group_evidence(rec["evidences"]),
        "misinformation_label":     "Misleading-NEI",
        "generated_misleading_claim": modified,
        "technique":                "NEI-spaCy-EntitySwap",
        "original_truthfulness":    rec["truthfulness"],
        "swapped_entity_type":      swap_type,
        "swapped_from":             swap_from,
        "swapped_to":               swap_to,
    }


# --- 5d. HYBRID (OOC + NEI) ---
def make_hybrid(claim_id, rec, img_path, donor_img, donor_cid,
                ent_cache, type_pools, rng):
    # Apply NEI to get the modified claim text
    nei = make_nei(claim_id, rec, img_path, ent_cache, type_pools, rng)
    return {
        "claim_id":                 claim_id,
        "original_claim":           rec["claim"],
        "image_path":               donor_img,          # OOC image swap
        "grouped_text_evidence":    group_evidence(rec["evidences"]),
        "misinformation_label":     "Misleading-Hybrid",
        "generated_misleading_claim": nei["generated_misleading_claim"],  # NEI text
        "technique":                "Hybrid-OOC+NEI-spaCy",
        "original_truthfulness":    rec["truthfulness"],
        "swapped_entity_type":      nei["swapped_entity_type"],
        "swapped_from":             nei["swapped_from"],
        "swapped_to":               nei["swapped_to"],
    }


# ─────────────────────────────────────────────
# 6.  MAIN GENERATION LOOP
# ─────────────────────────────────────────────
def generate(args):
    rng = random.Random(args.seed)

    print(f"[1/6] Loading spaCy model: {args.spacy_model}")
    load_spacy(args.spacy_model)

    print(f"[2/6] Loading Corpus2.csv ...")
    claims = load_corpus2(args.corpus2)
    print(f"      {len(claims)} unique claims loaded.")

    print(f"[3/6] Loading image qrels ...")
    img_map = load_img_qrels(args.img_qrels)

    print(f"[4/6] Resolving image paths ...")
    img_resolved = {cid: resolve_image(cid, img_map, args.images_dir)
                    for cid in claims}

    valid_ids = [
        cid for cid in claims
        if img_resolved.get(cid) and claims[cid]["claim"] and claims[cid]["evidences"]
    ]
    print(f"      Valid claims (claim + image + evidence): {len(valid_ids)}")

    if len(valid_ids) < 10:
        print("ERROR: Too few valid claims. Check your file paths.")
        sys.exit(1)

    print(f"[5/6] Building entity donor pools via spaCy NER ...")
    ent_cache, type_pools = build_donor_pools(
        {cid: claims[cid] for cid in valid_ids}
    )

    donor_pool = [(cid, img_resolved[cid])
                  for cid in valid_ids if img_resolved[cid]]

    print(f"[6/6] Generating {args.target} samples × 4 types ...")
    rows   = []
    counts = {"Truthful": 0, "OOC": 0, "NEI": 0, "Hybrid": 0}
    target = args.target
    ids_shuffled = valid_ids[:]
    rng.shuffle(ids_shuffled)

    idx = 0
    attempts = 0
    max_attempts = target * 4 * 15

    while sum(counts.values()) < target * 4 and attempts < max_attempts:
        attempts += 1
        cid      = ids_shuffled[idx % len(ids_shuffled)]
        idx     += 1
        rec      = claims[cid]
        img_path = img_resolved[cid]

        type_order = ["Truthful", "OOC", "NEI", "Hybrid"]
        rng.shuffle(type_order)

        for t in type_order:
            if counts[t] >= target:
                continue

            if t == "Truthful":
                rows.append(make_truthful(cid, rec, img_path))
                counts["Truthful"] += 1

            elif t == "OOC":
                for _ in range(30):
                    d_cid, d_img = rng.choice(donor_pool)
                    if d_cid != cid and d_img != img_path:
                        break
                else:
                    continue
                rows.append(make_ooc(cid, rec, img_path, d_img, d_cid))
                counts["OOC"] += 1

            elif t == "NEI":
                rows.append(make_nei(cid, rec, img_path,
                                     ent_cache, type_pools, rng))
                counts["NEI"] += 1

            elif t == "Hybrid":
                for _ in range(30):
                    d_cid, d_img = rng.choice(donor_pool)
                    if d_cid != cid and d_img != img_path:
                        break
                else:
                    continue
                rows.append(make_hybrid(cid, rec, img_path, d_img, d_cid,
                                        ent_cache, type_pools, rng))
                counts["Hybrid"] += 1
            break

        if sum(counts.values()) % 500 == 0 and sum(counts.values()) > 0:
            print(f"      Progress: {counts}")

    rng.shuffle(rows)

    print(f"\n      Final counts: {counts}")
    print(f"      Total rows  : {len(rows)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "claim_id", "original_claim", "image_path",
        "grouped_text_evidence", "misinformation_label",
        "generated_misleading_claim", "technique",
        "original_truthfulness", "swapped_entity_type",
        "swapped_from", "swapped_to",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓  Saved to: {args.output}")
    print(f"   Label distribution:")
    lc = collections.Counter(r["misinformation_label"] for r in rows)
    for lbl, cnt in sorted(lc.items()):
        print(f"    {lbl:<28} {cnt:>6}")
    swap_types = collections.Counter(
        r["swapped_entity_type"] for r in rows if r["swapped_entity_type"]
    )
    print(f"\n   NEI/Hybrid swapped entity types:")
    for k, v in swap_types.most_common():
        print(f"    {k:<12} {v:>6}")


if __name__ == "__main__":
    args = parse_args()
    generate(args)
