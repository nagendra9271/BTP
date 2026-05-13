"""
MOCHEG Synthetic Misinformer Dataset Generator — CLIP-Guided
=============================================================
Implements all four techniques from the paper faithfully:

  Truthful   — original (image, claim, evidence)
  OOC        — CLIP hard-negative image swap  (CSt-alt strategy)
  NEI        — CLIP-NESt-alt  (find semantically similar claim → swap named entity)
  Hybrid     — OOC image swap  +  NEI entity swap  (CLIP-NESt-alt + CSt-alt)

Setup (run once on your machine):
    pip install torch torchvision clip-by-openai spacy
    python -m spacy download en_core_web_trf
    # CLIP installs as:  pip install git+https://github.com/openai/CLIP.git

Usage:
    python generate_dataset_clip.py \\
        --corpus2     train/Corpus2.csv \\
        --img_qrels   train/img_evidence_qrels.csv \\
        --images_dir  train/images \\
        --output      mocheg_synthetic_12k.csv \\
        --target      3000 \\
        --spacy_model en_core_web_trf \\
        --clip_model  ViT-L/14 \\
        --top_k       10

Paper reference: Papadopoulos et al., "Synthetic Misinformers", MAD'23
"""

import argparse
import csv
import os
import re
import sys
import random
import collections
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ──────────────────────────────────────────────────────────────
# 0.  OPTIONAL BACKENDS  (CLIP, spaCy)
#     Both degrade gracefully so the script always runs.
# ──────────────────────────────────────────────────────────────

_clip_model = None
_clip_preproc = None
_clip_device = "cpu"
_nlp = None


def load_clip(model_name):
    global _clip_model, _clip_preproc, _clip_device
    try:
        import torch
        import clip
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preproc = clip.load(model_name, device=_clip_device)
        _clip_model.eval()
        print(f"      CLIP '{model_name}' loaded on {_clip_device}.")
        return True
    except Exception as e:
        print(f"      WARNING: CLIP unavailable ({e}).")
        print("      Falling back to pixel-histogram similarity (install CLIP for proper results).")
        return False


def load_spacy(model_name):
    global _nlp
    try:
        import spacy
        _nlp = spacy.load(model_name)
        print(f"      spaCy model '{model_name}' loaded.")
    except Exception as e:
        print(
            f"      WARNING: spaCy unavailable ({e}). Using regex NER fallback.")
        _nlp = None


# ──────────────────────────────────────────────────────────────
# 1.  EMBEDDING FUNCTIONS
#     clip_embed_text  : (list[str])  → np.array [N, D]
#     clip_embed_image : (list[path]) → np.array [N, D]
#     If CLIP not available, falls back to cheap proxies that
#     preserve the pipeline structure while being clearly labelled.
# ──────────────────────────────────────────────────────────────

def _hash_embed(text, dim=512):
    """Deterministic pseudo-embedding from text hash (fallback only)."""
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for i, w in enumerate(words[:dim]):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[i % dim] += (h % 1000) / 1000.0
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _pixel_embed(img_path, dim=512):
    """Pixel-histogram embedding (fallback only)."""
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        arr = np.array(img, dtype=np.float32).flatten()
        # reduce to dim via strided sampling
        step = max(1, len(arr) // dim)
        arr = arr[::step][:dim]
        if len(arr) < dim:
            arr = np.pad(arr, (0, dim - len(arr)))
        norm = np.linalg.norm(arr)
        return arr / (norm + 1e-9)
    except Exception:
        return np.random.default_rng(abs(hash(img_path)) % (2**31)).random(dim).astype(np.float32)


def clip_embed_texts(texts, batch=256):
    """Return np.array [N, D].  Uses real CLIP if available."""
    if _clip_model is not None:
        import torch
        import clip
        all_vecs = []
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            tokens = clip.tokenize(batch_texts, truncate=True).to(_clip_device)
            with torch.no_grad():
                vecs = _clip_model.encode_text(tokens).float().cpu().numpy()
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)
    # fallback
    return np.stack([_hash_embed(t) for t in texts])


def clip_embed_images(paths, batch=64):
    """Return np.array [N, D].  Uses real CLIP if available."""
    if _clip_model is not None:
        import torch
        from PIL import Image as PILImage
        all_vecs = []
        for i in range(0, len(paths), batch):
            batch_paths = paths[i:i+batch]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(_clip_preproc(PILImage.open(p).convert("RGB")))
                except Exception:
                    imgs.append(_clip_preproc(PILImage.new("RGB", (224, 224))))
            tensor = __import__("torch").stack(imgs).to(_clip_device)
            with __import__("torch").no_grad():
                vecs = _clip_model.encode_image(tensor).float().cpu().numpy()
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)
    # fallback
    return np.stack([_pixel_embed(p) for p in paths])


# ──────────────────────────────────────────────────────────────
# 2.  NER  (spaCy preferred, regex fallback)
# ──────────────────────────────────────────────────────────────

_PERSON_RE = re.compile(
    r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Sen\.|Rep\.|Gov\.|President|Vice President)?\s*'
    r'[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b')
_DATE_RE = re.compile(
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|'
    r'Dec(?:ember)?)\.?\s+\d{1,2}(?:,\s*\d{4})?\b'
    r'|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4}\b'
    r'|\b(?:19|20)\d{2}\b')
_LOC_RE = re.compile(
    r'\b(?:Michigan|California|Texas|New York|Florida|Washington|Alabama|'
    r'Alaska|Arizona|Arkansas|Colorado|Connecticut|Delaware|Georgia|Hawaii|'
    r'Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|'
    r'Massachusetts|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|'
    r'New Hampshire|New Jersey|New Mexico|North Carolina|North Dakota|Ohio|'
    r'Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|'
    r'Tennessee|Utah|Vermont|Virginia|Wisconsin|Wyoming|'
    r'United States|U\.S\.|Canada|Mexico|UK|United Kingdom|France|Germany|'
    r'China|Russia|Afghanistan|Iraq|Iran|Australia|India|Pakistan|'
    r'Los Angeles|New York City|Chicago|Houston|Phoenix|Philadelphia|'
    r'San Francisco|Washington D\.C\.|Boston|Seattle|Nashville|Miami|Atlanta)\b')

_SPACY_TYPE_MAP = {
    "PERSON": "PERSON", "DATE": "DATE", "TIME": "DATE",
    "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "ORG": "ORG", "EVENT": "EVENT",
    "NORP": "ORG", "ORDINAL": "DATE", "CARDINAL": "DATE",
}


def extract_entities(text):
    if _nlp is not None:
        doc = _nlp(text)
        ents = collections.defaultdict(list)
        for e in doc.ents:
            mapped = _SPACY_TYPE_MAP.get(e.label_)
            if mapped and len(e.text.strip()) > 1:
                ents[mapped].append(e.text.strip())
        return {k: list(dict.fromkeys(v)) for k, v in ents.items()}
    return {
        "PERSON": list(dict.fromkeys(_PERSON_RE.findall(text))),
        "DATE":   list(dict.fromkeys(_DATE_RE.findall(text))),
        "LOC":    list(dict.fromkeys(_LOC_RE.findall(text))),
        "ORG": [], "EVENT": [],
    }


# ──────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_corpus2(path):
    claims = collections.defaultdict(lambda: {
        "claim": "", "truthfulness": "", "ruling_outline": "",
        "evidences": [], "snopes_url": ""
    })
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("claim_id", "")).strip()
            if not cid:
                continue
            rec = claims[cid]
            if not rec["claim"]:
                rec["claim"] = row.get("Claim", "").strip()
                rec["truthfulness"] = row.get(
                    "cleaned_truthfulness", "").strip()
                rec["ruling_outline"] = row.get("ruling_outline", "").strip()
                rec["snopes_url"] = row.get("Snopes URL", "").strip()
            ev = re.sub(r"<[^>]+>", " ", row.get("Evidence", ""))
            ev = re.sub(r"\s+", " ", ev).strip()
            if ev:
                rec["evidences"].append(ev)
    return dict(claims)


def load_img_qrels(path):
    """
    Returns img_map: claim_id -> list[image_filename]
    Uses evidence_id column (actual proof image filename) as primary key.
    Falls back to DOCUMENT# for negative relevancy.
    """
    img_map = collections.defaultdict(list)
    img_map_neg = collections.defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("TOPIC", "")).strip()
            # evidence_id is the ACTUAL proof image filename
            ev_id = row.get("evidence_id", "").strip()
            doc = row.get("DOCUMENT#",   "").strip()
            rel = str(row.get("RELEVANCY", "0")).strip()

            if not cid:
                continue

            # Primary: use evidence_id (proof image) for positive relevancy
            target_img = ev_id if ev_id else doc
            if target_img:
                if rel == "1":
                    if target_img not in img_map[cid]:
                        img_map[cid].append(target_img)
                else:
                    img_map_neg[cid].append(target_img)

    # Fallback: use negative-relevancy images when no positive ones found
    for cid, imgs in img_map_neg.items():
        if cid not in img_map:
            img_map[cid] = imgs
    return dict(img_map)


def resolve_image(claim_id, img_map, images_dir):
    """Find the first existing image file for a claim."""
    for fname in img_map.get(str(claim_id), []):
        # evidence_id may be just a filename, e.g. "10324-proof-01-maps.jpg"
        # strip any leading path components
        bare = os.path.basename(fname)
        for candidate in [bare, fname]:
            full = os.path.join(images_dir, candidate)
            if os.path.isfile(full):
                return bare
    # Fallback: scan for prefix match
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


# ──────────────────────────────────────────────────────────────
# 4.  EMBEDDING INDEX
#     Builds two matrices:
#       text_matrix  [N, D]  — CLIP text embeddings of claims
#       image_matrix [N, D]  — CLIP image embeddings of proof images
#     Both are indexed by position in valid_ids list.
# ──────────────────────────────────────────────────────────────

def build_embedding_index(valid_ids, claims, img_resolved, images_dir,
                          cache_path=None):
    """
    Returns (text_matrix, image_matrix) both np.array [N, D].
    Saves/loads from cache_path (pickle) to avoid re-embedding.
    """
    if cache_path and os.path.isfile(cache_path):
        print(f"      Loading embedding cache from {cache_path} ...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("ids") == valid_ids:
            print("      Cache hit — skipping re-embedding.")
            return data["text_mat"], data["img_mat"]
        print("      Cache mismatch — re-embedding.")

    N = len(valid_ids)
    print(f"      Embedding {N} claim texts with CLIP ...")
    texts = [claims[cid]["claim"] for cid in valid_ids]
    text_mat = clip_embed_texts(texts)

    print(f"      Embedding {N} proof images with CLIP ...")
    img_paths = [os.path.join(images_dir, img_resolved[cid])
                 for cid in valid_ids]
    image_mat = clip_embed_images(img_paths)

    if cache_path:
        print(f"      Saving embedding cache to {cache_path} ...")
        with open(cache_path, "wb") as f:
            pickle.dump({"ids": valid_ids,
                         "text_mat": text_mat,
                         "img_mat":  image_mat}, f)
    return text_mat, image_mat


# ──────────────────────────────────────────────────────────────
# 5.  HARD-NEGATIVE SELECTION  (CSt-alt / CLIP-NESt-alt)
#
#   CSt-alt  (OOC):
#     For each query claim i, find Top-K images from OTHER claims
#     ranked by CLIP image-image cosine similarity.
#     Alternates between image-image and text-text similarity
#     (alt = we pick whichever gives a different claim).
#
#   CLIP-NESt-alt  (NEI):
#     For each query claim i, find Top-K OTHER claims ranked by
#     CLIP text-text cosine similarity.
#     Then swap named entities between query claim and best donor
#     that has a different entity of the same type.
# ──────────────────────────────────────────────────────────────

def precompute_topk_neighbors(matrix, top_k, exclude_self=True):
    """
    For every row i, find top_k most similar rows (excluding self).
    Returns neighbors: list[list[int]]  — indices into matrix rows.
    Done in batches to avoid OOM on large datasets.
    """
    N = len(matrix)
    neighbors = [[] for _ in range(N)]
    batch = 512

    for start in range(0, N, batch):
        end = min(start + batch, N)
        block = matrix[start:end]           # [B, D]
        sims = cos_sim(block, matrix)      # [B, N]
        if exclude_self:
            for local_i, global_i in enumerate(range(start, end)):
                sims[local_i, global_i] = -2.0  # mask self
        top_idx = np.argsort(-sims, axis=1)[:, :top_k]
        for local_i, global_i in enumerate(range(start, end)):
            neighbors[global_i] = top_idx[local_i].tolist()

    return neighbors


# ──────────────────────────────────────────────────────────────
# 6.  ENTITY DONOR POOLS
# ──────────────────────────────────────────────────────────────

def build_entity_cache(valid_ids, claims):
    print(
        f"      Extracting spaCy/regex entities from {len(valid_ids)} claims ...")
    ent_cache = {}
    type_pools = collections.defaultdict(list)   # type -> [(idx, ent_text)]

    for i, cid in enumerate(valid_ids):
        if i % 1000 == 0:
            print(f"        {i}/{len(valid_ids)}...", end="\r", flush=True)
        ents = extract_entities(claims[cid]["claim"])
        ent_cache[cid] = ents
        for etype, vals in ents.items():
            for v in vals:
                # store index for O(1) lookup
                type_pools[etype].append((i, cid, v))

    print(f"\n      Pool sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in type_pools.items()))
    return ent_cache, dict(type_pools)


# ──────────────────────────────────────────────────────────────
# 7.  FOUR MISINFORMER TECHNIQUES
# ──────────────────────────────────────────────────────────────

def _base_row(claim_id, rec, img_path):
    return {
        "claim_id":              claim_id,
        "original_claim":        rec["claim"],
        "image_path":            img_path,
        "grouped_text_evidence": group_evidence(rec["evidences"]),
        "original_truthfulness": rec["truthfulness"],
    }


# ── 7a. TRUTHFUL ──────────────────────────────────────────────
def make_truthful(claim_id, rec, img_path):
    row = _base_row(claim_id, rec, img_path)
    row.update({
        "misinformation_label":     "Truthful",
        "generated_misleading_claim": rec["claim"],
        "technique":                "Truthful",
        "ooc_donor_claim_id":       "",
        "ooc_original_image":       "",
        "ooc_swapped_image":        "",
        "nei_swapped_entity_type":  "",
        "nei_swapped_from":         "",
        "nei_swapped_to":           "",
        "nei_donor_claim_id":       "",
        "clip_similarity_score":    "",
    })
    return row


# ── 7b. OOC  (CSt-alt: CLIP image-image hard negative) ────────
def make_ooc(claim_id, rec, img_path,
             idx, valid_ids, img_resolved,
             img_neighbors, txt_neighbors,    # both top-K lists
             rng):
    """
    CSt-alt strategy:
      Alternate between image-image and text-text neighbors.
      Pick the top-ranked neighbor (from either list) that has
      a DIFFERENT image than the query claim.
    """
    orig_img = img_path

    # Merge two neighbor lists (alternating = zip then flatten)
    merged = []
    for a, b in zip(img_neighbors[idx], txt_neighbors[idx]):
        if a not in merged:
            merged.append(a)
        if b not in merged:
            merged.append(b)

    donor_idx = None
    donor_img = None
    donor_cid = None
    sim_score = ""

    for ni in merged:
        d_cid = valid_ids[ni]
        d_img = img_resolved.get(d_cid, "")
        if d_cid != claim_id and d_img and d_img != orig_img:
            donor_idx = ni
            donor_cid = d_cid
            donor_img = d_img
            break

    if donor_img is None:
        return None   # skip this sample

    row = _base_row(claim_id, rec, donor_img)   # swapped image
    row.update({
        "misinformation_label":     "Misleading-OOC",
        "generated_misleading_claim": rec["claim"],  # text unchanged
        "technique":                "OOC-CSt-alt-CLIP",
        "ooc_donor_claim_id":       donor_cid,
        "ooc_original_image":       orig_img,
        "ooc_swapped_image":        donor_img,
        "nei_swapped_entity_type":  "",
        "nei_swapped_from":         "",
        "nei_swapped_to":           "",
        "nei_donor_claim_id":       "",
        "clip_similarity_score":    sim_score,
    })
    return row


# ── 7c. NEI  (CLIP-NESt-alt: semantic donor + entity swap) ────
def make_nei(claim_id, rec, img_path,
             idx, valid_ids,
             txt_neighbors, img_neighbors,    # both used for alt selection
             ent_cache, claims, rng):
    """
    CLIP-NESt-alt strategy:
      1. Find top-K semantically similar claims (text-text + img-img alternating).
      2. For each candidate donor (in similarity order):
         a. Extract entities from donor claim.
         b. Find a type where BOTH query and donor have entities AND
            donor entity ≠ query entity.
         c. Replace query entity with donor entity.
      3. One swap per claim (most salient entity).
    """
    original = rec["claim"]
    src_ents = ent_cache.get(claim_id, {})

    # Alternating neighbor list (same CSt-alt pattern)
    merged = []
    for a, b in zip(txt_neighbors[idx], img_neighbors[idx]):
        if a not in merged:
            merged.append(a)
        if b not in merged:
            merged.append(b)

    priority = ["PERSON", "LOC", "ORG", "DATE", "EVENT"]

    best = None   # (modified_text, etype, src_ent, repl_ent, donor_cid)

    for ni in merged:
        d_cid = valid_ids[ni]
        if d_cid == claim_id:
            continue
        d_ents = ent_cache.get(d_cid, {})

        for etype in priority:
            src_list = src_ents.get(etype, [])
            d_list = [e for e in d_ents.get(etype, [])
                      if e not in original]
            if not src_list or not d_list:
                continue
            src_ent = src_list[0]
            if src_ent not in original:
                continue
            repl_ent = d_list[0]
            modified = original.replace(src_ent, repl_ent, 1)
            if modified != original:
                best = (modified, etype, src_ent, repl_ent, d_cid)
                break

        if best:
            break

    if best is None:
        # Fallback: mark but do not silently drop
        modified = original + " [context may be incomplete or misrepresented]"
        etype = swap_from = swap_to = donor_cid = "FALLBACK"
    else:
        modified, etype, swap_from, swap_to, donor_cid = best

    row = _base_row(claim_id, rec, img_path)
    row.update({
        "misinformation_label":     "Misleading-NEI",
        "generated_misleading_claim": modified,
        "technique":                "NEI-CLIP-NESt-alt",
        "ooc_donor_claim_id":       "",
        "ooc_original_image":       "",
        "ooc_swapped_image":        "",
        "nei_swapped_entity_type":  etype,
        "nei_swapped_from":         swap_from,
        "nei_swapped_to":           swap_to,
        "nei_donor_claim_id":       donor_cid,
        "clip_similarity_score":    "",
    })
    return row


# ── 7d. HYBRID  (CSt-alt image  +  CLIP-NESt-alt entity) ──────
def make_hybrid(claim_id, rec, img_path,
                idx, valid_ids, img_resolved,
                img_neighbors, txt_neighbors,
                ent_cache, claims, rng):
    ooc = make_ooc(claim_id, rec, img_path,
                   idx, valid_ids, img_resolved,
                   img_neighbors, txt_neighbors, rng)
    nei = make_nei(claim_id, rec, img_path,
                   idx, valid_ids,
                   txt_neighbors, img_neighbors,
                   ent_cache, claims, rng)

    if ooc is None or nei is None:
        return None

    row = _base_row(claim_id, rec, ooc["image_path"])  # OOC image
    row.update({
        "misinformation_label":     "Misleading-Hybrid",
        # NEI text
        "generated_misleading_claim": nei["generated_misleading_claim"],
        "technique":                "Hybrid-CLIP-NESt-alt+CSt-alt",
        "ooc_donor_claim_id":       ooc["ooc_donor_claim_id"],
        "ooc_original_image":       ooc["ooc_original_image"],
        "ooc_swapped_image":        ooc["ooc_swapped_image"],
        "nei_swapped_entity_type":  nei["nei_swapped_entity_type"],
        "nei_swapped_from":         nei["nei_swapped_from"],
        "nei_swapped_to":           nei["nei_swapped_to"],
        "nei_donor_claim_id":       nei["nei_donor_claim_id"],
        "clip_similarity_score":    "",
    })
    return row


# ──────────────────────────────────────────────────────────────
# 8.  ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus2",     required=True)
    p.add_argument("--img_qrels",   required=True)
    p.add_argument("--images_dir",  required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--target",      type=int,   default=3000,
                   help="Samples per type (default 3000 → 12k total)")
    p.add_argument("--spacy_model", default="en_core_web_trf")
    p.add_argument("--clip_model",  default="ViT-L/14",
                   help="CLIP model: ViT-B/32 | ViT-L/14 (default)")
    p.add_argument("--top_k",       type=int,   default=10,
                   help="Top-K neighbors for hard-negative selection (default 10)")
    p.add_argument("--embed_cache", default="",
                   help="Optional path to cache CLIP embeddings (pickle). "
                        "Set to e.g. clip_cache.pkl to reuse on reruns.")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print(" MOCHEG CLIP-Guided Synthetic Misinformer Generator")
    print("=" * 60)

    # ── Step 1: load backends ──────────────────────────────────
    print("\n[1/7] Loading CLIP ...")
    clip_ok = load_clip(args.clip_model)
    print(f"[2/7] Loading spaCy ({args.spacy_model}) ...")
    load_spacy(args.spacy_model)

    # ── Step 2: load data ──────────────────────────────────────
    print(f"\n[3/7] Loading Corpus2.csv ...")
    claims = load_corpus2(args.corpus2)
    print(f"      {len(claims)} unique claims.")

    print(f"[3/7] Loading img_evidence_qrels.csv ...")
    img_map = load_img_qrels(args.img_qrels)

    # ── Step 3: filter valid claims ───────────────────────────
    print(f"[4/7] Resolving proof images ...")
    img_resolved = {}
    for cid in claims:
        img_resolved[cid] = resolve_image(cid, img_map, args.images_dir)

    valid_ids = [
        cid for cid in claims
        if img_resolved.get(cid)
        and claims[cid]["claim"]
        and claims[cid]["evidences"]
    ]
    print(
        f"      Valid claims (claim + proof image + evidence): {len(valid_ids)}")

    if len(valid_ids) < 10:
        print("ERROR: Too few valid claims. Check paths.")
        sys.exit(1)

    # ── Step 4: CLIP embeddings ────────────────────────────────
    print(f"\n[5/7] Building CLIP embedding index ...")
    cache_path = args.embed_cache if args.embed_cache else None
    text_mat, img_mat = build_embedding_index(
        valid_ids, claims, img_resolved, args.images_dir, cache_path)
    print(f"      text_matrix shape : {text_mat.shape}")
    print(f"      img_matrix  shape : {img_mat.shape}")

    print(f"      Pre-computing top-{args.top_k} image-image neighbors ...")
    img_neighbors = precompute_topk_neighbors(img_mat,  args.top_k)
    print(f"      Pre-computing top-{args.top_k} text-text neighbors ...")
    txt_neighbors = precompute_topk_neighbors(text_mat, args.top_k)

    # ── Step 5: entity extraction ──────────────────────────────
    print(f"\n[6/7] Building entity cache ...")
    ent_cache, _ = build_entity_cache(valid_ids, claims)

    # ── Step 6: generate ──────────────────────────────────────
    print(f"\n[7/7] Generating {args.target} × 4 samples ...")
    idx_map = {cid: i for i, cid in enumerate(valid_ids)}

    rows = []
    counts = {"Truthful": 0, "OOC": 0, "NEI": 0, "Hybrid": 0}
    target = args.target

    ids_shuffled = valid_ids[:]
    rng.shuffle(ids_shuffled)

    ptr = 0
    attempts = 0
    max_att = target * 4 * 20

    while sum(counts.values()) < target * 4 and attempts < max_att:
        attempts += 1
        cid = ids_shuffled[ptr % len(ids_shuffled)]
        ptr += 1
        rec = claims[cid]
        img_path = img_resolved[cid]
        idx = idx_map[cid]

        type_order = ["Truthful", "OOC", "NEI", "Hybrid"]
        rng.shuffle(type_order)

        for t in type_order:
            if counts[t] >= target:
                continue

            row = None
            if t == "Truthful":
                row = make_truthful(cid, rec, img_path)

            elif t == "OOC":
                row = make_ooc(cid, rec, img_path, idx, valid_ids,
                               img_resolved, img_neighbors, txt_neighbors, rng)

            elif t == "NEI":
                row = make_nei(cid, rec, img_path, idx, valid_ids,
                               txt_neighbors, img_neighbors,
                               ent_cache, claims, rng)

            elif t == "Hybrid":
                row = make_hybrid(cid, rec, img_path, idx, valid_ids,
                                  img_resolved, img_neighbors, txt_neighbors,
                                  ent_cache, claims, rng)

            if row is not None:
                rows.append(row)
                counts[t] += 1
                break

        if sum(counts.values()) % 1000 == 0 and sum(counts.values()) > 0:
            print(f"      Progress → {counts}")

    rng.shuffle(rows)
    print(f"\n      Final counts : {counts}")
    print(f"      Total rows   : {len(rows)}")

    # add new labels to rows for clarity
    for r in rows:
        if r["misinformation_label"] == "Truthful":
            r["new_label"] = r["original_truthfulness"]
        else:
            r["new_label"] = "refuted"

    # ── Step 7: write CSV ─────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "claim_id", "original_claim", "image_path",
        "grouped_text_evidence", "misinformation_label",
        "generated_misleading_claim", "technique",
        "original_truthfulness",
        "new_label",
        "ooc_donor_claim_id", "ooc_original_image", "ooc_swapped_image",
        "nei_swapped_entity_type", "nei_swapped_from", "nei_swapped_to",
        "nei_donor_claim_id", "clip_similarity_score",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓  Dataset saved → {args.output}")
    print(f"   {'Label':<30} {'Count':>6}")
    print(f"   {'-'*38}")
    lc = collections.Counter(r["misinformation_label"] for r in rows)
    for lbl, cnt in sorted(lc.items()):
        print(f"   {lbl:<30} {cnt:>6}")

    swap_types = collections.Counter(
        r["nei_swapped_entity_type"] for r in rows
        if r["nei_swapped_entity_type"] not in ("", "FALLBACK")
    )
    if swap_types:
        print(f"\n   NEI entity types swapped:")
        for k, v in swap_types.most_common():
            print(f"   {k:<14} {v:>6}")

    if not clip_ok:
        print("\n   ⚠  CLIP was NOT available — embeddings used pixel/hash proxies.")
        print("      Install CLIP for proper hard-negative selection:")
        print("      pip install torch torchvision")
        print("      pip install git+https://github.com/openai/CLIP.git")


if __name__ == "__main__":
    main()
