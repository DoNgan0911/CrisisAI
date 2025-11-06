# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Auto-generate Crisis Need (CN) queries from SA tweets with few-shot prompting.

# Input:
#   - SA_enriched.csv   (must contain columns: tweet_id, augmented_text_0)
# Output:
#   - cn_queries.csv    (tweet_id, augmented_text_0, cn_query, reason, flags)
#   - cn_queries.xlsx   (pretty Excel, best-effort)

# Model:
#   - meta-llama/Meta-Llama-3.1-8B-Instruct (transformers, local or HF cache)

# Notes:
#   - Fixes HF padding error by assigning pad_token.
#   - Uses left padding (better for decoder-only LMs).
#   - Caps prompt length to avoid overflow.
# """

# import os, json, re, argparse, math
# import pandas as pd
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# # ---------------- Config ----------------
# # DEFAULT_IN  = "SA_enriched.csv"
# # DEFAULT_OUT = "FULL_cn_queries.csv"
# # DEFAULT_IN  = "sa_samples_564.csv"
# # DEFAULT_OUT = "564_cn_queries.csv"
# DEFAULT_IN  = "551_samples.csv"
# DEFAULT_OUT = "551_cn_queries.csv"
# MODEL_ID    = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# BATCH_SIZE  = 8
# MAX_NEW     = 72
# TEMP, TOP_P, TOP_K = 0.2, 0.9, 40  # conservative for consistency
# SEED = int(os.getenv("SEED", "42"))

# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE       = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# MAX_INPUT_CHARS = 6000   # guardrail for very long rows
# MAX_MODEL_TOKENS = 4096  # adjust if your model has a different context window

# # -------------- Few-shot prompt --------------
# SYSTEM_PROMPT = """You are a crisis-response assistant.

# Task:
# - Convert a descriptive Situational Awareness (SA) tweet into ONE concrete Crisis Need (CN) query a person would ask.
# - Keep location/time cues if present. Do NOT invent new places/times/resources.
# - Be short (<= 25 words), natural, and action-oriented (question or first-person need).
# - Target common need types: shelter, food/water, medical help, missing persons, donations, volunteering, animal rescue, recovery centers, services.
# - If the SA is purely political/opinion/summary with no actionable cue, output SKIP.

# Output strictly as JSON:
# {"cn_query": "<one sentence>", "reason": "<why this is actionable in 1 short clause>"}
# """

# FEW_SHOTS = [
#     {
#         "sa": "Disaster Recovery Center is open daily 9am–7pm at Chico Mall (former Sears), 1982 E. 20th St, Chico, starting Nov 19.",
#         "cn": {"cn_query": "Where can Camp Fire survivors get help at the Chico Disaster Recovery Center?", "reason": "SA announces DRC availability and location"}
#     },
#     {
#         "sa": "Convoy of Hope distributing aid quickly in Los Angeles County during wildfires.",
#         "cn": {"cn_query": "Where can I receive Convoy of Hope assistance in Los Angeles County today?", "reason": "SA mentions specific aid provider and area"}
#     },
#     {
#         "sa": "Lost pets from Camp Fire are at Silver Dollar Fairgrounds in Chico; check 9am–5pm Wed Nov 21.",
#         "cn": {"cn_query": "Where can I check for rescued pets near Chico after the Camp Fire?", "reason": "Pet reunification service with time/place"}
#     },
#     {
#         "sa": "Free Thanksgiving dinner for evacuees hosted in Chico on Nov 21 evening.",
#         "cn": {"cn_query": "Where and when is the free Thanksgiving meal for evacuees in Chico?", "reason": "Direct service and schedule"}
#     },
#     {
#         "sa": "Article: Death toll rises; no service or resource mentioned.",
#         "cn": {"cn_query": "SKIP", "reason": "Purely informational; no actionable service"}
#     },
#     {
#         "sa": "Far Northern Regional Center seeking help for clients and staff affected by Camp Fire.",
#         "cn": {"cn_query": "How can I donate or volunteer to support the Far Northern Regional Center after the Camp Fire?", "reason": "Call for assistance"}
#     },
# ]

# def build_prompt(sa_text: str) -> str:
#     # Trim overly long SA text (keep head and tail)
#     s = (sa_text or "").strip()
#     if len(s) > MAX_INPUT_CHARS:
#         s = s[: int(MAX_INPUT_CHARS * 0.6)] + " … " + s[-int(MAX_INPUT_CHARS * 0.3):]

#     shots = []
#     for ex in FEW_SHOTS:
#         shots.append(f"SA: {ex['sa']}\nCN: {json.dumps(ex['cn'], ensure_ascii=False)}")
#     shots_block = "\n\n".join(shots)
#     return f"{shots_block}\n\nSA: {s}\nCN:"

# # -------------- JSON parser with fallback --------------
# def parse_json(s: str):
#     if not s:
#         return None
#     # Take the first {...} block (greedy across lines)
#     m = re.search(r"\{.*\}", s, flags=re.DOTALL)
#     if not m:
#         return None
#     block = m.group(0)
#     try:
#         return json.loads(block)
#     except Exception:
#         # light cleanup for quotes/trailing commas
#         block2 = re.sub(r",\s*}", "}", block)
#         block2 = re.sub(r"[\u201c\u201d]", '"', block2)
#         try:
#             return json.loads(block2)
#         except Exception:
#             return None

# def generate(model, tok, prompts):
#     # Tokenize with left padding and safe max length
#     inputs = tok(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max(256, MAX_MODEL_TOKENS - MAX_NEW - 16),
#     ).to(DEVICE)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW,
#             do_sample=True if TEMP > 0 else False,
#             temperature=TEMP,
#             top_p=TOP_P,
#             top_k=TOP_K,
#             pad_token_id=tok.pad_token_id,
#             eos_token_id=tok.eos_token_id,
#         )

#     decoded = tok.batch_decode(outputs, skip_special_tokens=True)
#     # keep only assistant completion part after last "CN:"
#     cleaned = []
#     for full, prompt in zip(decoded, prompts):
#         idx = full.rfind("CN:")
#         cleaned.append(full[idx+3:].strip() if idx != -1 else full.strip())
#     return cleaned

# def main(args):
#     set_seed(SEED)

#     # Load data
#     df = pd.read_csv(args.input)
#     assert "augmented_text_0" in df.columns, "Input must contain column 'augmented_text_0'."
#     if "tweet_id" not in df.columns:
#         df["tweet_id"] = [f"row_{i}" for i in range(len(df))]

#     # -------- Tokenizer
#     tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

#     # Ensure pad token exists; prefer reusing EOS to avoid resizing embeddings
#     added_new_pad = False
#     if tok.pad_token is None:
#         if tok.eos_token is not None:
#             tok.pad_token = tok.eos_token
#         else:
#             tok.add_special_tokens({"pad_token": "[PAD]"})
#             added_new_pad = True
#     tok.padding_side = "left"

#     # -------- Model
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=DTYPE,
#         device_map="auto" if DEVICE == "cuda" else None,
#     )

#     # If we added a brand-new token (rare), resize embeddings
#     if added_new_pad:
#         model.resize_token_embeddings(len(tok))

#     # -------- Build prompts
#     rows = []
#     prompts = []
#     for _, row in df.iterrows():
#         sa = str(row["augmented_text_0"]) if pd.notna(row["augmented_text_0"]) else ""
#         p = build_prompt(sa)
#         chat = tok.apply_chat_template(
#             [{"role": "system", "content": SYSTEM_PROMPT},
#              {"role": "user", "content": p}],
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         prompts.append(chat)
#         rows.append(row)

#     # -------- Generate
#     results = []
#     for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating CN"):
#         batch = prompts[i:i+BATCH_SIZE]
#         outs  = generate(model, tok, batch)
#         for out in outs:
#             obj = parse_json(out) or {"cn_query": "SKIP", "reason": "parse_fail"}
#             cn = (obj.get("cn_query") or "").strip()
#             reason = (obj.get("reason") or "").strip()

#             flags = []
#             if not cn or cn.upper() == "SKIP":
#                 flags.append("SKIP")
#             if len(cn) > 120:
#                 flags.append("long")
#             if len(cn) == 0:
#                 flags.append("empty")
#             results.append((cn, reason, ",".join(flags)))

#     # -------- Write outputs
#     out_df = df.copy()
#     out_df["cn_query"] = [r[0] for r in results]
#     out_df["reason"]   = [r[1] for r in results]
#     out_df["flags"]    = [r[2] for r in results]

#     cols = ["tweet_id","augmented_text_0","cn_query","reason","flags"]
#     out_df[cols].to_csv(args.output, index=False)

#     try:
#         xlsx_path = args.output.replace(".csv", ".xlsx")
#         out_df.to_excel(xlsx_path, index=False)
#     except Exception:
#         pass

#     # -------- Report
#     total = len(out_df)
#     skipped = (out_df["cn_query"].str.upper() == "SKIP").sum()
#     print(f"\nDone. Wrote {args.output}.")
#     print(f"Generated: {total - skipped} | Skipped: {skipped} | Total: {total}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input",  default=DEFAULT_IN,  help="Path to SA_enriched.csv")
#     parser.add_argument("--output", default=DEFAULT_OUT, help="Path to write cn_queries.csv")
#     args = parser.parse_args()
#     main(args)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use Llama (Meta-Llama-3.1-8B-Instruct) to assign one fine-grained need_type
based on humAID_class and CN query.

Input : cn_queries.csv (must contain columns: cn_query, humAID_class)
Output: cn_with_needtype.csv / .xlsx
"""

import os, json, re, argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== CONFIG =====================
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
BATCH_SIZE = 6
MAX_NEW = 64
TEMP, TOP_P, TOP_K = 0.1, 0.9, 40
INPUT_CSV  = "305_queries.csv"
OUTPUT_CSV = "305_cn_with_needtype.csv"

SA_TO_NEED_MAP = {
    "injured_or_dead_people": ["medical_help"],
    "displaced_people_and_evacuations": ["shelter", "food_water", "transportation"],
    "infrastructure_and_utility_damage": ["transportation", "power_restoration", "logistics"],
    "rescue_volunteering_or_donation_effort": ["resource_coordination", "logistics"],
    "missing_or_found_people": ["family_reunification", "protection"]
}

ALL_NEEDS = sorted(set(sum(SA_TO_NEED_MAP.values(), [])))

SYSTEM_PROMPT = f"""
You are a humanitarian crisis analyst.

Given a crisis-need query and its HumAID class, 
choose ONE best-matching need_type from the candidate list provided.

Rules:
- Pick only from the candidate list.
- If no clear clue, pick the *most general* one that fits.
- Base your choice on the intent and content of the CN query.
- Respond STRICTLY as JSON: {{"need_type": "<label>", "reason": "<short reason>"}}
"""

# ===============================================================
def build_prompt(cn_query, humaid):
    humaid = str(humaid).strip()
    candidates = SA_TO_NEED_MAP.get(humaid, ALL_NEEDS)
    return f"""
HumAID class: {humaid}
Candidate need types: {candidates}

CN query: "{cn_query}"

Choose exactly one need_type that best represents the query.
Return JSON only.
"""

def parse_json(s):
    m = re.search(r"\{.*?\}", s, re.DOTALL)
    if not m:
        return None
    txt = m.group(0)
    txt = re.sub(r"[\u201c\u201d]", '"', txt)
    try:
        return json.loads(txt)
    except Exception:
        return None

def generate_batch(model, tok, prompts):
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=TEMP,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id
        )
    decoded = tok.batch_decode(outputs, skip_special_tokens=True)
    cleaned = []
    for full, prompt in zip(decoded, prompts):
        out = full[len(prompt):].strip()
        if not out:
            out = full.strip()
        cleaned.append(out)
    return cleaned

# ===============================================================
def main():
    df = pd.read_csv(INPUT_CSV)
    if "cn_query" not in df.columns or "humAID_class" not in df.columns:
        raise ValueError("Input file must contain columns: cn_query and humAID_class")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    prompts = []
    for _, row in df.iterrows():
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row["cn_query"], row["humAID_class"])}
        ]
        chat = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prompts.append(chat)

    results = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Assigning need_type"):
        batch = prompts[i:i+BATCH_SIZE]
        outs = generate_batch(model, tok, batch)
        for out in outs:
            obj = parse_json(out) or {"need_type": "unknown", "reason": "parse_fail"}
            results.append((obj.get("need_type", ""), obj.get("reason", "")))

    df["need_type"] = [r[0] for r in results]
    df["need_reason"] = [r[1] for r in results]

    df.to_csv(OUTPUT_CSV, index=False)
    try:
        df.to_excel(OUTPUT_CSV.replace(".csv", ".xlsx"), index=False)
    except Exception:
        pass

    print(f"✅ Done. Saved to {OUTPUT_CSV}")
    print(df["need_type"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
