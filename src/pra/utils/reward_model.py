import math
import requests


def call_mu_phi(messages, tokenizer, reward_model_path, reward_model_url, mu_phi_params):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    payload = {"model": reward_model_path, "prompt": prompt, **mu_phi_params}
    resp = requests.post(reward_model_url, json=payload)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    choice["_prompt"] = prompt
    return choice


def parse_mu_phi_output(output):
    """Parse reward model output to extract quality and dependency scores.

    Returns a dict with ``hat_r_qual`` and ``hat_r_dep`` floats. When per-token
    logprobs are available for "0"/"1", they're converted to a calibrated
    probability of "1"; otherwise the raw decoded "q,d" string is used.
    """
    text = output["text"] or ""
    try:
        parts = [float(x.strip()) for x in text.split(",")][:2]
        r_qual_raw, r_dep_raw = (parts + [0.0, 0.0])[:2]
    except ValueError:
        r_qual_raw, r_dep_raw = 0.0, 0.0

    logprobs = output["logprobs"]
    tokens = logprobs["tokens"]
    top_logprobs = logprobs["top_logprobs"]

    probs = []
    for tok, top in zip(tokens, top_logprobs):
        if tok.strip() not in ("0", "1"):
            continue
        p1 = p0 = math.exp(-100)
        for t, lp in (top or {}).items():
            if t.strip() == "1":
                p1 = math.exp(lp)
            elif t.strip() == "0":
                p0 = math.exp(lp)
        probs.append(p1 / (p1 + p0) if p1 + p0 > 0 else 0.0)
        if len(probs) == 2:
            break

    return {
        "hat_r_qual": probs[0] if probs else r_qual_raw,
        "hat_r_dep": probs[1] if len(probs) > 1 else r_dep_raw,
    }
