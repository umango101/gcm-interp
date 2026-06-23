#!/usr/bin/env python
"""Build >=200 minimal pairs of gender-stereotyped professions that differ by
exactly ONE token.
 
Construction: a stereotyped HEAD word (single token: nurse, welder, ...) is
wrapped in a shared, gender-neutral MODIFIER that is identical on both sides of
the pair ("senior", "trainee", "part-time", ...). The modifier may be any number
of tokens; because it is identical across the pair, the only differing token is
the head. So:
 
    "The senior nurse said that"   -> [The, senior, _nurse,  _said, _that]
    "The senior welder said that"  -> [The, senior, _welder, _said, _that]
                                                      ^ the single differing token
 
A few single-token heads x a few shared modifiers = hundreds of distinct,
verified pairs. Every emitted pair is checked directly: equal token length and
Hamming distance exactly 1 in the carrier template.
 
Run on the cluster where the tokenizer lives:
    python build_minimal_pairs_200.py --model Qwen/Qwen1.5-14B-Chat --target 200
"""
import json
import argparse
 
DEFAULT_TEMPLATE = "The {role} said that"
DEFAULT_TARGET = 200
 
# Gender-neutral modifiers, identical on both sides of every pair. Their own
# token count is irrelevant; they never contribute the differing token.
# "" yields the bare head pair. Add more here to scale the pair count up.
SCAFFOLDS = [
    "", "senior", "junior", "head", "lead", "chief", "trainee", "apprentice",
    "veteran", "experienced", "skilled", "licensed", "certified", "professional",
    "seasoned", "rookie", "novice", "young", "older", "retired", "former",
    "aspiring", "part-time", "full-time", "freelance", "union", "local",
    "night-shift", "day-shift", "off-duty",
]
 
# Stereotyped head candidates. Short single words are most likely to survive the
# single-token filter; multi-token ones are dropped automatically. Be generous.
FEMALE_HEADS = [
    "nurse", "midwife", "nanny", "maid", "housemaid", "nursemaid", "secretary",
    "receptionist", "waitress", "hostess", "cashier", "librarian", "florist",
    "seamstress", "dressmaker", "milliner", "babysitter", "caregiver",
    "housekeeper", "homemaker", "housewife", "hairdresser", "hairstylist",
    "stylist", "beautician", "manicurist", "esthetician", "cosmetologist",
    "dietitian", "nutritionist", "governess", "stewardess", "typist",
    "stenographer", "nun", "ballerina", "actress", "masseuse", "dancer",
    "cheerleader", "gymnast", "matron", "maiden", "mistress", "soprano",
    "songstress", "diva", "debutante", "choreographer", "clerk",
]
 
MALE_HEADS = [
    "carpenter", "electrician", "plumber", "welder", "mason", "roofer",
    "mechanic", "machinist", "blacksmith", "miner", "logger", "lumberjack",
    "trucker", "soldier", "marine", "sailor", "pilot", "firefighter", "officer",
    "sheriff", "guard", "bouncer", "butcher", "brewer", "barber", "lineman",
    "fisherman", "hunter", "trapper", "ranger", "engineer", "programmer",
    "mover", "laborer", "foreman", "contractor", "surveyor", "locksmith",
    "smith", "boxer", "wrestler", "bodybuilder", "athlete", "referee", "captain",
    "conductor", "technician", "bricklayer", "driller", "roughneck", "gladiator",
]
 
 
def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]
 
 
def filled(tok, role, template):
    return encode(tok, template.format(role=role))
 
 
def role_span_len(tok, role, template):
    left, right = template.split("{role}")
    return len(encode(tok, left + role + right)) - len(encode(tok, left.rstrip() + right))
 
 
def single_token_heads(tok, pool, template):
    seen, out = set(), []
    for r in pool:
        if r not in seen:
            seen.add(r)
            if role_span_len(tok, r, template) == 1:
                out.append(r)
    return out
 
 
def is_minimal_pair(tok, role_f, role_m, template):
    tf, tm = filled(tok, role_f, template), filled(tok, role_m, template)
    if len(tf) != len(tm):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(tf, tm)) if a != b]
    return diffs[0] if len(diffs) == 1 else None
 
 
def compose(scaffold, head):
    return head if scaffold == "" else f"{scaffold} {head}"
 
 
def build(tok, template=DEFAULT_TEMPLATE, target=DEFAULT_TARGET,
          female_heads=FEMALE_HEADS, male_heads=MALE_HEADS, scaffolds=SCAFFOLDS,
          balance=False):
    F = single_token_heads(tok, female_heads, template)
    M = single_token_heads(tok, male_heads, template)
    if not F or not M:
        raise SystemExit("No single-token heads survived; widen the head pools.")
    if balance:
        # Equalize unique-head vocab on both sides for a symmetric contrast.
        n = min(len(F), len(M))
        F, M = F[:n], M[:n]
    k = min(len(F), len(M))
 
    pairs, seen = [], set()
 
    def try_add(rf, rm):
        if (rf, rm) in seen:
            return
        idx = is_minimal_pair(tok, rf, rm, template)
        if idx is not None:
            seen.add((rf, rm))
            tf = filled(tok, rf, template)
            pairs.append({"female": rf, "male": rm, "diff_index": idx,
                          "female_token": tf[idx],
                          "male_token": filled(tok, rm, template)[idx]})
 
    # Primary: scaffold x head, rotating the male partner per scaffold so the
    # head contrast varies rather than repeating the same f/m mapping.
    for j, s in enumerate(scaffolds):
        for i in range(k):
            try_add(compose(s, F[i]), compose(s, M[(i + j) % len(M)]))
            if len(pairs) >= target:
                return pairs, F, M
    # Top-up: bare-template cross product, in case scaffolds x heads fell short.
    for f in F:
        for m in M:
            try_add(f, m)
            if len(pairs) >= target:
                return pairs, F, M
    return pairs, F, M
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-14B-Chat")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="carrier with a {role} slot")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--balance", action="store_true",
                    help="trim heads so female/male use equal unique-word counts")
    ap.add_argument("--out-prefix", default="professions")
    args = ap.parse_args()
 
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
 
    pairs, F, M = build(tok, args.template, args.target, balance=args.balance)
    print(f"single-token heads -> female: {len(F)}  male: {len(M)}")
    print(f"verified minimal pairs: {len(pairs)} (target {args.target})")
    uf = len({p["female"].split()[-1] for p in pairs})
    um = len({p["male"].split()[-1] for p in pairs})
    print(f"distinct head words actually used -> female: {uf}  male: {um}")
    if len(pairs) < args.target:
        print("  shortfall: add more SCAFFOLDS or single-token-friendly heads.")
 
    females = [p["female"] for p in pairs]
    males = [p["male"] for p in pairs]
 
    # Drop-in for the existing loader: data[i][0] -> role; female[i]/male[i] pair.
    with open(f"{args.out_prefix}_female_minpair.json", "w", encoding="utf-8") as fh:
        json.dump([[r] for r in females], fh, ensure_ascii=False, indent=2)
    with open(f"{args.out_prefix}_male_minpair.json", "w", encoding="utf-8") as fh:
        json.dump([[r] for r in males], fh, ensure_ascii=False, indent=2)
    with open(f"{args.out_prefix}_pairs.json", "w", encoding="utf-8") as fh:
        json.dump({"template": args.template, "pairs": pairs}, fh,
                  ensure_ascii=False, indent=2)
 
    for p in pairs[:8]:
        print(f"  {p['female']:>22} / {p['male']:<22} diff@{p['diff_index']}")
 
 
if __name__ == "__main__":
    main()