#!/usr/bin/env python
"""Pair 200 distinct female-stereotyped professions with 200 distinct
male-stereotyped professions, matched by in-context token length.

These are NOT one-token-minimal pairs -- that is impossible at this scale (only
~15 single-token female-stereo words exist in the vocab). Instead each pair is
length-matched: the female and male professions occupy the same number of
tokens in the carrier, so their profession spans align position-for-position.
That is the achievable control for a difference-in-means / span-patch design
when you need lexical diversity rather than a single differing token.

Pairing = sort both lists by profession-span token length, then zip. This keeps
all 200 pairs and maximizes the number that are exactly length-equal.

    python pair_stereo_professions.py --model Qwen/Qwen1.5-14B-Chat
"""
import json
import argparse

DEFAULT_TEMPLATE = "The {role} said that"

FEMALE = [
    "nurse", "registered nurse", "nurse practitioner", "licensed practical nurse",
    "nurse midwife", "midwife", "doula", "pediatric nurse", "neonatal nurse",
    "hospice nurse", "school nurse", "obstetric nurse", "fertility nurse",
    "women's health nurse", "nursing assistant", "certified nursing assistant",
    "home health aide", "personal care aide", "caregiver", "elder care worker",
    "dental hygienist", "dental assistant", "medical assistant", "medical secretary",
    "medical receptionist", "phlebotomist", "ultrasound technician", "sonographer",
    "occupational therapist", "speech therapist", "speech-language pathologist",
    "physical therapy assistant", "dietitian", "nutritionist", "lactation consultant",
    "art therapist", "music therapist", "recreational therapist", "play therapist",
    "dietary aide", "pharmacy technician", "veterinary technician", "veterinary assistant",
    "pet groomer", "kennel attendant", "preschool teacher", "kindergarten teacher",
    "elementary school teacher", "special education teacher", "early childhood educator",
    "preschool director", "daycare worker", "childcare worker", "infant care provider",
    "nanny", "babysitter", "au pair", "teacher's aide", "teaching assistant", "tutor",
    "reading specialist", "school counselor", "guidance counselor", "librarian",
    "school librarian", "dance teacher", "art teacher", "music teacher", "secretary",
    "administrative assistant", "executive assistant", "personal assistant",
    "virtual assistant", "receptionist", "front desk clerk", "office manager",
    "office clerk", "data entry clerk", "file clerk", "typist", "stenographer",
    "court reporter", "transcriptionist", "medical transcriptionist", "bookkeeper",
    "payroll clerk", "billing clerk", "accounts payable clerk", "scheduling coordinator",
    "switchboard operator", "telephone operator", "legal secretary", "legal assistant",
    "paralegal", "notary public", "real estate assistant", "hairdresser", "hairstylist",
    "cosmetologist", "beautician", "manicurist", "nail technician", "pedicurist",
    "esthetician", "skincare specialist", "makeup artist", "spa attendant", "spa manager",
    "massage therapist", "lash technician", "eyebrow technician", "waxing specialist",
    "salon receptionist", "beauty advisor", "perfume consultant", "cosmetics salesperson",
    "housekeeper", "maid", "housemaid", "house cleaner", "cleaner", "cleaning lady",
    "domestic worker", "laundress", "home organizer", "waitress", "server", "hostess",
    "flight attendant", "cabin crew member", "cashier", "retail sales associate",
    "sales clerk", "store clerk", "personal shopper", "fashion consultant",
    "wardrobe stylist", "cake decorator", "bakery clerk", "social worker", "caseworker",
    "child welfare worker", "community health worker", "marriage counselor",
    "family therapist", "grief counselor", "mental health counselor",
    "human resources specialist", "human resources coordinator", "recruiter",
    "diversity and inclusion officer", "customer service representative",
    "call center agent", "telemarketer", "survey interviewer", "fashion designer",
    "interior designer", "interior decorator", "seamstress", "dressmaker",
    "bridal seamstress", "bridal consultant", "embroiderer", "knitter", "weaver",
    "textile worker", "milliner", "pattern maker", "quilter", "florist",
    "flower arranger", "jewelry designer", "event planner", "wedding planner",
    "party planner", "event coordinator", "public relations specialist", "copy editor",
    "proofreader", "greeting card writer", "romance novelist", "food blogger",
    "lifestyle blogger", "birth photographer", "newborn photographer", "mommy blogger",
    "yoga instructor", "pilates instructor", "aerobics instructor", "zumba instructor",
    "dance instructor", "ballet dancer", "ballerina", "choreographer",
    "figure skating coach", "cheerleading coach", "gymnastics instructor",
    "nutrition coach", "health coach", "life coach",
]

MALE = [
    "construction worker", "carpenter", "electrician", "plumber", "pipefitter",
    "welder", "mason", "bricklayer", "stonemason", "roofer", "drywaller", "plasterer",
    "tiler", "glazier", "ironworker", "steelworker", "boilermaker", "sheet metal worker",
    "scaffolder", "paver", "concrete finisher", "cement mason", "demolition worker",
    "construction foreman", "site foreman", "general contractor", "machinist",
    "tool and die maker", "fabricator", "metalworker", "blacksmith", "farrier",
    "millwright", "mechanic", "auto mechanic", "diesel mechanic", "aircraft mechanic",
    "motorcycle mechanic", "automotive technician", "tire technician", "auto body repairer",
    "small engine mechanic", "heavy equipment operator", "crane operator",
    "forklift operator", "bulldozer operator", "excavator operator", "truck driver",
    "long-haul trucker", "delivery driver", "bus driver", "taxi driver", "freight handler",
    "dock worker", "longshoreman", "stevedore", "warehouse worker", "mover",
    "tow truck driver", "garbage collector", "sanitation worker", "landscaper",
    "groundskeeper", "lawn care worker", "tree surgeon", "arborist", "logger",
    "lumberjack", "forester", "miner", "coal miner", "oil rig worker", "roughneck",
    "driller", "oil field worker", "pipeline worker", "power lineman", "lineman",
    "utility worker", "HVAC technician", "furnace installer", "elevator mechanic",
    "locksmith", "rigger", "wind turbine technician", "solar panel installer",
    "land surveyor", "quantity surveyor", "building inspector", "home inspector",
    "pest control technician", "exterminator", "railroad worker", "train conductor",
    "locomotive engineer", "railroad engineer", "ship captain", "sea captain",
    "boat captain", "ferry operator", "fisherman", "commercial fisherman", "deckhand",
    "merchant mariner", "harbor pilot", "airline pilot", "fighter pilot",
    "helicopter pilot", "air traffic controller", "firefighter", "fire chief",
    "police officer", "police detective", "sheriff", "highway patrol officer",
    "security guard", "bodyguard", "bouncer", "prison guard", "correctional officer",
    "soldier", "marine", "infantryman", "army officer", "navy officer", "sniper",
    "combat engineer", "paratrooper", "special forces operator", "drill sergeant",
    "tank operator", "bomb disposal technician", "mercenary", "hunter", "trapper",
    "game warden", "park ranger", "mountain guide", "wilderness guide",
    "rock climbing instructor", "ski instructor", "scuba diving instructor",
    "surf instructor", "personal trainer", "strength coach", "boxing coach",
    "wrestling coach", "football coach", "basketball coach", "baseball coach",
    "hockey coach", "sports coach", "athletic director", "referee", "umpire",
    "sportscaster", "sports analyst", "professional athlete", "boxer", "wrestler",
    "mixed martial artist", "weightlifter", "bodybuilder", "race car driver",
    "motocross rider", "stunt performer", "stuntman", "demolition derby driver",
    "drone operator", "mechanical engineer", "civil engineer", "electrical engineer",
    "aerospace engineer", "petroleum engineer", "structural engineer", "architect",
    "software engineer", "systems administrator", "network engineer",
    "computer programmer", "video game developer", "IT technician",
    "computer repair technician", "butcher", "meat cutter", "brewer", "distiller",
    "brewmaster", "grill cook", "pitmaster", "barber", "tattoo artist", "car salesman",
    "used car dealer", "stockbroker", "investment banker", "hedge fund manager",
    "commodities trader", "venture capitalist", "chief executive officer",
]


def span_len(tok, role, template):
    left, right = template.split("{role}")
    full = tok(left + role + right, add_special_tokens=False)["input_ids"]
    base = tok(left.rstrip() + right, add_special_tokens=False)["input_ids"]
    return len(full) - len(base)


def build(tok, template=DEFAULT_TEMPLATE, female=FEMALE, male=MALE):
    assert len(female) == len(set(female)) == 200, "female list must be 200 unique"
    assert len(male) == len(set(male)) == 200, "male list must be 200 unique"
    assert not (set(female) & set(male)), "lists overlap"

    f = sorted(female, key=lambda r: (span_len(tok, r, template), r))
    m = sorted(male, key=lambda r: (span_len(tok, r, template), r))

    pairs = []
    for rf, rm in zip(f, m):
        lf, lm = span_len(tok, rf, template), span_len(tok, rm, template)
        pairs.append({"female": rf, "male": rm,
                      "female_len": lf, "male_len": lm, "len_match": lf == lm})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-14B-Chat")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--out-prefix", default="professions")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    pairs = build(tok, args.template)
    matched = sum(p["len_match"] for p in pairs)
    print(f"pairs: {len(pairs)}  exactly length-matched: {matched}/{len(pairs)}")

    with open(f"{args.out_prefix}_female_stereo_2.json", "w", encoding="utf-8") as fh:
        json.dump([[p["female"]] for p in pairs], fh, ensure_ascii=False, indent=2)
    with open(f"{args.out_prefix}_male_stereo_2.json", "w", encoding="utf-8") as fh:
        json.dump([[p["male"]] for p in pairs], fh, ensure_ascii=False, indent=2)
    with open(f"{args.out_prefix}_pairs.json", "w", encoding="utf-8") as fh:
        json.dump({"template": args.template, "pairs": pairs}, fh,
                  ensure_ascii=False, indent=2)

    for p in pairs[:6] + pairs[-3:]:
        flag = "" if p["len_match"] else "  (len differs)"
        print(f"  [{p['female_len']}] {p['female']:>26} / {p['male']:<26}{flag}")


if __name__ == "__main__":
    main()