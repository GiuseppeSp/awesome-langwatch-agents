"""
Deterministic dataset writer for the mixture-of-agents agent.

Each row is a question asking for a well-defined SET, plus the canonical gold set.
Sizes span small (single-pass usually complete -> no headroom) to large (single-pass
under-recalls -> room for the ensemble to add coverage, and room for drafts to add
wrong items -> a precision job for the aggregator). Golds are hand-verified here so the
dataset is self-contained; run this once to (re)generate dataset.csv.
"""

from __future__ import annotations

import csv
import json

# question, gold set (canonical spellings), size bucket
DATA = [
    ("List all the countries that border Germany.",
     ["France","Belgium","Netherlands","Luxembourg","Denmark","Poland","Czechia","Austria","Switzerland"], "medium"),
    ("List all the countries that border metropolitan France (in Europe).",
     ["Belgium","Luxembourg","Germany","Switzerland","Italy","Spain","Andorra","Monaco"], "medium"),
    ("List all the countries that border China.",
     ["Russia","Mongolia","North Korea","Vietnam","Laos","Myanmar","India","Bhutan","Nepal","Pakistan","Afghanistan","Tajikistan","Kyrgyzstan","Kazakhstan"], "large"),
    ("List all the countries that border Russia by land.",
     ["Norway","Finland","Estonia","Latvia","Lithuania","Poland","Belarus","Ukraine","Georgia","Azerbaijan","Kazakhstan","Mongolia","China","North Korea"], "large"),
    ("List all the countries in South America.",
     ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"], "medium"),
    ("List all the countries in Central America.",
     ["Belize","Costa Rica","El Salvador","Guatemala","Honduras","Nicaragua","Panama"], "medium"),
    ("List all fifty states of the United States of America.",
     ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"], "large"),
    ("List all the member countries of the European Union.",
     ["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"], "large"),
    ("List all the member countries of ASEAN.",
     ["Brunei","Cambodia","Indonesia","Laos","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam"], "medium"),
    ("List all the provinces of Canada.",
     ["Alberta","British Columbia","Manitoba","New Brunswick","Newfoundland and Labrador","Nova Scotia","Ontario","Prince Edward Island","Quebec","Saskatchewan"], "medium"),
    ("List all thirteen of the original American colonies.",
     ["Delaware","Pennsylvania","New Jersey","Georgia","Connecticut","Massachusetts","Maryland","South Carolina","New Hampshire","Virginia","New York","North Carolina","Rhode Island"], "large"),
    ("List all the nations that have won the FIFA World Cup.",
     ["Brazil","Germany","Italy","Argentina","France","Uruguay","England","Spain"], "medium"),
    ("List all the Ivy League universities.",
     ["Harvard","Yale","Princeton","Columbia","Brown","Dartmouth","Cornell","Pennsylvania"], "medium"),
    ("List all twelve signs of the zodiac.",
     ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"], "medium"),
    ("List all the chemical elements with atomic numbers 1 through 20.",
     ["hydrogen","helium","lithium","beryllium","boron","carbon","nitrogen","oxygen","fluorine","neon","sodium","magnesium","aluminium","silicon","phosphorus","sulfur","chlorine","argon","potassium","calcium"], "large"),
    ("List all the chemical elements with atomic numbers 21 through 36.",
     ["scandium","titanium","vanadium","chromium","manganese","iron","cobalt","nickel","copper","zinc","gallium","germanium","arsenic","selenium","bromine","krypton"], "large"),
    ("List all the chemical elements whose symbol is a single letter.",
     ["hydrogen","boron","carbon","nitrogen","oxygen","fluorine","phosphorus","potassium","sulfur","vanadium","yttrium","iodine","tungsten","uranium"], "large"),
    ("List all twenty standard amino acids.",
     ["alanine","arginine","asparagine","aspartic acid","cysteine","glutamic acid","glutamine","glycine","histidine","isoleucine","leucine","lysine","methionine","phenylalanine","proline","serine","threonine","tryptophan","tyrosine","valine"], "large"),
    ("List all the planets in the Solar System.",
     ["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"], "medium"),
]


def main() -> None:
    with open("dataset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "gold_json", "size"])
        for q, gold, size in DATA:
            w.writerow([q, json.dumps(gold, ensure_ascii=False), size])
    print(f"wrote dataset.csv — {len(DATA)} questions "
          f"(gold sizes {min(len(g) for _,g,_ in DATA)}–{max(len(g) for _,g,_ in DATA)})")


if __name__ == "__main__":
    main()
