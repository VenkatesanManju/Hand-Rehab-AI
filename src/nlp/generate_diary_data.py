"""
Diary Dataset Generator (Synthetic)
===================================
Generates synthetic patient diary entries for hand rehabilitation monitoring.
Labels 4 symptom dimensions: pain, stiffness, mobility limitation, grip limitation (each 0-3).
Also provides token-level BIO tags (SYMPTOM, SEVERITY_WORD) to support sequence labelling.

DISCLAIMER:
This dataset is purely SYNTHETIC, generated for NLP prototyping and research.
It has NOT been clinically validated.
The system is NOT a diagnostic tool.

Seed: 42 (fixed throughout)
"""

import os
import re
import csv
import json
import random
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

# Fixed random seed for strict reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# BIO Tag categories:
# O: Outside
# B-SYMPTOM / I-SYMPTOM: Mentions of symptoms or anatomical functions (pain, ache, stiffness, mobility, movement, grip, hold)
# B-SEVERITY_WORD / I-SEVERITY_WORD: Quantifiers, negations, or severity adjectives (no, not, zero, mild, moderate, severe, sharp, cannot)

# Phrase banks: symptom -> severity level -> list of token-tag pairs
# Each phrase is a list of tuples: (token, bio_tag)
PHRASE_BANK: Dict[str, Dict[int, List[List[Tuple[str, str]]]]] = {
    "pain": {
        0: [
            [("No", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("today", "O")],
            [("felt", "O"), ("zero", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("in", "O"), ("my", "O"), ("hand", "O")],
            [("no", "B-SEVERITY_WORD"), ("soreness", "B-SYMPTOM"), ("or", "O"), ("ache", "B-SYMPTOM")],
            [("completely", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("free", "B-SEVERITY_WORD")],
            [("not", "B-SEVERITY_WORD"), ("feeling", "O"), ("any", "O"), ("pain", "B-SYMPTOM")],
            [("pain", "B-SYMPTOM"), ("is", "O"), ("completely", "B-SEVERITY_WORD"), ("gone", "I-SEVERITY_WORD")],
            [("without", "B-SEVERITY_WORD"), ("any", "O"), ("pain", "B-SYMPTOM"), ("during", "O"), ("movement", "O")],
            [("no", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("in", "O"), ("wrist", "O"), ("or", "O"), ("fingers", "O")],
            [("felt", "O"), ("comfortable", "O"), ("without", "B-SEVERITY_WORD"), ("any", "O"), ("ache", "B-SYMPTOM")],
            [("pain", "B-SYMPTOM"), ("was", "O"), ("entirely", "B-SEVERITY_WORD"), ("absent", "I-SEVERITY_WORD")],
            [("did", "O"), ("not", "B-SEVERITY_WORD"), ("feel", "O"), ("any", "O"), ("hurt", "B-SYMPTOM")],
            [("zero", "B-SEVERITY_WORD"), ("ache", "B-SYMPTOM"), ("after", "O"), ("exercises", "O")],
            [("no", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("noticed", "O")],
            [("free", "B-SEVERITY_WORD"), ("of", "O"), ("pain", "B-SYMPTOM"), ("all", "O"), ("day", "O")],
        ],
        1: [
            [("Mild", "B-SEVERITY_WORD"), ("ache", "B-SYMPTOM"), ("after", "O"), ("typing", "O")],
            [("slight", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("in", "O"), ("wrist", "O")],
            [("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("of", "O"), ("soreness", "B-SYMPTOM"), ("in", "O"), ("my", "O"), ("knuckles", "O")],
            [("minor", "B-SEVERITY_WORD"), ("discomfort", "B-SYMPTOM"), ("in", "O"), ("fingers", "O")],
            [("faint", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("when", "O"), ("moving", "O")],
            [("mild", "B-SEVERITY_WORD"), ("soreness", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("evening", "O")],
            [("low", "B-SEVERITY_WORD"), ("level", "I-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("that", "O"), ("was", "O"), ("manageable", "B-SEVERITY_WORD")],
            [("tiny", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("of", "O"), ("throbbing", "B-SYMPTOM")],
            [("slight", "B-SEVERITY_WORD"), ("ache", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("palm", "O")],
            [("manageable", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("after", "O"), ("rehab", "O")],
            [("barely", "B-SEVERITY_WORD"), ("noticeable", "I-SEVERITY_WORD"), ("ache", "B-SYMPTOM")],
            [("dull", "B-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("ache", "B-SYMPTOM")],
        ],
        2: [
            [("Moderate", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("when", "O"), ("bending", "O")],
            [("noticeable", "B-SEVERITY_WORD"), ("ache", "B-SYMPTOM"), ("throughout", "O"), ("the", "O"), ("day", "O")],
            [("quite", "B-SEVERITY_WORD"), ("sore", "B-SYMPTOM"), ("after", "O"), ("doing", "O"), ("exercises", "O")],
            [("fairly", "B-SEVERITY_WORD"), ("bad", "I-SEVERITY_WORD"), ("throbbing", "B-SYMPTOM"), ("in", "O"), ("joints", "O")],
            [("moderate", "B-SEVERITY_WORD"), ("soreness", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("evening", "O")],
            [("persistent", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("that", "O"), ("bothered", "O"), ("me", "O")],
            [("considerable", "B-SEVERITY_WORD"), ("discomfort", "B-SYMPTOM"), ("in", "O"), ("fingers", "O")],
            [("throbbing", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("that", "O"), ("slowed", "O"), ("me", "O"), ("down", "O")],
            [("aching", "B-SEVERITY_WORD"), ("joints", "O"), ("with", "O"), ("moderate", "B-SEVERITY_WORD"), ("discomfort", "B-SYMPTOM")],
            [("steady", "B-SEVERITY_WORD"), ("ache", "B-SYMPTOM"), ("across", "O"), ("my", "O"), ("knuckles", "O")],
            [("pain", "B-SYMPTOM"), ("was", "O"), ("quite", "B-SEVERITY_WORD"), ("uncomfortable", "I-SEVERITY_WORD")],
            [("bothersome", "B-SEVERITY_WORD"), ("soreness", "B-SYMPTOM"), ("in", "O"), ("hand", "O")],
        ],
        3: [
            [("Sharp", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("when", "O"), ("I", "O"), ("tried", "O"), ("to", "O"), ("close", "O"), ("my", "O"), ("fist", "O")],
            [("severe", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("in", "O"), ("my", "O"), ("fingers", "O")],
            [("excruciating", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("when", "O"), ("moving", "O"), ("hand", "O")],
            [("unbearable", "B-SEVERITY_WORD"), ("throbbing", "B-SYMPTOM"), ("in", "O"), ("wrist", "O")],
            [("intense", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("that", "O"), ("stopped", "O"), ("my", "O"), ("exercises", "O")],
            [("Pain", "B-SYMPTOM"), ("is", "O"), ("back", "O"), ("and", "O"), ("worse", "B-SEVERITY_WORD"), ("than", "O"), ("last", "O"), ("week", "O")],
            [("terrible", "B-SEVERITY_WORD"), ("burning", "B-SYMPTOM"), ("pain", "I-SYMPTOM"), ("in", "O"), ("hand", "O")],
            [("sharp", "B-SEVERITY_WORD"), ("shooting", "B-SYMPTOM"), ("ache", "I-SYMPTOM"), ("in", "O"), ("joints", "O")],
            [("extreme", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("throughout", "O"), ("the", "O"), ("day", "O")],
            [("awful", "B-SEVERITY_WORD"), ("pain", "B-SYMPTOM"), ("making", "O"), ("hand", "O"), ("unusable", "O")],
            [("severe", "B-SEVERITY_WORD"), ("stabbing", "B-SYMPTOM"), ("pain", "I-SYMPTOM")],
            [("hurts", "B-SYMPTOM"), ("terribly", "B-SEVERITY_WORD"), ("at", "O"), ("every", "O"), ("touch", "O")],
        ],
    },
    "stiffness": {
        0: [
            [("No", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("joints", "O")],
            [("not", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("at", "O"), ("all", "O"), ("today", "O")],
            [("fingers", "O"), ("were", "O"), ("not", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("morning", "O")],
            [("zero", "B-SEVERITY_WORD"), ("tightness", "B-SYMPTOM"), ("in", "O"), ("hand", "O")],
            [("no", "B-SEVERITY_WORD"), ("morning", "O"), ("stiffness", "B-SYMPTOM")],
            [("joints", "O"), ("felt", "O"), ("flexible", "B-SEVERITY_WORD"), ("without", "B-SEVERITY_WORD"), ("any", "O"), ("stiffness", "B-SYMPTOM")],
            [("stiffness", "B-SYMPTOM"), ("has", "O"), ("completely", "B-SEVERITY_WORD"), ("disappeared", "I-SEVERITY_WORD")],
            [("fingers", "O"), ("felt", "O"), ("loose", "B-SEVERITY_WORD"), ("with", "O"), ("no", "B-SEVERITY_WORD"), ("rigidity", "B-SYMPTOM")],
            [("no", "B-SEVERITY_WORD"), ("joint", "O"), ("tightness", "B-SYMPTOM"), ("felt", "O")],
            [("woke", "O"), ("up", "O"), ("with", "O"), ("no", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM")],
            [("stiffness", "B-SYMPTOM"), ("was", "O"), ("zero", "B-SEVERITY_WORD")],
            [("no", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("feeling", "O"), ("after", "O"), ("resting", "O")],
        ],
        1: [
            [("wrist", "O"), ("is", "O"), ("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("tight", "B-SYMPTOM")],
            [("slight", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("upon", "O"), ("waking", "O")],
            [("a", "B-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("knuckles", "O")],
            [("minor", "B-SEVERITY_WORD"), ("joint", "O"), ("tightness", "B-SYMPTOM")],
            [("fingers", "O"), ("were", "O"), ("slightly", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("for", "O"), ("a", "O"), ("few", "O"), ("minutes", "O")],
            [("Not", "B-SEVERITY_WORD"), ("as", "O"), ("stiff", "B-SYMPTOM"), ("as", "O"), ("yesterday", "O")],
            [("mild", "B-SEVERITY_WORD"), ("morning", "O"), ("stiffness", "B-SYMPTOM"), ("that", "O"), ("eased", "O"), ("quickly", "O")],
            [("subtle", "B-SEVERITY_WORD"), ("tightness", "B-SYMPTOM"), ("in", "O"), ("fingers", "O")],
            [("somewhat", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("after", "O"), ("resting", "O")],
            [("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("of", "O"), ("stiffness", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("palm", "O")],
            [("mildly", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("joints", "O")],
            [("just", "B-SEVERITY_WORD"), ("a", "I-SEVERITY_WORD"), ("touch", "I-SEVERITY_WORD"), ("of", "O"), ("stiffness", "B-SYMPTOM")],
        ],
        2: [
            [("Moderate", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("in", "O"), ("fingers", "O")],
            [("quite", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("morning", "O")],
            [("fairly", "B-SEVERITY_WORD"), ("tight", "B-SYMPTOM"), ("joints", "O"), ("taking", "O"), ("an", "O"), ("hour", "O"), ("to", "O"), ("loosen", "O")],
            [("noticeable", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("throughout", "O"), ("the", "O"), ("day", "O")],
            [("joints", "O"), ("felt", "O"), ("rigid", "B-SYMPTOM"), ("and", "O"), ("slow", "B-SEVERITY_WORD"), ("to", "O"), ("bend", "O")],
            [("considerable", "B-SEVERITY_WORD"), ("morning", "O"), ("tightness", "B-SYMPTOM")],
            [("stiffness", "B-SYMPTOM"), ("lingered", "B-SEVERITY_WORD"), ("for", "O"), ("several", "O"), ("hours", "O")],
            [("fingers", "O"), ("were", "O"), ("fairly", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("when", "O"), ("typing", "O")],
            [("moderate", "B-SEVERITY_WORD"), ("joint", "O"), ("rigidity", "B-SYMPTOM"), ("after", "O"), ("exercises", "O")],
            [("persistent", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("limiting", "O"), ("movement", "O")],
            [("pretty", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("and", "O"), ("tight", "B-SYMPTOM")],
            [("stiff", "B-SYMPTOM"), ("joints", "O"), ("with", "O"), ("medium", "B-SEVERITY_WORD"), ("resistance", "O")],
        ],
        3: [
            [("my", "O"), ("fingers", "O"), ("were", "O"), ("very", "B-SEVERITY_WORD"), ("stiff", "B-SYMPTOM"), ("in", "O"), ("the", "O"), ("morning", "O")],
            [("severe", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("locking", "B-SEVERITY_WORD"), ("my", "O"), ("joints", "O")],
            [("fingers", "O"), ("completely", "B-SEVERITY_WORD"), ("locked", "I-SEVERITY_WORD"), ("and", "O"), ("rigid", "B-SYMPTOM")],
            [("extreme", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("unable", "B-SEVERITY_WORD"), ("to", "O"), ("bend", "O"), ("knuckles", "O")],
            [("joints", "O"), ("are", "O"), ("totally", "B-SEVERITY_WORD"), ("rigid", "B-SYMPTOM"), ("and", "O"), ("frozen", "B-SEVERITY_WORD")],
            [("terrible", "B-SEVERITY_WORD"), ("morning", "O"), ("stiffness", "B-SYMPTOM"), ("lasting", "O"), ("all", "O"), ("day", "O")],
            [("severe", "B-SEVERITY_WORD"), ("joint", "O"), ("locking", "B-SYMPTOM"), ("preventing", "B-SEVERITY_WORD"), ("flexion", "O")],
            [("awful", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("leaving", "O"), ("hand", "O"), ("like", "O"), ("a", "O"), ("board", "O")],
            [("intense", "B-SEVERITY_WORD"), ("rigidity", "B-SYMPTOM"), ("in", "O"), ("whole", "O"), ("hand", "O")],
            [("extreme", "B-SEVERITY_WORD"), ("stiffness", "B-SYMPTOM"), ("that", "O"), ("did", "O"), ("not", "B-SEVERITY_WORD"), ("loosen", "O"), ("at", "O"), ("all", "O")],
            [("knuckles", "O"), ("felt", "O"), ("completely", "B-SEVERITY_WORD"), ("fused", "B-SYMPTOM"), ("and", "O"), ("stiff", "B-SYMPTOM")],
            [("unbearable", "B-SEVERITY_WORD"), ("joint", "O"), ("stiffness", "B-SYMPTOM")],
        ],
    },
    "mobility": {
        0: [
            [("moving", "B-SYMPTOM"), ("fine", "B-SEVERITY_WORD"), ("without", "O"), ("restriction", "O")],
            [("Full", "B-SEVERITY_WORD"), ("finger", "O"), ("movement", "B-SYMPTOM"), ("for", "O"), ("the", "O"), ("first", "O"), ("time", "O")],
            [("able", "O"), ("to", "O"), ("bend", "B-SYMPTOM"), ("fingers", "O"), ("easily", "B-SEVERITY_WORD")],
            [("hand", "O"), ("moving", "B-SYMPTOM"), ("freely", "B-SEVERITY_WORD")],
            [("full", "B-SEVERITY_WORD"), ("range", "B-SYMPTOM"), ("of", "I-SYMPTOM"), ("motion", "I-SYMPTOM"), ("achieved", "O")],
            [("no", "B-SEVERITY_WORD"), ("trouble", "O"), ("flexing", "B-SYMPTOM"), ("my", "O"), ("wrist", "O")],
            [("could", "O"), ("straighten", "B-SYMPTOM"), ("all", "O"), ("fingers", "O"), ("normally", "B-SEVERITY_WORD")],
            [("flexibility", "B-SYMPTOM"), ("felt", "O"), ("completely", "B-SEVERITY_WORD"), ("restored", "I-SEVERITY_WORD")],
            [("moving", "B-SYMPTOM"), ("fingers", "O"), ("smoothly", "B-SEVERITY_WORD")],
            [("full", "B-SEVERITY_WORD"), ("motion", "B-SYMPTOM"), ("without", "B-SEVERITY_WORD"), ("any", "O"), ("restriction", "O")],
            [("normal", "B-SEVERITY_WORD"), ("movement", "B-SYMPTOM"), ("in", "O"), ("all", "O"), ("joints", "O")],
            [("fingers", "O"), ("can", "O"), ("flex", "B-SYMPTOM"), ("and", "O"), ("extend", "B-SYMPTOM"), ("fully", "B-SEVERITY_WORD")],
        ],
        1: [
            [("slight", "B-SEVERITY_WORD"), ("trouble", "O"), ("bending", "B-SYMPTOM"), ("my", "O"), ("index", "O"), ("finger", "O")],
            [("moving", "B-SYMPTOM"), ("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("slow", "I-SEVERITY_WORD")],
            [("minor", "B-SEVERITY_WORD"), ("restriction", "B-SYMPTOM"), ("when", "O"), ("making", "O"), ("a", "O"), ("fist", "O")],
            [("slight", "B-SEVERITY_WORD"), ("limitation", "B-SYMPTOM"), ("in", "O"), ("thumb", "O"), ("movement", "B-SYMPTOM")],
            [("could", "O"), ("bend", "B-SYMPTOM"), ("fingers", "O"), ("with", "O"), ("a", "B-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("hesitation", "O")],
            [("wrist", "O"), ("movement", "B-SYMPTOM"), ("was", "O"), ("slightly", "B-SEVERITY_WORD"), ("restricted", "B-SYMPTOM")],
            [("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("hard", "O"), ("to", "O"), ("fully", "O"), ("straighten", "B-SYMPTOM"), ("fingers", "O")],
            [("small", "B-SEVERITY_WORD"), ("limitation", "B-SYMPTOM"), ("in", "O"), ("finger", "O"), ("extension", "B-SYMPTOM")],
            [("moving", "B-SYMPTOM"), ("fingers", "O"), ("with", "O"), ("slight", "B-SEVERITY_WORD"), ("delay", "O")],
            [("mild", "B-SEVERITY_WORD"), ("resistance", "O"), ("when", "O"), ("bending", "B-SYMPTOM"), ("joints", "O")],
            [("just", "B-SEVERITY_WORD"), ("a", "I-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("slow", "O"), ("to", "O"), ("open", "B-SYMPTOM"), ("hand", "O")],
            [("mildly", "B-SEVERITY_WORD"), ("reduced", "B-SYMPTOM"), ("flexibility", "B-SYMPTOM")],
        ],
        2: [
            [("Moderate", "B-SEVERITY_WORD"), ("movement", "B-SYMPTOM"), ("restriction", "I-SYMPTOM"), ("in", "O"), ("hand", "O")],
            [("struggling", "B-SEVERITY_WORD"), ("to", "O"), ("close", "B-SYMPTOM"), ("my", "O"), ("fist", "O")],
            [("fingers", "O"), ("only", "B-SEVERITY_WORD"), ("bend", "B-SYMPTOM"), ("halfway", "B-SEVERITY_WORD")],
            [("noticeable", "B-SEVERITY_WORD"), ("difficulty", "O"), ("extending", "B-SYMPTOM"), ("wrist", "O")],
            [("hard", "B-SEVERITY_WORD"), ("to", "O"), ("make", "O"), ("a", "O"), ("full", "O"), ("fist", "B-SYMPTOM")],
            [("range", "B-SYMPTOM"), ("of", "I-SYMPTOM"), ("motion", "I-SYMPTOM"), ("was", "O"), ("noticeably", "B-SEVERITY_WORD"), ("limited", "B-SYMPTOM")],
            [("fairly", "B-SEVERITY_WORD"), ("restricted", "B-SYMPTOM"), ("thumb", "O"), ("flexion", "B-SYMPTOM")],
            [("slow", "B-SEVERITY_WORD"), ("and", "O"), ("restricted", "B-SEVERITY_WORD"), ("hand", "O"), ("movements", "B-SYMPTOM")],
            [("could", "O"), ("not", "B-SEVERITY_WORD"), ("fully", "O"), ("open", "B-SYMPTOM"), ("hand", "O")],
            [("moderate", "B-SEVERITY_WORD"), ("limitation", "B-SYMPTOM"), ("when", "O"), ("flexing", "B-SYMPTOM"), ("fingers", "O")],
            [("motion", "B-SYMPTOM"), ("is", "O"), ("clearly", "B-SEVERITY_WORD"), ("impaired", "B-SYMPTOM")],
            [("cannot", "B-SEVERITY_WORD"), ("reach", "B-SYMPTOM"), ("full", "O"), ("extension", "B-SYMPTOM")],
        ],
        3: [
            [("thumb", "O"), ("barely", "B-SEVERITY_WORD"), ("moves", "B-SYMPTOM")],
            [("completely", "B-SEVERITY_WORD"), ("unable", "I-SEVERITY_WORD"), ("to", "O"), ("bend", "B-SYMPTOM"), ("my", "O"), ("fingers", "O")],
            [("cannot", "B-SEVERITY_WORD"), ("make", "O"), ("a", "O"), ("fist", "B-SYMPTOM"), ("at", "O"), ("all", "O")],
            [("severe", "B-SEVERITY_WORD"), ("mobility", "B-SYMPTOM"), ("limitation", "I-SYMPTOM"), ("in", "O"), ("hand", "O")],
            [("zero", "B-SEVERITY_WORD"), ("finger", "O"), ("movement", "B-SYMPTOM"), ("today", "O")],
            [("could", "O"), ("not", "B-SEVERITY_WORD"), ("move", "B-SYMPTOM"), ("fingers", "O"), ("even", "O"), ("slightly", "O")],
            [("hand", "O"), ("is", "O"), ("completely", "B-SEVERITY_WORD"), ("immobile", "B-SYMPTOM")],
            [("impossible", "B-SEVERITY_WORD"), ("to", "O"), ("flex", "B-SYMPTOM"), ("or", "O"), ("extend", "B-SYMPTOM"), ("fingers", "O")],
            [("total", "B-SEVERITY_WORD"), ("loss", "I-SEVERITY_WORD"), ("of", "O"), ("finger", "O"), ("movement", "B-SYMPTOM")],
            [("fingers", "O"), ("would", "O"), ("not", "B-SEVERITY_WORD"), ("budge", "B-SYMPTOM"), ("at", "O"), ("all", "O")],
            [("zero", "B-SEVERITY_WORD"), ("mobility", "B-SYMPTOM"), ("in", "O"), ("wrist", "O")],
            [("cannot", "B-SEVERITY_WORD"), ("move", "B-SYMPTOM"), ("hand", "O"), ("whatsoever", "B-SEVERITY_WORD")],
        ],
    },
    "grip": {
        0: [
            [("gripped", "B-SYMPTOM"), ("a", "O"), ("bottle", "O"), ("easily", "B-SEVERITY_WORD")],
            [("grip", "B-SYMPTOM"), ("strength", "I-SYMPTOM"), ("felt", "O"), ("normal", "B-SEVERITY_WORD")],
            [("able", "O"), ("to", "O"), ("hold", "B-SYMPTOM"), ("items", "O"), ("firmly", "B-SEVERITY_WORD"), ("without", "O"), ("dropping", "O")],
            [("no", "B-SEVERITY_WORD"), ("trouble", "O"), ("gripping", "B-SYMPTOM"), ("cups", "O")],
            [("firm", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("when", "O"), ("holding", "B-SYMPTOM"), ("heavy", "O"), ("objects", "O")],
            [("opened", "B-SYMPTOM"), ("jars", "O"), ("with", "O"), ("no", "B-SEVERITY_WORD"), ("difficulty", "O")],
            [("grasped", "B-SYMPTOM"), ("utensils", "O"), ("securely", "B-SEVERITY_WORD")],
            [("grip", "B-SYMPTOM"), ("was", "O"), ("strong", "B-SEVERITY_WORD"), ("and", "O"), ("reliable", "B-SEVERITY_WORD")],
            [("held", "B-SYMPTOM"), ("phone", "O"), ("securely", "B-SEVERITY_WORD"), ("with", "O"), ("full", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM")],
            [("no", "B-SEVERITY_WORD"), ("weakness", "O"), ("when", "O"), ("pinching", "B-SYMPTOM"), ("small", "O"), ("items", "O")],
            [("full", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("strength", "I-SYMPTOM"), ("returned", "O")],
            [("able", "O"), ("to", "O"), ("grasp", "B-SYMPTOM"), ("objects", "O"), ("without", "B-SEVERITY_WORD"), ("problem", "O")],
        ],
        1: [
            [("grip", "B-SYMPTOM"), ("is", "O"), ("slightly", "B-SEVERITY_WORD"), ("weak", "B-SYMPTOM")],
            [("a", "B-SEVERITY_WORD"), ("bit", "I-SEVERITY_WORD"), ("of", "O"), ("trouble", "O"), ("opening", "B-SYMPTOM"), ("tight", "O"), ("jars", "O")],
            [("mild", "B-SEVERITY_WORD"), ("weakness", "B-SYMPTOM"), ("when", "O"), ("holding", "B-SYMPTOM"), ("heavy", "O"), ("mugs", "O")],
            [("pinching", "B-SYMPTOM"), ("small", "O"), ("coins", "O"), ("felt", "O"), ("a", "B-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("awkward", "O")],
            [("slight", "B-SEVERITY_WORD"), ("reduction", "B-SYMPTOM"), ("in", "O"), ("grip", "B-SYMPTOM"), ("strength", "I-SYMPTOM")],
            [("held", "B-SYMPTOM"), ("objects", "O"), ("but", "O"), ("grip", "B-SYMPTOM"), ("felt", "O"), ("slightly", "B-SEVERITY_WORD"), ("insecure", "B-SYMPTOM")],
            [("minor", "B-SEVERITY_WORD"), ("difficulty", "O"), ("holding", "B-SYMPTOM"), ("a", "O"), ("pen", "O"), ("firmly", "O")],
            [("grip", "B-SYMPTOM"), ("was", "O"), ("a", "B-SEVERITY_WORD"), ("little", "I-SEVERITY_WORD"), ("shaky", "B-SYMPTOM"), ("after", "O"), ("exercises", "O")],
            [("trouble", "O"), ("with", "O"), ("tight", "O"), ("bottle", "O"), ("caps", "O"), ("due", "O"), ("to", "O"), ("mild", "B-SEVERITY_WORD"), ("weakness", "B-SYMPTOM")],
            [("slight", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("fatigue", "B-SYMPTOM"), ("by", "O"), ("evening", "O")],
            [("grip", "B-SYMPTOM"), ("felt", "O"), ("a", "B-SEVERITY_WORD"), ("tad", "I-SEVERITY_WORD"), ("feeble", "B-SYMPTOM")],
            [("little", "B-SEVERITY_WORD"), ("loss", "O"), ("in", "O"), ("grasp", "B-SYMPTOM"), ("firmness", "O")],
        ],
        2: [
            [("grip", "B-SYMPTOM"), ("is", "O"), ("weak", "B-SEVERITY_WORD"), ("when", "O"), ("opening", "B-SYMPTOM"), ("jars", "O")],
            [("dropped", "B-SEVERITY_WORD"), ("a", "O"), ("cup", "O"), ("due", "O"), ("to", "O"), ("weak", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM")],
            [("moderate", "B-SEVERITY_WORD"), ("difficulty", "O"), ("holding", "B-SYMPTOM"), ("everyday", "O"), ("objects", "O")],
            [("struggling", "B-SEVERITY_WORD"), ("to", "O"), ("hold", "B-SYMPTOM"), ("a", "O"), ("bottle", "O"), ("securely", "O")],
            [("grip", "B-SYMPTOM"), ("was", "O"), ("noticeably", "B-SEVERITY_WORD"), ("reduced", "B-SYMPTOM")],
            [("cannot", "B-SEVERITY_WORD"), ("pick", "B-SYMPTOM"), ("up", "I-SYMPTOM"), ("small", "O"), ("coins", "O"), ("easily", "O")],
            [("hard", "B-SEVERITY_WORD"), ("to", "O"), ("hold", "B-SYMPTOM"), ("a", "O"), ("fork", "O"), ("or", "O"), ("spoon", "O")],
            [("weak", "B-SEVERITY_WORD"), ("grasp", "B-SYMPTOM"), ("making", "O"), ("carrying", "B-SYMPTOM"), ("items", "O"), ("difficult", "B-SEVERITY_WORD")],
            [("struggling", "B-SEVERITY_WORD"), ("to", "O"), ("turn", "B-SYMPTOM"), ("doorknobs", "O")],
            [("moderate", "B-SEVERITY_WORD"), ("weakness", "B-SYMPTOM"), ("in", "O"), ("pinching", "B-SYMPTOM"), ("items", "O")],
            [("unsteady", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("causing", "O"), ("slips", "O")],
            [("difficulty", "O"), ("grasping", "B-SYMPTOM"), ("handles", "O"), ("firmly", "O")],
        ],
        3: [
            [("could", "B-SEVERITY_WORD"), ("not", "I-SEVERITY_WORD"), ("hold", "B-SYMPTOM"), ("a", "O"), ("cup", "O")],
            [("still", "O"), ("cannot", "B-SEVERITY_WORD"), ("pick", "B-SYMPTOM"), ("up", "I-SYMPTOM"), ("small", "O"), ("coins", "O")],
            [("completely", "B-SEVERITY_WORD"), ("unable", "I-SEVERITY_WORD"), ("to", "O"), ("grip", "B-SYMPTOM"), ("anything", "O")],
            [("zero", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("strength", "I-SYMPTOM"), ("in", "O"), ("hand", "O")],
            [("severe", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("weakness", "I-SYMPTOM"), ("dropping", "B-SEVERITY_WORD"), ("everything", "O")],
            [("cannot", "B-SEVERITY_WORD"), ("hold", "B-SYMPTOM"), ("a", "O"), ("pen", "O"), ("or", "O"), ("phone", "O"), ("at", "O"), ("all", "O")],
            [("impossible", "B-SEVERITY_WORD"), ("to", "O"), ("grasp", "B-SYMPTOM"), ("utensils", "O")],
            [("total", "B-SEVERITY_WORD"), ("inability", "I-SEVERITY_WORD"), ("to", "O"), ("hold", "B-SYMPTOM"), ("objects", "O")],
            [("dropped", "B-SEVERITY_WORD"), ("everything", "O"), ("I", "O"), ("tried", "O"), ("to", "O"), ("carry", "B-SYMPTOM")],
            [("could", "B-SEVERITY_WORD"), ("not", "I-SEVERITY_WORD"), ("hold", "B-SYMPTOM"), ("even", "O"), ("lightweight", "O"), ("items", "O")],
            [("no", "B-SEVERITY_WORD"), ("grip", "B-SYMPTOM"), ("power", "O"), ("whatsoever", "B-SEVERITY_WORD")],
            [("unable", "B-SEVERITY_WORD"), ("to", "O"), ("clench", "B-SYMPTOM"), ("or", "O"), ("hold", "B-SYMPTOM"), ("anything", "O")],
        ],
    },
}

# Connectors for joining clauses: tuple of (token, tag)
CONNECTORS: List[List[Tuple[str, str]]] = [
    [(",", "O"), ("but", "O")],
    [(",", "O"), ("however", "O")],
    [(",", "O"), ("although", "O")],
    [("and", "O")],
    [(",", "O"), ("though", "O")],
    [(".", "O"), ("Also", "O"), (",", "O")],
    [(".", "O"), ("On", "O"), ("the", "O"), ("other", "O"), ("hand", "O"), (",", "O")],
    [(",", "O"), ("while", "O")],
    [(";", "O")],
    [(".", "O")],
]

# Time / Context Prefixes
TIME_PREFIXES: List[List[Tuple[str, str]]] = [
    [("This", "O"), ("morning", "O"), (",", "O")],
    [("In", "O"), ("the", "O"), ("evening", "O"), (",", "O")],
    [("After", "O"), ("waking", "O"), ("up", "O"), (",", "O")],
    [("After", "O"), ("my", "O"), ("exercises", "O"), (",", "O")],
    [("All", "O"), ("day", "O"), ("today", "O"), (",", "O")],
    [("By", "O"), ("afternoon", "O"), (",", "O")],
    [("Late", "O"), ("tonight", "O"), (",", "O")],
    [("During", "O"), ("physio", "O"), ("practice", "O"), (",", "O")],
    [("Felt", "O"), ("different", "O"), ("today", "O"), (":", "O")],
]

# Typo / Informal word mapping
TYPO_MAP: Dict[str, str] = {
    "morning": "mornin",
    "fingers": "fingrs",
    "cannot": "cant",
    "could not": "couldnt",
    "exercises": "excercises",
    "physio": "physioo",
    "really": "reallly",
    "a lot": "alot",
    "though": "tho",
    "because": "cuz",
    "without": "w/o",
    "before": "b4",
    "today": "2day",
}


def apply_informal_typos(tokens: List[str], tags: List[str], typo_prob: float = 0.15) -> Tuple[List[str], List[str]]:
    """Probabilistically applies informal phrasing or typos while preserving BIO tags."""
    new_tokens = []
    new_tags = []
    for token, tag in zip(tokens, tags):
        lower_token = token.lower()
        if random.random() < typo_prob and lower_token in TYPO_MAP:
            rep = TYPO_MAP[lower_token]
            if token.isupper():
                rep = rep.upper()
            elif token[0].isupper():
                rep = rep.capitalize()
            new_tokens.append(rep)
            new_tags.append(tag)
        else:
            new_tokens.append(token)
            new_tags.append(tag)
    return new_tokens, new_tags


def detokenize(tokens: List[str]) -> str:
    """Joins tokens into natural English text with clean punctuation spacing."""
    text = " ".join(tokens)
    # Fix punctuation attachments
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    text = re.sub(r'\(\s+', r'(', text)
    text = re.sub(r'\s+\)', r')', text)
    # Ensure capitalization at start of sentences
    sentences = re.split(r'([.?!]\s+)', text)
    capitalized = "".join(s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences))
    return capitalized.strip()


def generate_single_entry(
    pain_level: int,
    stiffness_level: int,
    mobility_level: int,
    grip_level: int,
    active_symptoms: List[str],
    add_prefix: bool = True,
    add_typos: bool = True
) -> Dict[str, Any]:
    """Generates a single synthetic diary entry with token-level BIO tags aligned to active symptoms."""
    tokens: List[str] = []
    tags: List[str] = []

    # Optional temporal prefix
    if add_prefix and random.random() < 0.45:
        prefix = random.choice(TIME_PREFIXES)
        for tok, tag in prefix:
            tokens.append(tok)
            tags.append(tag)

    # Symptom levels map
    levels = {
        "pain": pain_level,
        "stiffness": stiffness_level,
        "mobility": mobility_level,
        "grip": grip_level,
    }

    # Assemble symptom clauses
    clauses = []
    for sym in active_symptoms:
        lvl = levels[sym]
        phrase_options = PHRASE_BANK[sym][lvl]
        phrase = random.choice(phrase_options)
        clauses.append(phrase)

    # Join clauses with connectors
    for idx, clause in enumerate(clauses):
        if idx > 0:
            connector = random.choice(CONNECTORS)
            for tok, tag in connector:
                tokens.append(tok)
                tags.append(tag)
        for tok, tag in clause:
            tokens.append(tok)
            tags.append(tag)

    # Add final punctuation if missing
    if tokens and tokens[-1] not in [".", "!", "?"]:
        tokens.append(".")
        tags.append("O")

    # Apply typos
    if add_typos:
        tokens, tags = apply_informal_typos(tokens, tags, typo_prob=0.10)

    # Human-readable text
    natural_text = detokenize(tokens)
    tokens_str = " ".join(tokens)
    bio_tags_str = " ".join(tags)

    return {
        "text": natural_text,
        "pain": pain_level,
        "stiffness": stiffness_level,
        "mobility": mobility_level,
        "grip": grip_level,
        "source": "synthetic_generator",
        "bio_tags": bio_tags_str,
        "tokens": tokens_str,
    }


def generate_synthetic_dataset(n_samples: int = 2500, seed: int = SEED) -> pd.DataFrame:
    """
    Generates n_samples synthetic diary entries with balanced class distributions (0-3)
    across all 4 symptom dimensions, ensuring every class has >= 10% per symptom.
    """
    random.seed(seed)
    np.random.seed(seed)

    all_symptoms = ["pain", "stiffness", "mobility", "grip"]
    rows = []

    # Target: ensure uniform representation for each class (0, 1, 2, 3) across all symptoms.
    # We partition into varied entry styles:
    # 1. 4-symptom comprehensive entries (mixed & balanced): 35%
    # 2. 2-symptom entries (paired symptoms): 35%
    # 3. 3-symptom entries: 15%
    # 4. Single-symptom entries: 10%
    # 5. Full recovery entries (all 0s): 5%

    for i in range(n_samples):
        r = random.random()
        if r < 0.05:
            # Full recovery (all 0)
            active = all_symptoms.copy()
            random.shuffle(active)
            p, s, m, g = 0, 0, 0, 0
        elif r < 0.40:
            # 4 symptoms mentioned, each severity sampled uniformly 0..3
            active = all_symptoms.copy()
            random.shuffle(active)
            p = random.randint(0, 3)
            s = random.randint(0, 3)
            m = random.randint(0, 3)
            g = random.randint(0, 3)
        elif r < 0.75:
            # 2 symptoms mentioned, active symptoms sampled 1..3 to balance against unmentioned 0s
            active = random.sample(all_symptoms, 2)
            # Default unmentioned to 0
            p = random.choice([1, 2, 3]) if "pain" in active else 0
            s = random.choice([1, 2, 3]) if "stiffness" in active else 0
            m = random.choice([1, 2, 3]) if "mobility" in active else 0
            g = random.choice([1, 2, 3]) if "grip" in active else 0
        elif r < 0.90:
            # 3 symptoms mentioned
            active = random.sample(all_symptoms, 3)
            p = random.randint(0, 3) if "pain" in active else 0
            s = random.randint(0, 3) if "stiffness" in active else 0
            m = random.randint(0, 3) if "mobility" in active else 0
            g = random.randint(0, 3) if "grip" in active else 0
        else:
            # 1 symptom mentioned
            active = random.sample(all_symptoms, 1)
            p = random.choice([1, 2, 3]) if "pain" in active else 0
            s = random.choice([1, 2, 3]) if "stiffness" in active else 0
            m = random.choice([1, 2, 3]) if "mobility" in active else 0
            g = random.choice([1, 2, 3]) if "grip" in active else 0

        entry = generate_single_entry(p, s, m, g, active_symptoms=active)
        rows.append(entry)

    df = pd.DataFrame(rows)
    return df


def validate_label_distribution(df: pd.DataFrame, min_class_pct: float = 10.0) -> Dict[str, Dict[int, float]]:
    """
    Validates that every class (0-3) has >= min_class_pct (default 10.0%) for every symptom.
    Returns distribution dictionary.
    """
    symptoms = ["pain", "stiffness", "mobility", "grip"]
    dist_report: Dict[str, Dict[int, float]] = {}

    for sym in symptoms:
        counts = df[sym].value_counts().to_dict()
        total = len(df)
        dist_report[sym] = {}
        for c in range(4):
            cnt = counts.get(c, 0)
            pct = (cnt / total) * 100.0
            dist_report[sym][c] = round(pct, 2)
            if pct < min_class_pct:
                raise ValueError(
                    f"Acceptance criteria FAILED: symptom '{sym}' class {c} has {pct:.2f}% (< {min_class_pct}%)."
                )

    return dist_report


ADDITIONAL_HAND_WRITTEN: List[Dict[str, Any]] = [
    {"text": "Woke up with zero ache, though my knuckles feel like rusty hinges and will not bend.", "pain": 0, "stiffness": 3, "mobility": 2, "grip": 0, "source": "agent_written"},
    {"text": "Throbbing wrist after yard work, had to drop the watering can because grip gave out.", "pain": 2, "stiffness": 0, "mobility": 0, "grip": 3, "source": "agent_written"},
    {"text": "Completely pain free all day! Managed to button my work shirt without any help.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "A bit of an ache near the surgical scar, but fingers are flexing quite smoothly.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Excruciating flare up when turning the car ignition key, whole hand locked up.", "pain": 3, "stiffness": 3, "mobility": 3, "grip": 3, "source": "agent_written"},
    {"text": "No stiffness at all today, but sharp pain shoots up my forearm whenever I pinch a pen.", "pain": 3, "stiffness": 0, "mobility": 0, "grip": 2, "source": "agent_written"},
    {"text": "Slight tightness in my palm after squeezing the stress ball, but strength is improving.", "pain": 0, "stiffness": 1, "mobility": 0, "grip": 1, "source": "agent_written"},
    {"text": "Could barely hold my toothbrush this morning, hand feels completely dead.", "pain": 0, "stiffness": 1, "mobility": 3, "grip": 3, "source": "agent_written"},
    {"text": "No pain whatsoever, yet my pinky finger stays stubbornly crooked and will not straighten.", "pain": 0, "stiffness": 1, "mobility": 2, "grip": 0, "source": "agent_written"},
    {"text": "Pretty bad throbbing after physio session, took an ice pack to settle the swelling.", "pain": 2, "stiffness": 1, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Fingers moved like clockwork today! Sliced bread and carried grocery bags with ease.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Terrible burning pain woke me at 3am, could not move my thumb an inch.", "pain": 3, "stiffness": 0, "mobility": 3, "grip": 0, "source": "agent_written"},
    {"text": "Wrist feels a little stiff after cold weather, but grip is strong enough to open soda cans.", "pain": 0, "stiffness": 1, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Pain is gone, but I dropped a coffee mug because my index finger suddenly slipped.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 2, "source": "agent_written"},
    {"text": "A dull ache lingers in my palm, though I made a full fist for the first time.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Joints are stiff as wood this morning, taking over an hour of warm water soaks to loosen.", "pain": 0, "stiffness": 3, "mobility": 2, "grip": 0, "source": "agent_written"},
    {"text": "Felt great today, no aches or tightness, played cards with family without struggle.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Severe shooting pain when spreading my fingers apart, could not type at all today.", "pain": 3, "stiffness": 0, "mobility": 2, "grip": 2, "source": "agent_written"},
    {"text": "Not hurting today, but grip feels strangely weak when lifting a heavy skillet.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 2, "source": "agent_written"},
    {"text": "Moderate stiffness throughout the afternoon; grasping doorknobs feels slightly awkward.", "pain": 0, "stiffness": 2, "mobility": 0, "grip": 1, "source": "agent_written"},
    {"text": "Horrible pain radiating to my elbow, cannot clench fingers even halfway.", "pain": 3, "stiffness": 1, "mobility": 3, "grip": 2, "source": "agent_written"},
    {"text": "Knuckles were a tiny bit puffy and stiff upon waking, but loosened up after coffee.", "pain": 0, "stiffness": 1, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Pain free day! Cut up vegetables for dinner and had no trouble holding the knife.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Extreme soreness after yesterday stretching exercises, hand is exhausted.", "pain": 3, "stiffness": 0, "mobility": 1, "grip": 1, "source": "agent_written"},
    {"text": "Grip failed me completely while trying to turn the key in the front door.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 3, "source": "agent_written"},
    {"text": "Mild discomfort while writing notes, but joint flexibility was surprisingly good.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Fingers are locked straight, cannot curl them enough to hold a fork.", "pain": 0, "stiffness": 3, "mobility": 3, "grip": 3, "source": "agent_written"},
    {"text": "Slight ache in the thumb joint, but managed to tie my shoelaces independently.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "No pain and no swelling today, fingers are bending nicely during exercises.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Intense throbbing after carrying a light backpack, hand felt weak and shaky.", "pain": 3, "stiffness": 0, "mobility": 0, "grip": 2, "source": "agent_written"},
    {"text": "Stiffness is lingering longer than usual, struggled to pick up paperclips from the desk.", "pain": 0, "stiffness": 2, "mobility": 1, "grip": 2, "source": "agent_written"},
    {"text": "Zero soreness, but thumb movement remains restricted to half of its normal range.", "pain": 0, "stiffness": 0, "mobility": 2, "grip": 0, "source": "agent_written"},
    {"text": "Sharp pinch of pain when grabbing a heavy plate, dropped it right back onto the counter.", "pain": 3, "stiffness": 0, "mobility": 0, "grip": 3, "source": "agent_written"},
    {"text": "Morning stiffness was mild and cleared in ten minutes; gripped steering wheel firmly.", "pain": 0, "stiffness": 1, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Unbearable pain in wrist joint, unable to support any weight or bend hand.", "pain": 3, "stiffness": 1, "mobility": 3, "grip": 3, "source": "agent_written"},
    {"text": "Hand feels somewhat sluggish and tight, but not painful.", "pain": 0, "stiffness": 2, "mobility": 1, "grip": 0, "source": "agent_written"},
    {"text": "Did all prescribed finger extension stretches without any ache or resistance.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Weak grasp made it tricky to pull my socks up, though joints are not sore.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 2, "source": "agent_written"},
    {"text": "Constant dull ache in knuckles, typing on laptop is getting frustrating.", "pain": 2, "stiffness": 1, "mobility": 1, "grip": 1, "source": "agent_written"},
    {"text": "Pain subsided completely by afternoon; range of motion feels almost back to normal.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Hand feels stiff and uncooperative, could not write legibly with a ballpoint pen.", "pain": 0, "stiffness": 2, "mobility": 2, "grip": 2, "source": "agent_written"},
    {"text": "Sharp ache when flexing wrist backwards, though fingers grip things without issue.", "pain": 2, "stiffness": 0, "mobility": 1, "grip": 0, "source": "agent_written"},
    {"text": "Fingers felt completely normal this morning, no morning stiffness and opened every jar easily.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Minor twitching pain in palm, but range of motion was totally unaffected.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Wrist is locked solid, cannot twist my arm to turn on the faucet.", "pain": 0, "stiffness": 3, "mobility": 3, "grip": 2, "source": "agent_written"},
    {"text": "Excruciating pain when bumping my knuckles on the table, felt sick to my stomach.", "pain": 3, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Grip strength is definitely returning; held a heavy frying pan without shaking.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "No pain, but fingers feel swollen like sausages and refuse to make a fist.", "pain": 0, "stiffness": 2, "mobility": 3, "grip": 1, "source": "agent_written"},
    {"text": "Slight tenderness over the incision line, but moving my wrist freely in circles.", "pain": 1, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Dropped my house keys twice today because my grip slipped with no warning.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 3, "source": "agent_written"},
    {"text": "Moderate ache across the back of the hand after 30 minutes of computer work.", "pain": 2, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Stiffness cleared up quickly after a warm shower, able to grip shampoo bottles fine.", "pain": 0, "stiffness": 1, "mobility": 0, "grip": 0, "source": "agent_written"},
    {"text": "Severe burning sensation down the index finger, completely unable to pinch small beads.", "pain": 3, "stiffness": 0, "mobility": 1, "grip": 3, "source": "agent_written"},
    {"text": "Full recovery today! Zero pain, zero stiffness, held weights in physio without trouble.", "pain": 0, "stiffness": 0, "mobility": 0, "grip": 0, "source": "agent_written"},
]


def generate_hand_written_test_set(
    seed_jsonl_path: str = "data/raw/seed_examples.jsonl",
    output_csv_path: str = "data/raw/hand_written_test.csv"
) -> pd.DataFrame:
    """
    Loads hand-written seed examples from seed_examples.jsonl and merges with
    the curated 42 hand-written entries to form a comprehensive 50-entry test set.
    Assigns token-level BIO tags using rule-based alignment and saves to hand_written_test.csv.
    Leaves seed_examples.jsonl completely untouched.
    """
    if not os.path.exists(seed_jsonl_path):
        raise FileNotFoundError(f"Seed file not found: {seed_jsonl_path}")

    records = []
    with open(seed_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Append additional 42 hand-written examples
    records.extend(ADDITIONAL_HAND_WRITTEN)

    # Tokenizer regex
    tok_regex = re.compile(r"\w+|[^\w\s]")

    # Vocabulary for BIO tagging hand-written examples
    severity_vocab = {
        "no", "not", "zero", "none", "without", "free", "mild", "slight", "bit",
        "moderate", "quite", "fairly", "sharp", "worse", "severe", "extreme",
        "cannot", "could", "barely", "full", "much", "better", "easily", "weak",
        "horrible", "unbearable", "excruciating", "terrible", "bad", "dead",
        "exhausted", "hard", "difficult", "subsided", "loosened"
    }
    symptom_vocab = {
        "pain", "ache", "aches", "soreness", "stiff", "stiffness", "tight", "tightness",
        "moves", "movement", "move", "gripped", "grip", "grasp", "hold", "holding",
        "pick", "fist", "typing", "bend", "bending", "flexing", "flex", "straighten",
        "curl", "pinch", "throbbing", "burning", "locked", "swelling"
    }

    formatted_rows = []
    for r in records:
        text = r["text"]
        raw_tokens = tok_regex.findall(text)
        tags = []
        for tok in raw_tokens:
            low = tok.lower()
            if low in severity_vocab:
                tags.append("B-SEVERITY_WORD")
            elif low in symptom_vocab:
                tags.append("B-SYMPTOM")
            else:
                tags.append("O")

        formatted_rows.append({
            "text": text,
            "pain": r["pain"],
            "stiffness": r["stiffness"],
            "mobility": r["mobility"],
            "grip": r["grip"],
            "source": r.get("source", "hand_written"),
            "bio_tags": " ".join(tags),
            "tokens": " ".join(raw_tokens),
        })

    hw_df = pd.DataFrame(formatted_rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    hw_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    return hw_df


def save_dataset_and_reports(
    df: pd.DataFrame,
    csv_path: str = "data/raw/diary_synthetic.csv",
    results_dir: str = "results"
) -> Tuple[str, str]:
    """Saves synthetic dataset and distribution reports."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save synthetic dataset
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Distribution report
    dist = validate_label_distribution(df)
    dist_df = pd.DataFrame(dist).T
    dist_df.columns = [f"Class_{c}_pct" for c in range(4)]
    dist_df["Total_Samples"] = len(df)

    dist_csv = os.path.join(results_dir, "diary_synthetic_distribution.csv")
    dist_df.to_csv(dist_csv, index_label="Symptom")

    dist_txt = os.path.join(results_dir, "diary_synthetic_distribution.txt")
    with open(dist_txt, "w", encoding="utf-8") as f:
        f.write("Synthetic Diary Dataset Label Distribution Report\n")
        f.write("=================================================\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write("Notice: Synthetic data generated for research and demo prototype only.\n")
        f.write("No clinical validity claimed. System is not a diagnostic tool.\n\n")
        f.write(dist_df.to_string())
        f.write("\n\nAcceptance Status: PASSED (Every class 0-3 >= 10% per symptom)\n")

    return dist_csv, dist_txt


def main():
    print("Generating synthetic diary dataset...")
    df = generate_synthetic_dataset(n_samples=2500, seed=SEED)
    print(f"Generated {len(df)} synthetic entries.")

    csv_path = "data/raw/diary_synthetic.csv"
    dist_csv, dist_txt = save_dataset_and_reports(df, csv_path=csv_path)
    print(f"Dataset saved to: {csv_path}")
    print(f"Distribution saved to: {dist_csv} and {dist_txt}")

    # Generate hand-written test set without touching seed_examples.jsonl
    print("Formatting hand-written test set...")
    hw_df = generate_hand_written_test_set()
    print(f"Hand-written test set saved ({len(hw_df)} rows) to data/raw/hand_written_test.csv")

    # Quick preview
    dist = validate_label_distribution(df)
    for sym, class_pcts in dist.items():
        print(f"  {sym.capitalize()}: {class_pcts}")


if __name__ == "__main__":
    main()
