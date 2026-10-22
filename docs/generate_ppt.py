"""
Generate Professional 16:9 Presentation Slides for HandRehab-AI
Covers: Problem & Objectives, Literature Survey, Proposed System, System Architecture,
Implementation & Demo, Result & Discussion, Conclusion & Future Work.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "docs", "HandRehab_AI_Presentation.pptx")

# --- Color Palette ---
COLOR_NAVY = RGBColor(15, 23, 42)        # #0F172A - Deep Slate / Navy
COLOR_NAVY_LIGHT = RGBColor(30, 41, 59)  # #1E293B
COLOR_PRIMARY = RGBColor(2, 132, 199)    # #0284C7 - Medical Sky Blue
COLOR_PRIMARY_DARK = RGBColor(3, 105, 161)# #0369A1
COLOR_TEAL = RGBColor(13, 148, 136)      # #0D9488 - Modern Teal
COLOR_EMERALD = RGBColor(16, 185, 129)   # #10B981 - Success Green
COLOR_AMBER = RGBColor(245, 158, 11)     # #F59E0B - Warning
COLOR_ROSE = RGBColor(225, 29, 72)       # #E11D48 - Alert / Relapse
COLOR_BG_LIGHT = RGBColor(248, 250, 252) # #F8FAFC - Off-white
COLOR_CARD_BG = RGBColor(255, 255, 255)  # #FFFFFF - Pure White
COLOR_BORDER = RGBColor(226, 232, 240)   # #E2E8F0 - Light Gray Border
COLOR_TEXT_MAIN = RGBColor(15, 23, 42)   # #0F172A
COLOR_TEXT_MUTED = RGBColor(71, 85, 105) # #475569
COLOR_TEXT_LIGHT = RGBColor(148, 163, 184)# #94A3B8
COLOR_WHITE = RGBColor(255, 255, 255)


def add_slide_header(slide, section_num, section_name, title_text):
    """Adds a standard professional header with section pill and main title."""
    # Top Accent Line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.1))
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_PRIMARY
    line.line.color.rgb = COLOR_PRIMARY

    # Section Pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.35), Inches(3.2), Inches(0.38))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(224, 242, 254) # #E0F2FE
    pill.line.color.rgb = RGBColor(186, 230, 253)      # #BAE6FD
    tf = pill.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{section_num} | {section_name.upper()}"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.alignment = PP_ALIGN.CENTER

    # Main Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_TEXT_MAIN


def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_BORDER):
    """Draws a rounded card box to group information cleanly."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Navy Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_NAVY
    bg1.line.fill.background()

    # Top Cyan Line
    s1_line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    s1_line.fill.solid()
    s1_line.fill.fore_color.rgb = COLOR_PRIMARY
    s1_line.line.fill.background()

    # Badge Pill
    badge1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.9), Inches(4.5), Inches(0.42))
    badge1.fill.solid()
    badge1.fill.fore_color.rgb = RGBColor(12, 74, 110)
    badge1.line.color.rgb = COLOR_PRIMARY
    tf = badge1.text_frame
    p = tf.paragraphs[0]
    p.text = "RESEARCH PROJECT PRESENTATION • HEALTHCARE AI"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248)
    p.alignment = PP_ALIGN.CENTER

    # Project Title
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "HandRehab-AI: A Multimodal Framework Fusing Attentive Patient Diaries and Markerless Joint Kinematics"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(10)

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Longitudinal Telerehabilitation Monitoring via Multi-Task Recurrent NLP and Browser-Native Computer Vision"
    p1_sub.font.size = Pt(18)
    p1_sub.font.color.rgb = RGBColor(147, 197, 253)

    # Info Grid / Metadata Box
    meta_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.0), Inches(11.333), Inches(1.8))
    meta_card.fill.solid()
    meta_card.fill.fore_color.rgb = COLOR_NAVY_LIGHT
    meta_card.line.color.rgb = RGBColor(51, 65, 85)
    tf_meta = meta_card.text_frame
    tf_meta.word_wrap = True
    
    p = tf_meta.paragraphs[0]
    p.text = "PROJECT HIGHLIGHTS & ARCHITECTURE OVERVIEW"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248)
    p.space_after = Pt(6)

    bullets = [
        "Clinical NLP: Multi-head BiLSTM + Additive Attention extracting Pain, Stiffness, Mobility, and Grip severities with causal attribution.",
        "Computer Vision: Real-time client-side MediaPipe Tasks Vision (21 3D landmarks, PIP flexion angles, active ROM 0°–100°, rep counter).",
        "Temporal Forecasting: 14-day sliding window Multi-Task GRU predicting next-day physical recovery and detecting plateaus/flare-ups.",
        "System Architecture: Privacy-preserving offline browser runtime with zero video stream transmission to remote servers."
    ]
    for b in bullets:
        pb = tf_meta.add_paragraph()
        pb.text = "•  " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = RGBColor(226, 232, 240)
        pb.space_after = Pt(3)

    # Bottom Clinical Disclaimer Pill
    disc_pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(6.1), Inches(11.333), Inches(0.55))
    disc_pill.fill.solid()
    disc_pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    disc_pill.line.color.rgb = RGBColor(245, 158, 11)
    tf_d = disc_pill.text_frame
    pd = tf_d.paragraphs[0]
    pd.text = "⚠️ Clinical Notice: Observational telerehabilitation monitoring aid, not a diagnostic medical device. Synthetic/simulated datasets utilized. Alerts instruct patients: 'Consult your physiotherapist'."
    pd.font.size = Pt(10)
    pd.font.color.rgb = RGBColor(254, 243, 199)
    pd.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: PROBLEM & CLINICAL MOTIVATION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "01", "Problem & Objectives", "Clinical Background & Motivation: The Post-Surgical Rehab Void")

    # 3 Cards Layout
    card_w = Inches(3.65)
    gap = Inches(0.38)
    top_y = Inches(1.65)
    card_h = Inches(5.15)

    # Card 1: Critical Window
    c1 = add_card(s2, Inches(0.8), top_y, card_w, card_h)
    tf = c1.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Post-Surgical Healing Window"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(10)
    
    c1_points = [
        ("High-Risk Conditions", "Flexor & extensor tendon repairs, distal radius wrist fractures, and metacarpal fractures."),
        ("Early Mobilization Paradox", "Gentle active tendon gliding is mandatory within days 1–30 to avoid permanent scar tissue adhesions."),
        ("Catastrophic Failure Modes", "Over-straining risks acute tendon rupture; under-straining leads to permanent joint contracture and stiffness."),
        ("Daily Precision Required", "Consistent home exercises must adhere strictly to safe angular thresholds.")
    ]
    for title, desc in c1_points:
        p = tf.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(6)

    # Card 2: Monitoring Void
    c2 = add_card(s2, Inches(0.8) + card_w + gap, top_y, card_w, card_h)
    tf = c2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. The Telerehabilitation Void"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ROSE
    p.space_after = Pt(10)

    c2_points = [
        ("Infrequent Clinic Visits", "Patients typically see their physical therapist only once every 2 to 4 weeks due to clinic capacity and travel costs."),
        ("Blind Recovery Between Visits", "Clinicians have zero visibility into daily exercises performed during the 14–28 days at home."),
        ("Silent Flare-Ups & Plateaus", "Secondary inflammatory flare-ups or recovery plateaus go unnoticed until irreversible loss of mobility occurs."),
        ("Adherence Deterioration", "Over 60% of home patients gradually drop adherence without interactive, objective feedback.")
    ]
    for title, desc in c2_points:
        p = tf.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(6)

    # Card 3: Limitations of Current Tools
    c3 = add_card(s2, Inches(0.8) + (card_w + gap) * 2, top_y, card_w, card_h)
    tf = c3.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. Shortcomings of Existing Tools"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(10)

    c3_points = [
        ("Paper / Unstructured Diaries", "Subjective, prone to retrospective recall bias, irregular logging, and completely non-actionable."),
        ("Mechanical Goniometers", "Only usable by trained clinicians during in-person visits; impossible for patients to measure their own hand angles."),
        ("Sensorized Gloves", "Commercial cybergloves cost $1,000–$5,000, require frequent calibration, and cause tactile discomfort on injured tissue."),
        ("Siloed Solutions", "Current vision tools ignore subjective pain/stiffness; NLP tools ignore physical Range of Motion (ROM).")
    ]
    for title, desc in c3_points:
        p = tf.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(6)

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT & CORE OBJECTIVES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "01", "Problem & Objectives", "Problem Formulation & Four Core Research Objectives")

    # Left: Problem Statement Card (Width: 4.8)
    p_left = add_card(s3, Inches(0.8), Inches(1.65), Inches(4.8), Inches(5.15))
    tf_pl = p_left.text_frame
    tf_pl.word_wrap = True
    p = tf_pl.paragraphs[0]
    p.text = "Problem Statement"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ROSE
    p.space_after = Pt(12)

    prob_desc = (
        "Post-surgical hand telerehabilitation lacks an integrated, accessible framework that simultaneously captures "
        "subjective patient narratives and objective physical joint kinematics.\n\n"
        "Key Unresolved Challenges:\n"
        "1. Semantic Distortion in NLP: Standard clinical text processing strips negations and qualifiers "
        "(e.g., 'no pain but very stiff' misclassified as severe pain).\n\n"
        "2. High Hardware Barrier: Existing kinematic measurement demands costly wearable sensors, preventing equitable home monitoring.\n\n"
        "3. Anomaly Blindness: Tele-rehab platforms rely on static slope thresholds that collapse under daily physiological sensor noise, failing to catch plateaus and flare-ups."
    )
    p = tf_pl.add_paragraph()
    p.text = prob_desc
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.line_spacing = 1.25

    # Right: Four Objectives Cards (Width: 6.5)
    obj_x = Inches(5.95)
    obj_w = Inches(6.58)
    obj_h = Inches(1.15)
    obj_gap = Inches(0.18)

    objs = [
        ("Objective 1: Multi-Head Explainable Clinical NLP",
         "Develop a multi-task sequence architecture (BiLSTM + Attention) predicting 4 discrete symptom severities (0–3) for Pain, Stiffness, Mobility, and Grip, maintaining negations and causal token attribution.",
         COLOR_PRIMARY),
        ("Objective 2: Markerless Real-Time Optical Kinematics",
         "Implement client-side MediaPipe Tasks Vision in JavaScript/WASM to compute 21 3D joint landmarks, PIP flexion angles, active ROM (0°–100°), and rep counting without specialized hardware.",
         COLOR_TEAL),
        ("Objective 3: Longitudinal Forecasting & Flare-Up Detection",
         "Train multi-task recurrent sequence models over 14-day historical trajectories to forecast next-day physical recovery and classify trajectory state (Normal vs. Plateau vs. Acute Relapse).",
         COLOR_AMBER),
        ("Objective 4: Integrated Privacy-First Web Architecture",
         "Build an offline-ready, responsive dashboard (Flask + SQLite + Chart.js) with zero external video streaming, condition-specific exercise guidance, and ethical non-diagnostic transparency.",
         COLOR_EMERALD)
    ]

    for i, (title, desc, col) in enumerate(objs):
        card = add_card(s3, obj_x, Inches(1.65) + i * (obj_h + obj_gap), obj_w, obj_h)
        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(3)

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(10.5)
        p_desc.font.color.rgb = COLOR_TEXT_MUTED
        p_desc.line_spacing = 1.15

    # =========================================================================
    # SLIDE 4: LITERATURE SURVEY & RESEARCH GAPS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "02", "Literature Survey", "Critical Review of Prior Works & Identified Research Gaps")

    # Table of Prior Works
    rows, cols = 5, 4
    left = Inches(0.8)
    top = Inches(1.65)
    width = Inches(11.733)
    height = Inches(3.6)

    table_shape = s4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(2.9)
    table.columns[2].width = Inches(3.4)
    table.columns[3].width = Inches(3.133)

    headers = ["Research Domain", "Representative Works", "Methodology & Strengths", "Critical Limitations & Gaps"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ("Sensorized Gloves & Hardware",
         "Simone et al. (2007)\nKramer et al. (CyberGlove)\nLin et al. (2020)",
         "High physical precision; measures strain gages and bend angles directly on hand joints.",
         "Prohibitive cost ($1,000–$5,000); discomfort on swollen tissue; requires frequent calibration; non-scalable."),
        ("Clinical NLP & Sentiment Models",
         "BioBERT (Lee et al., 2020)\nClinicalBERT (Alsentzer)\nDernoncourt et al. (2017)",
         "Extracts clinical entities from hospital EHR notes; high general linguistic capability.",
         "Generic stopword lists drop vital negations ('no pain'); lack hand-specific functional severity heads (0–3)."),
        ("Markerless Computer Vision",
         "Lugaresi et al. (MediaPipe 2019)\nCao et al. (OpenPose)\nMathis et al. (DeepLabCut)",
         "Zero hardware cost; monocular RGB landmark tracking; accessible via standard webcams.",
         "Operates in a physical silo; completely blind to internal patient symptoms (pain, stiffness, grip fatigue)."),
        ("Longitudinal Trend Analytics",
         "Rule-Based Tele-Rehab (2018)\nMoving Average Models\nStandard Linear Slopes",
         "Simple heuristics for clinical reporting; easy to interpret in dashboards.",
         "Fails under daily sensor noise; cannot distinguish temporary rest from genuine recovery plateaus.")
    ]

    for i, row in enumerate(lit_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_TEXT_MAIN

    # Bottom Synthesis Callout Card
    gap_card = add_card(s4, Inches(0.8), Inches(5.5), Inches(11.733), Inches(1.3),
                        bg_color=RGBColor(240, 253, 250), border_color=RGBColor(94, 234, 212))
    tf_g = gap_card.text_frame
    tf_g.word_wrap = True
    p = tf_g.paragraphs[0]
    p.text = "🎯 Core Research Gap Addressed by HandRehab-AI:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(3)

    p_body = tf_g.add_paragraph()
    p_body.text = (
        "No prior framework unifies daily patient narrative self-reports with markerless optical joint kinematics in an offline, "
        "privacy-first web architecture. HandRehab-AI bridges this divide by fusing explainable NLP severity extraction with 3D joint "
        "biomechanics and longitudinal recurrent sequence forecasting."
    )
    p_body.font.size = Pt(10.5)
    p_body.font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 5: PROPOSED SYSTEM OVERVIEW
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "03", "Proposed System", "HandRehab-AI: Multimodal Rehabilitation Framework")

    # 3 Pillar Cards
    p_w = Inches(3.65)
    p_gap = Inches(0.38)
    p_top = Inches(1.65)
    p_h = Inches(5.15)

    # Pillar 1
    p1 = add_card(s5, Inches(0.8), p_top, p_w, p_h)
    tf = p1.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Pillar 1: Negation-Preserving Diary NLP"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(8)

    pts1 = [
        "Clinical Multi-Task Heads: Simultaneous prediction of 4 severity scores (0 to 3) for Pain, Stiffness, Mobility Limitation, and Grip Weakness.",
        "Negation & Qualifier Integrity: Strict retention of negation operators ('no', 'not', 'never', 'without') and qualifiers ('bit', 'slightly', 'barely').",
        "Frozen Semantic Embeddings: GloVe 100d word vectors frozen to evaluate true generalization under fixed semantic spaces.",
        "Explainable Causal Attribution: Leave-one-out token occlusion and attention weights highlighting causal symptom words (e.g., 'sharp', 'throbbing', 'stiff') directly in the UI."
    ]
    for pt in pts1:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(6)

    # Pillar 2
    p2 = add_card(s5, Inches(0.8) + p_w + p_gap, p_top, p_w, p_h)
    tf = p2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Pillar 2: Markerless Joint Kinematics"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(8)

    pts2 = [
        "Edge WebAssembly Execution: MediaPipe Tasks Vision running locally inside JavaScript; zero raw video streamed to remote servers.",
        "3D Anatomical Trigonometry: Real-time PIP joint angle computation (Index: 5-6-7, Middle: 9-10-11) via 3D vector dot products.",
        "Active Range of Motion (ROM): Functional flexion computed in degrees (0°–100°) dynamically tracking finger curl depth.",
        "Adaptive Exercise Guidance: Condition-specific routines tailored to 6 distinct post-surgical conditions (Flexor Tendon, Extensor Tendon, Wrist Fracture, Carpal Tunnel, Metacarpal, General)."
    ]
    for pt in pts2:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(6)

    # Pillar 3
    p3 = add_card(s5, Inches(0.8) + (p_w + p_gap) * 2, p_top, p_w, p_h)
    tf = p3.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Pillar 3: Longitudinal Forecasting"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(8)

    pts3 = [
        "14-Day Temporal Window: Sequence models process 5 daily features [Pain, Stiffness, Mobility, Grip, ROM] over a two-week historical context.",
        "Dual Multi-Task Objectives: Simultaneous 1-day ahead physical trajectory regression (MAE, RMSE) and 3-state clinical status classification.",
        "Robust Anomaly Classification: Detects Normal Progression, Recovery Plateaus, and Acute Flare-Ups while filtering daily noise spikes.",
        "Patient-Centric Transparency: Separates objective real data charts from AI forecasts; alerts strictly instruct consultation with physiotherapist."
    ]
    for pt in pts3:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 6: SYSTEM ARCHITECTURE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "04", "System Architecture", "End-to-End Multimodal System Architecture")

    # Left: Embedded System Architecture Image
    arch_img_path = os.path.join(FIGURES_DIR, "figure_3_1_system_architecture.jpg")
    if os.path.exists(arch_img_path):
        s6.shapes.add_picture(arch_img_path, Inches(0.8), Inches(1.65), Inches(6.8), Inches(5.15))
    else:
        # Fallback card
        c_fall = add_card(s6, Inches(0.8), Inches(1.65), Inches(6.8), Inches(5.15))
        tf = c_fall.text_frame
        tf.paragraphs[0].text = "[System Architecture Diagram: Ingestion -> Modeling -> Clinical Presentation]"

    # Right: Architectural Breakdown Cards
    r_x = Inches(7.8)
    r_w = Inches(4.733)
    
    tiers = [
        ("Layer 1: Data Ingestion & Preprocessing",
         "• Text Tokenization with Negation & Qualifier Preservation ('no', 'not', 'slightly').\n• Local WebAssembly Video Stream decoding 21 3D landmarks at 30+ FPS.\n• SQLite Patient Profile and Longitudinal Historical Repository.",
         COLOR_PRIMARY),
        ("Layer 2: AI Core & Deep Learning Engine",
         "• Multi-Head BiLSTM + Additive Attention NLP (Frozen GloVe 100d).\n• 3D Trigonometric Joint Angle & Active Range of Motion (ROM) Engine.\n• Multi-Task Temporal GRU (14-day window for regression + anomaly classification).",
         COLOR_TEAL),
        ("Layer 3: Clinical Presentation & Telerehabilitation",
         "• Flask REST API exposing POST /api/diary, /api/trend, and /api/user/record.\n• Responsive Dashboard with Chart.js visualization (3 dedicated patient charts).\n• Explainable causal word attribution chips and non-diagnostic alert banners.",
         COLOR_NAVY)
    ]

    t_h = Inches(1.58)
    t_gap = Inches(0.2)
    for i, (t_title, t_desc, col) in enumerate(tiers):
        card = add_card(s6, r_x, Inches(1.65) + i * (t_h + t_gap), r_w, t_h)
        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = t_title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(3)

        p_desc = tf.add_paragraph()
        p_desc.text = t_desc
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = COLOR_TEXT_MUTED
        p_desc.line_spacing = 1.15

    # =========================================================================
    # SLIDE 7: METHODOLOGY - NLP PIPELINE & EXPLAINABILITY
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "05", "Implementation & Demo", "Methodology: Negation-Preserving NLP & Causal Attribution")

    # Images: Preprocessing Pipeline & Recurrent Architecture
    img_pipe = os.path.join(FIGURES_DIR, "figure_3_2_preprocessing_pipeline.png")
    img_arch = os.path.join(FIGURES_DIR, "figure_3_3_recurrent_architecture.png")

    if os.path.exists(img_pipe):
        s7.shapes.add_picture(img_pipe, Inches(0.8), Inches(1.65), Inches(5.6), Inches(2.6))
    if os.path.exists(img_arch):
        s7.shapes.add_picture(img_arch, Inches(6.8), Inches(1.65), Inches(5.7), Inches(2.6))

    # Bottom Text Cards
    b_card1 = add_card(s7, Inches(0.8), Inches(4.45), Inches(5.6), Inches(2.45))
    tf1 = b_card1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Negation Preservation & Multi-Head Scheme"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(4)

    nlp_desc = (
        "• Tokenization & Lemmatization with Negation Guard: Words like 'no', 'not', 'never', 'without' "
        "and degree modifiers ('bit', 'slightly', 'barely') are explicitly exempt from stopword pruning.\n"
        "• Formulation: 4 symptom dimensions (Pain, Stiffness, Mobility, Grip) × 4 severity levels (0=None, 1=Mild, 2=Moderate, 3=Severe) = 16 classification heads.\n"
        "• Frozen GloVe 100d Embeddings ensure fair, fixed semantic evaluation without vocabulary drift."
    )
    p = tf1.add_paragraph()
    p.text = nlp_desc
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.line_spacing = 1.2

    b_card2 = add_card(s7, Inches(6.8), Inches(4.45), Inches(5.7), Inches(2.45))
    tf2 = b_card2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Attention & Leave-One-Out Causal Attribution"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(4)

    att_desc = (
        "• Additive Sequence Attention: Context vector c = Σ α_t h_t allows the network to weight high-salience symptom tokens.\n"
        "• Causal Attribution via Leave-One-Out Occlusion:\n"
        "   Δ_i = Σ_s || P(Y_s | X) - P(Y_s | X_\\i) ||_1\n"
        "• Quantifies exact probability shift induced by removing word i; tokens exceeding the causal threshold are rendered as interactive highlighted chips in the UI."
    )
    p = tf2.add_paragraph()
    p.text = att_desc
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.line_spacing = 1.2

    # =========================================================================
    # SLIDE 8: METHODOLOGY - COMPUTER VISION & KINEMATICS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "05", "Implementation & Demo", "Methodology: Markerless Computer Vision & 3D Kinematics")

    card_w8 = Inches(5.65)
    gap8 = Inches(0.433)

    # Left: Vision Math & Landmarks
    v_left = add_card(s8, Inches(0.8), Inches(1.65), card_w8, Inches(5.15))
    tf_v = v_left.text_frame
    tf_v.word_wrap = True
    p = tf_v.paragraphs[0]
    p.text = "1. Markerless Optical Kinematics"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(8)

    v_points = [
        ("MediaPipe Tasks Vision Edge Inference",
         "Executes natively in client browser using WebAssembly. Detects 21 3D hand landmarks at 30+ FPS without transmitting video to any cloud server."),
        ("3D PIP Joint Angle Calculation",
         "Computed via 3D vector dot product across proximal interphalangeal joints (Index: 5-6-7, Middle: 9-10-11):\n"
         "θ = arccos((BA · BC) / (||BA|| ||BC||))\n"
         "Robust against camera perspective skew and depth variations."),
        ("Active Range of Motion (ROM)",
         "Quantifies functional active flexion (0°–100°):\n"
         "ROM = clip(180.0° - θ_PIP, 0.0, 100.0)"),
        ("Hysteresis Repetition State Machine",
         "FLEX_THRESH ≤ 85.0° triggers FLEXED state;\n"
         "EXTEND_THRESH ≥ 140.0° registers +1 Rep and resets to EXTENDED state, preventing false bounces.")
    ]
    for title, desc in v_points:
        p = tf_v.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf_v.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(5)

    # Right: Multitask Temporal GRU
    v_right = add_card(s8, Inches(0.8) + card_w8 + gap8, Inches(1.65), card_w8, Inches(5.15))
    tf_r = v_right.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "2. Multi-Task Longitudinal Modeling"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(8)

    t_points = [
        ("14-Day Sliding Window",
         "Processes sequence X = [x_{t-13}, ..., x_t] containing 5 daily physical features: [Pain, Stiffness, Mobility, Grip, ROM]."),
        ("Dual Prediction Head",
         "• Head 1 (Continuous Regression): Predicts next-day physical recovery vector y_{t+1} (Loss: Mean Squared Error).\n"
         "• Head 2 (Clinical Status Classification): Predicts trajectory health state c_{t+1} ∈ {Normal Recovery, Plateau, Flare-Up} (Loss: Cross-Entropy)."),
        ("Multi-Task Loss Balancing",
         "L_total = L_regression + λ * L_classification\n"
         "Trained with early stopping on validation loss across fixed seeds (42, 43, 44)."),
        ("Noise Robustness",
         "GRU temporal recurrence acts as an adaptive low-pass filter, maintaining alert accuracy where rule-based heuristics fail under noisy real-world data.")
    ]
    for title, desc in t_points:
        p = tf_r.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf_r.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(5)

    # =========================================================================
    # SLIDE 9: IMPLEMENTATION & UI DEMO WALKTHROUGH
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "05", "Implementation & Demo", "System Implementation: Three-Step Clinical Patient Workflow")

    ui_w = Inches(3.65)
    ui_gap = Inches(0.38)
    ui_top = Inches(1.65)

    # Step 1: Diary
    img_ui1 = os.path.join(FIGURES_DIR, "figure_3_4_diary_interface.png")
    if os.path.exists(img_ui1):
        s9.shapes.add_picture(img_ui1, Inches(0.8), ui_top, ui_w, Inches(2.8))
    c_u1 = add_card(s9, Inches(0.8), Inches(4.6), ui_w, Inches(2.2))
    tf = c_u1.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Step 1: Daily Rehab Diary"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(4)
    p = tf.add_paragraph()
    p.text = "• Natural language self-report input.\n• Instant extraction of 4 symptom scores.\n• Explainable causal attention chips.\n• Today's Session Summary & Save actions."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Step 2: Camera Tracker
    img_ui2 = os.path.join(FIGURES_DIR, "figure_3_5_camera_tracking.png")
    if os.path.exists(img_ui2):
        s9.shapes.add_picture(img_ui2, Inches(0.8) + ui_w + ui_gap, ui_top, ui_w, Inches(2.8))
    c_u2 = add_card(s9, Inches(0.8) + ui_w + ui_gap, Inches(4.6), ui_w, Inches(2.2))
    tf = c_u2.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Step 2: Camera Exercise Tracker"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(4)
    p = tf.add_paragraph()
    p.text = "• Real-time MediaPipe hand landmark tracking.\n• Live joint angle & active ROM display.\n• Curl depth progress bar & rep counter.\n• Condition-guided 3-step exercise cues."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Step 3: Recovery Progress
    img_ui3 = os.path.join(FIGURES_DIR, "figure_3_6_recovery_dashboard.png")
    if os.path.exists(img_ui3):
        s9.shapes.add_picture(img_ui3, Inches(0.8) + (ui_w + ui_gap) * 2, ui_top, ui_w, Inches(2.8))
    c_u3 = add_card(s9, Inches(0.8) + (ui_w + ui_gap) * 2, Inches(4.6), ui_w, Inches(2.2))
    tf = c_u3.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Step 3: Recovery Progress"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_after = Pt(4)
    p = tf.add_paragraph()
    p.text = "• 3 real-data charts (Pain/Stiff, ROM, Grip).\n• 6 patient-friendly recovery cards.\n• Plain-English ROM educational definitions.\n• AI-estimated recovery & clinical alert banner."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 10: EXPERIMENTAL RESULTS - DIARY NLP BENCHMARKS
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "06", "Result & Discussion", "Experimental Results: Multi-Head Diary NLP Benchmarks")

    # Left: NLP Benchmark Table (Width: 6.8)
    rows, cols = 7, 5
    t_shape = s10.shapes.add_table(rows, cols, Inches(0.8), Inches(1.65), Inches(6.8), Inches(3.2))
    t = t_shape.table
    t.columns[0].width = Inches(1.9)
    t.columns[1].width = Inches(1.2)
    t.columns[2].width = Inches(1.2)
    t.columns[3].width = Inches(1.2)
    t.columns[4].width = Inches(1.3)

    nlp_headers = ["Model", "Synth Acc", "Synth F1", "Held-Out Acc", "Held-Out F1 (95% CI)"]
    for j, h in enumerate(nlp_headers):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    nlp_rows = [
        ("TF-IDF + LogReg", "0.985 ± 0.000", "0.985", "0.726 ± 0.000", "0.556 [0.47, 0.62]"),
        ("Simple RNN", "0.714 ± 0.002", "0.685", "0.586 ± 0.018", "0.455 [0.38, 0.51]"),
        ("LSTM", "0.901 ± 0.009", "0.891", "0.667 ± 0.007", "0.518 [0.44, 0.58]"),
        ("GRU", "0.927 ± 0.008", "0.919", "0.695 ± 0.008", "0.545 [0.46, 0.61]"),
        ("BiLSTM", "0.939 ± 0.008", "0.933", "0.696 ± 0.007", "0.546 [0.46, 0.61]"),
        ("BiLSTM + Attention", "0.930 ± 0.008", "0.924", "0.679 ± 0.019", "0.534 [0.45, 0.60]")
    ]

    for i, row in enumerate(nlp_rows):
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_MAIN
            if j == 0:
                p.font.bold = True
            if j > 0:
                p.alignment = PP_ALIGN.CENTER

    # Right: Macro F1 Comparison Plot
    img_f1 = os.path.join(FIGURES_DIR, "figure_4_1_macro_f1_comparison.png")
    if os.path.exists(img_f1):
        s10.shapes.add_picture(img_f1, Inches(7.8), Inches(1.65), Inches(4.733), Inches(3.2))

    # Bottom Findings Card
    bot_card = add_card(s10, Inches(0.8), Inches(5.05), Inches(11.733), Inches(1.75))
    tf_b = bot_card.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Key Findings & Analytical Discussion:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(3)

    nlp_takeaways = [
        "Synthetic-to-Real Generalization Gap: Models achieve >90% Macro F1 on synthetic test templates, but drop to 51.8%–55.6% on the held-out set (N=62) due to complex compounding syntax and colloquial expressions.",
        "Overlapping Bootstrap 95% CIs: The held-out confidence intervals for TF-IDF ([0.47, 0.62]), BiLSTM ([0.46, 0.61]), and GRU ([0.46, 0.61]) heavily overlap, demonstrating that n-gram baselines remain competitive on short clinical texts, while BiLSTM provides token-level attention explainability.",
        "Symptom Sensitivity Asymmetry: Pain achieved highest sensitivity (F1 up to 0.897 with GRU), while Mobility and Grip exhibited lower F1 (0.32–0.52) due to clinical symptom overlap in patient descriptions."
    ]
    for pt in nlp_takeaways:
        p = tf_b.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 11: EXPERIMENTAL RESULTS - LONGITUDINAL TREND BENCHMARKS
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "06", "Result & Discussion", "Experimental Results: Recovery Trajectory Forecasting & Alerts")

    # Left: Trend Table
    rows, cols = 7, 5
    t_shape11 = s11.shapes.add_table(rows, cols, Inches(0.8), Inches(1.65), Inches(6.8), Inches(3.2))
    t11 = t_shape11.table
    t11.columns[0].width = Inches(2.0)
    t11.columns[1].width = Inches(1.2)
    t11.columns[2].width = Inches(1.2)
    t11.columns[3].width = Inches(1.2)
    t11.columns[4].width = Inches(1.2)

    t_headers = ["Model", "Forecast MAE", "ROM MAE (°)", "Status Acc", "Status Macro F1"]
    for j, h in enumerate(t_headers):
        cell = t11.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    trend_rows = [
        ("Tomorrow = Today (Naive)", "0.284", "1.21°", "0.745", "0.285 (Fails Anomaly)"),
        ("7-Day Moving Avg", "0.673", "2.92°", "0.745", "0.285 (Fails Anomaly)"),
        ("Rule-Based Threshold", "0.284", "1.21°", "0.720", "0.509 (Clean Data)"),
        ("Rule-Based (Noisy)", "0.701", "2.87°", "0.473", "0.307 (Degrades Under Noise)"),
        ("LSTM (Standard)", "0.563 ± 0.016", "2.37 ± 0.09°", "0.737 ± 0.013", "0.582 ± 0.006"),
        ("GRU (Selected Best)", "0.475 ± 0.027", "1.99 ± 0.12°", "0.758 ± 0.006", "0.611 ± 0.009 (Best)")
    ]

    for i, row in enumerate(trend_rows):
        for j, val in enumerate(row):
            cell = t11.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_MAIN
            if j == 0:
                p.font.bold = True
            if j > 0:
                p.alignment = PP_ALIGN.CENTER

    # Right: Forecast Plot
    img_trend = os.path.join(FIGURES_DIR, "figure_4_3_trend_forecast_plot.png")
    if os.path.exists(img_trend):
        s11.shapes.add_picture(img_trend, Inches(7.8), Inches(1.65), Inches(4.733), Inches(3.2))

    # Bottom Findings Card
    bot_card11 = add_card(s11, Inches(0.8), Inches(5.05), Inches(11.733), Inches(1.75))
    tf_b11 = bot_card11.text_frame
    tf_b11.word_wrap = True
    p = tf_b11.paragraphs[0]
    p.text = "Forecasting vs Alerting Asymmetry & Key Takeaways:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(3)

    trend_takeaways = [
        "Forecasting Asymmetry: Naive Persistence ('Tomorrow = Today') achieves lowest MAE (0.284) because biological healing progresses smoothly day-by-day. Neural models exhibit slight regression shrinkage that inflates 1-day point MAE.",
        "Anomaly Classification Superiority: Persistence fails completely on clinical alerts (Macro F1 = 0.285, unable to detect state transitions). Rule-based slopes collapse under realistic sensor noise (F1 drops 0.509 -> 0.307).",
        "Selected Architecture (GRU): Multi-Task GRU achieves highest Status Macro F1 (0.611 on standard, 0.438 on out-of-distribution noisy patients) by capturing 14-day temporal context and filtering daily noise spikes."
    ]
    for pt in trend_takeaways:
        p = tf_b11.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 12: DISCUSSION & CLINICAL IMPLICATIONS
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "06", "Result & Discussion", "Discussion: Clinical Insights, Ethical Positioning & Trade-offs")

    # 4 Thematic Cards (2x2 Grid)
    g_w = Inches(5.65)
    g_h = Inches(2.4)
    g_x1 = Inches(0.8)
    g_x2 = Inches(6.883)
    g_y1 = Inches(1.65)
    g_y2 = Inches(4.35)

    themes = [
        ("1. Synthetic-to-Real Domain Shift",
         "The empirical drop from >90% to ~55% F1 on held-out text mirrors real clinical deployment challenges. Template generation guarantees grammatical purity, while actual patient notes contain compound sentences, misspellings, and mixed-sentiment clauses. This underscores the necessity of continuous feedback loops in post-surgical monitoring.",
         COLOR_PRIMARY, g_x1, g_y1),
        ("2. Explainability & Causal Attribution",
         "Black-box AI is unacceptable in healthcare tele-monitoring. By coupling additive sequence attention with leave-one-out probability shifts (Δ_i), HandRehab-AI renders the exact symptom tokens driving scores into interactive UI chips, allowing physical therapists to verify that clinical reasoning aligns with medical intent.",
         COLOR_TEAL, g_x2, g_y1),
        ("3. Privacy-Preserving Edge Computer Vision",
         "Transmitting private patient home webcam video over the internet introduces immense regulatory, HIPAA, and ethical hurdles. Executing MediaPipe Tasks Vision entirely client-side via WebAssembly provides zero-trust privacy: landmarks and ROM are computed locally, and only numerical vectors are saved.",
         COLOR_AMBER, g_x1, g_y2),
        ("4. Honest Limitations & Non-Diagnostic Stance",
         "HandRehab-AI is deliberately positioned as an observational pair-monitoring aid, not a diagnostic medical device. Synthetic and simulated datasets are explicitly disclosed. Warning banners systematically instruct patients: 'Consult your physiotherapist', maintaining clinical safety and ethical compliance.",
         COLOR_ROSE, g_x2, g_y2)
    ]

    for title, desc, col, x, y in themes:
        c = add_card(s12, x, y, g_w, g_h)
        tf = c.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13.5)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(6)

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(10.5)
        p_desc.font.color.rgb = COLOR_TEXT_MUTED
        p_desc.line_spacing = 1.2

    # =========================================================================
    # SLIDE 13: CONCLUSION & FUTURE WORK
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "07", "Conclusion & Future Work", "Concluding Remarks & Strategic Future Roadmap")

    c_w = Inches(5.65)
    c_h = Inches(5.15)

    # Left: Conclusion
    c_left = add_card(s13, Inches(0.8), Inches(1.65), c_w, c_h)
    tf_cl = c_left.text_frame
    tf_cl.word_wrap = True
    p = tf_cl.paragraphs[0]
    p.text = "Summary of Contributions"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_DARK
    p.space_after = Pt(8)

    conc_bullets = [
        ("Multimodal Fusion", "Demonstrated that combining subjective patient diaries with objective markerless joint kinematics bridges the tele-rehab monitoring gap."),
        ("Negation & Semantic Integrity", "Preserving negations ('no', 'not', 'without') eliminates critical clinical misclassification in patient self-reports."),
        ("Explainable Architecture", "BiLSTM + Attention with leave-one-out causal attribution makes deep learning predictions transparent and auditable for therapists."),
        ("Robust Anomaly Detection", "Multi-Task GRU successfully detects clinical plateaus and inflammatory relapses while filtering everyday sensor noise spikes."),
        ("Zero-Cost Accessible Delivery", "Fully client-side browser runtime provides accessible, offline-capable tracking without requiring expensive sensorized gloves.")
    ]
    for title, desc in conc_bullets:
        p = tf_cl.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf_cl.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(4)

    # Right: Future Work Roadmap (Permitted Scope for Presentation)
    c_right = add_card(s13, Inches(6.883), Inches(1.65), c_w, c_h)
    tf_cr = c_right.text_frame
    tf_cr.word_wrap = True
    p = tf_cr.paragraphs[0]
    p.text = "Strategic Future Roadmap"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL
    p.space_after = Pt(8)

    future_bullets = [
        ("1. Sign-Language Dexterity Gamification",
         "Transform finger curl exercises into interactive sign recognition challenges (ASL/ISL finger spelling), boosting patient motivation through motor gamification."),
        ("2. Whisper Multilingual Voice Diary",
         "Integrate hands-free automatic speech recognition (OpenAI Whisper) to enable effortless voice self-reporting for patients with severe bilateral hand trauma."),
        ("3. Pretrained Biomedical LLM Fine-Tuning",
         "Benchmark domain-specific transformers (BioBERT, ClinicalBERT, Med-PaLM) against lightweight recurrent models to evaluate inference latency vs semantic nuances."),
        ("4. Dynamic Exercise Progression Engine",
         "Develop automated adaptive recommendation algorithms that adjust rep targets and flexion goals as patient recovery progresses."),
        ("5. Prospective Clinical Cohort Validation",
         "Partner with post-surgical orthopedic clinics to benchmark HandRehab-AI against physical therapist goniometric evaluations on real recovering patients.")
    ]
    for title, desc in future_bullets:
        p = tf_cr.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p_sub = tf_cr.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED
        p_sub.space_after = Pt(4)

    # =========================================================================
    # SLIDE 14: THANK YOU & Q&A (Dark Navy Theme)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    bg14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg14.fill.solid()
    bg14.fill.fore_color.rgb = COLOR_NAVY
    bg14.line.fill.background()

    # Top Cyan Line
    s14_line = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    s14_line.fill.solid()
    s14_line.fill.fore_color.rgb = COLOR_PRIMARY
    s14_line.line.fill.background()

    # Thank you box
    tb_ty = s14.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(1.5))
    tf_ty = tb_ty.text_frame
    p = tf_ty.paragraphs[0]
    p.text = "Thank You!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.space_after = Pt(8)

    p_sub = tf_ty.add_paragraph()
    p_sub.text = "HandRehab-AI: Open, Objective & Attentive Post-Surgical Tele-Monitoring"
    p_sub.font.size = Pt(20)
    p_sub.font.color.rgb = RGBColor(147, 197, 253)

    # Deliverables & Verification Card
    d_card = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.4), Inches(11.333), Inches(2.5))
    d_card.fill.solid()
    d_card.fill.fore_color.rgb = COLOR_NAVY_LIGHT
    d_card.line.color.rgb = RGBColor(51, 65, 85)
    tf_d = d_card.text_frame
    tf_d.word_wrap = True

    p = tf_d.paragraphs[0]
    p.text = "VERIFIED PROJECT ARTIFACTS & DELIVERABLES"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248)
    p.space_after = Pt(8)

    delivs = [
        "Complete Project Documentation: docs/HandRehab_AI_Project_Report.docx (Full IEEE-formatted academic project report).",
        "Automated Test Suite: 59 passed unit tests (100% passing rate across NLP, Vision, Trend, and API endpoints).",
        "Functional Web Dashboard: Local Flask backend + JavaScript frontend running on http://127.0.0.1:5000 (offline ready).",
        "Rigorous Reproducibility: PyTorch, fixed random seeds (42, 43, 44), 1,000 bootstrap confidence intervals, and benchmark CSVs in results/."
    ]
    for d in delivs:
        p = tf_d.add_paragraph()
        p.text = "✔  " + d
        p.font.size = Pt(11.5)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.space_after = Pt(4)

    # Bottom Pill
    b_pill = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(6.2), Inches(11.333), Inches(0.55))
    b_pill.fill.solid()
    b_pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    b_pill.line.color.rgb = COLOR_PRIMARY
    tf_bp = b_pill.text_frame
    p = tf_bp.paragraphs[0]
    p.text = "Questions & Discussion • HandRehab-AI Research Group"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(224, 242, 254)
    p.alignment = PP_ALIGN.CENTER

    prs.save(OUTPUT_PPTX)
    print(f"Presentation generated successfully: {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_presentation()
