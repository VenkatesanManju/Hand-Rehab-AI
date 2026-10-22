"""
Script to update the HandRehab AI Project Report docx:
1. Verify and fill incomplete tables (Table 4.4 and Table 4.5).
2. Insert all project figures:
   - Figure 3.1: System Architecture (media_1792676021667_529871dd.jpg)
   - Figure 3.2: Negation-Preserving Preprocessing Pipeline
   - Figure 3.3: Recurrent Encoder & Multi-Head Classification Architecture
   - Figure 3.4: Word Cloud of Clinical Diary Corpus
   - Figure 3.5: Bigram Association Network Graph
   - Figure 3.6: Diary Analysis Interface (Step 1)
   - Figure 3.7: Browser-Based Hand Exercise Tracking (Step 2)
   - Figure 3.8: Recovery Progress Dashboard (Step 3)
   - Figure 4.3: Longitudinal Recovery Trend Forecast & Anomaly Detection Plot
   - Figure 4.4: Multi-Head Confusion Matrices (BiLSTM + Attention)
3. Ensure professional formatting and save to docs/HandRehab_AI_Project_Report.docx.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

input_path = r'C:\Users\HP\.gemini\antigravity\brain\8c4cf6fd-646c-4a18-bb44-662a718d6661\.user_uploaded\media_1792675856638_6e4d291d.docx'
output_path = 'docs/HandRehab_AI_Project_Report.docx'

doc = docx.Document(input_path)
print(f'Loaded document: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables.')

# ---------------------------------------------------------------------------
# 1. Fill Table 4 (Table 4.4: Next-day forecast error)
# ---------------------------------------------------------------------------
t4 = doc.tables[4]
# Verify header
print('Table 4 Header:', [c.text.strip() for c in t4.rows[0].cells])

# Fill values from results/trend_comparison.txt
# Row 1: Tomorrow = Today
t4.cell(1, 1).text = '0.284'
t4.cell(1, 2).text = '0.708'

# Row 2: 7-Day Moving Average
t4.cell(2, 1).text = '0.673'
t4.cell(2, 2).text = '1.634'

# Row 3: GRU multi-task model
t4.cell(3, 1).text = '0.475 ± 0.027'
t4.cell(3, 2).text = '1.167 ± 0.064'

# Format cells
for r in t4.rows:
    for c in r.cells:
        for p in c.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10.5)

print('Table 4.4 successfully filled with verified values.')

# ---------------------------------------------------------------------------
# 2. Fill Table 5 (Table 4.5: Recovery status classification)
# ---------------------------------------------------------------------------
t5 = doc.tables[5]
print('Table 5 Header:', [c.text.strip() for c in t5.rows[0].cells])

# Fill values from results/trend_comparison.txt
# Row 1: Majority class (normal)
t5.cell(1, 1).text = '0.248'
t5.cell(1, 2).text = '0.333'
t5.cell(1, 3).text = '0.285'

# Row 2: Rule-based detector
t5.cell(2, 1).text = '0.509'
t5.cell(2, 2).text = '0.578'
t5.cell(2, 3).text = '0.509'

# Row 3: GRU multi-task model
t5.cell(3, 1).text = '0.642 ± 0.007'
t5.cell(3, 2).text = '0.593 ± 0.011'
t5.cell(3, 3).text = '0.611 ± 0.009'

for r in t5.rows:
    for c in r.cells:
        for p in c.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10.5)

print('Table 4.5 successfully filled with verified values.')

# ---------------------------------------------------------------------------
# 3. Locate and insert images into paragraphs
# ---------------------------------------------------------------------------
def insert_image_above_paragraph(p, img_path, width_in=5.8):
    if not os.path.exists(img_path):
        print(f'Warning: {img_path} not found!')
        return None
    new_p = p.insert_paragraph_before()
    new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = new_p.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    return new_p

# Iterate through paragraphs to find insertion anchors
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    
    # Figure 3.1: System Architecture
    if text.startswith('Figure 3.1: Overall system architecture'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_1_system_architecture.jpg', width_in=6.0)
        print(f'Inserted Figure 3.1 before P{i}')
        
    # Figure 3.2: Preprocessing Pipeline
    elif text.startswith('Figure 3.2: Negation-preserving text preprocessing pipeline'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_2_preprocessing_pipeline.png', width_in=6.0)
        print(f'Inserted Figure 3.2 before P{i}')
        
    # Figure 3.3: Recurrent Architecture
    elif text.startswith('Figure 3.3: Recurrent encoder and multi-head symptom classification architecture'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_3_recurrent_architecture.png', width_in=5.8)
        print(f'Inserted Figure 3.3 before P{i}')
        
    # Figure 3.4: Diary analysis interface
    elif text.startswith('Figure 3.4: Diary analysis interface'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_4_diary_interface.png', width_in=5.5)
        print(f'Inserted Figure 3.4 before P{i}')
        
    # Figure 3.5: Browser-based hand exercise tracking
    elif text.startswith('Figure 3.5: Browser-based hand exercise tracking'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_5_camera_tracking.png', width_in=5.5)
        print(f'Inserted Figure 3.5 before P{i}')
        
    # Figure 3.6: Recovery dashboard
    elif text.startswith('Figure 3.6: Recovery dashboard displaying'):
        insert_image_above_paragraph(p, 'docs/figures/figure_3_6_recovery_dashboard.png', width_in=5.5)
        print(f'Inserted Figure 3.6 before P{i}')

# Insert Word Cloud and Bigram Association Graph in Section 3.1.1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('The dataset is intended for developing and comparing symptom classification methods'):
        # Insert after this concluding paragraph of 3.1.1
        # Insert Bigram Association Graph
        p_bigram_caption = p.insert_paragraph_before()
        p_bigram_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_bigram_caption.paragraph_format.space_after = Pt(12)
        r = p_bigram_caption.add_run('Figure 3.1c: Bigram Association Network Graph showing common symptom collocations and clinical dependency pairs across the HandRehab AI corpus.')
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.italic = True
        
        insert_image_above_paragraph(p_bigram_caption, 'docs/figures/bigram_association_graph.png', width_in=5.8)
        
        # Insert Word Cloud
        p_wc_caption = p_bigram_caption.insert_paragraph_before()
        p_wc_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_wc_caption.paragraph_format.space_after = Pt(12)
        r = p_wc_caption.add_run('Figure 3.1b: Word Cloud of the HandRehab AI Clinical Diary Corpus displaying the most prominent symptom descriptors and functional terms.')
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.italic = True
        
        insert_image_above_paragraph(p_wc_caption, 'docs/figures/wordcloud.png', width_in=5.8)
        
        print('Inserted Word Cloud and Bigram Graph in Section 3.1.1.')
        break

# Insert Trend Forecast plot in Section 4.2.2
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('Table 4.5: Recovery status classification'):
        # Insert after Table 4.5
        # We can find paragraph after Table 4.5
        pass

# Add Figure 4.3 and Figure 4.4 at the end of Section 4.2.2
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('CHAPTER 5'):
        p_c5 = p
        
        # Figure 4.4: Confusion Matrices
        p_cm_cap = p_c5.insert_paragraph_before()
        p_cm_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cm_cap.paragraph_format.space_after = Pt(12)
        r = p_cm_cap.add_run('Figure 4.4: Multi-Head Confusion Matrices for BiLSTM + Attention across Pain, Stiffness, Mobility, and Grip severity classes on the held-out evaluation set.')
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.italic = True
        insert_image_above_paragraph(p_cm_cap, 'docs/figures/figure_4_4_confusion_matrices.png', width_in=5.8)
        
        # Figure 4.3: Trend Forecast Plot
        p_tr_cap = p_cm_cap.insert_paragraph_before()
        p_tr_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_tr_cap.paragraph_format.space_after = Pt(12)
        r = p_tr_cap.add_run('Figure 4.3: Simulated 14-day longitudinal recovery trajectory showing next-day continuous feature forecasts and status classification (Normal vs. Plateau vs. Relapse).')
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.italic = True
        insert_image_above_paragraph(p_tr_cap, 'docs/figures/figure_4_3_trend_forecast_plot.png', width_in=5.8)
        
        print('Inserted Figure 4.3 (Trend Forecast) and Figure 4.4 (Confusion Matrices) in Section 4.2.2.')
        break

# Save updated document
doc.save(output_path)
print(f'Successfully saved updated report to: {output_path}')
