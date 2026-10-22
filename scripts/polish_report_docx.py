"""
Polish and clean up all encoding anomalies, tables, captions, and Table of Contents
in HandRehab_AI_Final_Complete_Report.docx.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc_path = 'docs/HandRehab_AI_Final_Complete_Report.docx'
doc = docx.Document(doc_path)

# Helper function to clean text
def clean_text(txt):
    if not txt:
        return txt
    # Fix smart quotes and dashes
    t = txt.replace('CHENNAI \ufffd 600116', 'CHENNAI – 600116')
    t = t.replace('encoder\ufffddecoder', 'encoder–decoder')
    t = t.replace('\ufffd \ufffd', '–')
    t = t.replace(' \ufffd ', ' – ')
    t = t.replace('\ufffd', ' ')
    return t

# 1. Clean all paragraphs
for p in doc.paragraphs:
    if '\ufffd' in p.text:
        # Check specific patterns
        orig = p.text
        # Replace plus-minus in tables/text
        if re_match := re.search(r'(\d+\.\d+)\s*\ufffd\s*(\d+\.\d+)', orig):
            orig = re.sub(r'(\d+\.\d+)\s*\ufffd\s*(\d+\.\d+)', r'\1 ± \2', orig)
        orig = orig.replace('CHENNAI \ufffd 600116', 'CHENNAI – 600116')
        orig = orig.replace('encoder\ufffddecoder', 'encoder–decoder')
        orig = orig.replace(' \ufffd ', ' – ')
        orig = orig.replace('\ufffd', '"')
        p.text = orig

# 2. Clean all tables
for ti, t in enumerate(doc.tables):
    for ri, r in enumerate(t.rows):
        for ci, c in enumerate(r.cells):
            if '\ufffd' in c.text:
                txt = c.text
                # Fix plus-minus
                txt = re.sub(r'(\d+\.\d+)\s*\ufffd\s*(\d+\.\d+)', r'\1 ± \2', txt)
                txt = txt.replace('\ufffd', '')
                c.text = txt.strip()
                # Re-apply font
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(10)

# 3. Clean Table 0 (Table of Contents)
t0 = doc.tables[0]
# Fix row 12
if len(t0.rows) > 12:
    t0.cell(12, 2).text = '2.1.1'
    t0.cell(12, 3).text = 'Review of related works'

# Fix row 16 & 17
if len(t0.rows) > 16:
    t0.cell(16, 2).text = '3.1.1'
    t0.cell(16, 3).text = 'Dataset Description and Exploratory Analysis'

if len(t0.rows) > 17:
    t0.cell(17, 2).text = '3.1.2'
    t0.cell(17, 3).text = 'Negation-Preserving Preprocessing Pipeline'

# Add remaining sub-sections if rows exist or add rows
subsections = [
    ('3.1.3', 'Embedding and Recurrent Encoder Layers', '15'),
    ('3.1.4', 'Multi-Head Classification and Loss Function', '17'),
    ('3.1.5', 'Explainability via Token Occlusion', '18'),
    ('3.1.6', 'Longitudinal Recovery Trend Module', '19'),
    ('3.1.7', 'Web Application and Markerless Hand Tracking', '20')
]

# Check existing rows in Table 0
print(f"Table 0 current rows: {len(t0.rows)}")

# Re-format Table 0 fonts
for r in t0.rows:
    for c in r.cells:
        for p in c.paragraphs:
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10)

# Save polished document
doc.save('docs/HandRehab_AI_Final_Complete_Report.docx')
print("Successfully polished docs/HandRehab_AI_Final_Complete_Report.docx")

# Also attempt to save to docs/HandRehab_AI_Project_Report.docx if unlocked
try:
    doc.save('docs/HandRehab_AI_Project_Report.docx')
    print("Also successfully saved to docs/HandRehab_AI_Project_Report.docx!")
except Exception as e:
    print("HandRehab_AI_Project_Report.docx still locked by Word, keeping Final_Complete_Report.")
