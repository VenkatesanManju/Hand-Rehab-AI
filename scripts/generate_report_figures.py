"""
Generate All Report Figures and copy project output images for HandRehab AI.
"""
import os
import shutil
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
import nltk
from nltk.corpus import stopwords

os.makedirs('docs/figures', exist_ok=True)
up_dir = r'C:\Users\HP\.gemini\antigravity\brain\8c4cf6fd-646c-4a18-bb44-662a718d6661\.user_uploaded'

# 1. Copy user and project images
if os.path.exists(os.path.join(up_dir, 'media_1792676021667_529871dd.jpg')):
    shutil.copyfile(os.path.join(up_dir, 'media_1792676021667_529871dd.jpg'), 'docs/figures/figure_3_1_system_architecture.jpg')
if os.path.exists(os.path.join(up_dir, 'media_1792413528529_c307be53.png')):
    shutil.copyfile(os.path.join(up_dir, 'media_1792413528529_c307be53.png'), 'docs/figures/figure_3_4_diary_interface.png')
if os.path.exists(os.path.join(up_dir, 'media_1792413549477_ca6b3e52.png')):
    shutil.copyfile(os.path.join(up_dir, 'media_1792413549477_ca6b3e52.png'), 'docs/figures/figure_3_5_camera_tracking.png')
if os.path.exists(os.path.join(up_dir, 'media_1792413579728_5177e8e1.png')):
    shutil.copyfile(os.path.join(up_dir, 'media_1792413579728_5177e8e1.png'), 'docs/figures/figure_3_6_recovery_dashboard.png')
if os.path.exists('results/trend_forecast_plot.png'):
    shutil.copyfile('results/trend_forecast_plot.png', 'docs/figures/figure_4_3_trend_forecast_plot.png')
if os.path.exists('results/confusion_matrices/confusion_matrices_bilstm_attention.png'):
    shutil.copyfile('results/confusion_matrices/confusion_matrices_bilstm_attention.png', 'docs/figures/figure_4_4_confusion_matrices.png')

print('Images copied.')

# 2. Figure 3.2: Negation-Preserving Text Preprocessing Pipeline
fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

steps = [
    ('Raw Diary\nEntry', 1.0, '#f1f5f9', '#475569'),
    ('Contraction &\nInformal Expansion', 2.6, '#e0f2fe', '#0284c7'),
    ('Lowercasing &\nTokenisation', 4.2, '#e0f2fe', '#0284c7'),
    ('Stop-Word Removal\n(Negations Protected)', 5.8, '#fef08a', '#ca8a04'),
    ('WordNet\nLemmatisation', 7.4, '#e0f2fe', '#0284c7'),
    ('Vocabulary\nIndexing', 9.0, '#e0f2fe', '#0284c7'),
    ('Padding / Truncation\n(Max 40 Tokens)', 10.6, '#e0f2fe', '#0284c7'),
    ('Frozen GloVe\nEmbedding (100d)', 12.2, '#dcfce7', '#16a34a')
]

for title, x, bg, border in steps:
    rect = patches.FancyBboxPatch((x - 0.65, 2.2), 1.3, 1.6, boxstyle='round,pad=0.1,rounding_size=0.15',
                                 facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 3.0, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')

for i in range(len(steps) - 1):
    x1 = steps[i][1] + 0.65
    x2 = steps[i+1][1] - 0.65
    ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#64748b'))

ax.text(7.0, 4.8, 'Design Rule: Strictly Preserve Negations ("no", "not", "never", "without") & Modifiers ("bit", "very")',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#b45309',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef3c7', edgecolor='#f59e0b', lw=1.5))

ax.text(7.0, 1.2, 'Example: "not stiff, mild pain"  ==>  Tokens: ["not", "stiff", "mild", "pain"] (Negations Preserved)',
        ha='center', va='center', fontsize=9.5, style='italic', color='#334155')

plt.title('Figure 3.2: Negation-Preserving Text Preprocessing Pipeline', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('docs/figures/figure_3_2_preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 3.2 saved.')

# 3. Figure 3.3: Recurrent Encoder & Multi-Head Symptom Classification Architecture
fig, ax = plt.subplots(figsize=(13, 6.8), dpi=300)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis('off')

ax.add_patch(patches.FancyBboxPatch((0.5, 2.5), 1.6, 2.0, boxstyle='round,pad=0.1', facecolor='#f1f5f9', edgecolor='#64748b', lw=2))
ax.text(1.3, 3.5, 'Input Token\nIndices\n(Length <= 40)', ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a')

ax.add_patch(patches.FancyBboxPatch((2.7, 2.5), 1.8, 2.0, boxstyle='round,pad=0.1', facecolor='#dcfce7', edgecolor='#16a34a', lw=2))
ax.text(3.6, 3.5, 'Frozen GloVe\nEmbedding\n(100 Dimensions)', ha='center', va='center', fontsize=9, fontweight='bold', color='#14532d')

ax.add_patch(patches.FancyBboxPatch((5.1, 2.3), 2.2, 2.4, boxstyle='round,pad=0.1', facecolor='#dbeafe', edgecolor='#2563eb', lw=2))
ax.text(6.2, 3.5, 'Recurrent\nEncoder Layer\n(RNN/LSTM/GRU/\nBiLSTM+Attention)\n128 Hidden Units', ha='center', va='center', fontsize=9, fontweight='bold', color='#1e3a8a')

ax.add_patch(patches.FancyBboxPatch((7.9, 2.5), 1.7, 2.0, boxstyle='round,pad=0.1', facecolor='#fef3c7', edgecolor='#d97706', lw=2))
ax.text(8.75, 3.5, 'Pooling /\nAttention Context\n+ Dropout (0.3)', ha='center', va='center', fontsize=9, fontweight='bold', color='#78350f')

heads = [
    ('Pain Head\n(Classes 0-3)', 5.5, '#fee2e2', '#dc2626', '#7f1d1d'),
    ('Stiffness Head\n(Classes 0-3)', 4.2, '#fed7aa', '#ea580c', '#7c2d12'),
    ('Mobility Head\n(Classes 0-3)', 2.9, '#f3e8ff', '#9333ea', '#581c87'),
    ('Grip Head\n(Classes 0-3)', 1.6, '#ccfbf1', '#0d9488', '#134e4a')
]

for title, y, bg, border, tc in heads:
    ax.add_patch(patches.FancyBboxPatch((10.3, y - 0.45), 2.1, 0.9, boxstyle='round,pad=0.1', facecolor=bg, edgecolor=border, lw=2))
    ax.text(11.35, y, title, ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
    ax.annotate('', xy=(10.3, y), xytext=(9.6, 3.5), arrowprops=dict(arrowstyle='->', lw=1.8, color='#64748b'))

ax.annotate('', xy=(2.7, 3.5), xytext=(2.1, 3.5), arrowprops=dict(arrowstyle='->', lw=2, color='#64748b'))
ax.annotate('', xy=(5.1, 3.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle='->', lw=2, color='#64748b'))
ax.annotate('', xy=(7.9, 3.5), xytext=(7.3, 3.5), arrowprops=dict(arrowstyle='->', lw=2, color='#64748b'))

plt.title('Figure 3.3: Recurrent Encoder and Multi-Head Symptom Classification Architecture', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('docs/figures/figure_3_3_recurrent_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 3.3 saved.')
