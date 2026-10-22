# Progress log
- [x] 1. Phase 1: diary dataset generator + BIO tags + hand-written test set (see docs/MASTER_PLAN.md)
- [x] 2. Preprocessing module (src/nlp/preprocess.py) + tests
- [x] 3. Embeddings loader (GloVe/FastText)
- [x] 4. BiLSTM + attention classifier (src/nlp/)
- [x] 5. RNN/LSTM/GRU/BiLSTM comparison (src/compare/) -> results/diary_comparison.csv
- [x] 6. Simulated recovery curves (data/simulated) with plateau/relapse labels
- [x] 7. Trend model LSTM/GRU (src/trend/) -> results/trend_comparison.csv
- [x] 8. Browser-based hand tracking with MediaPipe Tasks Vision JS (app/frontend/vendor/)
- [x] 9. Flask API backend (app/backend/) & HTML/CSS/JS dashboard (app/frontend/)
- [x] 10. Report tables/plots and presentation outline
- [x] 11. Patient-friendly Step 3 Recovery Progress overhaul: 3 real-data charts (Pain/Stiffness, ROM, Mobility/Grip), 6 recovery summary cards, ROM educational guide, 7/14-day history filters, and separated AI estimates.
- [x] 12. Conference paper title and metadata added to documentation (README.md & docs/REPORT_NOTES.md).
- [x] 13. Final Project Report compilation: verified all table metrics, filled Tables 4.4 & 4.5, embedded System Architecture (Fig 3.1), generated Word Cloud & Bigram Association Graph from real project data, embedded all UI screenshots and empirical plots into docs/HandRehab_AI_Project_Report.docx.
- [x] 14. Integrated "Today's Session Ready" box and save actions directly into Step 1: Daily Rehab Diary with real-time 6-metric summary, day advance/update controls, Day 0 baseline handling, and automated NLP analysis on direct save.
- [x] 15. Enhanced registration condition selection with plain-English descriptions & dynamic hint card; adapted Step 2 Camera Exercise Tracker to provide condition-specific guided exercises (rehab goals, 3-step cues, and live clinical tips) tailored to each patient's selected surgery.
- [x] 16. Fixed uncaught JavaScript ReferenceError (undefined 'webcamStream' replaced with 'webcamRunning') in updateExerciseGuide that halted execution on page load and blocked Sign In / Quick Demo submissions.
- [x] 17. Generated comprehensive 14-slide conference presentation (docs/HandRehab_AI_Presentation.pptx) covering Problem & Objectives, Literature Survey, Proposed System, System Architecture, Implementation & Demo, Result & Discussion, and Conclusion & Future Work with embedded high-resolution empirical plots and UI figures.
- [x] 18. Removed "Advance & Log as Next Day" button from both Step 1: Diary and Step 2: Camera Exercise Tracker as requested, keeping streamlined "Save / Update Today (Day X)" workflow.
- [x] 19. Removed "Broken Hand Bone Recovery (Metacarpal)" option from both registration and camera condition selector dropdowns.
- [x] 20. Configured condition-specific candidate camera exercises/gestures for each of the 5 rehabilitation conditions (Hook Fist & Individual Finger Bend for Flexor Tendon; Straight Finger Extension & Tabletop Hand Position for Extensor Tendon; Wrist Flexion & Wrist Extension for Distal Radius; Thumb-to-Index & Thumb-to-Little Fingertip Touch for Carpal Tunnel; Finger Spreading/Closing & Thumb Opposition for General Mobility), interactive gesture switcher toggles, and embedded the mandatory clinical advisory notice across registration and camera tracking UI.
- [x] 21. Patient-Friendly Clarity Overhaul: replaced confusing anatomical jargon with memorable everyday analogies ("Pirate Hook", "Piano Key", "Flat High-Five", "Tabletop Roof", "Wave Down", "Lift Up", "OK Sign", "Reach to Pinky", "Starfish", "Tap 1-2-3-4"), created dynamic SVG hand posture illustration cards for all 10 candidate gestures, integrated Do's & Don'ts guidance cards, and added an on-screen live camera target alignment silhouette with real-time feedback.
- [x] 22. Fixed inverted canvas orientation ("ulta" backwards text) by setting #vision-canvas to transform: none and mirroring landmark drawing coordinates to match the webcam mirror view; implemented real-time studio Hand Focus mode featuring 14px background room blur with crisp, unblurred hand spotlighting, glowing cyan focus contours, and interactive toggle control.
- [x] 23. Real-Time Form Validation & Repetition Blocking: implemented biomechanical form monitoring (detecting shallow/incomplete curls, incomplete extension before next rep, and jerky/too-fast transitions <0.55s), added prominent crimson warning alerts on both the feedback banner and the on-screen target guide, penalized form score, and blocked invalid repetitions from incrementing; added an interactive "Test Bad Form" demo button.
- [x] 24. Sign In Issue Resolution & Zero-Friction Authentication: pre-populated demo patient credentials (demo_patient / rehab123) directly into the login form inputs, added an interactive credentials helper badge with a 1-click "Fill In" button, and hardened loadUserSession() with top-level try/catch fail-safes so authentication transitions instantly to the rehabilitation dashboard without silent promise rejections.
- [x] 25. Permanent Bulletproof Auth Architecture: resolved ES module lifecycle race condition where DOMContentLoaded fired before module execution leaving listeners unattached, converted form submissions to explicit onsubmit="return false;" preventing GET reloads, added an immediate self-contained auth controller in index.html for zero-latency login, and connected window.loadUserSession seamlessly with app.js.
- [x] 26. Fixed SyntaxError in app.js (duplicate ctx.restore() in drawLiveAlignmentGuide) that previously blocked entire frontend script parsing; fully verified end-to-end browser execution (tab navigation, 'Proceed to Step 2', condition gesture switching, and NLP symptom analysis) with headless Edge browser testing.





