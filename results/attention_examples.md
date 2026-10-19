# Attention Explainability Analysis (BiLSTM + Attention)

The additive attention mechanism dynamically scores the relevance of each token in the diary entry.
Below are sample test entries showing top-attended words behind the multi-head severity decisions:

### Example 1: "No pain today but my fingers were very stiff in the morning."
- **Ground Truth**: Pain=0, Stiffness=3, Mobility=1, Grip=0
- **Predicted**: Pain=0, Stiffness=3, Mobility=0, Grip=0
- **Top Attended Words**: `stiff` (0.904), `pain` (0.076), `no` (0.012), `very` (0.003), `morning` (0.003)

---
### Example 2: "Sharp pain when I tried to close my fist, could not hold a cup."
- **Ground Truth**: Pain=3, Stiffness=1, Mobility=2, Grip=3
- **Predicted**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Top Attended Words**: `pain` (0.830), `sharp` (0.156), `fist` (0.006), `try` (0.006), `hold` (0.001)

---
### Example 3: "Mild ache after typing, wrist is a bit tight but moving fine."
- **Ground Truth**: Pain=1, Stiffness=1, Mobility=0, Grip=0
- **Predicted**: Pain=1, Stiffness=1, Mobility=0, Grip=0
- **Top Attended Words**: `ache` (0.482), `mild` (0.451), `tight` (0.052), `bit` (0.008), `wrist` (0.005)

---
### Example 4: "Not as stiff as yesterday, but still cannot pick up small coins."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=3
- **Predicted**: Pain=0, Stiffness=0, Mobility=0, Grip=3
- **Top Attended Words**: `stiff` (0.554), `small` (0.336), `pick` (0.031), `coin` (0.025), `cannot` (0.025)

---
### Example 5: "A bit of an ache near the surgical scar, but fingers are flexing quite smoothly."
- **Ground Truth**: Pain=1, Stiffness=0, Mobility=0, Grip=0
- **Predicted**: Pain=1, Stiffness=0, Mobility=0, Grip=0
- **Top Attended Words**: `bit` (0.523), `ache` (0.444), `scar` (0.020), `surgical` (0.011), `near` (0.001)

---
### Example 6: "Slight tightness in my palm after squeezing the stress ball, but strength is improving."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=1
- **Predicted**: Pain=0, Stiffness=1, Mobility=0, Grip=1
- **Top Attended Words**: `tightness` (0.553), `stress` (0.338), `slight` (0.082), `squeeze` (0.021), `ball` (0.002)

---
### Example 7: "Wrist feels a little stiff after cold weather, but grip is strong enough to open soda cans."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=0
- **Predicted**: Pain=0, Stiffness=2, Mobility=0, Grip=2
- **Top Attended Words**: `stiff` (0.600), `soda` (0.096), `enough` (0.093), `cold` (0.082), `weather` (0.027)

---
### Example 8: "Grip failed me completely while trying to turn the key in the front door."
- **Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=3
- **Predicted**: Pain=0, Stiffness=0, Mobility=0, Grip=2
- **Top Attended Words**: `try` (0.228), `turn` (0.221), `completely` (0.179), `grip` (0.161), `fail` (0.105)

---
### Example 9: "Hand feels somewhat sluggish and tight, but not painful."
- **Ground Truth**: Pain=0, Stiffness=2, Mobility=1, Grip=0
- **Predicted**: Pain=0, Stiffness=1, Mobility=0, Grip=1
- **Top Attended Words**: `tight` (0.697), `sluggish` (0.139), `painful` (0.048), `somewhat` (0.047), `feel` (0.038)

---
### Example 10: "Hand feels stiff and uncooperative, could not write legibly with a ballpoint pen."
- **Ground Truth**: Pain=0, Stiffness=2, Mobility=2, Grip=2
- **Predicted**: Pain=0, Stiffness=1, Mobility=0, Grip=3
- **Top Attended Words**: `stiff` (0.487), `uncooperative` (0.467), `could` (0.015), `ballpoint` (0.012), `feel` (0.011)

---
