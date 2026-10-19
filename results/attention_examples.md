# Attention Explainability Analysis (BiLSTM + Attention)

The additive attention mechanism dynamically scores the relevance of each token in the diary entry.
Below are sample test entries showing top-attended words behind the multi-head severity decisions:

### Example 1: "No pain today but my fingers were very stiff in the morning."
- **Ground Truth**: Pain=0, Stiffness=3, Mobility=1, Grip=0
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=0
- **Top Attended Words**: `stiff` (0.437), `in` (0.126), `No` (0.095), `my` (0.066), `fingers` (0.064)

---
### Example 2: "Sharp pain when I tried to close my fist, could not hold a cup."
- **Ground Truth**: Pain=3, Stiffness=1, Mobility=2, Grip=3
- **Predicted**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Top Attended Words**: `Sharp` (0.192), `pain` (0.164), `when` (0.147), `I` (0.143), `tried` (0.112)

---
### Example 3: "Mild ache after typing, wrist is a bit tight but moving fine."
- **Ground Truth**: Pain=1, Stiffness=1, Mobility=0, Grip=0
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=0
- **Top Attended Words**: `ache` (0.200), `typing,` (0.155), `after` (0.150), `tight` (0.123), `Mild` (0.122)

---
### Example 4: "Not as stiff as yesterday, but still cannot pick up small coins."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=3
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Top Attended Words**: `stiff` (0.600), `as` (0.144), `Not` (0.098), `as` (0.094), `yesterday,` (0.023)

---
### Example 5: "A bit of an ache near the surgical scar, but fingers are flexing quite smoothly."
- **Ground Truth**: Pain=1, Stiffness=0, Mobility=0, Grip=0
- **Predicted**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Top Attended Words**: `bit` (0.256), `of` (0.144), `ache` (0.130), `an` (0.101), `surgical` (0.061)

---
### Example 6: "Slight tightness in my palm after squeezing the stress ball, but strength is improving."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=1
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Top Attended Words**: `tightness` (0.286), `after` (0.112), `squeezing` (0.100), `in` (0.092), `stress` (0.089)

---
### Example 7: "Wrist feels a little stiff after cold weather, but grip is strong enough to open soda cans."
- **Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=0
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Top Attended Words**: `stiff` (0.432), `after` (0.119), `little` (0.112), `Wrist` (0.067), `feels` (0.059)

---
### Example 8: "Grip failed me completely while trying to turn the key in the front door."
- **Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=3
- **Predicted**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Top Attended Words**: `trying` (0.147), `while` (0.146), `failed` (0.144), `me` (0.139), `Grip` (0.115)

---
### Example 9: "Hand feels somewhat sluggish and tight, but not painful."
- **Ground Truth**: Pain=0, Stiffness=2, Mobility=1, Grip=0
- **Predicted**: Pain=3, Stiffness=0, Mobility=0, Grip=1
- **Top Attended Words**: `sluggish` (0.475), `somewhat` (0.280), `and` (0.110), `Hand` (0.065), `feels` (0.055)

---
### Example 10: "Hand feels stiff and uncooperative, could not write legibly with a ballpoint pen."
- **Ground Truth**: Pain=0, Stiffness=2, Mobility=2, Grip=2
- **Predicted**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Top Attended Words**: `stiff` (0.528), `and` (0.155), `feels` (0.090), `Hand` (0.087), `legibly` (0.040)

---
