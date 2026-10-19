# Error Analysis: 20 Misclassified Held-Out Cases

Analysis of failure modes on the manually authored held-out test set:

### Case 1
- **Entry**: "No pain today but my fingers were very stiff in the morning."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Stiffness (True: 3 vs Pred: 1); Mobility (True: 1 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=3, Mobility=1, Grip=0
- **Full Prediction**: Pain=3, Stiffness=1, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 2
- **Entry**: "Sharp pain when I tried to close my fist, could not hold a cup."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=1, Mobility=2, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling, Functional task extrapolation (grip vs mobility).

### Case 3
- **Entry**: "Felt much better, did all my exercises and gripped a bottle easily."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 4
- **Entry**: "Mild ache after typing, wrist is a bit tight but moving fine."
- **Mismatched Heads**: Pain (True: 1 vs Pred: 3)
- **Full Ground Truth**: Pain=1, Stiffness=1, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=1, Mobility=0, Grip=0
- **Root Cause Analysis**: Complex contrastive clause coordination.

### Case 5
- **Entry**: "Pain is back and worse than last week, thumb barely moves."
- **Mismatched Heads**: Stiffness (True: 2 vs Pred: 0); Mobility (True: 3 vs Pred: 0); Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=2, Mobility=3, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 6
- **Entry**: "Not as stiff as yesterday, but still cannot pick up small coins."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=3
- **Full Prediction**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination, Functional task extrapolation (grip vs mobility).

### Case 7
- **Entry**: "Moderate soreness in the evening, grip is weak when opening jars."
- **Mismatched Heads**: Pain (True: 2 vs Pred: 3); Stiffness (True: 1 vs Pred: 0); Mobility (True: 1 vs Pred: 0)
- **Full Ground Truth**: Pain=2, Stiffness=1, Mobility=1, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=2
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 8
- **Entry**: "No pain, no stiffness. Full finger movement for the first time since surgery."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling.

### Case 9
- **Entry**: "Woke up with zero ache, though my knuckles feel like rusty hinges and will not bend."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Stiffness (True: 3 vs Pred: 0); Mobility (True: 2 vs Pred: 0); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=3, Mobility=2, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 10
- **Entry**: "Throbbing wrist after yard work, had to drop the watering can because grip gave out."
- **Mismatched Heads**: Pain (True: 2 vs Pred: 3)
- **Full Ground Truth**: Pain=2, Stiffness=0, Mobility=0, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 11
- **Entry**: "Completely pain free all day! Managed to button my work shirt without any help."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 12
- **Entry**: "A bit of an ache near the surgical scar, but fingers are flexing quite smoothly."
- **Mismatched Heads**: Pain (True: 1 vs Pred: 3)
- **Full Ground Truth**: Pain=1, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Complex contrastive clause coordination.

### Case 13
- **Entry**: "Excruciating flare up when turning the car ignition key, whole hand locked up."
- **Mismatched Heads**: Stiffness (True: 3 vs Pred: 0); Mobility (True: 3 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=3, Mobility=3, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 14
- **Entry**: "No stiffness at all today, but sharp pain shoots up my forearm whenever I pinch a pen."
- **Mismatched Heads**: Stiffness (True: 0 vs Pred: 3); Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=0, Mobility=0, Grip=2
- **Full Prediction**: Pain=3, Stiffness=3, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 15
- **Entry**: "Slight tightness in my palm after squeezing the stress ball, but strength is improving."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Grip (True: 1 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=1
- **Full Prediction**: Pain=3, Stiffness=1, Mobility=0, Grip=3
- **Root Cause Analysis**: Complex contrastive clause coordination.

### Case 16
- **Entry**: "Could barely hold my toothbrush this morning, hand feels completely dead."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Stiffness (True: 1 vs Pred: 0); Mobility (True: 3 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=3, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 17
- **Entry**: "No pain whatsoever, yet my pinky finger stays stubbornly crooked and will not straighten."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Stiffness (True: 1 vs Pred: 0); Mobility (True: 2 vs Pred: 0); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling.

### Case 18
- **Entry**: "Pretty bad throbbing after physio session, took an ice pack to settle the swelling."
- **Mismatched Heads**: Pain (True: 2 vs Pred: 3); Stiffness (True: 1 vs Pred: 0); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=2, Stiffness=1, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 19
- **Entry**: "Fingers moved like clockwork today! Sliced bread and carried grocery bags with ease."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 20
- **Entry**: "Terrible burning pain woke me at 3am, could not move my thumb an inch."
- **Mismatched Heads**: Mobility (True: 3 vs Pred: 0); Grip (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=3, Stiffness=0, Mobility=3, Grip=0
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling.

