# Error Analysis: 20 Misclassified Held-Out Cases

Analysis of failure modes on the manually authored held-out test set:

### Case 1
- **Entry**: "No pain today but my fingers were very stiff in the morning."
- **Mismatched Heads**: Stiffness (True: 3 vs Pred: 2); Mobility (True: 1 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=3, Mobility=1, Grip=0
- **Full Prediction**: Pain=0, Stiffness=2, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 2
- **Entry**: "Sharp pain when I tried to close my fist, could not hold a cup."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=1, Mobility=2, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling, Functional task extrapolation (grip vs mobility).

### Case 3
- **Entry**: "Pain is back and worse than last week, thumb barely moves."
- **Mismatched Heads**: Stiffness (True: 2 vs Pred: 0); Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=2, Mobility=3, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=3, Grip=0
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 4
- **Entry**: "Not as stiff as yesterday, but still cannot pick up small coins."
- **Mismatched Heads**: Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=3
- **Full Prediction**: Pain=0, Stiffness=1, Mobility=0, Grip=3
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination, Functional task extrapolation (grip vs mobility).

### Case 5
- **Entry**: "Moderate soreness in the evening, grip is weak when opening jars."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 1 vs Pred: 0)
- **Full Ground Truth**: Pain=2, Stiffness=1, Mobility=1, Grip=2
- **Full Prediction**: Pain=2, Stiffness=0, Mobility=0, Grip=2
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 6
- **Entry**: "Woke up with zero ache, though my knuckles feel like rusty hinges and will not bend."
- **Mismatched Heads**: Stiffness (True: 3 vs Pred: 0); Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=3, Mobility=2, Grip=0
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 7
- **Entry**: "Excruciating flare up when turning the car ignition key, whole hand locked up."
- **Mismatched Heads**: Stiffness (True: 3 vs Pred: 0); Mobility (True: 3 vs Pred: 0); Grip (True: 3 vs Pred: 2)
- **Full Ground Truth**: Pain=3, Stiffness=3, Mobility=3, Grip=3
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=2
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 8
- **Entry**: "No stiffness at all today, but sharp pain shoots up my forearm whenever I pinch a pen."
- **Mismatched Heads**: Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=0, Mobility=0, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 9
- **Entry**: "Could barely hold my toothbrush this morning, hand feels completely dead."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 3 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=3, Grip=3
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=0, Grip=3
- **Root Cause Analysis**: Functional task extrapolation (grip vs mobility).

### Case 10
- **Entry**: "No pain whatsoever, yet my pinky finger stays stubbornly crooked and will not straighten."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 2 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=2, Grip=0
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=3, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling.

### Case 11
- **Entry**: "Pretty bad throbbing after physio session, took an ice pack to settle the swelling."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0)
- **Full Ground Truth**: Pain=2, Stiffness=1, Mobility=0, Grip=0
- **Full Prediction**: Pain=2, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 12
- **Entry**: "Fingers moved like clockwork today! Sliced bread and carried grocery bags with ease."
- **Mismatched Heads**: Mobility (True: 0 vs Pred: 1); Grip (True: 0 vs Pred: 2)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=1, Grip=2
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 13
- **Entry**: "Wrist feels a little stiff after cold weather, but grip is strong enough to open soda cans."
- **Mismatched Heads**: Grip (True: 0 vs Pred: 2)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=0
- **Full Prediction**: Pain=0, Stiffness=1, Mobility=0, Grip=2
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 14
- **Entry**: "Pain is gone, but I dropped a coffee mug because my index finger suddenly slipped."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 3)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=2
- **Root Cause Analysis**: Complex contrastive clause coordination, Functional task extrapolation (grip vs mobility).

### Case 15
- **Entry**: "Joints are stiff as wood this morning, taking over an hour of warm water soaks to loosen."
- **Mismatched Heads**: Stiffness (True: 3 vs Pred: 2); Mobility (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=0, Stiffness=3, Mobility=2, Grip=0
- **Full Prediction**: Pain=0, Stiffness=2, Mobility=0, Grip=0
- **Root Cause Analysis**: Subtle severity nuance (e.g. mild vs moderate distinction).

### Case 16
- **Entry**: "Severe shooting pain when spreading my fingers apart, could not type at all today."
- **Mismatched Heads**: Mobility (True: 2 vs Pred: 0); Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=0, Mobility=2, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling.

### Case 17
- **Entry**: "Not hurting today, but grip feels strangely weak when lifting a heavy skillet."
- **Mismatched Heads**: Grip (True: 2 vs Pred: 1)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=2
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=0, Grip=1
- **Root Cause Analysis**: Negation boundary / modifier handling, Complex contrastive clause coordination.

### Case 18
- **Entry**: "Horrible pain radiating to my elbow, cannot clench fingers even halfway."
- **Mismatched Heads**: Stiffness (True: 1 vs Pred: 0); Mobility (True: 3 vs Pred: 0); Grip (True: 2 vs Pred: 0)
- **Full Ground Truth**: Pain=3, Stiffness=1, Mobility=3, Grip=2
- **Full Prediction**: Pain=3, Stiffness=0, Mobility=0, Grip=0
- **Root Cause Analysis**: Negation boundary / modifier handling.

### Case 19
- **Entry**: "Knuckles were a tiny bit puffy and stiff upon waking, but loosened up after coffee."
- **Mismatched Heads**: Pain (True: 0 vs Pred: 1); Grip (True: 0 vs Pred: 1)
- **Full Ground Truth**: Pain=0, Stiffness=1, Mobility=0, Grip=0
- **Full Prediction**: Pain=1, Stiffness=1, Mobility=0, Grip=1
- **Root Cause Analysis**: Complex contrastive clause coordination.

### Case 20
- **Entry**: "Pain free day! Cut up vegetables for dinner and had no trouble holding the knife."
- **Mismatched Heads**: Grip (True: 0 vs Pred: 2)
- **Full Ground Truth**: Pain=0, Stiffness=0, Mobility=0, Grip=0
- **Full Prediction**: Pain=0, Stiffness=0, Mobility=0, Grip=2
- **Root Cause Analysis**: Negation boundary / modifier handling, Functional task extrapolation (grip vs mobility).

