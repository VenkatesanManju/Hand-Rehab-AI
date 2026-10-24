# Contributing to HandRehab-AI

Thank you for your interest in contributing to **HandRehab-AI**! We welcome improvements, bug fixes, and scientific extensions to our multimodal hand rehabilitation framework.

---

## Code of Conduct & Development Principles

1. **Python & Framework Standards**:
   - Python 3.10 / 3.11 compatibility.
   - Deep learning code must use **PyTorch**. Do not introduce TensorFlow dependencies.
   - Maintain a deterministic random seed (`seed = 42`) across all data loaders, model initializations, and splits.
2. **Preprocessing Integrity**:
   - Natural language preprocessing **must preserve clinical negations** (e.g., `"no"`, `"not"`, `"never"`, `"without"`). In medical contexts, stripping negation reverses clinical meaning (e.g., `"no pain"` -> `"pain"`).
3. **Data Transparency & Safety**:
   - All synthetic datasets must remain labeled as synthetic. Never claim direct clinical trial validation without prospective ethics committee (IRB) review.
   - System interfaces must always state: *"Rehabilitation monitoring aid, not a diagnostic medical device; consult your physiotherapist."*
4. **Testing Requirements**:
   - Run the complete pytest test suite before submitting pull requests:
     ```bash
     pytest
     ```
   - Ensure all unit, integration, and API tests pass (60/60 passing).

---

## Development Workflow

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/hand-rehab-ai.git
   cd hand-rehab-ai
   ```
2. Create a clean virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(nlp): add attention entropy calculation"
   ```
5. Run the test suite:
   ```bash
   pytest
   ```
6. Open a Pull Request detailing your changes and experimental results.
