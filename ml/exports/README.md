# TremorTrace — Export Package

Backend integration guide for the TremorTrace ML pipeline.

**TremorTrace** is an AI-powered Parkinson's disease screening tool that analyzes spiral and wave drawings using two MobileNetV2 CNNs in a weighted ensemble. It outputs a **probability percentage (0–100%)** with a risk tier — never a binary diagnosis.

---

## Package Contents

| File | Description |
|------|-------------|
| `spiral_final.keras` | Trained Spiral CNN model (MobileNetV2 + classification head) |
| `wave_final.keras` | Trained Wave CNN model (MobileNetV2 + classification head) |
| `ensemble.py` | 2-model weighted ensemble — combines predictions into percentage output |
| `gradcam.py` | Grad-CAM heatmap generation for model explainability |
| `input_handler.py` | **Main entry point** — the only file the backend needs to call |
| `metadata.json` | Model metadata, training metrics, and configuration |

All three Python files (`ensemble.py`, `gradcam.py`, `input_handler.py`) are **self-contained** with absolute imports. They must be deployed together in the same directory.

---

## Quick Start

```python
import tensorflow as tf
from input_handler import process_input

# Load models once at startup
spiral_model = tf.keras.models.load_model('spiral_final.keras')
wave_model = tf.keras.models.load_model('wave_final.keras')

# Run prediction (same call for both drawn and uploaded input)
result = process_input(
    spiral_image_base64="<base64-encoded PNG>",
    wave_image_base64="<base64-encoded PNG>",
    spiral_cnn_model=spiral_model,
    wave_cnn_model=wave_model,
    input_mode='drawn',  # or 'uploaded' — metadata only, no processing difference
)

# Result is a dictionary:
print(f"PD Probability: {result['pd_probability_percent']}%")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Risk Color: {result['risk_color']}")
```

---

## Model Loading

```python
import tensorflow as tf

# Load both models (do this ONCE at server startup, not per-request)
spiral_model = tf.keras.models.load_model('spiral_final.keras')
wave_model = tf.keras.models.load_model('wave_final.keras')

# Both models expect: input shape (None, 224, 224, 3), float32, values [0, 1]
# Both models output: shape (None, 1), sigmoid probability P(Parkinson's)
```

---

## Input Requirements

| Parameter | Requirement |
|-----------|-------------|
| Image format | PNG or JPG, base64-encoded string |
| Data URI prefix | Optional — `"data:image/png;base64,..."` is handled automatically |
| Image size | Any size (automatically resized to 224×224) |
| Color mode | Any (automatically converted to RGB) |
| Normalization | Automatic — raw image bytes are fine |

### WARNING: Canvas Color Inversion

HTML5 canvas drawings typically have a **dark/black background** with light strokes, but the training data has **dark strokes on white backgrounds**. The `input_handler.py` automatically detects and fixes this inversion:

- If the mean pixel intensity is < 50% (dark background), the image is inverted
- This is handled inside `preprocess_image_from_base64()` — no action needed from the backend
- If you need to call it manually: `from input_handler import detect_and_fix_inversion`

---

## Using `process_input()`

This is the **single entry point** — the backend only needs to call this one function.

```python
def process_input(
    spiral_image_base64: str,       # base64 PNG of spiral drawing
    wave_image_base64: str,         # base64 PNG of wave drawing
    spiral_cnn_model=None,          # loaded Keras model
    wave_cnn_model=None,            # loaded Keras model
    input_mode: str = 'drawn',      # 'drawn' or 'uploaded' (metadata only)
) -> dict:
```

**Both input modes are processed identically.** The `input_mode` parameter is only recorded in the output dictionary for logging/analytics — it does NOT affect predictions.

**Error handling:** On failure, returns `{"error": "<message>", "error_type": "<exception type>"}`.

---

## Output Schema

```python
{
    # PRIMARY OUTPUT
    "pd_probability_percent": 73.2,          # float, 0.0–100.0 — THE MAIN RESULT
    "risk_tier": "Elevated Risk",            # str, see Risk Tier Definitions below
    "risk_color": "#E74C3C",                 # str, hex color for UI display

    # PER-MODEL BREAKDOWN
    "spiral_cnn_percent": 81.4,              # float, 0.0–100.0
    "wave_cnn_percent": 65.0,                # float, 0.0–100.0

    # ENSEMBLE METADATA
    "input_mode": "drawn",                   # str, "drawn" or "uploaded"
    "weights_used": {
        "spiral_cnn": 0.5,                   # float, weight applied to spiral model
        "wave_cnn": 0.5,                     # float, weight applied to wave model
    },
    "model_agreement": 1.0,                  # float, 1.0 if both agree, 0.5 if split
    "unanimous": true,                       # bool, whether both models agree

    # CONFIDENCE
    "confidence_score": 0.464,               # float, 0.0–1.0 — distance from 50%
    "confidence_label": "Moderate",          # str, "Low" / "Moderate" / "High"

    # EXPLAINABILITY (Grad-CAM)
    "spiral_gradcam_base64": "data:image/png;base64,...",  # str or null
    "wave_gradcam_base64": "data:image/png;base64,...",    # str or null

    # LEGAL
    "disclaimer": "This is a screening tool only. ..."     # str, MUST be displayed
}
```

---

## Risk Tier Definitions

| Range | Label | Hex Color | UI Meaning |
|-------|-------|-----------|------------|
| 0–25% | Low Risk | `#27AE60` (green) | Minimal indicators detected |
| 25–45% | Mild Risk | `#F1C40F` (yellow) | Some indicators present |
| 45–65% | Moderate Risk | `#E67E22` (orange) | Notable indicators detected |
| 65–85% | Elevated Risk | `#E74C3C` (red-orange) | Significant indicators detected |
| 85–100% | High Risk | `#C0392B` (red) | Strong indicators detected |

Use `risk_color` directly in CSS for UI display.

---

## Confidence Labels

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 0.0–0.3 | Low | Prediction near 50% — model is uncertain |
| 0.3–0.6 | Moderate | Prediction leans one way but not strongly |
| 0.6–1.0 | High | Prediction is far from 50% — model is confident |

Formula: `confidence_score = abs(probability - 0.5) * 2`

---

## Medical Disclaimer

The following disclaimer text is included in every response and **MUST** be displayed to the user in the UI:

> This is a screening tool only. It does not constitute a medical diagnosis. Please consult a healthcare professional for proper evaluation.

---

## Performance Metrics

*These values are populated during training. See `metadata.json` for exact numbers.*

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| Spiral CNN | See metadata.json | See metadata.json | Trained on ~72 spiral drawings |
| Wave CNN | See metadata.json | See metadata.json | Trained on ~72 wave drawings |

---

## Important Notes

1. **Percentage output, NOT binary.** The system outputs a probability percentage (0–100%) with a risk tier. It does NOT output "healthy" or "parkinson's" as a primary label.

2. **Both input modes are identical.** There is no difference in processing, weighting, or confidence between drawn and uploaded images. The `input_mode` field is metadata only.

3. **Two models only.** This system uses exactly 2 CNN models (Spiral + Wave) in a 50/50 weighted ensemble. There are no XGBoost models, no feature extraction, and no temporal analysis.

4. **Grad-CAM may be null.** If heatmap generation fails for any reason, the `*_gradcam_base64` fields will be `null`. The prediction result is still valid.

5. **Deploy files together.** `input_handler.py` imports from `ensemble.py` and `gradcam.py`. All three must be in the same directory.

6. **TensorFlow required.** The backend needs `tensorflow>=2.15.0`, `numpy`, `Pillow`, and `opencv-python-headless`.
