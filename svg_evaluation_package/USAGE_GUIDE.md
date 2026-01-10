# SVG Pipeline Evaluation System - Complete Guide
## What This System Does

### 1. Visual Similarity Evaluation
- **CLIP Score**: Compares rendered PNG images using CLIP vision model
- **BLIP Score**: Generates captions and compares semantic similarity
- Range: 0 (completely different) to 1 (identical)

### 2. SVG Code Quality Assessment

**Validity Score (40% weight)**
- XML parsing success
- Proper SVG namespace
- Valid attributes

**Structure Score (25% weight)**
- Has viewBox attribute
- Uses <defs> for reusable elements
- Semantic grouping with <g> tags
- Layered organization (background/content/foreground)

**Optimization Score (20% weight)**
- File size < 50KB (good), < 100KB (acceptable)
- Coordinate precision (1-3 decimals optimal)
- No redundant elements

**Readability Score (15% weight)**
- Proper indentation
- Semantic IDs
- Clean formatting

**Overall Code Quality** = weighted average of above scores

### 3. Combined Score

**Overall Score = (CLIP × 0.35) + (BLIP × 0.35) + (Code Quality × 0.30)**

This balances visual fidelity (70%) with code quality (30%)

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
pip install --break-system-packages \
    pandas numpy matplotlib pillow \
    torch torchvision transformers \
    cairosvg tqdm numbers-parser
```

**OR** use the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Prepare Your Data

Make sure it has these columns:
- `uuid`: Unique identifier for each SVG (UUID format)
- `Id`: Integer Id for SVG filenames (e.g., 12, 6, 8)
- `name`: Name/description
- `blip_caption`: Original BLIP caption
- `svg_code`: SVG code (optional if image_path exists)
- `image_path`: Path to PNG (optional if svg_code exists)

### Step 3: Organize Your Generated SVGs

```
your-dataset/
├── input/
│   ├── 5-agents-namecolumn/
│   │   ├── chatgpt-5.1/
│   │   │   ├── 12.svg
│   │   │   ├── 6.svg
│   │   │   └── ...
│   │   ├── gemini-3-pro-preview/
│   │   │   └── *.svg
│   │   └── sonnet-4.5/
│   │       └── *.svg
│   ├── 5-agents-sentence/
│   │   ├── chatgpt-5.1/
│   │   ├── gemini-3-pro-preview/
│   │   └── sonnet-4.5/
│   └── single-shot-namecolumn/
│       ├── chatgpt-5.1/
│       ├── gemini-3-pro-preview/
│       └── sonnet-4.5/
```

### Step 4: Configure Paths

Edit `svg_pipeline_evaluator.py`, in the `main()` function:

```python
ORIGINALS_CSV = '/home/claude/svgx_samples_main_sources.csv'
GENERATED_ROOT = '/path/to/your-dataset'  # Contains input/ folder
OUTPUT_DIR = '/home/claude/outputs/pipeline_evaluation'
```

### Step 5: Run Evaluation

```bash
python svg_pipeline_evaluator.py
```


## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Excellent - Nearly identical to original |
| 0.80 - 0.89 | Very Good - High fidelity with minor differences |
| 0.70 - 0.79 | Good - Recognizable but noticeable differences |
| 0.60 - 0.69 | Fair - Substantial differences |
| 0.00 - 0.59 | Poor - Major discrepancies |



