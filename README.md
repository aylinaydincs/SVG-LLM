# SVG Generation with LLMs - Evaluation Code

Evaluation framework and prompts for generating Scalable Vector Graphics (SVG) using ChatGPT, Gemini, and Claude **without any training**.

## 🎯 Overview

This repository contains the code and prompts used to evaluate whether frontier large language models can generate high-quality SVGs through prompt engineering alone. We compare two approaches:

- **Zero-Shot Prompt**: Direct generation in one step
- **Agentic Prompt**: 5-stage pipeline (Planning → Design → Generation → Validation → Refinement)

**Results**: Multi-agent prompting achieves **0.922 overall quality** (+3.6% over single-shot) with zero training cost.

## 📁 Repository Structure

```
SVG-LLM/
├── svg_evaluation_package/
│   ├── svg_pipeline_evaluator.py      # Main evaluation script
│   ├── svgx_samples_main_sources.csv  # Test samples metadata (12 SVGs)
│   ├── setup.sh                       # Dependency installation
│   ├── test_system.py                 # System check script
│   ├── USAGE_GUIDE.md                 # Detailed usage instructions
│   └── FOLDER_STRUCTURE_EXAMPLE.txt   # Expected input folder structure
│
├── visualization/                     # Generated SVGs
│
├── agentic-prompt.MD                  # Multi-agent prompt 
├── zero-shot-prompt.MD                # Single-shot prompt
└── README.md
```

## 🚀 Try the Prompts

### Option 1: Gemini Studio (Interactive UI - No Setup)

**Try our prompts directly in a pre-configured notebook:**

[**🌟 Launch Gemini Studio →**](https://ai.studio/apps/drive/16jO5RVF9L2HV-_BpDgHOExXqua341AvJ?fullscreenApplet=true)

Just type a description (e.g., "smartwatch icon") and get validated SVG code instantly!

### Option 2: Use with Any LLM

Copy prompts from the repository:
- `agentic-prompt.MD` - Multi-agent pipeline (recommended for quality)
- `zero-shot-prompt.MD` - Single-shot generation (faster)

Use with:
- **ChatGPT**: [chat.openai.com](https://chat.openai.com/)
- **Claude**: [claude.ai](https://claude.ai/)
- **Gemini**: [gemini.google.com](https://gemini.google.com/)

All three models work with the same prompts.

## 📊 Evaluation System

The `svg_pipeline_evaluator.py` evaluates generated SVGs using:

### Visual Similarity
- **CLIP**: Image embedding similarity (colors, shapes, layout)
- **BLIP**: Caption embedding similarity (semantic content)

### Code Quality
- **Validity** (40%): XML parsing, namespaces, valid attributes
- **Structure** (25%): viewBox, defs, semantic grouping
- **Optimization** (20%): File size, coordinate precision
- **Readability** (15%): Indentation, IDs, formatting

### Combined Score
```
Overall = 0.35×CLIP + 0.35×BLIP + 0.30×Code Quality
```

## 🛠️ Running Evaluation

### 1. Install Dependencies

```bash
cd svg_evaluation_package
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
pip install pandas numpy matplotlib pillow torch transformers cairosvg tqdm
```

### 2. Organize Your Generated SVGs

Follow the structure in `svg_evaluation_package/FOLDER_STRUCTURE_EXAMPLE.txt`:

```
dataset/
└── input/
    ├── 5-agents-namecolumn/
    │   ├── chatgpt-5.1/
    │   │   ├── 12.svg
    │   │   ├── 6.svg
    │   │   └── ...
    │   ├── gemini-3-pro-preview/
    │   └── sonnet-4.5/
    ├── 5-agents-sentence/
    └── single-shot-namecolumn/
```

### 3. Run Evaluation

```bash
cd svg_evaluation_package
python svg_pipeline_evaluator.py
```

### 4. Check Results

```bash
# View summary statistics
cat ../outputs/pipeline_evaluation/summary_statistics.json

# View visualizations
open ../outputs/pipeline_evaluation/visualizations/
```

## 📋 Results Summary

### By Pipeline (36 samples each)

| Pipeline | CLIP | BLIP | Code | Overall |
|----------|------|------|------|---------|
| **5-Agents-Sentence** | **0.937** | **0.891** | 0.940 | **0.922** |
| 5-Agents-NameColumn | 0.922 | 0.879 | **0.963** | 0.919 |
| Single-Shot | 0.919 | 0.875 | 0.875 | 0.890 |

### By Model (36 samples each)

| Model | Overall | Visual | Code |
|-------|---------|--------|------|
| **Gemini-3-Pro** | **0.921** | **0.930** | **0.946** |
| Sonnet-4.5 | 0.906 | 0.915 | 0.930 |
| ChatGPT-5.1 | 0.905 | 0.932 | 0.903 |

**Key Findings**:
- ✅ Multi-agent improves quality by +3.6% overall
- ✅ Code quality gains +7.4% with multi-agent
- ✅ Gemini leads in both visual similarity and code quality
- ✅ Zero training cost, fully reproducible

## 🎨 Visualizations

The evaluation generates side-by-side comparisons showing:
- Original SVG (from SVGX-SFT dataset)
- Generated versions from each configuration
- Scores: Overall, CLIP, BLIP, Code quality

Example: `outputs/pipeline_evaluation/visualizations/id_12.png`

## 📝 Dataset

12 test samples from [SVGX-SFT dataset](https://github.com/ximinng/LLM4SVG) (250K SVGs):
- 4 samples from Google Noto Emoji
- 5 samples from Twitter Twemoji  
- 3 samples from community sources

Categories: emoji, icons, symbols, illustrations

## 📖 Documentation

- **Usage Guide**: `svg_evaluation_package/USAGE_GUIDE.md` - Complete setup and usage instructions
- **Folder Structure**: `svg_evaluation_package/FOLDER_STRUCTURE_EXAMPLE.txt` - Expected input organization

## 📧 Contact

**Aylin Aydın**  
Bogazici University  
aylinaydin216@gmail.com


**Quick Links:**
- [Try in Gemini Studio →](https://ai.studio/apps/drive/16jO5RVF9L2HV-_BpDgHOExXqua341AvJ?fullscreenApplet=true)
- [SVGX-SFT Dataset →](https://github.com/ximinng/LLM4SVG)
- [Usage Guide →](svg_evaluation_package/USAGE_GUIDE.md)
