#!/bin/bash
# SVG Pipeline Evaluation - Complete Setup Script

echo "==============================================="
echo "SVG Pipeline Evaluation System - Setup"
echo "==============================================="

# Install required packages
echo -e "\n[1/4] Installing Python packages..."
pip install --break-system-packages \
    pandas \
    numpy \
    matplotlib \
    pillow \
    torch \
    torchvision \
    transformers \
    cairosvg \
    tqdm \
    numbers-parser

# Convert .numbers to CSV if needed
echo -e "\n[2/4] Converting .numbers file to CSV..."
python3 << 'EOF'
try:
    from numbers_parser import Document
    import pandas as pd
    from pathlib import Path
    
    numbers_file = Path('/mnt/user-data/uploads/svgx_samples_main_sources.numbers')
    output_csv = Path('/home/claude/svgx_samples_main_sources.csv')
    
    if numbers_file.exists() and not output_csv.exists():
        doc = Document(str(numbers_file))
        table = doc.sheets[0].tables[0]
        data = table.rows(values_only=True)
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_csv(output_csv, index=False)
        print(f"✅ Converted: {len(df)} rows saved to {output_csv}")
    elif output_csv.exists():
        print(f"✅ CSV already exists: {output_csv}")
    else:
        print(f"⚠️  .numbers file not found at {numbers_file}")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
EOF

# Test system
echo -e "\n[3/4] Testing system..."
python3 test_system.py

# Instructions
echo -e "\n[4/4] Setup complete!"
echo -e "\n==============================================="
echo "NEXT STEPS:"
echo "==============================================="
echo "1. Update paths in svg_pipeline_evaluator.py:"
echo "   - GENERATED_ROOT: Path to your dataset folder (contains input/)"
echo ""
echo "2. Run evaluation:"
echo "   python svg_pipeline_evaluator.py"
echo ""
echo "3. Check results in: /home/claude/outputs/pipeline_evaluation/"
echo "   - summary_statistics.json"
echo "   - pipeline_comparison.png"
echo "   - visualizations/*.png"
echo "==============================================="
