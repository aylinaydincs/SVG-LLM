#!/usr/bin/env python3
"""
Quick test script for SVG Pipeline Evaluator
Checks if all components work correctly
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import cairosvg
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_svg_rendering():
    """Test SVG to PNG rendering"""
    print("\nTesting SVG rendering...")
    
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="40" fill="blue"/>
</svg>'''
    
    try:
        import cairosvg
        from PIL import Image
        from io import BytesIO
        
        png_data = cairosvg.svg2png(bytestring=test_svg.encode('utf-8'))
        img = Image.open(BytesIO(png_data))
        print(f"✅ SVG rendering works (output: {img.size})")
        return True
    except Exception as e:
        print(f"❌ SVG rendering failed: {e}")
        return False


def test_code_evaluator():
    """Test SVG code quality evaluator"""
    print("\nTesting code quality evaluator...")
    
    from svg_pipeline_evaluator import SVGCodeQualityEvaluator
    
    test_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="grad1">
      <stop offset="0%" stop-color="red"/>
      <stop offset="100%" stop-color="blue"/>
    </linearGradient>
  </defs>
  <g id="background">
    <rect width="512" height="512" fill="url(#grad1)"/>
  </g>
  <g id="content">
    <circle cx="256" cy="256" r="100" fill="white"/>
  </g>
</svg>'''
    
    try:
        evaluator = SVGCodeQualityEvaluator()
        result = evaluator.evaluate_svg_code(test_svg)
        
        scores = result['scores']
        print(f"  Validity: {scores['validity']:.2f}")
        print(f"  Structure: {scores['structure']:.2f}")
        print(f"  Optimization: {scores['optimization']:.2f}")
        print(f"  Overall: {scores['overall']:.2f}")
        print("✅ Code evaluator works")
        return True
    except Exception as e:
        print(f"❌ Code evaluator failed: {e}")
        return False


def test_csv_reading():
    """Test if CSV can be read"""
    print("\nTesting CSV reading...")
    
    import pandas as pd
    csv_path = Path('/home/claude/svgx_samples_main_sources.csv')
    
    if not csv_path.exists():
        print(f"⚠️  CSV not found at {csv_path}")
        print("   Run the numbers-parser conversion first")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns[:5])}...")
        return True
    except Exception as e:
        print(f"❌ CSV reading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("SVG Pipeline Evaluator - System Check")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("SVG Rendering", test_svg_rendering()))
    results.append(("Code Evaluator", test_code_evaluator()))
    results.append(("CSV Reading", test_csv_reading()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ All tests passed! System ready to run.")
        print("\nNext steps:")
        print("1. Update paths in svg_pipeline_evaluator.py")
        print("2. Run: python svg_pipeline_evaluator.py")
    else:
        print("\n⚠️  Some tests failed. Fix issues before running evaluation.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install --break-system-packages <package>")
        print("- Install cairo: apt-get install libcairo2-dev")
        print("- Convert .numbers file to CSV first")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
