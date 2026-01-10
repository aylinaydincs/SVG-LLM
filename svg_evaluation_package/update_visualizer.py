#!/usr/bin/env python3
"""
Update svg_pipeline_evaluator.py with improved horizontal visualization
Run this in the same directory as svg_evaluation_package/
"""

import re
from pathlib import Path

# Path to the evaluator
eval_path = Path('svg_evaluation_package/svg_pipeline_evaluator.py')

if not eval_path.exists():
    print(f"❌ File not found: {eval_path}")
    print("Make sure you run this from the directory containing svg_evaluation_package/")
    exit(1)

# Read the file
with open(eval_path, 'r') as f:
    content = f.read()

# New visualization function
new_function = '''    def visualize_comparison(self, results, output_path):
        """
        Create horizontal comparison showing original + all generated versions
        Layout: Original | Model1 | Model2 | Model3
        """
        generations = list(results['generations'].keys())
        num_images = len(generations) + 1  # +1 for original
        
        # Create figure with horizontal layout
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 6))
        
        # Handle different axes formats
        if num_images == 1:
            axes = [axes]
        elif num_images == 2:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        
        # Plot original (leftmost)
        ax = axes[0]
        ax.imshow(results['original']['image'])
        ax.set_title('Original', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        
        # Add original's BLIP caption below
        caption = results['original']['blip_caption']
        ax.text(0.5, -0.08, f"BLIP: {caption[:60]}", 
               ha='center', va='top', transform=ax.transAxes, 
               fontsize=9, wrap=True, style='italic')
        
        # Plot generated versions
        for idx, (gen_key, data) in enumerate(results['generations'].items(), start=1):
            ax = axes[idx]
            ax.imshow(data['image'])
            
            # Title with model name and average score
            model_name = data['model']
            # Format model names nicely
            if 'chatgpt' in model_name.lower():
                model_display = 'ChatGPT-5.1'
            elif 'gemini' in model_name.lower():
                model_display = 'Gemini-3-Pro'
            elif 'sonnet' in model_name.lower():
                model_display = 'Sonnet-4.5'
            else:
                model_display = model_name.replace('-', ' ').title()
            
            title = f"{model_display}\\nAvg: {data['overall_score']:.3f}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
            
            # Add scores and caption below
            score_text = f"CLIP: {data['clip_score']:.3f} | BLIP: {data['blip_score']:.3f}"
            caption_text = f"{data['blip_caption'][:50]}"
            
            combined_text = f"{score_text}\\n{caption_text}"
            ax.text(0.5, -0.08, combined_text,
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=9, wrap=True)
        
        # Add super title
        if generations:
            pipeline = list(results['generations'].values())[0]['pipeline'].replace('-', ' ').title()
        else:
            pipeline = 'Unknown'
        
        fig.suptitle(f"SVG Comparison: {results['name']}\\n"
                    f"Pipeline: {pipeline} | Id: {results['svg_id']} | UUID: {results['uuid'][:16]}...",
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()'''

# Find and replace the visualize_comparison function
# Look for the function definition and replace everything until the next def or class
pattern = r'(    def visualize_comparison\(self, results, output_path\):.*?)(\n    def |\n\nclass )'
replacement = new_function + r'\2'

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content == content:
    print("⚠️  Could not find visualize_comparison function to replace")
    print("Manual update needed")
else:
    # Write back
    with open(eval_path, 'w') as f:
        f.write(new_content)
    print("✅ Successfully updated visualization function!")
    print("   Now the output will show: Original | Model1 | Model2 | Model3")
    print("\nRun the evaluator again:")
    print("   python svg_evaluation_package/svg_pipeline_evaluator.py")
