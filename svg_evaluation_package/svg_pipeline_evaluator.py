#!/usr/bin/env python3
"""
SVG Pipeline Evaluation System
Compares 3 different generation methods against original SVGs:
- 5-agents-namecolumn
- 5-agents-sentence  
- single-shot-namecolumn

Evaluates:
1. Visual similarity (CLIP, BLIP)
2. SVG code quality (validity, structure, optimization)
3. Cross-method comparison
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re


class SVGCodeQualityEvaluator:
    """Evaluate SVG code quality and structure"""
    
    @staticmethod
    def evaluate_svg_code(svg_code):
        """
        Comprehensive SVG code quality evaluation
        
        Returns:
            dict with scores and metrics
        """
        scores = {
            'validity': 0.0,
            'structure': 0.0,
            'optimization': 0.0,
            'readability': 0.0,
            'overall': 0.0
        }
        
        metrics = {
            'total_elements': 0,
            'path_count': 0,
            'group_count': 0,
            'gradient_count': 0,
            'has_defs': False,
            'has_viewbox': False,
            'file_size_kb': len(svg_code) / 1024,
            'avg_coordinate_precision': 0,
            'errors': []
        }
        
        # 1. VALIDITY CHECK (0-1)
        try:
            # Parse XML
            root = ET.fromstring(svg_code)
            scores['validity'] = 1.0
            
            # Check namespace
            if 'xmlns' in root.attrib or root.tag.startswith('{http://www.w3.org/2000/svg}'):
                scores['validity'] = 1.0
            else:
                scores['validity'] = 0.8
                metrics['errors'].append('Missing xmlns namespace')
                
        except ET.ParseError as e:
            scores['validity'] = 0.0
            metrics['errors'].append(f'XML parse error: {str(e)[:50]}')
            return {'scores': scores, 'metrics': metrics}
        
        # 2. STRUCTURE CHECK (0-1)
        structure_points = 0
        max_structure_points = 5
        
        # Has proper SVG root
        if root.tag.endswith('svg'):
            structure_points += 1
        
        # Has viewBox
        if 'viewBox' in root.attrib:
            metrics['has_viewbox'] = True
            structure_points += 1
        
        # Has <defs> section
        defs = root.find('.//{http://www.w3.org/2000/svg}defs') or root.find('.//defs')
        if defs is not None:
            metrics['has_defs'] = True
            structure_points += 1
        
        # Has semantic grouping
        groups = root.findall('.//{http://www.w3.org/2000/svg}g') or root.findall('.//g')
        metrics['group_count'] = len(groups)
        if len(groups) > 0:
            structure_points += 1
        
        # Proper layering (background, content, foreground groups)
        group_ids = [g.get('id', '') for g in groups]
        has_layering = any('background' in gid.lower() or 'foreground' in gid.lower() for gid in group_ids)
        if has_layering:
            structure_points += 1
        
        scores['structure'] = structure_points / max_structure_points
        
        # 3. OPTIMIZATION CHECK (0-1)
        optimization_points = 0
        max_optimization_points = 5
        
        # Count elements
        all_elements = list(root.iter())
        metrics['total_elements'] = len(all_elements)
        
        paths = root.findall('.//{http://www.w3.org/2000/svg}path') or root.findall('.//path')
        metrics['path_count'] = len(paths)
        
        # Check coordinate precision (should be ~2 decimals)
        coordinates = re.findall(r'[-+]?\d*\.\d+', svg_code)
        if coordinates:
            precisions = [len(c.split('.')[1]) if '.' in c else 0 for c in coordinates[:100]]
            avg_precision = np.mean(precisions)
            metrics['avg_coordinate_precision'] = avg_precision
            
            # Optimal precision: 1-3 decimals
            if 1 <= avg_precision <= 3:
                optimization_points += 2
            elif avg_precision <= 5:
                optimization_points += 1
        
        # File size check (<50KB is good)
        if metrics['file_size_kb'] < 50:
            optimization_points += 2
        elif metrics['file_size_kb'] < 100:
            optimization_points += 1
        
        # No redundant attributes (simplified check)
        if svg_code.count('fill="none" stroke="none"') == 0:
            optimization_points += 1
        
        scores['optimization'] = optimization_points / max_optimization_points
        
        # 4. READABILITY CHECK (0-1)
        readability_points = 0
        max_readability_points = 4
        
        # Proper indentation (check for newlines and spaces)
        lines = svg_code.split('\n')
        if len(lines) > 5:
            readability_points += 1
        
        # Has semantic IDs
        has_ids = bool(re.search(r'id="[a-zA-Z]', svg_code))
        if has_ids:
            readability_points += 1
        
        # Consistent formatting (no excessive whitespace)
        if not re.search(r'\s{10,}', svg_code):
            readability_points += 1
        
        # Comments or structure (optional but good)
        if '<!--' in svg_code or metrics['group_count'] > 2:
            readability_points += 1
        
        scores['readability'] = readability_points / max_readability_points
        
        # 5. GRADIENTS AND ADVANCED FEATURES
        gradients = (root.findall('.//{http://www.w3.org/2000/svg}linearGradient') or 
                    root.findall('.//linearGradient') or
                    root.findall('.//{http://www.w3.org/2000/svg}radialGradient') or
                    root.findall('.//radialGradient'))
        metrics['gradient_count'] = len(gradients)
        
        # OVERALL SCORE
        scores['overall'] = (
            scores['validity'] * 0.4 +
            scores['structure'] * 0.25 +
            scores['optimization'] * 0.2 +
            scores['readability'] * 0.15
        )
        
        return {
            'scores': scores,
            'metrics': metrics
        }


class SVGPipelineEvaluator:
    """Evaluate different SVG generation pipelines"""
    
    def __init__(self, originals_csv, generated_root):
        """
        Initialize evaluator
        
        Args:
            originals_csv: Path to CSV with original SVG metadata
            generated_root: Root folder with input/ subfolder
        """
        self.originals_csv = Path(originals_csv)
        self.generated_root = Path(generated_root)
        
        # Load original data
        self.df_originals = pd.read_csv(originals_csv)
        self.df_lookup = self.df_originals.set_index('uuid')
        
        # Pipeline methods to evaluate
        self.pipelines = [
            '5-agents-namecolumn',
            '5-agents-sentence',
            'single-shot-namecolumn'
        ]
        
        # Initialize ML models
        print("Loading models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.code_evaluator = SVGCodeQualityEvaluator()
        
        print(f"Using device: {self.device}")
    
    def svg_to_png(self, svg_code, size=512, bg_color='white'):
        """Convert SVG to PNG with proper background"""
        try:
            if 'xmlns' not in svg_code:
                svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
            
            png_data = cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                output_width=size,
                output_height=size
            )
            
            img = Image.open(BytesIO(png_data)).convert('RGBA')
            
            if bg_color:
                background = Image.new('RGB', (size, size), bg_color)
                background.paste(img, (0, 0), img)
                return background
            
            return img.convert('RGB')
            
        except Exception as e:
            print(f"    SVG render failed: {str(e)[:60]}")
            return Image.new('RGB', (size, size), 'white')
    
    def generate_blip_caption(self, image):
        """Generate BLIP caption"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
    
    def compute_clip_similarity(self, image1, image2):
        """Compute CLIP visual similarity"""
        inputs = self.clip_processor(images=[image1, image2], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        
        features = features / features.norm(dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
        
        return (similarity + 1) / 2
    
    def compute_blip_text_similarity(self, caption1, caption2):
        """Compute BLIP caption similarity using CLIP text embeddings"""
        inputs = self.clip_processor(text=[caption1, caption2], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        
        features = features / features.norm(dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(features[0:1], features[1:2]).item()
        
        return (similarity + 1) / 2
    
    def scan_generated_svgs(self):
        """
        Scan all pipeline folders and map SVGs by Id
        Structure: input/pipeline_name/model_name/id.svg (where id is integer)
        
        Returns:
            dict: {id_str: {(pipeline, model): svg_path, 'uuid': actual_uuid}}
        """
        input_folder = self.generated_root / 'input'
        svg_map = {}
        
        if not input_folder.exists():
            print(f"  Input folder not found: {input_folder}")
            return svg_map
        
        # Create Id to UUID mapping from CSV
        id_to_uuid = {}
        for _, row in self.df_originals.iterrows():
            svg_id = str(int(row['Id'])) if pd.notna(row['Id']) else None
            if svg_id:
                id_to_uuid[svg_id] = row['uuid']
        
        print(f"  Created Id->UUID mapping for {len(id_to_uuid)} entries")
        
        for pipeline in self.pipelines:
            pipeline_folder = input_folder / pipeline
            
            if not pipeline_folder.exists():
                print(f"  Pipeline folder not found: {pipeline}")
                continue
            
            # Scan model subfolders
            model_folders = [f for f in pipeline_folder.iterdir() if f.is_dir()]
            
            for model_folder in model_folders:
                model_name = model_folder.name
                
                # Scan for SVG files
                svg_files = list(model_folder.glob('*.svg'))
                
                for svg_path in svg_files:
                    # Extract Id from filename (should be integer like "12.svg")
                    svg_id = svg_path.stem
                    
                    # Get corresponding UUID
                    uuid = id_to_uuid.get(svg_id)
                    
                    if not uuid:
                        print(f"    No UUID found for Id={svg_id}")
                        continue
                    
                    if svg_id not in svg_map:
                        svg_map[svg_id] = {'uuid': uuid}
                    
                    # Use (pipeline, model) as key
                    svg_map[svg_id][(pipeline, model_name)] = svg_path
            
            total_svgs = sum(len(list(m.glob('*.svg'))) for m in model_folders)
            print(f"  {pipeline}: {total_svgs} SVGs across {len(model_folders)} models")
        
        return svg_map
    
    def load_original_data(self, uuid):
        """
        Load original SVG data, prioritizing pre-rendered PNGs
        
        Priority:
        1. Pre-rendered PNG from dataset/SVGX-rendering-data/uuid.png
        2. image_path from CSV
        3. Render from svg_code
        """
        if uuid not in self.df_lookup.index:
            return None
        
        row = self.df_lookup.loc[uuid]
        original_image = None
        
        # PRIORITY 1: Check for pre-rendered PNG in dataset
        png_render_path = Path('dataset/SVGX-rendering-data') / f"{uuid}.png"
        if png_render_path.exists():
            try:
                img = Image.open(png_render_path)
                # Convert RGBA to RGB with white background if needed
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, 'white')
                    background.paste(img, mask=img.split()[3])
                    original_image = background
                else:
                    original_image = img.convert('RGB')
            except Exception as e:
                print(f"  Error loading pre-rendered PNG for {uuid[:8]}: {str(e)[:40]}")
        
        # PRIORITY 2: Try image_path from CSV
        if original_image is None and 'image_path' in row and pd.notna(row['image_path']):
            img_path = row['image_path']
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, 'white')
                        background.paste(img, mask=img.split()[3])
                        original_image = background
                    else:
                        original_image = img.convert('RGB')
                except Exception as e:
                    print(f"  Error loading image_path for {uuid[:8]}: {str(e)[:40]}")
        
        # PRIORITY 3: Fallback - render from svg_code
        if original_image is None and 'svg_code' in row and pd.notna(row['svg_code']):
            try:
                original_image = self.svg_to_png(row['svg_code'], size=512, bg_color='white')
            except Exception as e:
                print(f"  Error rendering SVG for {uuid[:8]}: {str(e)[:40]}")
        
        # Last resort: white square
        if original_image is None:
            print(f"  No valid image source for {uuid[:8]}, using white square")
            original_image = Image.new('RGB', (512, 512), 'white')
        
        return {
            'image': original_image,
            'blip_caption': row.get('blip_caption', ''),
            'name': row.get('name', 'unknown'),
            'svg_code': row.get('svg_code', '')
        }
    
    def evaluate_single_svg(self, svg_id, uuid, pipeline_paths, original_data):
        """
        Evaluate all pipelines/models for a single SVG
        
        Args:
            svg_id: The Id from CSV (integer as string)
            uuid: The UUID from CSV
            pipeline_paths: dict with (pipeline, model) tuples as keys
            original_data: Original SVG data
        
        Returns:
            dict with comparison results
        """
        results = {
            'svg_id': svg_id,
            'uuid': uuid,
            'name': original_data['name'],
            'original': {
                'image': original_data['image'],
                'blip_caption': original_data['blip_caption']
            },
            'generations': {}
        }
        
        for (pipeline_name, model_name), svg_path in pipeline_paths.items():
            try:
                # Load generated SVG
                with open(svg_path, 'r', encoding='utf-8') as f:
                    gen_svg_code = f.read()
                
                # Render to image
                gen_image = self.svg_to_png(gen_svg_code)
                
                # Generate caption
                gen_blip_caption = self.generate_blip_caption(gen_image)
                
                # Compute visual similarities
                clip_score = self.compute_clip_similarity(original_data['image'], gen_image)
                blip_score = self.compute_blip_text_similarity(
                    original_data['blip_caption'], 
                    gen_blip_caption
                )
                
                # Evaluate SVG code quality
                code_quality = self.code_evaluator.evaluate_svg_code(gen_svg_code)
                
                # Create combined key
                gen_key = f"{pipeline_name}_{model_name}"
                
                # Store results
                results['generations'][gen_key] = {
                    'pipeline': pipeline_name,
                    'model': model_name,
                    'image': gen_image,
                    'svg_code': gen_svg_code,
                    'blip_caption': gen_blip_caption,
                    'clip_score': clip_score,
                    'blip_score': blip_score,
                    'visual_avg': (clip_score + blip_score) / 2,
                    'code_quality': code_quality['scores'],
                    'code_metrics': code_quality['metrics'],
                    'overall_score': (
                        clip_score * 0.35 +
                        blip_score * 0.35 +
                        code_quality['scores']['overall'] * 0.30
                    )
                }
                
            except Exception as e:
                print(f"    ⚠ Error processing {pipeline_name}/{model_name}/Id={svg_id}: {str(e)[:60]}")
                continue
        
        return results if results['generations'] else None
    
    def visualize_comparison(self, results, output_path):
        """
        Simple horizontal comparison with numeric scores only
        Layout: Original | Model1 | Model2 | Model3
        Shows: Images + numeric scores with clear pipeline/model labels
        """
        generations = list(results['generations'].keys())
        num_images = len(generations) + 1  # +1 for original
        
        # Create figure with horizontal layout
        fig, axes = plt.subplots(1, num_images, figsize=(4.5 * num_images, 5.5))
        
        # Handle different axes formats
        if num_images == 1:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot original (leftmost)
        ax = axes[0]
        ax.imshow(results['original']['image'])
        ax.set_title('Original', fontsize=14, fontweight='bold', pad=10, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax.axis('off')
        
        # Plot generated versions
        for idx, (gen_key, data) in enumerate(results['generations'].items(), start=1):
            ax = axes[idx]
            ax.imshow(data['image'])
            
            # Get pipeline and model names
            pipeline_name = data['pipeline']
            model_name = data['model']
            
            # Format model name
            if 'chatgpt' in model_name.lower():
                model_display = 'ChatGPT-5.1'
            elif 'gemini' in model_name.lower():
                model_display = 'Gemini-3-Pro'
            elif 'sonnet' in model_name.lower():
                model_display = 'Sonnet-4.5'
            else:
                model_display = model_name.replace('-', ' ').title()
            
            # Format pipeline name for display
            if '5-agents-namecolumn' in pipeline_name:
                pipeline_display = '5-Agents-NameColumn'
            elif '5-agents-sentence' in pipeline_name:
                pipeline_display = '5-Agents-Sentence'
            elif 'single-shot' in pipeline_name:
                pipeline_display = 'Single-Shot'
            else:
                pipeline_display = pipeline_name.replace('-', ' ').title()
            
            # Title with pipeline and model
            title = f"{pipeline_display}\n{model_display}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
            ax.axis('off')
            
            # Show scores as simple numbers below the image
            scores_text = (
                f"Overall: {data['overall_score']:.3f}\n"
                f"CLIP: {data['clip_score']:.3f} | BLIP: {data['blip_score']:.3f}\n"
                f"Code: {data['code_quality']['overall']:.3f}"
            )
            
            ax.text(0.5, -0.09, scores_text,
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        # Title with both Id and name
        fig.suptitle(f"{results['name']} (Id: {results['svg_id']})",
                    fontsize=14, fontweight='bold', y=0.97)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def process_all(self, output_dir='pipeline_evaluation_results'):
        """Process all SVGs and create comprehensive evaluation"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Scan generated SVGs
        print("\n[STEP 1] Scanning generated SVGs...")
        svg_map = self.scan_generated_svgs()
        print(f"  Found {len(svg_map)} unique Ids with generated versions")
        
        # Process each SVG by Id
        print(f"\n[STEP 2] Evaluating SVGs...")
        all_results = []
        
        for svg_id, svg_data in tqdm(svg_map.items(), desc="Processing"):
            uuid = svg_data['uuid']
            
            # Get pipeline paths (exclude 'uuid' key)
            pipeline_paths = {k: v for k, v in svg_data.items() if k != 'uuid'}
            
            # Load original
            original_data = self.load_original_data(uuid)
            if not original_data:
                print(f"  Original not found for Id={svg_id} (UUID={uuid[:8]})")
                continue
            
            # Evaluate
            results = self.evaluate_single_svg(svg_id, uuid, pipeline_paths, original_data)
            
            if results and results['generations']:
                all_results.append(results)
                
                # Create visualization
                viz_path = viz_dir / f"id_{results['svg_id']}.png"
                self.visualize_comparison(results, viz_path)
        
        # Create summary
        print(f"\n[STEP 3] Creating summary...")
        self.create_summary(all_results, output_dir)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total evaluations: {len(all_results)}")
        print(f"Results saved to: {output_dir}")
        
        return all_results
    
    def create_summary(self, all_results, output_dir):
        """Create comprehensive summary statistics grouped by pipeline and model"""
        # Aggregate by (pipeline, model)
        combo_stats = {}
        pipeline_stats = {}
        model_stats = {}
        
        for result in all_results:
            for gen_key, data in result['generations'].items():
                pipeline = data['pipeline']
                model = data['model']
                
                # Combined stats
                if gen_key not in combo_stats:
                    combo_stats[gen_key] = {
                        'pipeline': pipeline,
                        'model': model,
                        'clip_scores': [],
                        'blip_scores': [],
                        'visual_avg': [],
                        'code_validity': [],
                        'code_structure': [],
                        'code_optimization': [],
                        'code_readability': [],
                        'code_overall': [],
                        'overall_scores': []
                    }
                
                combo_stats[gen_key]['clip_scores'].append(data['clip_score'])
                combo_stats[gen_key]['blip_scores'].append(data['blip_score'])
                combo_stats[gen_key]['visual_avg'].append(data['visual_avg'])
                combo_stats[gen_key]['code_validity'].append(data['code_quality']['validity'])
                combo_stats[gen_key]['code_structure'].append(data['code_quality']['structure'])
                combo_stats[gen_key]['code_optimization'].append(data['code_quality']['optimization'])
                combo_stats[gen_key]['code_readability'].append(data['code_quality']['readability'])
                combo_stats[gen_key]['code_overall'].append(data['code_quality']['overall'])
                combo_stats[gen_key]['overall_scores'].append(data['overall_score'])
                
                # Pipeline-level aggregation
                if pipeline not in pipeline_stats:
                    pipeline_stats[pipeline] = {
                        'clip_scores': [],
                        'blip_scores': [],
                        'overall_scores': [],
                        'code_overall': []
                    }
                
                pipeline_stats[pipeline]['clip_scores'].append(data['clip_score'])
                pipeline_stats[pipeline]['blip_scores'].append(data['blip_score'])
                pipeline_stats[pipeline]['overall_scores'].append(data['overall_score'])
                pipeline_stats[pipeline]['code_overall'].append(data['code_quality']['overall'])
                
                # Model-level aggregation
                if model not in model_stats:
                    model_stats[model] = {
                        'clip_scores': [],
                        'blip_scores': [],
                        'overall_scores': [],
                        'code_overall': []
                    }
                
                model_stats[model]['clip_scores'].append(data['clip_score'])
                model_stats[model]['blip_scores'].append(data['blip_score'])
                model_stats[model]['overall_scores'].append(data['overall_score'])
                model_stats[model]['code_overall'].append(data['code_quality']['overall'])
        
        # Calculate statistics for combinations
        summary = {'combinations': {}, 'pipelines': {}, 'models': {}}
        
        for combo_key, stats in combo_stats.items():
            summary['combinations'][combo_key] = {
                'pipeline': stats['pipeline'],
                'model': stats['model'],
                'count': len(stats['clip_scores']),
                'visual_similarity': {
                    'clip_mean': float(np.mean(stats['clip_scores'])),
                    'clip_std': float(np.std(stats['clip_scores'])),
                    'blip_mean': float(np.mean(stats['blip_scores'])),
                    'blip_std': float(np.std(stats['blip_scores'])),
                },
                'code_quality': {
                    'overall_mean': float(np.mean(stats['code_overall'])),
                    'overall_std': float(np.std(stats['code_overall']))
                },
                'combined_score': {
                    'mean': float(np.mean(stats['overall_scores'])),
                    'std': float(np.std(stats['overall_scores']))
                }
            }
        
        # Pipeline-level summary
        for pipeline, stats in pipeline_stats.items():
            summary['pipelines'][pipeline] = {
                'count': len(stats['clip_scores']),
                'clip_mean': float(np.mean(stats['clip_scores'])),
                'blip_mean': float(np.mean(stats['blip_scores'])),
                'code_mean': float(np.mean(stats['code_overall'])),
                'overall_mean': float(np.mean(stats['overall_scores'])),
                'overall_std': float(np.std(stats['overall_scores']))
            }
        
        # Model-level summary
        for model, stats in model_stats.items():
            summary['models'][model] = {
                'count': len(stats['clip_scores']),
                'clip_mean': float(np.mean(stats['clip_scores'])),
                'blip_mean': float(np.mean(stats['blip_scores'])),
                'code_mean': float(np.mean(stats['code_overall'])),
                'overall_mean': float(np.mean(stats['overall_scores'])),
                'overall_std': float(np.std(stats['overall_scores']))
            }
        
        # Save JSON
        summary_path = output_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY BY PIPELINE")
        print("="*80)
        
        for pipeline, stats in summary['pipelines'].items():
            print(f"\n{pipeline.upper().replace('-', ' ')}:")
            print(f"  Samples: {stats['count']}")
            print(f"  CLIP:    {stats['clip_mean']:.3f}")
            print(f"  BLIP:    {stats['blip_mean']:.3f}")
            print(f"  Code:    {stats['code_mean']:.3f}")
            print(f"  Overall: {stats['overall_mean']:.3f} ± {stats['overall_std']:.3f}")
        
        print("\n" + "="*80)
        print("SUMMARY BY MODEL")
        print("="*80)
        
        for model, stats in summary['models'].items():
            print(f"\n{model.upper()}:")
            print(f"  Samples: {stats['count']}")
            print(f"  CLIP:    {stats['clip_mean']:.3f}")
            print(f"  BLIP:    {stats['blip_mean']:.3f}")
            print(f"  Code:    {stats['code_mean']:.3f}")
            print(f"  Overall: {stats['overall_mean']:.3f} ± {stats['overall_std']:.3f}")
        
        print(f"\nSummary saved to: {summary_path}")
    
    def plot_pipeline_comparison(self, pipeline_stats, model_stats, combo_stats, output_dir):
        """Create comprehensive comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Multi-Agent Pipeline Evaluation Results', fontsize=18, fontweight='bold')
        
        # 1. Pipeline Comparison
        ax = axes[0, 0]
        pipelines = list(pipeline_stats.keys())
        x = np.arange(len(pipelines))
        
        overall_means = [np.mean(pipeline_stats[p]['overall_scores']) for p in pipelines]
        overall_stds = [np.std(pipeline_stats[p]['overall_scores']) for p in pipelines]
        
        bars = ax.bar(x, overall_means, yerr=overall_stds, capsize=5, alpha=0.8, 
                     color=['#3498db', '#e74c3c', '#2ecc71'][:len(pipelines)])
        ax.set_title('Overall Score by Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('-', '\n') for p in pipelines], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, overall_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Model Comparison
        ax = axes[0, 1]
        models = list(model_stats.keys())
        x = np.arange(len(models))
        
        model_means = [np.mean(model_stats[m]['overall_scores']) for m in models]
        model_stds = [np.std(model_stats[m]['overall_scores']) for m in models]
        
        bars = ax.bar(x, model_means, yerr=model_stds, capsize=5, alpha=0.8,
                     color=['#9b59b6', '#f39c12', '#1abc9c'][:len(models)])
        ax.set_title('Overall Score by Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, model_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. CLIP vs BLIP by Pipeline
        ax = axes[0, 2]
        x = np.arange(len(pipelines))
        width = 0.35
        
        clip_means = [np.mean(pipeline_stats[p]['clip_scores']) for p in pipelines]
        blip_means = [np.mean(pipeline_stats[p]['blip_scores']) for p in pipelines]
        
        ax.bar(x - width/2, clip_means, width, label='CLIP', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, blip_means, width, label='BLIP', color='#e74c3c', alpha=0.8)
        
        ax.set_title('Visual Similarity by Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('-', '\n') for p in pipelines], fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Code Quality by Pipeline
        ax = axes[1, 0]
        code_means = [np.mean(pipeline_stats[p]['code_overall']) for p in pipelines]
        
        bars = ax.bar(x, code_means, alpha=0.8, color='#16a085')
        ax.set_title('Code Quality by Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Code Quality Score')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('-', '\n') for p in pipelines], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, code_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 5. Detailed combination heatmap
        ax = axes[1, 1]
        
        # Create matrix: pipelines x models
        pipeline_list = sorted(set(combo_stats[k]['pipeline'] for k in combo_stats))
        model_list = sorted(set(combo_stats[k]['model'] for k in combo_stats))
        
        matrix = np.zeros((len(pipeline_list), len(model_list)))
        
        for i, pipeline in enumerate(pipeline_list):
            for j, model in enumerate(model_list):
                key = f"{pipeline}_{model}"
                if key in combo_stats:
                    matrix[i, j] = np.mean(combo_stats[key]['overall_scores'])
                else:
                    matrix[i, j] = np.nan
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(model_list)))
        ax.set_yticks(np.arange(len(pipeline_list)))
        ax.set_xticklabels([m.replace('-', '\n') for m in model_list], fontsize=9)
        ax.set_yticklabels([p.replace('-', '\n') for p in pipeline_list], fontsize=9)
        ax.set_title('Overall Score: Pipeline × Model', fontsize=14, fontweight='bold')
        
        # Add values
        for i in range(len(pipeline_list)):
            for j in range(len(model_list)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax)
    
        
        output_path = output_dir / 'pipeline_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison chart saved to: {output_path}")
        plt.close()


def main():
    """Main execution"""
    # Configuration
    ORIGINALS_CSV = 'svg_evaluation_package/svgx_samples_main_sources.csv'
    GENERATED_ROOT = 'dataset'  # UPDATE THIS - should contain input/ folder
    ORIGINAL_SVGS_PATH = 'dataset/SVGX-rendering-data'  # Pre-rendered PNG folder
    OUTPUT_DIR = 'outputs/pipeline_evaluation'
    
    print("="*80)
    print("SVG PIPELINE EVALUATION SYSTEM")
    print("="*80)
    print(f"\nOriginals CSV: {ORIGINALS_CSV}")
    print(f"Generated SVGs root: {GENERATED_ROOT}/input/")
    print(f"Original PNG renderings: {ORIGINAL_SVGS_PATH}/")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Initialize
    evaluator = SVGPipelineEvaluator(ORIGINALS_CSV, GENERATED_ROOT)
    
    # Process
    results = evaluator.process_all(output_dir=OUTPUT_DIR)
    
    print(f"Done! Check {OUTPUT_DIR} for results")
    print(f"   - summary_statistics.json: Detailed statistics")
    print(f"   - visualizations/: SVG comparisons (no graphs, just images + scores)")
    print(f"   - visualizations/: Individual SVG comparisons")


if __name__ == "__main__":
    main()
