#!/usr/bin/env python3
import argparse
import sys
from ocr_manager import UnifiedOCR, OCREngine


def main():
    parser = argparse.ArgumentParser(description="Offline OCR Suite - Multi-Engine OCR")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-e", "--engine", choices=['tesseract', 'easyocr', 'paddleocr', 'auto'],
                       default='auto', help="OCR engine to use")
    parser.add_argument("-o", "--output", help="Output file for results (JSON)")
    parser.add_argument("-l", "--language", help="Target language (e.g., 'en', 'ch')")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip image preprocessing")
    parser.add_argument("--compare", action="store_true", help="Compare all engines")
    parser.add_argument("--best", action="store_true", help="Return only best result")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = UnifiedOCR()
    
    # Map engine string to enum
    engine_map = {
        'tesseract': OCREngine.TESSERACT,
        'easyocr': OCREngine.EASYOCR,
        'paddleocr': OCREngine.PADDLEOCR,
        'auto': OCREngine.AUTO
    }
    selected_engine = engine_map[args.engine]
    
    # Process
    import os
    from glob import glob
    
    inputs = []
    if os.path.isdir(args.input):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.pdf')
        for ext in extensions:
            inputs.extend(glob(os.path.join(args.input, ext)))
    else:
        inputs = [args.input]
    
    if not inputs:
        print("No valid image files found")
        sys.exit(1)
    
    print(f"Processing {len(inputs)} file(s)...")
    
    if args.compare and len(inputs) == 1:
        # Compare mode
        comparison = ocr.compare_engines(inputs[0], preprocess=not args.no_preprocess)
        print("\n" + "="*60)
        print("ENGINE COMPARISON RESULTS")
        print("="*60)
        for eng_name, results in comparison.items():
            print(f"\n{eng_name.upper()}:")
            print(f"  Results: {len(results)}")
            if results:
                avg_conf = sum(r.confidence for r in results) / len(results)
                print(f"  Avg Confidence: {avg_conf:.2%}")
                print(f"  Time: {results[0].processing_time:.3f}s")
                print(f"  Sample: {results[0].text[:100]}...")
    else:
        # Normal processing
        results = ocr.batch_process(
            inputs, 
            engine=selected_engine,
            output_file=args.output
        )
        
        # Print summary
        for path, res_list in results.items():
            print(f"\n{path}:")
            if args.best and res_list:
                best = max(res_list, key=lambda x: x.confidence)
                print(f"  Best [{best.engine}]: {best.text[:100]}... (conf: {best.confidence:.2%})")
            else:
                for i, r in enumerate(res_list[:3]):  # Show top 3
                    print(f"  {i+1}. [{r.engine}] {r.text[:80]}... (conf: {r.confidence:.2%})")
    
    print(f"\nDone! Processed {len(inputs)} file(s).")


if __name__ == "__main__":
    main()
