"""
Example usage of the Offline OCR Suite
Demonstrates various features and API usage patterns
"""

from ocr_manager import UnifiedOCR, OCREngine
import os


def basic_ocr_example():
    """Basic OCR with automatic engine selection"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic OCR with Auto Engine Selection")
    print("="*60)
    
    ocr = UnifiedOCR()
    
    # Replace with your test image
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("Please create a test image or update the path")
        return
    
    results = ocr.recognize(image_path)
    
    print(f"\n✓ Found {len(results)} text regions:\n")
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. [{r.engine}] {r.text[:80]}...")
        print(f"   Confidence: {r.confidence:.2%} | Time: {r.processing_time:.3f}s\n")


def specific_engine_example():
    """Using a specific OCR engine"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Using Specific Engines")
    print("="*60)
    
    ocr = UnifiedOCR()
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Try each engine
    for engine in [OCREngine.TESSERACT, OCREngine.EASYOCR, OCREngine.PADDLEOCR]:
        print(f"\n--- {engine.value.upper()} ---")
        results = ocr.recognize(image_path, engine=engine)
        
        if results:
            best = max(results, key=lambda x: x.confidence)
            print(f"✓ {len(results)} results")
            print(f"Best: {best.text[:60]}... ({best.confidence:.2%})")
        else:
            print("❌ No results or engine unavailable")


def best_result_example():
    """Get single best result using voting"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Best Result with Voting")
    print("="*60)
    
    ocr = UnifiedOCR()
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Confidence-based
    best_conf = ocr.recognize_best(image_path, strategy="confidence")
    print(f"\n📊 Confidence Strategy:")
    print(f"Engine: {best_conf.engine}")
    print(f"Text: {best_conf.text[:100]}")
    print(f"Confidence: {best_conf.confidence:.2%}")
    
    # Voting-based
    best_vote = ocr.recognize_best(image_path, strategy="voting")
    print(f"\n🗳️  Voting Strategy:")
    print(f"Engine: {best_vote.engine}")
    print(f"Text: {best_vote.text[:100]}")
    print(f"Confidence: {best_vote.confidence:.2%}")


def compare_engines_example():
    """Compare all available engines"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Engine Comparison")
    print("="*60)
    
    ocr = UnifiedOCR()
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    comparison = ocr.compare_engines(image_path)
    
    print()
    for engine_name, results in comparison.items():
        print(f"\n{engine_name.upper()}:")
        print(f"  📝 Results: {len(results)}")
        
        if results:
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"  📊 Avg Confidence: {avg_conf:.2%}")
            print(f"  ⏱️  Time: {results[0].processing_time:.3f}s")
            print(f"  📄 Sample: {results[0].text[:60]}...")


def batch_processing_example():
    """Process multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)
    
    ocr = UnifiedOCR()
    
    # Create list of image paths
    import glob
    image_paths = glob.glob("*.png") + glob.glob("*.jpg")
    
    if not image_paths:
        print("⚠️  No images found in current directory")
        print("Add some .png or .jpg files to test batch processing")
        return
    
    print(f"\nProcessing {len(image_paths)} images...\n")
    
    results = ocr.batch_process(
        image_paths[:5],  # Process first 5
        engine=OCREngine.AUTO,
        output_file="batch_results.json"
    )
    
    print(f"\n✓ Processed {len(results)} images")
    print("Results saved to: batch_results.json")


def preprocessing_example():
    """Demonstrate preprocessing effects"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Preprocessing Comparison")
    print("="*60)
    
    ocr = UnifiedOCR()
    image_path = "test_image.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Without preprocessing
    print("\n--- Without Preprocessing ---")
    results_raw = ocr.recognize(image_path, preprocess=False)
    if results_raw:
        print(f"Results: {len(results_raw)}")
        print(f"Best confidence: {max(r.confidence for r in results_raw):.2%}")
    
    # With preprocessing
    print("\n--- With Preprocessing ---")
    results_processed = ocr.recognize(image_path, preprocess=True)
    if results_processed:
        print(f"Results: {len(results_processed)}")
        print(f"Best confidence: {max(r.confidence for r in results_processed):.2%}")


def engine_info_example():
    """Display engine information"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Engine Information")
    print("="*60)
    
    ocr = UnifiedOCR()
    info = ocr.get_engine_info()
    
    print()
    for engine, details in info.items():
        print(f"\n{engine.upper()}:")
        print(f"  Available: {'✅' if details['available'] else '❌'}")
        print(f"  Enabled: {'✅' if details['enabled'] else '❌'}")
        print(f"  Languages: {', '.join(details['languages'])}")
        print(f"  GPU: {'✅' if details['gpu'] else '❌'}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Offline OCR Suite - Example Usage")
    print("="*60)
    
    try:
        # Run all examples
        engine_info_example()
        
        # Uncomment individual examples as needed:
        # basic_ocr_example()
        # specific_engine_example()
        # best_result_example()
        # compare_engines_example()
        # batch_processing_example()
        # preprocessing_example()
        
        print("\n" + "="*60)
        print(" ✓ Examples Complete!")
        print("="*60)
        print("\nTip: Uncomment specific examples in the code to test them")
        print("Make sure to have test images in the current directory\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Dependencies are installed (pip install -r requirements.txt)")
        print("2. Tesseract is installed and in PATH")
        print("3. Test images are available")
