import os
import cv2
import yaml
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json
import pickle
from PIL import Image

# OCR Engines
import pytesseract
# Global availability flags will be set during engine initialization


class OCREngine(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AUTO = "auto"


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    bbox: Optional[List[int]] = None
    processing_time: float = 0.0
    language: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "bbox": self.bbox,
            "processing_time": round(self.processing_time, 3),
            "language": self.language
        }


class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    def __init__(self, config: Dict):
        self.config = config.get('preprocessing', {})
        self.enabled = self.config.get('enabled', True)
        
    def process(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
            
        original = image.copy()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Deskew
        if self.config.get('deskew', True):
            gray = self._deskew(gray)
            
        # Denoise
        if self.config.get('denoise', True):
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
        # Contrast enhancement (CLAHE)
        if self.config.get('contrast_enhance', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
        # Resize to target DPI if needed
        if self.config.get('target_dpi'):
            gray = self._set_dpi(gray, self.config['target_dpi'])
            
        return gray
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 100:  # Too few points to determine angle
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) < 0.5:  # Nearly straight
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def _set_dpi(self, image: np.ndarray, target_dpi: int) -> np.ndarray:
        """Resize image to approximate target DPI (assuming standard 96 DPI input)"""
        scale = target_dpi / 96
        if abs(scale - 1.0) > 0.1:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return image


class TesseractEngine:
    """Tesseract OCR wrapper with advanced features"""
    
    def __init__(self, config: Dict):
        self.config = config.get('tesseract', {})
        self.enabled = self.config.get('enabled', True)
        
        # Set Tesseract path if specified
        path = self.config.get('path')
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            
        self.languages = self.config.get('languages', ['eng'])
        self.lang_string = '+'.join(self.languages)
        self.psm_modes = self.config.get('psm_modes', {})
        
        # Verify installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized: v{version}")
            self.available = True
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            self.available = False
    
    def recognize(self, image: np.ndarray, 
                  psm_mode: str = 'auto',
                  oem: int = 3,
                  whitelist: Optional[str] = None) -> List[OCRResult]:
        if not self.available or not self.enabled:
            return []
            
        start_time = time.time()
        
        # Build config string
        config_parts = [f'--psm {self.psm_modes.get(psm_mode, 3)}']
        config_parts.append(f'--oem {oem}')  # OCR Engine Mode
        
        if whitelist:
            config_parts.append(f'-c tessedit_char_whitelist={whitelist}')
            
        config_string = ' '.join(config_parts)
        
        try:
            # Get detailed data including bounding boxes
            data = pytesseract.image_to_data(
                image, 
                lang=self.lang_string,
                config=config_string,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:  # Valid confidence
                    text = data['text'][i].strip()
                    if text:
                        bbox = [
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ]
                        results.append(OCRResult(
                            text=text,
                            confidence=float(data['conf'][i]) / 100,
                            engine='tesseract',
                            bbox=bbox,
                            processing_time=time.time() - start_time,
                            language=self.languages[0] if self.languages else 'unknown'
                        ))
                        
            # Also get full text as single result if no boxes found
            if not results:
                text = pytesseract.image_to_string(
                    image, 
                    lang=self.lang_string,
                    config=config_string
                ).strip()
                if text:
                    results.append(OCRResult(
                        text=text,
                        confidence=0.8,  # Default confidence for block text
                        engine='tesseract',
                        processing_time=time.time() - start_time
                    ))
                        
            return results
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return []
    
    def recognize_pdf(self, pdf_path: str, dpi: int = 300) -> List[OCRResult]:
        """OCR a PDF file by converting to images"""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi)
            
            all_results = []
            for i, page in enumerate(images):
                page_array = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                results = self.recognize(page_array)
                for r in results:
                    r.text = f"[Page {i+1}] {r.text}"
                all_results.extend(results)
                
            return all_results
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return []


class EasyOCREngine:
    """EasyOCR wrapper with caching"""
    
    def __init__(self, config: Dict):
        self.config = config.get('easyocr', {})
        # Try to import easyocr locally
        global EASYOCR_AVAILABLE
        try:
            import easyocr
            EASYOCR_AVAILABLE = True
        except ImportError:
            EASYOCR_AVAILABLE = False

        self.enabled = self.config.get('enabled', True) and EASYOCR_AVAILABLE
        self.gpu = self.config.get('gpu', False)
        self.languages = self.config.get('languages', ['en'])
        self.model_storage = self.config.get('model_storage', './models/easyocr')
        
        self.reader = None
        self.available = False
        
        if self.enabled:
            try:
                os.makedirs(self.model_storage, exist_ok=True)
                logger.info(f"Initializing EasyOCR with languages: {self.languages}")
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    model_storage_directory=self.model_storage,
                    download_enabled=True
                )
                self.available = True
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"EasyOCR initialization failed: {e}")
    
    def recognize(self, image: np.ndarray, 
                  detail: int = 1,
                  paragraph: bool = False) -> List[OCRResult]:
        if not self.available or not self.enabled:
            return []
            
        start_time = time.time()
        
        try:
            # EasyOCR expects RGB or path
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            results = self.reader.readtext(
                rgb_image,
                detail=detail,
                paragraph=paragraph
            )
            
            ocr_results = []
            for detection in results:
                if detail == 1:
                    bbox, text, conf = detection
                    # Convert bbox format
                    bbox_flat = [
                        int(min([p[0] for p in bbox])),
                        int(min([p[1] for p in bbox])),
                        int(max([p[0] for p in bbox]) - min([p[0] for p in bbox])),
                        int(max([p[1] for p in bbox]) - min([p[1] for p in bbox]))
                    ]
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(conf),
                        engine='easyocr',
                        bbox=bbox_flat,
                        processing_time=time.time() - start_time,
                        language='+'.join(self.languages)
                    ))
                else:
                    ocr_results.append(OCRResult(
                        text=str(detection),
                        confidence=0.9,
                        engine='easyocr',
                        processing_time=time.time() - start_time
                    ))
                    
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []


class PaddleOCREngine:
    """PaddleOCR wrapper optimized for accuracy"""
    
    def __init__(self, config: Dict):
        self.config = config.get('paddleocr', {})
        # Try to import paddleocr locally
        global PADDLEOCR_AVAILABLE
        try:
            from paddleocr import PaddleOCR
            PADDLEOCR_AVAILABLE = True
        except (ImportError, RuntimeError, Exception):
            PADDLEOCR_AVAILABLE = False

        self.enabled = self.config.get('enabled', True) and PADDLEOCR_AVAILABLE
        self.gpu = self.config.get('gpu', False)
        self.languages = self.config.get('languages', ['en'])
        self.model_storage = self.config.get('model_storage', './models/paddleocr')
        self.use_angle_cls = self.config.get('angle_classification', True)
        
        self.ocr = None
        self.available = False
        
        if self.enabled:
            try:
                os.makedirs(self.model_storage, exist_ok=True)
                logger.info(f"Initializing PaddleOCR with languages: {self.languages}")
                
                # Map language codes
                lang_map = {
                    'en': 'en', 'ch': 'ch', 'fr': 'french', 'de': 'german',
                    'ja': 'japan', 'ko': 'korean', 'es': 'en'  # Fallback for Spanish
                }
                
                # Use first language as primary
                primary_lang = lang_map.get(self.languages[0], 'en')
                
                self.ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=primary_lang,
                    show_log=False,
                    use_gpu=self.gpu,
                    det_model_dir=os.path.join(self.model_storage, 'det'),
                    rec_model_dir=os.path.join(self.model_storage, 'rec'),
                    cls_model_dir=os.path.join(self.model_storage, 'cls')
                )
                self.available = True
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"PaddleOCR initialization failed: {e}")
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if not self.available or not self.enabled:
            return []
            
        start_time = time.time()
        
        try:
            # Save temp image (PaddleOCR works better with file paths)
            temp_path = f"temp_paddle_{int(time.time())}.png"
            cv2.imwrite(temp_path, image)
            
            result = self.ocr.ocr(temp_path, cls=self.use_angle_cls)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            ocr_results = []
            if result and result[0]:
                for line in result[0]:
                    bbox_points, (text, confidence) = line
                    
                    # Calculate bounding box
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    bbox = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    ]
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(confidence),
                        engine='paddleocr',
                        bbox=bbox,
                        processing_time=time.time() - start_time,
                        language=self.languages[0]
                    ))
                    
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return []


class UnifiedOCR:
    """
    Main interface combining all three OCR engines with intelligent selection
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._init_logging()
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(self.config)
        
        # Initialize engines
        self.engines = {
            OCREngine.TESSERACT: TesseractEngine(self.config),
            OCREngine.EASYOCR: EasyOCREngine(self.config),
            OCREngine.PADDLEOCR: PaddleOCREngine(self.config)
        }
        
        # Cache setup
        self.cache_enabled = self.config.get('performance', {}).get('cache_results', True)
        self.cache_dir = Path(self.config.get('performance', {}).get('cache_dir', './cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Unified OCR Suite initialized")
        self._print_status()
    
    def _load_config(self, path: str) -> Dict:
        default_config = {
            'preprocessing': {'enabled': True},
            'performance': {'cache_results': True, 'cache_dir': './cache'},
            'logging': {'level': 'INFO'}
        }
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return {**default_config, **yaml.safe_load(f)}
        except Exception as e:
            logger.warning(f"Could not load config from {path}: {e}. Using defaults.")
            return default_config
    
    def _init_logging(self):
        log_config = self.config.get('logging', {})
        level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', './logs/ocr_suite.log')
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=level, colorize=True)
        logger.add(log_file, rotation="10 MB", level=level)
    
    def _print_status(self):
        print("\n" + "="*50)
        print("UNIFIED OCR SUITE STATUS")
        print("="*50)
        for engine_type, engine in self.engines.items():
            status = "✅ Ready" if engine.available else "❌ Unavailable"
            print(f"{engine_type.value.upper():12} : {status}")
        print("="*50 + "\n")
    
    def _get_cache_key(self, image: np.ndarray, engine: OCREngine) -> str:
        """Generate cache key from image data"""
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        return f"{engine.value}_{img_hash}.pkl"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[OCRResult]]:
        if not self.cache_enabled:
            return None
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, results: List[OCRResult]):
        if self.cache_enabled:
            cache_path = self.cache_dir / cache_key
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
    
    def recognize(self, 
                  image_input: Union[str, np.ndarray, Image.Image],
                  engine: OCREngine = OCREngine.AUTO,
                  preprocess: bool = True,
                  language: Optional[str] = None) -> List[OCRResult]:
        """
        Main OCR method with automatic engine selection
        
        Args:
            image_input: Path to image, numpy array, or PIL Image
            engine: Specific engine to use, or AUTO for smart selection
            preprocess: Whether to apply image preprocessing
            language: Target language hint for engine selection
        """
        # Load image
        image = self._load_image(image_input)
        if image is None:
            return []
        
        # Preprocess if requested
        if preprocess:
            image = self.preprocessor.process(image)
        
        # Determine which engine(s) to use
        engines_to_try = self._select_engines(engine, language, image)
        
        all_results = []
        for eng_type in engines_to_try:
            engine_instance = self.engines[eng_type]
            
            if not engine_instance.available:
                continue
            
            # Check cache
            cache_key = self._get_cache_key(image, eng_type)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for {eng_type.value}")
                all_results.extend(cached)
                continue
            
            # Run OCR
            logger.info(f"Running {eng_type.value}...")
            start = time.time()
            results = engine_instance.recognize(image)
            elapsed = time.time() - start
            
            if results:
                logger.info(f"{eng_type.value} completed in {elapsed:.2f}s with {len(results)} results")
                self._save_to_cache(cache_key, results)
                all_results.extend(results)
                
                # If AUTO mode and we got good results, stop here
                if engine == OCREngine.AUTO and len(results) > 0:
                    avg_conf = np.mean([r.confidence for r in results])
                    if avg_conf > 0.85:
                        logger.info(f"AUTO mode: {eng_type.value} provided high confidence results, stopping")
                        break
        
        # Sort by confidence
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results
    
    def recognize_best(self, 
                       image_input: Union[str, np.ndarray, Image.Image],
                       strategy: str = "confidence") -> OCRResult:
        """
        Get single best result using multiple engines and voting/confidence
        """
        results = self.recognize(image_input, engine=OCREngine.AUTO)
        
        if not results:
            return OCRResult(text="", confidence=0.0, engine="none", processing_time=0.0)
        
        if strategy == "confidence":
            return results[0]  # Highest confidence
        
        elif strategy == "voting":
            # Group similar texts and vote
            text_groups = {}
            for r in results:
                normalized = r.text.lower().strip()
                if normalized not in text_groups:
                    text_groups[normalized] = []
                text_groups[normalized].append(r)
            
            # Find group with most votes, then highest confidence within group
            best_group = max(text_groups.items(), key=lambda x: (len(x[1]), max(r.confidence for r in x[1])))
            return max(best_group[1], key=lambda x: x.confidence)
        
        return results[0]
    
    def _load_image(self, image_input) -> Optional[np.ndarray]:
        """Convert various inputs to numpy array"""
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    logger.error(f"Image not found: {image_input}")
                    return None
                image = cv2.imread(image_input)
            elif isinstance(image_input, np.ndarray):
                image = image_input
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                logger.error(f"Unsupported image type: {type(image_input)}")
                return None
                
            if image is None or image.size == 0:
                logger.error("Failed to load image")
                return None
                
            return image
        except Exception as e:
            logger.error(f"Image loading error: {e}")
            return None
    
    def _select_engines(self, 
                       engine: OCREngine, 
                       language: Optional[str],
                       image: np.ndarray) -> List[OCREngine]:
        """Intelligent engine selection"""
        
        if engine != OCREngine.AUTO:
            return [engine]
        
        # Auto-selection logic
        candidates = []
        
        # Check image characteristics
        is_color = len(image.shape) == 3 and image.shape[2] == 3
        height, width = image.shape[:2]
        is_low_res = height < 1000 or width < 1000
        
        # Language-based selection
        asian_langs = ['ch', 'japan', 'korean', 'chi_sim', 'chi_tra']
        is_asian = language in asian_langs if language else False
        
        if is_asian:
            # For Asian languages, prefer PaddleOCR or EasyOCR
            candidates = [
                OCREngine.PADDLEOCR,
                OCREngine.EASYOCR,
                OCREngine.TESSERACT
            ]
        elif is_color and not is_low_res:
            # For high-res color images (photos, screenshots), EasyOCR is often best
            candidates = [
                OCREngine.EASYOCR,
                OCREngine.PADDLEOCR,
                OCREngine.TESSERACT
            ]
        else:
            # For scanned documents, Tesseract is reliable
            candidates = [
                OCREngine.TESSERACT,
                OCREngine.EASYOCR,
                OCREngine.PADDLEOCR
            ]
        
        # Filter to available engines
        available = [e for e in candidates if self.engines[e].available]
        return available if available else [OCREngine.TESSERACT]
    
    def compare_engines(self, 
                        image_input: Union[str, np.ndarray, Image.Image],
                        preprocess: bool = True) -> Dict[str, List[OCRResult]]:
        """Run all available engines and compare results"""
        image = self._load_image(image_input)
        if image is None:
            return {}
        
        if preprocess:
            image = self.preprocessor.process(image)
        
        comparison = {}
        for eng_type in OCREngine:
            if eng_type == OCREngine.AUTO:
                continue
            
            engine = self.engines[eng_type]
            if engine.available:
                results = engine.recognize(image)
                comparison[eng_type.value] = results
        
        return comparison
    
    def batch_process(self, 
                      image_paths: List[str],
                      engine: OCREngine = OCREngine.AUTO,
                      output_file: Optional[str] = None) -> Dict[str, List[OCRResult]]:
        """Process multiple images with progress tracking"""
        from tqdm import tqdm
        
        results = {}
        for path in tqdm(image_paths, desc="Processing images"):
            results[path] = self.recognize(path, engine=engine)
        
        if output_file:
            # Save as JSON
            output_data = {
                path: [r.to_dict() for r in res] 
                for path, res in results.items()
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def get_engine_info(self) -> Dict:
        """Get status and capabilities of all engines"""
        info = {}
        for eng_type, engine in self.engines.items():
            info[eng_type.value] = {
                'available': engine.available,
                'enabled': engine.enabled,
                'languages': getattr(engine, 'languages', []),
                'gpu': getattr(engine, 'gpu', False)
            }
        return info
