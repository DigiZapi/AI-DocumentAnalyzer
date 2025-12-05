import os
import base64
from typing import List, Dict
import requests
import time
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from config import (
        VISION_MODEL, OLLAMA_BASE_URL,
        MIN_IMAGE_SIZE, MAX_IMAGE_SIZE,
        CAPTION_RATE_LIMIT_DELAY, CAPTION_MAX_RETRIES, CAPTION_RETRY_DELAY
    )
    CAPTION_MODEL = VISION_MODEL
    OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
    RATE_LIMIT_DELAY = CAPTION_RATE_LIMIT_DELAY
    MAX_RETRIES = CAPTION_MAX_RETRIES
    RETRY_DELAY = CAPTION_RETRY_DELAY
except ImportError:
    # Fallback to hardcoded values if config not found
    CAPTION_MODEL = "qwen2.5vl:latest"
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    MIN_IMAGE_SIZE = 1000
    MAX_IMAGE_SIZE = 500_000
    RATE_LIMIT_DELAY = 0.5
    MAX_RETRIES = 2
    RETRY_DELAY = 2


def caption_images(image_infos: List[Dict], caption_model_name: str = CAPTION_MODEL) -> Dict[str, str]:
    """Generate captions for images using Qwen-VL via Ollama.
    
    Filters out small images (icons) and implements rate limiting
    to prevent overwhelming the Ollama server.

    Returns mapping {image_id: caption}.
    """
    print(f"Generating image captions with Qwen-VL via Ollama...")
    print(f"Model: {caption_model_name}")
    print(f"Number of images to process: {len(image_infos)}")
    print(f"Filtering: Images < {MIN_IMAGE_SIZE} bytes or > {MAX_IMAGE_SIZE} bytes will be skipped")

    captions: Dict[str, str] = {}
    skipped = 0
    processed = 0
    failed = 0

    for idx, info in enumerate(image_infos, 1):
        try:
            image_path = info["path"]
            
            # Check file size first
            if not os.path.exists(image_path):
                print(f"  ✗ Image file not found: {info['path']}")
                captions[info["id"]] = ""
                failed += 1
                continue
            
            file_size = os.path.getsize(image_path)
            
            # Skip tiny images (icons, buttons) and very large images
            if file_size < MIN_IMAGE_SIZE:
                print(f"⏭️  Skipping {idx}/{len(image_infos)}: {os.path.basename(image_path)} ({file_size} bytes - too small)")
                captions[info["id"]] = f"[Small image: {file_size} bytes]"
                skipped += 1
                continue
            
            if file_size > MAX_IMAGE_SIZE:
                print(f"⏭️  Skipping {idx}/{len(image_infos)}: {os.path.basename(image_path)} ({file_size} bytes - too large)")
                captions[info["id"]] = f"[Large image: {file_size} bytes]"
                skipped += 1
                continue
            
            print(f"\n🖼️  Processing {idx}/{len(image_infos)}: {os.path.basename(image_path)} ({file_size} bytes)")
            
            # Read and encode image to base64
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # Retry logic for failed requests
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    # Prepare request for Ollama API
                    payload = {
                        "model": caption_model_name,
                        "prompt": "Make a short description of the image in maximal 2 sentences.",
                        "images": [image_data],
                        "stream": False,
                        "options": {
                            "num_ctx": 2048,  # Reduce context to save memory
                            "temperature": 0.1
                        }
                    }
                    
                    # Call Ollama API with timeout
                    response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
                    response.raise_for_status()
                    
                    result = response.json()
                    caption = result.get("response", "").strip()
                    
                    if caption:
                        captions[info["id"]] = caption
                        print(f"  ✅ Caption: {caption[:80]}...")
                        processed += 1
                        success = True
                        break
                    else:
                        captions[info["id"]] = ""
                        print(f"  ⚠️  Empty caption returned")
                        processed += 1
                        success = True
                        break
                        
                except requests.exceptions.RequestException as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"  ⚠️  Attempt {attempt + 1} failed, retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"  ❌ All retries failed: {e}")
                        if hasattr(e, 'response') and e.response is not None:
                            print(f"     Status: {e.response.status_code}")
                        captions[info["id"]] = ""
                        failed += 1
                        # Wait longer after complete failure to let server recover
                        time.sleep(RETRY_DELAY * 2)
            
            # Rate limiting - give Ollama time to recover between requests
            if success:
                time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")
            captions[info["id"]] = ""
            failed += 1

    print(f"\n{'='*60}")
    print(f"✅ Captioning complete!")
    print(f"   Processed: {processed}/{len(image_infos)}")
    print(f"   Skipped: {skipped} (too small/large)")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {(processed/(len(image_infos)-skipped)*100) if (len(image_infos)-skipped) > 0 else 0:.1f}%")
    print(f"{'='*60}\n")
    
    return captions
