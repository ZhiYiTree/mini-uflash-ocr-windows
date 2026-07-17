"""
Unlimited-OCR 批量处理
用法: 修改下方的 IMG_DIR 和 OUTPUT_FILE，然后运行
"""
import os, torch, sys, time, glob, re, json, shutil
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models/PaddlePaddle/Unlimited-OCR')

# ═══ 修改这里 ═══
IMG_DIR = r'D:/Python/OCR/images'        # 图片目录
OUTPUT_FILE = r'D:/Python/OCR/result.md'  # 输出文件
PROGRESS_FILE = r'D:/Python/OCR/progress.txt'

TEMP_DIR = os.path.join(BASE_DIR, 'ocr_temp_page')
os.makedirs(TEMP_DIR, exist_ok=True)

# Chapter boundaries (zero-indexed pages)
CHAPTERS = [
    ("第一章 进入近代后中华民族的磨难与抗争", 12, 48),
    ("第二章 不同社会力量对国家出路的早期探索", 48, 67),
    ("第三章 辛亥革命与君主专制制度的终结", 67, 90),
    ("第四章 中国共产党成立和中国革命新局面", 90, 119),
    ("第五章 中国革命的新道路", 119, 137),
    ("第六章 中华民族的抗日战争", 137, 167),
    ("第七章 为建立新中国而奋斗", 167, 193),
]

def clean_text(text):
    """Remove <|det|> tags and cleanup"""
    text = re.sub(r'<\|det\|>[^<]+<\|/det\|>', '', text)
    return text.strip()

print("Loading model...", flush=True)
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    use_safetensors=True, torch_dtype=torch.bfloat16,
).eval().cuda()
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

# Get sorted image list
all_images = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))
print(f"Found {len(all_images)} images", flush=True)

# Check progress
completed = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        completed = set(int(line.strip()) for line in f if line.strip())
    print(f"Resuming: {len(completed)} pages already done", flush=True)

# Initialize output file if needed
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 中国近现代史纲要（2023年版）\n\n")
        f.write("> 使用百度 Unlimited-OCR 提取\n\n")

start_time = time.time()
pages_done_this_run = 0

for ch_title, start, end in CHAPTERS:
    chapter_started = False
    
    for img_path in all_images:
        fname = os.path.basename(img_path)
        page_idx = int(fname[1:5])
        
        # Skip pages outside this chapter
        if page_idx < start or page_idx >= end:
            continue
        # Skip already done pages
        if page_idx in completed:
            continue
        
        # Start chapter header
        if not chapter_started:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n\n## {ch_title}\n\n")
            chapter_started = True
            print(f"\n{'='*60}", flush=True)
            print(f"Chapter: {ch_title} (pages {start+1}-{end})", flush=True)
            print(f"{'='*60}", flush=True)
        
        try:
            print(f"  Page {page_idx+1}...", end=' ', flush=True)
            t0 = time.time()
            
            # Clean temp dir
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            
            # Run OCR (model saves to output_path and returns None)
            model.infer(
                tokenizer,
                prompt='<image>document parsing.',
                image_file=img_path,
                output_path=TEMP_DIR,
                base_size=1024, image_size=640, crop_mode=True,
                max_length=4096,
                no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )
            
            elapsed = time.time() - t0
            
            # Read saved result
            result_md = os.path.join(TEMP_DIR, 'result.md')
            if os.path.exists(result_md):
                with open(result_md, 'r', encoding='utf-8') as f:
                    raw = f.read()
                clean = clean_text(raw)
            else:
                clean = ''
            
            if clean:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"**[第{page_idx+1}页]**\n\n{clean}\n\n---\n\n")
                with open(PROGRESS_FILE, 'a') as f:
                    f.write(f"{page_idx}\n")
                completed.add(page_idx)
                pages_done_this_run += 1
                
                preview = clean.split('\n')[0][:60]
                print(f"OK ({elapsed:.0f}s): {preview}", flush=True)
            else:
                print(f"SKIP ({elapsed:.0f}s)", flush=True)
                
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        
        # Clear CUDA cache periodically
        if pages_done_this_run % 5 == 0:
            torch.cuda.empty_cache()

total_elapsed = time.time() - start_time
print(f"\n{'='*60}", flush=True)
print(f"Done! {pages_done_this_run} pages in {total_elapsed/60:.1f} min", flush=True)
print(f"Output: {OUTPUT_FILE}", flush=True)
