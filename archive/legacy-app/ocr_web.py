"""
Unlimited-OCR Web 界面
用法: python ocr_web.py 然后在浏览器打开 http://localhost:5000
"""
import os, sys, re, json, time, uuid, glob, shutil, tempfile, threading
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# --- 配置 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/PaddlePaddle/Unlimited-OCR')
WORK_DIR = os.path.join(os.path.dirname(__file__), 'ocr_workspace')
os.makedirs(WORK_DIR, exist_ok=True)

# 任务状态存储
tasks = {}
task_lock = threading.Lock()

# 懒加载模型
model = None
tokenizer = None
model_lock = threading.Lock()


def get_model():
    global model, tokenizer
    if model is None:
        with model_lock:
            if model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer
                torch.cuda.empty_cache()
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    MODEL_PATH, trust_remote_code=True,
                    use_safetensors=True, torch_dtype=torch.bfloat16,
                ).eval().cuda()
    return model, tokenizer


def do_ocr(task_id, pdf_path, start_page, end_page, dpi):
    """后台OCR线程"""
    try:
        import fitz, torch
        tasks[task_id]['status'] = 'preparing'
        tasks[task_id]['message'] = '正在将PDF转为图片...'

        # PDF → images
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        end_page = min(end_page, total_pages - 1)

        img_dir = os.path.join(WORK_DIR, task_id, 'images')
        os.makedirs(img_dir, exist_ok=True)
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        tasks[task_id]['total'] = end_page - start_page + 1
        page_images = []

        for i in range(start_page, end_page + 1):
            page = doc[i]
            out = os.path.join(img_dir, f'p{i:04d}.png')
            page.get_pixmap(matrix=mat).save(out)
            page_images.append(out)
        doc.close()

        # 加载模型
        tasks[task_id]['status'] = 'loading'
        tasks[task_id]['message'] = '正在加载OCR模型...'
        model, tokenizer = get_model()

        # 逐页OCR
        output_md = os.path.join(WORK_DIR, task_id, 'result.md')
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# OCR Result\n\n> {os.path.basename(pdf_path)} | DPI: {dpi} | Pages: {start_page+1}-{end_page+1}\n\n")

        for idx, img_path in enumerate(page_images):
            tasks[task_id]['status'] = 'processing'
            tasks[task_id]['done'] = idx
            tasks[task_id]['message'] = f'正在OCR第 {start_page+idx+1} 页 / {end_page+1} 页...'

            temp_dir = os.path.join(WORK_DIR, task_id, 'temp')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            model.infer(
                tokenizer,
                prompt='<image>document parsing.',
                image_file=img_path,
                output_path=temp_dir,
                base_size=1024, image_size=640, crop_mode=True,
                max_length=4096,
                no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )

            result_file = os.path.join(temp_dir, 'result.md')
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as rf:
                    raw = rf.read()
                clean = re.sub(r'<\|det\|>[^<]+<\|/det\|>', '', raw).strip()
                with open(output_md, 'a', encoding='utf-8') as f:
                    f.write(f"**[第{start_page+idx+1}页]**\n\n{clean}\n\n---\n\n")

            torch.cuda.empty_cache()

        tasks[task_id]['status'] = 'done'
        tasks[task_id]['done'] = tasks[task_id]['total']
        tasks[task_id]['message'] = 'OCR 完成！'
        tasks[task_id]['result_file'] = output_md

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)
        import traceback
        traceback.print_exc()


# --- 路由 ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files.get('pdf')
    if not file:
        return jsonify({'error': 'No file'}), 400

    task_id = str(uuid.uuid4())[:8]
    task_dir = os.path.join(WORK_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    pdf_path = os.path.join(task_dir, 'input.pdf')
    file.save(pdf_path)

    # 获取页数
    import fitz
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    return jsonify({
        'task_id': task_id,
        'filename': file.filename,
        'pages': page_count,
    })


@app.route('/api/start', methods=['POST'])
def start():
    data = request.json
    task_id = data['task_id']
    start_page = int(data.get('start_page', 0))
    end_page = int(data.get('end_page', 0))
    dpi = int(data.get('dpi', 200))

    pdf_path = os.path.join(WORK_DIR, task_id, 'input.pdf')

    tasks[task_id] = {
        'status': 'queued',
        'message': '排队中...',
        'done': 0,
        'total': 0,
        'result_file': None,
    }

    thread = threading.Thread(target=do_ocr, args=(task_id, pdf_path, start_page, end_page, dpi))
    thread.daemon = True
    thread.start()

    return jsonify({'ok': True})


@app.route('/api/progress/<task_id>')
def progress(task_id):
    task = tasks.get(task_id, {})
    return jsonify({
        'status': task.get('status', 'unknown'),
        'message': task.get('message', ''),
        'done': task.get('done', 0),
        'total': task.get('total', 0),
    })


@app.route('/api/result/<task_id>')
def result(task_id):
    task = tasks.get(task_id)
    if not task or not task.get('result_file'):
        return jsonify({'error': 'Not ready'}), 404
    return send_file(task['result_file'], as_attachment=True, download_name='ocr_result.md')


@app.route('/api/preview/<task_id>')
def preview(task_id):
    task = tasks.get(task_id)
    if not task or not task.get('result_file'):
        return jsonify({'error': 'Not ready'}), 404
    with open(task['result_file'], 'r', encoding='utf-8') as f:
        return jsonify({'content': f.read()})


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════╗
║   Unlimited-OCR Web Interface       ║
║   打开浏览器访问: http://localhost:5001  ║
╚══════════════════════════════════════╝
""")
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=False)
