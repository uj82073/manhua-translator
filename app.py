import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from deep_translator import GoogleTranslator
import easyocr
import io, os, textwrap, re, concurrent.futures, sys
from pdf2image import convert_from_bytes

# ==============================
# Paths & Config
# ==============================
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH = DEFAULT_FONT if os.path.exists(DEFAULT_FONT) else None

TEMP_DIR = "temp_pages"
BACKUP_DIR = "processed"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Poppler path auto-detect
POPLER_PATH = "/usr/bin" if os.name != "nt" else r"C:\path\to\poppler\bin"

# ==============================
# Helpers for cleaning & translation
# ==============================
def clean_text_for_translation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[~•·◦◆◇★☆…]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip("[](){}<>")
    return text

def polish_translation(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if len(text) > 2 and text[0].islower():
        text = text[0].upper() + text[1:]
    text = re.sub(r"\b(\w+)( \1){1,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return text

def translate_text(text):
    clean = clean_text_for_translation(text)
    if not clean:
        return "", clean
    elif re.match(r"^[A-Za-z0-9 .,!?'-]+$", clean):
        return polish_translation(clean), clean
    else:
        try:
            translated = GoogleTranslator(source='zh-CN', target='en').translate(clean)
            return polish_translation(translated), clean
        except Exception:
            return "[Error]", clean

# ==============================
# Bubble Detection
# ==============================
def detect_bubbles(image_pil):
    image = np.array(image_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500 and w * h < image.shape[0] * image.shape[1] * 0.5:
            bubbles.append((x, y, w, h))
    return bubbles

# ==============================
# Text fitting in bubbles
# ==============================
def fit_text_in_box(draw, text, box, max_font=28, min_font=12, padding=5):
    x, y, w, h = box
    if not FONT_PATH:
        font = ImageFont.load_default()
        return font, [text], font.getbbox("A")[3], 
    for font_size in range(max_font, min_font-1, -1):
        font = ImageFont.truetype(FONT_PATH, font_size)
        avg_char_width = sum(font.getbbox(c)[2]-font.getbbox(c)[0] for c in text)/max(len(text), 1)
        max_chars_per_line = max(int((w-2*padding)/(avg_char_width+1)), 1)
        lines = textwrap.wrap(text, width=max_chars_per_line)
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        total_height = line_height * len(lines)
        if total_height <= h - 2*padding:
            return font, lines, line_height
    font = ImageFont.truetype(FONT_PATH, min_font)
    avg_char_width = sum(font.getbbox(c)[2]-font.getbbox(c)[0] for c in text)/max(len(text), 1)
    max_chars_per_line = max(int((w-2*padding)/(avg_char_width+1)), 1)
    lines = textwrap.wrap(text, width=max_chars_per_line)
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    return font, lines, line_height

def draw_text_in_bubble(draw, text, bubble, padding=5):
    x, y, w, h = bubble
    if w <= 0 or h <= 0:
        return
    draw.rectangle([x, y, x+w, y+h], fill="white")
    font, lines, line_height = fit_text_in_box(draw, text, (x, y, w, h), padding=padding)
    total_height = line_height * len(lines)
    start_y = y + (h - total_height)//2
    for i, line in enumerate(lines):
        bbox2 = draw.textbbox((0,0), line, font=font)
        text_w = bbox2[2] - bbox2[0]
        text_x = x + (w - text_w)//2
        text_y = start_y + i*line_height
        draw.text((text_x, text_y), line, font=font, fill="black")

# ==============================
# Process one page
# ==============================
def process_page(fname, fbuf, reader, mode):
    image = Image.open(fbuf).convert("RGB")
    if mode.startswith("Fast"):
        max_dim = 1000
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            image = image.resize((int(image.width*ratio), int(image.height*ratio)), Image.LANCZOS)

    draw = ImageDraw.Draw(image)
    bubbles = detect_bubbles(image)
    results = reader.readtext(np.array(image))
    if not results:
        return fname, image, []

    mapped = []
    for bbox, text, prob in results:
        if text.strip():
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            closest_bubble = min(
                bubbles, key=lambda b: (b[0]-x_min)**2 + (b[1]-y_min)**2
            ) if bubbles else (x_min, y_min, bbox[1][0]-bbox[0][0], bbox[2][1]-bbox[0][1])
            mapped.append((closest_bubble, text))

    page_log = []
    translations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(translate_text, t): (bubble, t) for bubble, t in mapped}
        for future in concurrent.futures.as_completed(futures):
            bubble, original = futures[future]
            translated, cleaned = future.result()
            translations.append({"original": original, "cleaned": cleaned, "translated": translated})
            draw_text_in_bubble(draw, translated, bubble)

    # Save backup image
    backup_path = os.path.join(BACKUP_DIR, fname.replace("/", "_") + ".png")
    image.save(backup_path)

    return fname, image, translations

# ==============================
# Streamlit UI
# ==============================
st.title("🚀 Bubble-Aware Manhua Translator")

uploaded_files = st.file_uploader(
    "Upload Images or PDF",
    type=["jpg","png","jpeg","pdf"],
    accept_multiple_files=True
)

mode = st.radio("OCR Mode:", ["Fast (smaller images, quicker)", "Accurate (larger images, slower)"])

if 'processed_pages_dict' not in st.session_state:
    st.session_state['processed_pages_dict'] = {}
if 'translation_log' not in st.session_state:
    st.session_state['translation_log'] = []

if uploaded_files:
    try:
        reader = easyocr.Reader(['ch_sim','en'], gpu=(os.environ.get("USE_GPU","0")=="1"))
    except Exception:
        reader = easyocr.Reader(['ch_sim','en'], gpu=False)

    all_files = []

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            dpi = 100 if mode.startswith("Fast") else 200
            pdf_pages = convert_from_bytes(file.read(), dpi=dpi, poppler_path=POPLER_PATH)
            for i, page in enumerate(pdf_pages, start=1):
                fname = f"{file.name}_page{str(i).zfill(4)}.png"
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                buf.seek(0)
                all_files.append((fname, buf))
        else:
            all_files.append((file.name, file))

    # filter already processed
    all_files = [(fname, fbuf) for fname, fbuf in all_files if fname not in st.session_state['processed_pages_dict']]
    total_files = len(all_files)

    if total_files == 0:
        st.info("All pages already processed. You can download PDF below.")
    else:
        progress = st.progress(0)
        status_text = st.empty()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_page, fname, fbuf, reader, mode)
                for fname, fbuf in all_files
            ]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    fname, image, page_log = future.result()
                    st.session_state['processed_pages_dict'][fname] = image
                    st.session_state['translation_log'].extend(page_log)
                except Exception as e:
                    st.error(f"Error processing page {fname}: {e}")
                progress.progress(idx / total_files)
                status_text.text(f"Processed page {idx}/{total_files}")

    # Merge into PDF
    all_pages_sorted = [st.session_state['processed_pages_dict'][fname] for fname in sorted(st.session_state['processed_pages_dict'].keys())]
    if all_pages_sorted:
        pdf_buf = io.BytesIO()
        all_pages_sorted[0].save(pdf_buf, format="PDF", save_all=True, append_images=all_pages_sorted[1:])
        pdf_buf.seek(0)
        st.success("✅ All pages merged into PDF!")
        st.download_button(
            label="📕 Download Full Chapter PDF",
            data=pdf_buf,
            file_name="chapter_translated.pdf",
            mime="application/pdf"
        )

# Translation log
with st.expander("📝 Translation Log (Original → Clean → Translated)"):
    for entry in st.session_state['translation_log']:
        st.markdown(f"**Original:** {entry['original']}")
        st.markdown(f"**Cleaned:** {entry['cleaned']}")
        st.markdown(f"**Translated:** {entry['translated']}")
        st.markdown("---")
