import streamlit as st
import easyocr
from pdf2image import convert_from_bytes
from PIL import Image
import io
import concurrent.futures
import traceback

# === Constants ===
POPLER_PATH = "/usr/bin"  # Update if poppler is in a different location

# === Helper function: Process a single page ===
def process_page(fname, fbuf, reader, mode):
    logs = []
    try:
        image = Image.open(fbuf).convert("RGB")
        result = reader.readtext(image)

        translated_texts = []
        for bbox, text, prob in result:
            try:
                translated = text  # TODO: Replace with actual translator if needed
                translated_texts.append((bbox, translated))
            except Exception:
                logs.append(f"[ERROR] Translation failed for '{text}': {traceback.format_exc()}")

        # TODO: replace Chinese text in image with translated text
        # Currently returning original image
        return fname, image, logs
    except Exception:
        logs.append(f"[ERROR] OCR failed on {fname}: {traceback.format_exc()}")
        raise

# === Streamlit App ===
st.set_page_config(page_title="Comic Translator", layout="wide")
st.title("📖 Comic Translator")

uploaded_files = st.file_uploader("Upload PDF or images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    try:
        reader = easyocr.Reader(['ch_sim','en'], gpu=True)
        all_files = []

        for file in uploaded_files:
            if file.name.lower().endswith(".pdf"):
                try:
                    dpi = 100 if st.session_state.get("mode", "Accurate").startswith("Fast") else 200
                    pdf_pages = convert_from_bytes(file.read(), dpi=dpi, poppler_path=POPLER_PATH)
                    for i, page in enumerate(pdf_pages, start=1):
                        fname = f"{file.name}_page{str(i).zfill(4)}.png"
                        buf = io.BytesIO()
                        page.save(buf, format="PNG")
                        buf.seek(0)
                        all_files.append((fname, buf))
                except Exception:
                    st.error("❌ PDF conversion failed. See logs for details.")
                    print(f"[DEBUG] PDF conversion error for {file.name}: {traceback.format_exc()}")
            else:
                all_files.append((file.name, file))

        if "processed_pages_dict" not in st.session_state:
            st.session_state['processed_pages_dict'] = {}
        if "translation_log" not in st.session_state:
            st.session_state['translation_log'] = []

        all_files = [(fname, fbuf) for fname, fbuf in all_files
                     if fname not in st.session_state['processed_pages_dict']]
        total_files = len(all_files)

        if total_files == 0:
            st.info("All pages already processed. You can download PDF below.")
        else:
            progress = st.progress(0)
            status_text = st.empty()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_page, fname, fbuf, reader, "Accurate")
                           for fname, fbuf in all_files]
                for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    try:
                        fname, image, page_log = future.result()
                        st.session_state['processed_pages_dict'][fname] = image
                        st.session_state['translation_log'].extend(page_log)
                    except Exception:
                        st.error(f"❌ Error processing page {fname}. See logs for details.")
                        print(f"[DEBUG] Processing error in {fname}: {traceback.format_exc()}")
                    progress.progress(idx / total_files)
                    status_text.text(f"Processed page {idx}/{total_files}")

        # Merge into PDF
        all_pages_sorted = [
            st.session_state['processed_pages_dict'][fname]
            for fname in sorted(st.session_state['processed_pages_dict'].keys())
        ]
        if all_pages_sorted:
            try:
                pdf_buf = io.BytesIO()
                all_pages_sorted[0].save(
                    pdf_buf, format="PDF", save_all=True, append_images=all_pages_sorted[1:]
                )
                pdf_buf.seek(0)
                st.success("✅ All pages merged into PDF!")
                st.download_button(
                    label="📕 Download Full Chapter PDF",
                    data=pdf_buf,
                    file_name="chapter_translated.pdf",
                    mime="application/pdf"
                )
            except Exception:
                st.error("❌ PDF merge failed. See logs for details.")
                print(f"[DEBUG] PDF merge error: {traceback.format_exc()}")

    except Exception:
        st.error("❌ Top-level error. See logs for details.")
        print(f"[DEBUG] Top-level error: {traceback.format_exc()}")
