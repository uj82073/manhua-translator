import traceback
import io

# --- Safe global try/except for startup errors ---
try:
    import streamlit as st
    import easyocr
    from pdf2image import convert_from_bytes
    from PIL import Image
    import concurrent.futures

    # === Debug Log Collector ===
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []

    def log(msg):
        st.session_state["debug_logs"].append(msg)
        print(msg)  # also goes to server logs

    # === Streamlit Setup ===
    st.set_page_config(page_title="Comic Translator (Debug)", layout="wide")
    st.title("📖 Comic Translator (Debug Mode)")

    uploaded_files = st.file_uploader(
        "Upload PDF or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
    )

    POPLER_PATH = "/usr/bin"  # adjust if needed

    def process_page(fname, fbuf, reader):
        try:
            image = Image.open(fbuf).convert("RGB")
            result = reader.readtext(image)

            # Dummy translation (identity)
            translated_texts = [(bbox, text) for bbox, text, prob in result]

            return fname, image, translated_texts, None
        except Exception:
            err = traceback.format_exc()
            log(f"[ERROR] Processing {fname}: {err}")
            return fname, None, None, err

    if uploaded_files:
        try:
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
            all_files = []

            for file in uploaded_files:
                if file.name.lower().endswith(".pdf"):
                    try:
                        dpi = 200
                        pdf_pages = convert_from_bytes(file.read(), dpi=dpi, poppler_path=POPLER_PATH)
                        for i, page in enumerate(pdf_pages, start=1):
                            fname = f"{file.name}_page{str(i).zfill(4)}.png"
                            buf = io.BytesIO()
                            page.save(buf, format="PNG")
                            buf.seek(0)
                            all_files.append((fname, buf))
                        log(f"✅ Converted {file.name} to {len(pdf_pages)} pages")
                    except Exception:
                        err = traceback.format_exc()
                        log(f"[ERROR] PDF conversion failed for {file.name}: {err}")
                        st.error("❌ PDF conversion failed. See logs below.")
                else:
                    all_files.append((file.name, file))

            st.info(f"📂 {len(all_files)} files ready for processing")

            processed_images = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_page, fname, fbuf, reader) for fname, fbuf in all_files]
                for future in concurrent.futures.as_completed(futures):
                    fname, image, texts, err = future.result()
                    if image:
                        processed_images.append(image)
                        st.image(image, caption=f"✅ {fname}", use_container_width=True)
                    else:
                        st.error(f"❌ Failed {fname}")

            # Merge into PDF if pages exist
            if processed_images:
                try:
                    pdf_buf = io.BytesIO()
                    processed_images[0].save(
                        pdf_buf, format="PDF", save_all=True, append_images=processed_images[1:]
                    )
                    pdf_buf.seek(0)
                    st.success("✅ Merged all pages into one PDF")
                    st.download_button(
                        "📕 Download Translated PDF", data=pdf_buf, file_name="chapter_translated.pdf", mime="application/pdf"
                    )
                except Exception:
                    err = traceback.format_exc()
                    log(f"[ERROR] PDF merge failed: {err}")
                    st.error("❌ PDF merge failed. See logs below.")

        except Exception:
            err = traceback.format_exc()
            log(f"[ERROR] Top-level processing error: {err}")
            st.error("❌ App crashed during processing. See logs below.")

    # === Debug Log Panel ===
    with st.expander("🐞 Debug Logs"):
        if st.session_state["debug_logs"]:
            for line in st.session_state["debug_logs"]:
                st.text(line)
        else:
            st.text("No logs yet.")

except Exception:
    import streamlit as st
    st.error("🔥 App crashed at startup. See logs below.")
    st.text(traceback.format_exc())
    print("🔥 Startup crash:\n", traceback.format_exc())
