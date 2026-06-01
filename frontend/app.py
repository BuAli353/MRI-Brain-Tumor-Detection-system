import io, os, base64
import numpy as np
import gradio as gr
import requests
from PIL import Image
from report_generator import generate_pdf_report

BACKEND  = "http://127.0.0.1:8000"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
body, .gradio-container { background-color: #1a1b26 !important; font-family: 'Inter', sans-serif !important; color: #e0e0e0 !important; }
footer { display: none !important; }

/* Sidebar */
#sidebar { background-color: #24283b; border-right: 1px solid #1a1b26; padding: 25px; border-radius: 12px; }
#sidebar h2 { color: #ffffff; font-size: 1.4rem; font-weight: 700; margin-bottom: 20px; }
#sidebar h3 { color: #ffffff; font-size: 1.05rem; font-weight: 600; margin-top: 30px; margin-bottom: 15px; }
.user-profile { font-size: 0.9rem; margin-bottom: 15px; color: #a9b1d6; display: flex; align-items: center; gap: 8px; }
.info-box { background-color: #1f2335; border: 1px solid #292e42; padding: 18px; border-radius: 8px; font-size: 0.85rem; color: #7aa2f7; margin-top: 25px; line-height: 1.6; }

/* Main layout */
#main-content { padding: 20px 30px; }
.section-title { color: #ffffff; font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }

/* Buttons */
button, .gradio-button { border-radius: 6px !important; font-weight: 600 !important; transition: all 0.2s ease !important; }
.blue-btn { background: linear-gradient(135deg, #3d59a1 0%, #7aa2f7 100%) !important; color: white !important; border: none !important; }
.blue-btn:hover { opacity: 0.9 !important; transform: translateY(-1px); }
#analyze-btn { background: linear-gradient(135deg, #3d59a1 0%, #7aa2f7 100%) !important; color: white !important; border: none !important; padding: 12px !important; font-size: 1.1rem !important; margin-top: 30px !important; width: 100% !important; border-radius: 8px !important; }
#analyze-btn:hover { opacity: 0.9 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(122,162,247,0.3); }

/* Images */
#input-image, #heatmap-image { background-color: #000000 !important; border-radius: 12px !important; overflow: hidden !important; border: none !important; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }

/* Results UI */
.diagnosis-box { padding: 15px 20px; border-radius: 8px; font-weight: 600; font-size: 1rem; margin-bottom: 25px; text-align: left; box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
.diagnosis-box.pending { background-color: #24283b; color: #a9b1d6; border: 1px solid #292e42; }
.diagnosis-box.benign { background-color: #2e3a36; color: #9ece6a; border: 1px solid #414868; }
.diagnosis-box.malignant { background-color: #4a2730; color: #f7768e; border: 1px solid #542c36; }

.conf-label { font-size: 0.85rem; color: #a9b1d6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.conf-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-top: 5px; margin-bottom: 12px; }

.progress-bar { width: 100%; height: 6px; background-color: #24283b; border-radius: 3px; overflow: hidden; margin-bottom: 35px; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #3d59a1 0%, #7aa2f7 100%); border-radius: 3px; transition: width 0.5s ease; }
.progress-bar.small { height: 4px; margin-bottom: 12px; }

.probs-box { background-color: #24283b; padding: 20px; border-radius: 12px; border: 1px solid #292e42; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.prob-row { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 8px; }
.prob-class { color: #a9b1d6; }
.prob-pct { color: #ffffff; font-weight: 600; }

.stats-container { display: flex; gap: 15px; }
.stat-box { background-color: #1a1b26; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #292e42; }
.stat-label { font-size: 0.75rem; color: #a9b1d6; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-size: 1.4rem; font-weight: 700; color: #ffffff; display: block; margin-top: 5px; }
"""

def classify(image, total_scans, model_choice):
    if image is None:
        return (
            "<div class='diagnosis-box pending'>Diagnosis: Pending Scan Upload</div>",
            "<div class='conf-value'>--%</div><div class='progress-bar'><div class='progress-fill' style='width: 0%;'></div></div>",
            None,
            "<div class='probs-box' style='margin-top:0;'><div style='text-align:center; color:#a9b1d6; font-size:0.85rem; padding: 20px 0;'>Upload scan to see class probabilities</div></div>",
            total_scans,
            gr.update(),
            gr.update(value=None, interactive=False)
        )
    try:
        buf = io.BytesIO()
        image.save(buf, "JPEG")
        buf.seek(0)
        r = requests.post(f"{BACKEND}/predict", files={"file": ("scan.jpg", buf, "image/jpeg")}, data={"model_type": model_choice}, timeout=15)
        r.raise_for_status()
        d = r.json()
        
        gcam = Image.open(io.BytesIO(base64.b64decode(d["gradcam_b64"])))
        lbl = d["label"]
        conf = d["confidence"] * 100
        
        if lbl == "No Tumor":
            diag_class = "benign"
            diag_text = "Diagnosis: No Tumor Detected"
        else:
            diag_class = "malignant"
            diag_text = f"Diagnosis: YES ({lbl} Detected)"
            
        diag_html = f"<div class='diagnosis-box {diag_class}'>{diag_text}</div>"
        conf_html_str = f"<div class='conf-value'>{conf:.2f}%</div><div class='progress-bar'><div class='progress-fill' style='width: {conf}%;'></div></div>"
        
        probs_str = "<div class='probs-box' style='margin-top:0;'>"
        for c, p in sorted(d["scores"].items(), key=lambda x: -x[1]):
            pct = p * 100
            probs_str += f"<div class='prob-row'><span class='prob-class'>{c}</span><span class='prob-pct'>{pct:.1f}%</span></div><div class='progress-bar small'><div class='progress-fill' style='width: {pct}%;'></div></div>"
        probs_str += "</div>"
        
        new_total = int(total_scans) + 1
        stats_html = f"<div class='stats-container'><div class='stat-box'><span class='stat-label'>Total Scans</span><span class='stat-value'>{new_total}</span></div><div class='stat-box'><span class='stat-label'>Last Conf.</span><span class='stat-value'>{conf:.1f}%</span></div></div>"
        
        # Generate dynamic clinical PDF report
        pdf_path = generate_pdf_report(
            original_pil=image,
            heatmap_pil=gcam,
            label=lbl,
            confidence=d["confidence"],
            scores=d["scores"],
            latency=d["latency_ms"],
            model_choice=model_choice
        )
        
        return diag_html, conf_html_str, gcam, probs_str, new_total, stats_html, gr.update(value=pdf_path, interactive=True)
        
    except Exception as e:
        return (
            f"<div class='diagnosis-box malignant'>Error: {str(e)}</div>",
            "<div class='conf-value'>--%</div><div class='progress-bar'><div class='progress-fill' style='width: 0%;'></div></div>",
            None,
            "",
            total_scans,
            gr.update(),
            gr.update(value=None, interactive=False)
        )

with gr.Blocks(title="Brain Tumor Detection", css=CSS) as demo:
    total_scans = gr.State(0)
    
    with gr.Row():
        # --- SIDEBAR ---
        with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
            gr.HTML("<h2>Brain Tumor Detection</h2>")
            gr.HTML("<div class='user-profile'>👤 User: Doctor</div>")
            gr.Button("Logout", elem_classes="small-btn blue-btn")
            
            gr.HTML("<div class='info-box'>This app runs predictions using our fine-tuned EfficientNetB2 model. It provides highly accurate classifications and Grad-CAM explainability.</div>")
            
            gr.HTML("<h3>Model Selection</h3>")
            model_dropdown = gr.Dropdown(choices=["EfficientNetB2", "YOLOv8 Segmentation"], value="EfficientNetB2", label="Choose models to run:", interactive=True)
            
            gr.HTML("<h3>Settings</h3>")
            gr.Slider(minimum=0, maximum=1, value=0.70, step=0.01, label="Confidence Threshold")
            
            gr.HTML("<h3>Session Stats</h3>")
            stats_panel = gr.HTML("<div class='stats-container'><div class='stat-box'><span class='stat-label'>Total Scans</span><span class='stat-value'>0</span></div><div class='stat-box'><span class='stat-label'>Last Conf.</span><span class='stat-value'>--</span></div></div>")
            
            analyze_btn = gr.Button("🔍 Analyze the Scan", elem_id="analyze-btn")

        # --- MAIN CONTENT ---
        with gr.Column(scale=3, elem_id="main-content"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>Uploaded Scan</div>")
                with gr.Column(scale=1):
                    with gr.Row():
                        gr.HTML("<div class='section-title'>🔬 Analysis Results</div>")
                        download_btn = gr.DownloadButton("📄 Download Report (PDF)", elem_classes="small-btn blue-btn", interactive=False)
            
            with gr.Row():
                # LEFT: Original Image
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", show_label=False, elem_id="input-image")
                
                # RIGHT: Results
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:white; font-size:1.2rem; margin-bottom:15px; margin-top:0;'>EfficientNetB2</h3>")
                    diag_out = gr.HTML("<div class='diagnosis-box pending'>Diagnosis: Pending Scan Upload</div>")
                    
                    gr.HTML("<div class='conf-label'>Model Confidence</div>")
                    conf_out = gr.HTML("<div class='conf-value'>--%</div><div class='progress-bar'><div class='progress-fill' style='width: 0%;'></div></div>")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML("<div class='conf-label' style='margin-bottom:10px;'>Model Attention Heatmap</div>")
                            gcam_out = gr.Image(show_label=False, elem_id="heatmap-image", interactive=False)
                            gr.HTML("<p style='text-align:center; color:#a9b1d6; font-size:0.8rem; margin-top:5px;'>Red areas show model focus.</p>")
                        
                        with gr.Column(scale=1):
                            gr.HTML("<div class='conf-label' style='margin-bottom:10px;'>Class Probabilities</div>")
                            probs_out = gr.HTML("<div class='probs-box' style='margin-top:0;'><div style='text-align:center; color:#a9b1d6; font-size:0.85rem; padding: 20px 0;'>Upload scan to see class probabilities</div></div>")

    analyze_btn.click(
        classify, 
        inputs=[img_in, total_scans, model_dropdown], 
        outputs=[diag_out, conf_out, gcam_out, probs_out, total_scans, stats_panel, download_btn]
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port)