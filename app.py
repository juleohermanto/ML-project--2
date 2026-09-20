try:
    import spaces
except ImportError:
    class DummySpaces:
        def GPU(self):
            def decorator(fn):
                return fn
            return decorator
    class DummySpaces:
        def GPU(self, fn=None, **kwargs):
            if fn is None:
                def wrapper(f): return f
                return wrapper
            return fn
    spaces = DummySpaces()

import gradio as gr
import pandas as pd
from api.model_utils import load_models, predict_pipeline

print("Initializing models for Gradio...")
load_models()

@spaces.GPU
def analyze_single_text(text):
    if not text.strip():
        return "Please enter some text.", "", ""
    try:
        res = predict_pipeline(text)
        aspects = ", ".join(res["aspects"]) if res["aspects"] else "None"
        confidence = f"{res['confidence'] * 100:.1f}%"
        return res["sentiment"], confidence, aspects
    except Exception as e:
        return f"Error: {e}", "", ""

def analyze_csv(file):
    try:
        df = pd.read_csv(file.name)
        if "review" not in df.columns:
            return "Error: CSV must contain a 'review' column.", None

        sentiments = []
        aspects_list = []

        for text in df["review"]:
            res = predict_pipeline(str(text))
            sentiments.append(res["sentiment"])
            aspects_list.append(", ".join(res["aspects"]))

        df["Predicted Sentiment"] = sentiments
        df["Predicted Aspects"] = aspects_list

        return "Processing Complete!", df
    except Exception as e:
        return f"Error: {e}", None

with gr.Blocks(title="E-Commerce AI Dashboard") as demo:
    gr.Markdown("# 🛍️ E-Commerce Review Categorizer (IndoBERT)")
    gr.Markdown("Analyze customer reviews in real-time. Hosted for free on Hugging Face Spaces!")

    with gr.Tab("Single Text Tester"):
        with gr.Row():
            text_input = gr.Textbox(lines=4, label="Type a review here", value="Bagus, namun ternyata tidak bagus")
        with gr.Row():
            analyze_btn = gr.Button("Analyze Review", variant="primary")
        with gr.Row():
            out_sentiment = gr.Textbox(label="Sentiment")
            out_confidence = gr.Textbox(label="Confidence")
            out_aspects = gr.Textbox(label="Aspects Detected")

        analyze_btn.click(
            fn=analyze_single_text,
            inputs=text_input,
            outputs=[out_sentiment, out_confidence, out_aspects]
        )

    with gr.Tab("Batch CSV Upload"):
        with gr.Row():
            file_input = gr.File(label="Upload CSV with a 'review' column", file_types=[".csv"])
        with gr.Row():
            csv_btn = gr.Button("Process CSV", variant="primary")
        with gr.Row():
            csv_status = gr.Textbox(label="Status")
        with gr.Row():
            csv_output = gr.Dataframe(label="Results")

        csv_btn.click(
            fn=analyze_csv,
            inputs=file_input,
            outputs=[csv_status, csv_output]
        )

if __name__ == "__main__":
    demo.launch()
