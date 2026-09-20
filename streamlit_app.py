import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="E-Commerce Review Analyzer", layout="wide", page_icon="🛍️")

st.title("E-Commerce Review Categorizer & Sentiment Dashboard")
st.markdown("Analyze customer reviews in real-time. Connects to our FastAPI NLP Backend.")

tab1, tab2 = st.tabs(["Single Text Tester", "Batch CSV Upload"])

with tab1:
    st.subheader("Test a Single Review")
    user_input = st.text_area(
        "Type a review here:", 
        "Pengirimannya lama banget, tapi barangnya ori. Adminnya juga acuh."
    )

    if st.button("Analyze Review", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"text": user_input})

                if response.status_code == 200:
                    data = response.json()

                    col1, col2, col3 = st.columns(3)

                    sentiment_color = "" if data["sentiment"] == "Positive" else "" if data["sentiment"] == "Negative" else "🟡"

                    col1.metric("Sentiment", f"{sentiment_color} {data['sentiment']}")
                    col2.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
                    col3.metric("Aspects Detected", ", ".join(data["aspects"]))

                else:
                    st.error(f"API Error: {response.json().get('detail', response.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Please ensure your FastAPI server is running on http://127.0.0.1:8000!")

with tab2:
    st.subheader("Batch Review Processing & Analytics")
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'review_text' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'review_text' not in df.columns:
            st.error("Invalid CSV: The file must contain a column exactly named 'review_text'.")
        else:
            st.info(f"Loaded {len(df)} reviews. Ready to process.")

            if st.button("Run Batch Inference", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, row in df.iterrows():
                    text = str(row['review_text'])
                    try:
                        res = requests.post(API_URL, json={"text": text})
                        if res.status_code == 200:
                            data = res.json()
                            results.append({
                                "Review Text": text,
                                "Sentiment": data["sentiment"],
                                "Confidence": data["confidence"],
                                "Aspects": ", ".join(data["aspects"])
                            })
                    except requests.exceptions.ConnectionError:
                        st.error("API connection lost during batch processing!")
                        break

                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i+1} of {len(df)}...")

                if results:
                    results_df = pd.DataFrame(results)
                    st.success("Batch Processing Complete!")

                    st.markdown("---")
                    st.header("Batch Analytics Dashboard")

                    total = len(results_df)
                    pos_count = len(results_df[results_df['Sentiment'] == 'Positive'])
                    neg_count = len(results_df[results_df['Sentiment'] == 'Negative'])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Reviews Analyzed", total)
                    c2.metric("Positive Sentiment Split", f"{(pos_count/total)*100:.1f}%")
                    c3.metric("Negative Sentiment Split", f"{(neg_count/total)*100:.1f}%")

                    st.subheader("Most Frequent Complaint Categories (Negative Reviews Only)")
                    neg_df = results_df[results_df['Sentiment'] == 'Negative']

                    if not neg_df.empty:
                        all_aspects = []
                        for asp_str in neg_df['Aspects']:
                            all_aspects.extend([a.strip() for a in asp_str.split(',') if a.strip()])

                        aspect_counts = pd.Series(all_aspects).value_counts()
                        if 'General' in aspect_counts:
                            aspect_counts = aspect_counts.drop('General')

                        st.bar_chart(aspect_counts)
                    else:
                        st.info("Great news! No negative reviews found.")

                    st.subheader("High-Priority Reviews (Requires Immediate Action)")
                    st.markdown("These are reviews tagged as **Negative** and complain about **Customer Service** or **Product Quality**.")

                    priority_mask = (results_df['Sentiment'] == 'Negative') & \
                                    (results_df['Aspects'].str.contains("Customer Service|Product Quality", case=False))
                    priority_df = results_df[priority_mask]

                    if not priority_df.empty:
                        st.dataframe(priority_df, use_container_width=True)
                    else:
                        st.write("No severe priority reviews found. Here are standard negative reviews:")
                        st.dataframe(neg_df, use_container_width=True)
