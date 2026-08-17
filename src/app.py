import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import joblib
import shap
import numpy as np


@st.cache_resource
def load_results():
    return joblib.load("results/results.pkl")


# Load the results from the saved file
results = load_results()
evaluation_results = results["evaluation_results"]
shap_results = results["shap_results"]
feature_names = results["feature_names"]

st.set_page_config(page_title="Explainable AI Dashboard", layout="wide")

# CSS
css = f"""
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size:1.5rem;
        }}
    </style>
"""

st.markdown("""
    <style>

    .performance-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }

    .performance-table th {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        padding: 12px;
        background-color: #f0f2f6;
    }

    .performance-table td {
        font-size: 16px;
        text-align: center;
        padding: 10px;
    }
            
    .select-class-label {
        font-size: 20px;
        font-weight: bold;
        margin-top: 4px;
    }
            
    .st-key-class_selector {
        background-color: #EAF4FF;
        border: 1px solid #C8DFFF;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 15px;
    }
                   
    </style>
    """, unsafe_allow_html=True)


# Dashboard
st.header("🧠 Explainable AI Dashboard")
st.markdown("⚖️ Compare machine learning models using predictive performance metrics and SHAP explanations.")
st.markdown(css, unsafe_allow_html=True)
tab1, tab2 = st.tabs(
    [
        "📊 Model Comparison",
        "🔍 Dataset Exploration"
    ]
)

# MODE 1: Select 1 Dataset and Compare between three ML Models
with tab1:
    st.subheader("📁 Dataset")
    dataset = st.selectbox(
        label="",
        options=list(evaluation_results.keys()),
        label_visibility="collapsed"
    )

    st.header(f"📊 {dataset} Analysis")
    st.write("Compare predictive performance between different machine learning models")

    # Performance Matrix
    evaluation = []
    for model_name, metrics in evaluation_results[dataset].items():
        evaluation.append(
            {
                "Model": model_name,
                "Accuracy": f"{metrics['Accuracy']:.2%}",
                "Precision": f"{metrics['Precision']:.2%}",
                "Recall": f"{metrics['Recall']:.2%}",
                "F1-score": f"{metrics['F1-score']:.2%}"
            }
        )
    evaluation = pd.DataFrame(evaluation)
    html = evaluation.to_html(index=False, classes="performance-table")
    st.markdown(html, unsafe_allow_html=True)

    # Confusion Matrix
    st.subheader("📋 Confusion Matrix")
    cols = st.columns(3)
    for col, (model_name, metrics) in zip(cols, evaluation_results[dataset].items()):
        with col:
            st.markdown(f"**{model_name}**")
            fig, ax = plt.subplots(figsize=(4,4))

            num_classes = metrics["Confusion Matrix"].shape[0]
            if num_classes > 2:
                class_labels = list(range(1, num_classes + 1))
            else:
                class_labels = list(range(num_classes))

            sb.heatmap(metrics["Confusion Matrix"],annot=True,fmt="d",cmap="Blues",xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

    for model_name, shap_dict in shap_results["Fake News"].items():
        print(model_name)
        print(type(shap_dict["SHAP Values"]))
        print(shap_dict["SHAP Values"].shape)

    # SHAP Summary Plot
    st.subheader("🌟 SHAP Summary")

    # Get one SHAP object to determine the number of classes
    sample_shap = next(iter(shap_results[dataset].values()))["SHAP Values"]
    
    class_idx = None
    if len(sample_shap.shape) == 3:
        with st.container(key="class_selector"):
            col1, col2 = st.columns([0.5,5])
            with col1:
                st.markdown('<div class="select-class-label">Select Class:</div>',unsafe_allow_html=True)
            with col2:
                selected_class = st.radio(
                    "",
                    options=range(1, sample_shap.shape[2] + 1),
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"class_{dataset}"
                )
            class_idx = selected_class - 1
    
    cols = st.columns(3)
    for col, (model_name, shap_dict) in zip(cols, shap_results[dataset].items()):
        with col:
            st.markdown(f"**{model_name}**")
            shap_values = shap_dict["SHAP Values"]
            
            if dataset == "Medical":
                shap_plot = shap_values[:, :, class_idx]  

            # Fake News Random Forest
            elif len(shap_values.shape) == 3: 
                shap_plot = shap_values[:, :, 0]        

            # Logistic Regression & XGBoost
            else:
                shap_plot = shap_values

            fig = plt.figure(figsize=(5,4))
            shap.summary_plot(shap_plot, feature_names=feature_names[dataset], show=False)

            st.pyplot(fig)
            plt.close(fig)
    
    # TOP 10 Important Features
    st.subheader("🔝 Top 10 Important Features")
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_plot.values).mean(axis=0)
    feature_df = pd.DataFrame({
        "Feature": feature_names[dataset],
        "Importance": importance
    })
    top10 = feature_df.sort_values(by="Importance", ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(top10["Feature"][::-1], top10["Importance"][::-1])

    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_ylabel("")
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.pyplot(fig)
    plt.close(fig)



# MODE 2: Choose 1 ML Model and show results between both datasets
with tab2:

    st.subheader("🤖 Model")

    model = st.radio(
        label="",
        options=list(evaluation_results[dataset].keys()),
        horizontal=True,
        label_visibility="collapsed"
    )

    medical_metrics = evaluation_results["Medical"][model]
    fake_metrics = evaluation_results["Fake News"][model]

    comparison = pd.DataFrame({

        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],

        "Medical": [
            f"{medical_metrics['Accuracy']:.2%}",
            f"{medical_metrics['Precision']:.2%}",
            f"{medical_metrics['Recall']:.2%}",
            f"{medical_metrics['F1-score']:.2%}"
        ],

        "Fake News": [
            f"{fake_metrics['Accuracy']:.2%}",
            f"{fake_metrics['Precision']:.2%}",
            f"{fake_metrics['Recall']:.2%}",
            f"{fake_metrics['F1-score']:.2%}"
        ]
    })
    
    html = comparison.to_html(index=False, classes="performance-table")
    st.markdown(html, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.subheader("🩺 Medical")
    with right:
        st.subheader("📰 Fake News")

    # Confusion Matrix
    st.subheader("📋 Confusion Matrix")
    col1, col2 = st.columns(2)
    for col, dataset_name in zip([col1, col2], ["Medical", "Fake News"]):
        with col:
            cm = evaluation_results[dataset_name][model]["Confusion Matrix"]
            num_classes = cm.shape[0]
            if num_classes > 2:
                class_labels = list(range(1, num_classes + 1))
            else:
                class_labels = list(range(num_classes))

            fig, ax = plt.subplots(figsize=(4,4))

            sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

    # SHAP Summary
    st.subheader("🌟 SHAP Summary")
    # Get one SHAP object to determine the number of classes
    sample_shap = shap_results["Medical"][model]["SHAP Values"]

    class_idx = None
    if len(sample_shap.shape) == 3:
        col1, col2 = st.columns([0.5,5])
        with col1:
            st.markdown("Select Class:")
        with col2:
            selected_class1 = st.radio(
            "",
            options=range(1, sample_shap.shape[2] + 1),
            horizontal=True,
            label_visibility="collapsed",
            key=f"class_{model}"
            )
        class_idx = selected_class1 - 1

    col1, col2 = st.columns(2)
    for col, dataset_name in zip([col1, col2], ["Medical", "Fake News"]):
        with col:
            shap_values = shap_results[dataset_name][model]["SHAP Values"]
            fig = plt.figure(figsize=(5,4))
            
            if dataset_name == "Medical":
                shap_plot = shap_values[:, :, class_idx]  

            # Fake News Random Forest
            elif len(shap_values.shape) == 3: 
                shap_plot = shap_values[:, :, 0]        

            # Logistic Regression & XGBoost
            else:
                shap_plot = shap_values

            shap.summary_plot(shap_plot, feature_names=feature_names[dataset_name], show=False
            )
            st.pyplot(fig)
            plt.close(fig)
