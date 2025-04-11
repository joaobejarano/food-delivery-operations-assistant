# 🚀 Smart Operations Assistant – Food Delivery

This project is a **smart operations assistant MVP** for food delivery platforms. It predicts demand, identifies operational bottlenecks, and suggests intelligent actions using historical data, RAG (Retrieval-Augmented Generation), and GPT.

---

## 🎯 Purpose

Support logistics and operations teams in **making fast, informed decisions** using AI-powered forecasting and generative insights.

---

## 🧠 Tech Stack & Concepts

- Python + Pandas + Faker (simulated realistic data)
- **Streamlit** (interactive dashboard)
- **LangChain + ChromaDB** (RAG pipeline for contextual retrieval)
- **OpenAI GPT-4** (generating actionable suggestions)
- **Hugging Face Transformers** (embeddings)
- **Prophet** or **XGBoost** (time series forecasting)
- **Plotly** (data visualization)

---

## ⚙️ Features

- 📦 View operational data: orders, delivery times, courier load, etc.
- 📈 Predict demand by hour and region
- 🧠 Generate GPT-4 insights based on retrieved context
- 🔍 Retrieve similar past operational incidents using RAG
- ☁️ Streamlit-powered visual interface

---

## 🗂️ Project Structure

```bash
smart-ops-assistant-food-delivery/
│
├── data/                # Simulated datasets
├── src/                 # Core logic: RAG, GPT, forecasting
├── streamlit_app/       # Frontend app
├── notebooks/           # Data exploration and experimentation
```

## ⚙️ How to run the project

1. Create your virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Create a .env from .env.example and add your OpenAI key:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
3. Ingest the incidents (generates the semantic vectors):

```bash
python src/rag/ingest.py
```

4. Run the application:

```bash
streamlit run streamlit_app/app.py --server.runOnSave false
```