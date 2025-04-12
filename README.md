# 🚀 Smart Operations Assistant – Food Delivery

This project is a **smart operations assistant MVP** for food delivery platforms. It predicts demand, identifies operational bottlenecks, and suggests intelligent actions using historical data, RAG (Retrieval-Augmented Generation), and GPT.

---

## 🎯 Purpose

Support logistics and operations teams in **making fast, informed decisions** using AI-powered forecasting and generative insights.

---

## 🧠 Tech Stack & Concepts

- **Python**, **Pandas**, **Faker** – realistic simulated datasets
- **Streamlit** – visual dashboard
- **LangChain + ChromaDB** – RAG for contextual search
- **OpenAI GPT-4** – generates insights based on incidents
- **Hugging Face Transformers** or **OpenAI Embeddings**
- **Prophet** or **XGBoost** – demand forecasting
- **Plotly** – interactive charts and visualizations

---

## ⚙️ Features

- 📦 Explore operational metrics: delivery times, regions, ratings, etc.
- 📈 Predict hourly demand per region (Prophet with fallback)
- 🧠 Generate AI-powered recommendations for managers
- 🔎 Retrieve and summarize past incidents using RAG
- ✅ Fully interactive dashboard built with Streamlit
- 💡 GPT-generated insights rendered in real time

---

## 🗂️ Project Structure

```bash
smart-ops-assistant-food-delivery/
├── data/                  # Simulated .csv datasets
├── notebooks/             # Exploratory notebooks and analysis
├── src/
│   ├── forecasting.py     # Forecast logic (Prophet)
│   └── rag/
│       ├── ingest.py      # Vector ingestion (ChromaDB)
│       ├── qa.py          # RAG-powered Q&A with GPT
│       └── gpt_insights.py # GPT-4 insight generator
├── streamlit_app/
│   └── app.py             # Streamlit UI
├── .env.example           # Example of environment config
└── README.md              # Project overview
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