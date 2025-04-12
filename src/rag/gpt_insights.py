import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_operational_insight(incident_df: pd.DataFrame, region: str = None) -> str:
    """
    Generates an operational suggestion for the manager using GPT-4,
    based on the most common incident types in the filtered data.
    """
    if incident_df.empty:
        return "No incident data provided."

    top_incidents = incident_df['incident_type'].value_counts().head(3).to_dict()

    incident_summary = "\n".join([
        f"- {incident_type}: {count} occurrences"
        for incident_type, count in top_incidents.items()
    ])

    context = "The following data represents the most frequent operational incidents recently recorded"
    if region:
        context += f" in the region(s) of **{region}**"

    prompt = f"""
{context}:

{incident_summary}

Based on this data, provide a **clear and practical suggestion** for the delivery operations manager.
The suggestion should be concise, data-driven, and follow logistics best practices.
Avoid generic answers and tailor it to the situation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a logistics and operations expert helping managers take action based on delivery data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating suggestion: {str(e)}"
