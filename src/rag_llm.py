import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


RULES_PATH = "data/fraud_rag_rules.csv"
SCORES_PATH = "outputs/scores.csv"
DATA_PATH = "processed/X_test.csv"


def categorize_risk(score):
    if score < 0.03:
        return "LOW"
    elif score < 0.07:
        return "MEDIUM"
    else:
        return "HIGH"


def load_data():
    X = pd.read_csv(DATA_PATH)
    scores = pd.read_csv(SCORES_PATH)
    rules = pd.read_csv(RULES_PATH)
    return X, scores, rules


def build_rule_texts(rules_df):
    return (
        "Source: " + rules_df["source"].astype(str) +
        " | Category: " + rules_df["category"].astype(str) +
        " | Topic: " + rules_df["topic"].astype(str) +
        " | Rule: " + rules_df["text"].astype(str)
    ).tolist()


def retrieve_rules(query_text, embed_model, rule_embs, rules_df, k=3):
    q_emb = embed_model.encode([query_text])[0]
    sims = cosine_similarity([q_emb], rule_embs)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return rules_df.iloc[top_idx]


def call_llm(prompt):

    client = OpenAI(
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        res = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fraud detection assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return res.choices[0].message.content or "Empty LLM response"

    except Exception as e:
        return f"LLM ERROR: {str(e)}"

def process_transaction(idx, X, scores, rules, embed_model, rule_embs):

    tx = X.iloc[idx]
    sc = scores.iloc[idx]

    risk_level = categorize_risk(sc["final_score"])

    tx_text = f"""
    Transaction amount: {tx.get('TransactionAmt','N/A')}
    Isolation Forest score: {sc['iso_score']:.4f}
    XGBoost fraud probability: {sc['xgb_score']:.4f}
    Autoencoder error: {sc['ae_score']:.4f}
    Ensemble risk score: {sc['final_score']:.4f}
    """

    retrieved = retrieve_rules(tx_text, embed_model, rule_embs, rules)

    retrieved_text = "\n".join(
        f"- [{r.source}] {r.text}"
        for r in retrieved.itertuples()
    )

    prompt = f"""
Risk Level: {risk_level}

Transaction:
{tx_text}

Relevant Rules:
{retrieved_text}

Explain why this transaction is {risk_level} risk.
"""

    explanation = call_llm(prompt)

    return {
        "id": idx,
        "amount": tx.get("TransactionAmt", 0),
        "risk": risk_level,
        "score": float(sc["final_score"]),
        "explanation": explanation
    }

