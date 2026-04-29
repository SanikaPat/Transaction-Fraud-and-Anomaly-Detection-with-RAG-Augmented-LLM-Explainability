from flask import Flask, render_template, request
import pandas as pd
from rag_llm import load_data, build_rule_texts, retrieve_rules, categorize_risk, call_llm, process_transaction
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

X, scores, rules = load_data()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
rule_texts = build_rule_texts(rules)
rule_embs = embed_model.encode(rule_texts)


def process_transaction(i, X, scores, rules, embed_model, rule_embs):
    tx = X.iloc[i]
    sc = scores.iloc[i]

    risk = categorize_risk(sc["final_score"])

    tx_text = f"""
    Amount: {tx.get('TransactionAmt','N/A')}
    ISO: {sc['iso_score']:.4f}
    XGB: {sc['xgb_score']:.4f}
    AE: {sc['ae_score']:.4f}
    Final: {sc['final_score']:.4f}
    """

    retrieved = retrieve_rules(tx_text, embed_model, rule_embs, rules)

    rules_text = "\n".join(
        f"- {r.source}: {r.text}"
        for r in retrieved.itertuples()
    )

    prompt = f"""
Risk Level: {risk}

Transaction:
{tx_text}

Rules:
{rules_text}

Explain fraud risk clearly.
"""

    explanation = call_llm(prompt)

    return {
        "id": i,
        "amount": tx.get("TransactionAmt", 0),
        "risk": risk,
        "score": round(sc["final_score"], 4),
        "explanation": explanation
    }


@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":
        indices = request.form.get("indices")

        try:
            idx_list = [int(x.strip()) for x in indices.split(",")]

            for i in idx_list[:3]:  # limit to 3
                results.append(process_transaction(i, X, scores, rules, embed_model, rule_embs))

        except:
            results = []

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)