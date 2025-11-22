import json
import subprocess

def generate_explanation_local(data):
    """Genereaza explicatia verdictului folosind un LLM local prin Ollama."""

    # Extractie date reale din JSON-ul generat la pasul 4
    est = data["estimation"]
    comparables = data["comparables_used"]

    fair_price = est["fair_price"]
    fair_min = est["confidence_interval"]["lower"]
    fair_max = est["confidence_interval"]["upper"]
    label = est["verdict"]

    target_title = data.get("title", "Proprietatea analizata")
    listed_price = data.get("listed_price_eur", "NECUNOSCUT")

    # Construire lista comparabile
    comparables_text = "\n".join([
        f"- {c['id']} ({c['neighborhood']}): "
        f"{c['price_per_sqm']} €/mp, {c['size_sqm']} mp, scor {c['final_score']:.2f}"
        for c in comparables
    ])

    # Prompt
    prompt = f"""
Esti un asistent care explica evaluari imobiliare, fara sa inventezi informatii.
Datele de mai jos provin dintr-un sistem RAG.

Proprietatea analizata:
{target_title}
Pret listat: {listed_price} €
Pret corect estimat: {fair_price} € (interval {fair_min}–{fair_max} €)
Verdict: {label}

Comparabile relevante:
{comparables_text}

Scrie o explicatie concisa (maxim 5 propozitii) care justifica verdictul.
Raspunsul trebuie sa fie in limba romana.
"""

    #Rulare LLM local (Mistral/Ollama)
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )

    explanation = result.stdout.decode("utf-8").strip()

    return {
        "explanation_text": explanation,
        "disclaimer": "Proiect educațional; nu reprezintă consultanță imobiliară."
    }


if __name__ == "__main__":
    # Testare locala
    with open("pricing_output.json") as f:
        data = json.load(f)
    result = generate_explanation_local(data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
