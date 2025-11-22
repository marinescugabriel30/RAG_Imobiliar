import json
import numpy as np

# LOAD COMPARABLES

def load_comparables(path="comparables.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# FAIR PRICE ESTIMATION

def compute_fair_price(comparables, target_price=None, target_sqm=None):
    """
    comparables: lista cu id, final_score, price_per_sqm, size_sqm etc.
    target_price: pretul proprietatii analizate (din cererea userului)
    target_sqm: suprafata proprietatii (din cererea userului)
    """

    # Daca nu avem suprafata in query, luam media comparabilelor
    if target_sqm is None:
        target_sqm = np.mean([c["size_sqm"] for c in comparables])

    # Calculam media ponderata €/mp
    weights = np.array([c["final_score"] for c in comparables])
    prices_ppsqm = np.array([c["price_per_sqm"] for c in comparables])

    fair_ppsqm = np.sum(prices_ppsqm * weights) / np.sum(weights)
    fair_price = fair_ppsqm * target_sqm

    # Interval ±5%
    lower = fair_price * 0.95
    upper = fair_price * 1.05

    # Verdict
    verdict = None
    if target_price is None:
        verdict = "UNKNOWN"
    else:
        if target_price < lower:
            verdict = "UNDERPRICED"
        elif target_price > upper:
            verdict = "OVERPRICED"
            pass
        else:
            verdict = "FAIR"

    return {
        "fair_price": round(float(fair_price), 2),
        "fair_ppsqm": round(float(fair_ppsqm), 2),
        "confidence_interval": {
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2)
        },
        "verdict": verdict,
        "target_sqm": target_sqm
    }

# MAIN FUNCTION

def evaluate_property(
    comparables_path="comparables.json",
    target_price=None,
    target_sqm=None,
    output_path="pricing_output.json"
):
    comparables = load_comparables(comparables_path)

    estimation = compute_fair_price(
        comparables,
        target_price=target_price,
        target_sqm=target_sqm
    )

    output = {
        "estimation": estimation,
        "comparables_used": comparables
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved pricing report to {output_path}")
    return output

# TEST

if __name__ == "__main__":
    # Exemplu pentru cererea 1: apartament 2 camere, 54 mp, 60.000 euro
    evaluate_property(
        comparables_path="comparables.json",
        target_price=60000,
        target_sqm=54,
        output_path="pricing_output.json"
    )
