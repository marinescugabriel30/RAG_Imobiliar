import pandas as pd

# Load raw data

df = pd.read_csv("properties_raw.csv")

# Basic cleanup

# Remove duplicates by ID
df = df.drop_duplicates(subset=["id"])

# Remove rows with missing essential fields
essential_cols = ["price_eur", "size_sqm", "property_type", "city", "neighborhood"]
df = df.dropna(subset=essential_cols)

# Convert numeric columns safely
num_cols = ["price_eur", "size_sqm", "rooms", "year_built",
            "floor", "max_floor", "dist_to_metro_min", "dist_to_park_min"]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove absurd values (basic sanity checks)
df = df[(df["price_eur"] > 5000) & (df["price_eur"] < 2_000_000)]
df = df[(df["size_sqm"] > 10) & (df["size_sqm"] < 2000)]

# Derived features

# Price per sqm
df["price_per_sqm"] = df["price_eur"] / df["size_sqm"]

# Age of the property (if house/apartment)
current_year = 2025
df["age"] = current_year - df["year_built"]

# Is new build (year >= 2015)
df["is_new_build"] = df["year_built"].apply(lambda y: 1 if y and y >= 2015 else 0)

# Distance score (simplified)
# lower distance = higher score
df["distance_score"] = (
    (20 - df["dist_to_metro_min"].clip(0, 20)) * 0.6 +
    (15 - df["dist_to_park_min"].clip(0, 15)) * 0.4
)

# Text document for embeddings

def build_text_embedding(row):
    return (
        f"{row['property_type']} in {row['neighborhood']}, "
        f"{row['city']}, {row['size_sqm']} mp, "
        f"{row['rooms']} camere, construit in {row['year_built']}. "
        f"Detalii: {row['description']}"
    )

df["text_for_embedding"] = df.apply(build_text_embedding, axis=1)

output_path = "properties_clean.csv"
df.to_csv(output_path, index=False)

print(f"Dataset curatat generat: {output_path}")
print(f"Numar final de proprietati: {len(df)}")
