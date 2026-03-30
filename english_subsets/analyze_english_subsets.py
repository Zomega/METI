import pandas as pd

# List of files
files = ["globish.csv", "ogdens_basic_english.csv", "upgoer5.csv", "voa_special_english.csv"]

# The 22 valid categories requested
allowed_categories = [
    "People", "Business", "Visual", "Nature", "Time", "Mathematics", 
    "Education", "Direction", "Political", "Body Parts", "Household", 
    "War", "Transport", "Materials", "Food/Drink", "Industry", 
    "Agriculture", "Tools", "Buildings", "Animals", "Clothes", "Color"
]

# Read each CSV file into a DataFrame
df_list = []
for f in files:
    df = pd.read_csv(f, names=["word", "category"])
    df["source"] = f.replace(".csv", "")
    df_list.append(df)

# (a) Words in all subsets (based on presence in the files)
word_sets = [set(df["word"].dropna().unique()) for df in df_list]
common_words = sorted(list(set.intersection(*word_sets)))

print(f"Number of words present in all 4 subsets: {len(common_words)}")
with open("common_words.txt", "w") as f:
    f.write("\n".join(common_words))
print("Full list of common words saved to 'common_words.txt'.")

# (b) Master list of categorizations
# Concatenate and filter to allowed categories
master_df = pd.concat(df_list, ignore_index=True)
filtered_df = master_df[master_df["category"].isin(allowed_categories)]

# Resolve to a single category and comma-separated sources for each word
def resolve_entry(group):
    # Category resolution: most frequent, then alphabetical
    cat_counts = group["category"].value_counts()
    max_cat_count = cat_counts.max()
    top_categories = cat_counts[cat_counts == max_cat_count].index.tolist()
    final_category = sorted(top_categories)[0]
    
    # Sources: unique sources, comma-separated string
    unique_sources = sorted(list(set(group["source"].astype(str))))
    final_sources = ", ".join(unique_sources)
    
    return pd.Series({
        "category": final_category,
        "sources": final_sources
    })

final_master = filtered_df.groupby("word").apply(resolve_entry, include_groups=False).reset_index()

# Save to master_categorization.csv
final_master.to_csv("master_categorization.csv", index=False)
print(f"\nFinal master categorization saved to 'master_categorization.csv'.")

# Save to data.json for the web app
final_master.to_json("data.json", orient="records")
print("Data exported to 'data.json' for the web explorer.")

print(f"Total unique words: {len(final_master)}")

# Show a sample
print("\nSample of final master categorization:")
print(final_master.head())

# Optional: verify counts
print("\nCategory counts in the final master list (total occurrences):")
print(filtered_df["category"].value_counts())
