import os
import csv
from clip_explainer import CLIPExplainer

# Directories
original_dir = "/home/tpei0009/ACE/celebhq_s_5k_new/Original/Correct"
cf_dir = "/home/tpei0009/ACE/celebhq_s_5k_new/Results/CelebAHQ/explanation/CC/CCF/CF"

# Initialize explainer
explainer = CLIPExplainer()

results = []

for filename in os.listdir(original_dir):
    orig_path = os.path.join(original_dir, filename)
    cf_path = os.path.join(cf_dir, filename)
    if os.path.exists(cf_path):
        explanations = explainer.explain_transformation(orig_path, cf_path, top_k=1)
        # Find the highest similarity class probability
        top_score = -float('inf')
        top_category = None
        top_caption = ""
        for category, caption_scores in explanations.items():
            caption, score = caption_scores[0]
            if score > top_score:
                top_score = score
                top_category = category
                top_caption = caption
        results.append({
            "filename": filename,
            "category": top_category,
            "caption": top_caption,
            "score": top_score
        })
        print(f"{filename}: {top_category} - {top_caption} ({top_score:.2f}%)")

# Save results to a CSV file
csv_path = "highest_similarity_results.csv"
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "category", "caption", "score"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f"Results saved to {csv_path}")

# Print LaTeX table
print("\\begin{tabular}{l l l r}")
print("Filename & Category & Caption & Score \\ \\hline")
for row in results:
    print(f"{row['filename']} & {row['category']} & {row['caption']} & {row['score']:.2f} \\")
print("\\end{tabular}")
