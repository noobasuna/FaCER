import pandas as pd
import numpy as np

# Read both CSV files
ground_truth = pd.read_csv('/home/tpei0009/ACE/celeba_align_raw.csv')
predictions = pd.read_csv('/home/tpei0009/ACE/celeba_ace.csv')

# Extract base file names
ground_truth['file_name'] = ground_truth['file_path'].apply(lambda x: x.split('/')[-1])
predictions['file_name'] = predictions['file_path'].apply(lambda x: x.split('/')[-1])

# Merge the dataframes on file_name to ensure we're comparing the same images
merged_df = pd.merge(ground_truth, 
                    predictions, 
                    on='file_name', 
                    suffixes=('_gt', '_pred'))

# Convert probabilities to binary predictions using 0 as threshold
gt_predictions = (merged_df['score_20_gt'] > 0).astype(int)
model_predictions = (merged_df['score_20_pred'] > 0).astype(int)
protected_attribute = (merged_df['score_33_gt'] > 0).astype(int)

def calculate_fairness_metrics(y_true, y_pred, protected):
    metrics = {}
    
    # Demographic Parity
    pred_rate_prot = y_pred[protected == 1].mean()
    pred_rate_nonprot = y_pred[protected == 0].mean()
    metrics['demographic_parity'] = abs(pred_rate_prot - pred_rate_nonprot)
    
    # Equalized Odds
    # True Positive Rate
    tpr_prot = np.mean((y_pred == 1)[protected == 1][y_true[protected == 1] == 1])
    tpr_nonprot = np.mean((y_pred == 1)[protected == 0][y_true[protected == 0] == 1])
    # False Positive Rate
    fpr_prot = np.mean((y_pred == 1)[protected == 1][y_true[protected == 1] == 0])
    fpr_nonprot = np.mean((y_pred == 1)[protected == 0][y_true[protected == 0] == 0])
    
    metrics['equalized_odds_tpr'] = abs(tpr_prot - tpr_nonprot)
    metrics['equalized_odds_fpr'] = abs(fpr_prot - fpr_nonprot)
    
    # Counterfactual Fairness (simplified version - comparing predictions across groups)
    cf_diff = abs(pred_rate_prot - pred_rate_nonprot)
    metrics['counterfactual_fairness'] = cf_diff
    
    return metrics

# Calculate fairness metrics
metrics = calculate_fairness_metrics(gt_predictions.values, 
                                   model_predictions.values,
                                   protected_attribute.values)

print("Fairness Metrics:")
print(f"Demographic Parity Difference: {metrics['demographic_parity']:.4f}")
print(f"Equalized Odds - TPR Difference: {metrics['equalized_odds_tpr']:.4f}")
print(f"Equalized Odds - FPR Difference: {metrics['equalized_odds_fpr']:.4f}")
print(f"Counterfactual Fairness Difference: {metrics['counterfactual_fairness']:.4f}")

# Print additional information
print("\nDataset Statistics:")
print(f"Total samples after merging: {len(merged_df)}")
print(f"Protected group (score_33 > 0) size: {sum(protected_attribute)}")
print(f"Non-protected group (score_33 <= 0) size: {sum(protected_attribute == 0)}")

# Print confusion matrix for verification
from sklearn.metrics import confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(gt_predictions, model_predictions))