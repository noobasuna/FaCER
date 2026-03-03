import torch
import clip
from PIL import Image
import os
import numpy as np
from typing import List, Dict, Tuple

class CounterfactualEvaluator:
    def __init__(self, device=None):
        """Initialize the CLIP-based counterfactual evaluator."""
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def compute_image_embedding(self, image_path: str) -> torch.Tensor:
        """Compute CLIP embedding for an image."""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)
        
    def compute_text_embedding(self, text_prompts: List[str]) -> torch.Tensor:
        """Compute CLIP embeddings for text prompts."""
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)
        
    def compute_similarity(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings."""
        return (100.0 * image_embedding @ text_embedding.T).softmax(dim=-1)
    
    def evaluate_counterfactual(self, 
                               original_image: str, 
                               counterfactual_image: str,
                               attribute_descriptions: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate counterfactual explanation by comparing semantic alignment between images and descriptions.
        
        Parameters:
        - original_image: Path to the original image
        - counterfactual_image: Path to the counterfactual image
        - attribute_descriptions: Dictionary mapping attribute names to lists of textual descriptions
                                 e.g., {"sunglasses": ["a person wearing sunglasses", "sunglasses on face"]}
        
        Returns:
        - Dictionary with evaluation metrics for each attribute
        """
        results = {}
        
        # Compute image embeddings
        original_embedding = self.compute_image_embedding(original_image)
        counterfactual_embedding = self.compute_image_embedding(counterfactual_image)
        
        for attr_name, descriptions in attribute_descriptions.items():
            # Compute positive and negative text embeddings
            positive_descriptions = descriptions
            negative_descriptions = [f"a person without {attr.replace('with ', '')}" if "with" in attr else f"a person with no {attr}" 
                                    for attr in descriptions]
            
            all_descriptions = positive_descriptions + negative_descriptions
            text_embeddings = self.compute_text_embedding(all_descriptions)
            
            # Split embeddings for positive and negative
            n_desc = len(positive_descriptions)
            pos_embeddings = text_embeddings[:n_desc]
            neg_embeddings = text_embeddings[n_desc:]
            
            # Calculate similarities
            orig_pos_sim = self.compute_similarity(original_embedding, pos_embeddings).mean().item()
            orig_neg_sim = self.compute_similarity(original_embedding, neg_embeddings).mean().item()
            cf_pos_sim = self.compute_similarity(counterfactual_embedding, pos_embeddings).mean().item()
            cf_neg_sim = self.compute_similarity(counterfactual_embedding, neg_embeddings).mean().item()
            
            # Calculate directional change
            orig_direction = orig_pos_sim - orig_neg_sim
            cf_direction = cf_pos_sim - cf_neg_sim
            
            # Calculate effectiveness score (did the counterfactual change the attribute in the intended direction?)
            direction_change = cf_direction - orig_direction
            
            results[attr_name] = {
                "original_positive_similarity": orig_pos_sim,
                "original_negative_similarity": orig_neg_sim,
                "counterfactual_positive_similarity": cf_pos_sim,
                "counterfactual_negative_similarity": cf_neg_sim,
                "direction_change": direction_change
            }
            
        return results
    
    def visualize_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Simple visualization of the evaluation results."""
        print("\n===== Counterfactual Evaluation Results =====")
        
        for attr, metrics in results.items():
            print(f"\nAttribute: {attr}")
            print(f"  Original image - positive similarity: {metrics['original_positive_similarity']:.4f}")
            print(f"  Original image - negative similarity: {metrics['original_negative_similarity']:.4f}")
            print(f"  Counterfactual - positive similarity: {metrics['counterfactual_positive_similarity']:.4f}")
            print(f"  Counterfactual - negative similarity: {metrics['counterfactual_negative_similarity']:.4f}")
            print(f"  Direction change: {metrics['direction_change']:.4f}")
            
            if metrics['direction_change'] < -0.1:
                effectiveness = "Strong change in intended direction"
            elif metrics['direction_change'] < -0.05:
                effectiveness = "Moderate change in intended direction"
            elif metrics['direction_change'] > 0.05:
                effectiveness = "Change in opposite direction"
            else:
                effectiveness = "Minimal or no meaningful change"
                
            print(f"  Effectiveness: {effectiveness}")


if __name__ == "__main__":
    # Example usage
    evaluator = CounterfactualEvaluator()
    
    # Example paths (user should replace with actual paths)
    original_image = "path/to/original_image.jpg"
    counterfactual_image = "path/to/counterfactual_image.jpg"
    
    # Example attribute descriptions focusing on smile and young
    attribute_descriptions = {
        "smile": [
            "a person with a smile", 
            "a smiling face",
            "a portrait with a happy expression"
        ],
        "young": [
            "a young person",
            "a face with youthful features", 
            "a portrait of a young individual"
        ]
    }
    
    # Skip example evaluation if files don't exist
    if not os.path.exists(original_image) or not os.path.exists(counterfactual_image):
        print("Example image paths not found. Please replace with your actual image paths.")
    else:
        results = evaluator.evaluate_counterfactual(
            original_image, 
            counterfactual_image,
            attribute_descriptions
        )
        evaluator.visualize_results(results)
