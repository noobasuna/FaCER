import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import os

class CounterfactualCaptioner:
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        """
        Initialize the CounterfactualCaptioner with a CLIP model
        
        Args:
            clip_model_name: The CLIP model variant to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # Pre-defined captions for different types of changes
        self.counterfactual_captions = {
            "smile": [
                "with a smile",
                "showing a happy expression",
                "with a cheerful smile",
                "with an upbeat expression",
                "with a grin",
                "with a beaming smile",
                "with no smile",
                "with a serious expression",
                "with a neutral expression"
            ],
            "age": [
                "looking younger",
                "with a youthful appearance",
                "with fewer wrinkles",
                "looking older",
                "with more wrinkles",
                "with aged features",
                "with mature features",
                "with youthful features"
            ],
            "emotion": [
                "showing a sad expression",
                "showing an angry expression",
                "showing a neutral expression",
                "showing a surprised expression",
                "showing a fearful expression"
            ],
            "hair": [
                "with longer hair",
                "with shorter hair",
                "with curly hair",
                "with straight hair",
                "with blonde hair",
                "with brown hair",
                "with black hair",
                "with red hair"
            ],
            "facial_features": [
                "with glasses",
                "without glasses",
                "with a beard",
                "without a beard",
                "with a mustache",
                "without a mustache",
                "with narrower eyes",
                "with wider eyes"
            ],
            # Add more categories as needed
        }
    
    def get_image_embedding(self, image_path: str) -> torch.Tensor:
        """
        Get CLIP embedding for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image embedding tensor
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def compute_embedding_difference(self, original_embedding: torch.Tensor, 
                                     modified_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the difference between original and modified image embeddings
        
        Args:
            original_embedding: Embedding of the original image
            modified_embedding: Embedding of the modified image
            
        Returns:
            Difference vector
        """
        return modified_embedding - original_embedding
    
    def find_best_matching_captions(self, 
                                    difference_embedding: torch.Tensor,
                                    caption_categories: List[str] = None,
                                    top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find captions that best describe the change between images
        
        Args:
            difference_embedding: The embedding difference vector
            caption_categories: Which categories of captions to consider (None = all)
            top_k: Number of top captions to return
            
        Returns:
            List of (caption, similarity_score) tuples
        """
        if caption_categories is None:
            caption_categories = list(self.counterfactual_captions.keys())
            
        all_captions = []
        for category in caption_categories:
            if category in self.counterfactual_captions:
                all_captions.extend(self.counterfactual_captions[category])
            
        text_inputs = torch.cat([
            clip.tokenize(["A person " + caption]) for caption in all_captions
        ]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # Normalize the difference embedding
        normalized_diff = difference_embedding / difference_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarities between the difference vector and caption embeddings
        similarities = (100.0 * normalized_diff @ text_features.T).squeeze()
        
        # Get top_k captions
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        results = []
        for i in top_indices:
            results.append((all_captions[i], similarities[i].item()))
            
        return results
    
    def compare_images_and_generate_caption(self, 
                                           original_image_path: str, 
                                           modified_image_path: str,
                                           caption_categories: List[str] = None,
                                           top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Compare two images and generate captions describing the changes
        
        Args:
            original_image_path: Path to the original image
            modified_image_path: Path to the modified image
            caption_categories: Which categories of captions to consider
            top_k: Number of top captions to return
            
        Returns:
            List of (caption, similarity_score) tuples
        """
        original_embedding = self.get_image_embedding(original_image_path)
        modified_embedding = self.get_image_embedding(modified_image_path)
        
        difference = self.compute_embedding_difference(original_embedding, modified_embedding)
        
        return self.find_best_matching_captions(difference, caption_categories, top_k)
    
    def add_custom_captions(self, category: str, captions: List[str]):
        """
        Add custom captions to an existing or new category
        
        Args:
            category: The category name
            captions: List of caption strings (without "A person" prefix)
        """
        if category not in self.counterfactual_captions:
            self.counterfactual_captions[category] = []
            
        self.counterfactual_captions[category].extend(captions)
    
    def generate_detailed_caption(self, original_image_path: str, modified_image_path: str) -> str:
        """
        Generate a detailed caption describing the counterfactual changes
        
        Args:
            original_image_path: Path to the original image
            modified_image_path: Path to the modified image
            
        Returns:
            A detailed caption string
        """
        # Get top matches from all categories
        matches = self.compare_images_and_generate_caption(
            original_image_path, modified_image_path, top_k=5
        )
        
        # Create a detailed caption
        if matches:
            top_match = matches[0][0]
            return f"This is an image of a person that has been modified to be {top_match}."
        else:
            return "This is a modified image of a person."


# Example usage
def main():
    # Initialize the captioner
    captioner = CounterfactualCaptioner()
    
    # Add some custom captions for smile and age-related attributes
    captioner.add_custom_captions("smile_variations", [
        "with a subtle smile",
        "with a slight smirk",
        "with a closed-mouth smile",
        "with a toothy grin",
        "with no visible smile"
    ])
    
    captioner.add_custom_captions("age_indicators", [
        "with more visible age lines",
        "with smoother skin texture",
        "with more defined facial features",
        "with a more youthful complexion",
        "with age-related facial characteristics"
    ])
    
    # Example paths (replace with actual image paths)
    original_image_path = "/home/tpei0009/ACE/celebhq_s_5k_new/Original/Correct/0000002.png"
    modified_image_path = "/home/tpei0009/ACE/celebhq_s_5k_new/Results/CelebAHQ/explanation/CC/CCF/CF/0000002.png"
    
    # Get descriptions of changes focusing on smile and age
    captions = captioner.compare_images_and_generate_caption(
        original_image_path, 
        modified_image_path,
        caption_categories=["smile", "age", "smile_variations", "age_indicators"],
        top_k=3
    )
    
    print("Top captions describing the changes:")
    for caption, score in captions:
        print(f"- {caption} (score: {score:.2f})")
    
    # Generate a more detailed caption
    detailed_caption = captioner.generate_detailed_caption(original_image_path, modified_image_path)
    print("\nDetailed caption:")
    print(detailed_caption)


if __name__ == "__main__":
    main()