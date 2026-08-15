import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1. Load CLIP model
# --------------------------------------------------

print("Loading CLIP model...")

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()

print("CLIP model loaded successfully!")


# --------------------------------------------------
# 2. Load one extracted video frame
# --------------------------------------------------

image_path = r"C:\Users\Admin\Documents\SSSC AI ML Cohort\Final Capstone Project - Vision seek AI\frames\frame_00000.jpg"

image = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# 3. Define natural-language queries
# --------------------------------------------------

text_queries = [
    "a person",
    "a car",
    "a road",
    "a building",
    "an outdoor scene",
    "an indoor scene"
]


# --------------------------------------------------
# 4. Prepare image + text for CLIP
# --------------------------------------------------

inputs = processor(
    text=text_queries,
    images=image,
    return_tensors="pt",
    padding=True
)


# --------------------------------------------------
# 5. Run CLIP
# --------------------------------------------------

with torch.no_grad():

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image

    probabilities = logits_per_image.softmax(dim=1)


# --------------------------------------------------
# 6. Display similarity results
# --------------------------------------------------

print("\nCLIP Similarity Results")
print("-" * 40)

results = []

for query, probability in zip(
    text_queries,
    probabilities[0]
):

    score = probability.item()

    results.append((query, score))


results.sort(
    key=lambda x: x[1],
    reverse=True
)


for query, score in results:

    print(
        f"{query:<25} {score:.4f}"
    )


print("\nBest Match:")
print(results[0][0])
