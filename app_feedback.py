# --- 1. Installation ---
# pip install langchain langchain-community faiss-cpu torch torchvision transformers Pillow gradio sentence-transformers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
import gradio as gr

# LangChain and Model Imports
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoImageProcessor, AutoModel

# --- 2. Configuration---
class AppConfig:
    MODEL_NAME = "facebook/dinov2-base"
    DEVICE = "cpu" # Change to "cuda" if you have a GPU
    IMAGE_DB_PATH = "./images_full"
    FAISS_INDEX_PATH = "./faiss_index_full"
    TOP_K = 3
    # --- CHANGE: Added a similarity threshold ---
    SIMILARITY_THRESHOLD = 0.75

# --- 3. Custom Image Embedding Class ---
class DinoImageEmbeddings(Embeddings):
    """A custom LangChain Embeddings class for the DINOv2 model."""
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    def _embed_image(self, image_path: str) -> List[float]:
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Error opening or processing image file: {image_path}. Skipping. Error: {e}")
            return [0.0] * 768
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            normalized_embedding = self._normalize(embedding.cpu().numpy())
            return normalized_embedding.tolist()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_image(path) for path in texts]
    def embed_query(self, text: str) -> List[float]:
        return self._embed_image(text)

# --- 4. Helper Function to Prepare Image Database  ---
def setup_image_database(db_path: str) -> List[str]:
    if not os.path.exists(db_path):
        print(f"Database path '{db_path}' not found. Please check the path.")
        return []
    image_paths = []
    print(f"Scanning for images in '{db_path}'...")
    for root, _, files in os.walk(db_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    if not image_paths:
        print(f"No images found in '{db_path}'. Please add images.")
    else:
        print(f"Found {len(image_paths)} images across all subfolders.")
    return image_paths

# --- 5. Function to Load or Create the Vector Store ---
def load_or_create_vectorstore(image_paths: List[str], embedding_model: Embeddings, index_path: str) -> FAISS:
    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading existing FAISS index from: {index_path}")
        return FAISS.load_local(
            index_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        if not image_paths:
            return None
        print("No existing index found. Creating new FAISS index...")
        print(f"Embedding {len(image_paths)} images. This may take a while...")
        documents = []
        for path in image_paths:
            plant_name = os.path.basename(os.path.dirname(path))
            doc = Document(
                page_content=path,
                metadata={"plant_name": plant_name.replace("_", " ")}
            )
            documents.append(doc)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        print(f"Saving new FAISS index to: {index_path}")
        vectorstore.save_local(index_path)
        return vectorstore

# --- 6. Main Application Logic ---
print("Initializing application...")
print(f"Loading embedding model: {AppConfig.MODEL_NAME} on {AppConfig.DEVICE}")
embedding_model = DinoImageEmbeddings(model_name=AppConfig.MODEL_NAME, device=AppConfig.DEVICE)
image_paths = setup_image_database(AppConfig.IMAGE_DB_PATH)
vectorstore = load_or_create_vectorstore(image_paths, embedding_model, AppConfig.FAISS_INDEX_PATH)

# --- 7. Prediction and Feedback Functions ---
# --- CHANGE: The predict function is updated to handle the similarity threshold ---
def predict(query_image_path: str):
    """
    Finds similar images, returns results, and clears any old feedback messages.
    If the top similarity score is below a threshold, it returns 'Unknown'.
    """
    if not vectorstore:
        error_msg = {"Error": f"Vector store not available. Check DB at '{AppConfig.IMAGE_DB_PATH}'."}
        return error_msg, None, gr.update(visible=False), ""

    # Search for similar images
    docs_and_scores = vectorstore.similarity_search_with_score(query_image_path, k=AppConfig.TOP_K)

    # If no results are found, return "Unknown"
    if not docs_and_scores:
        return {"Unknown": 0.0}, query_image_path, gr.update(visible=True), ""

    # Process results to get unique plant names and their highest scores
    results = {}
    for doc, score in docs_and_scores:
        # The score from FAISS is L2 distance, convert it to cosine similarity
        cosine_similarity = 1 - (score**2) / 2
        plant_name = doc.metadata.get('plant_name', 'Unknown')
        # Ensure we only consider positive similarities
        if cosine_similarity > 0:
            # If we haven't seen this plant or the new score is higher, update it
            if plant_name not in results or cosine_similarity > results[plant_name]:
                results[plant_name] = round(cosine_similarity, 3)

    # If after processing, no valid results were found
    if not results:
        return {"Unknown": 0.0}, query_image_path, gr.update(visible=True), ""

    # Find the top prediction and its score from our processed results
    top_prediction_name = max(results, key=results.get)
    top_prediction_score = results[top_prediction_name]

    # Check if the highest score meets our confidence threshold
    if top_prediction_score < AppConfig.SIMILARITY_THRESHOLD:
        # If confidence is too low, classify as "Unknown"
        final_results = {"Unknown": top_prediction_score}
    else:
        # Otherwise, the predictions are confident enough
        final_results = results

    # Return results, path for state, make feedback UI visible, and clear old feedback message
    return final_results, query_image_path, gr.update(visible=True), ""

def handle_feedback(last_query_path: str, correct_name: str):
    """
    Adds user's correction to the vector store, saves it, and returns a success message.
    """
    global vectorstore
    if not last_query_path or not correct_name.strip():
        return "Error: Cannot submit feedback without a query image and a correct name.", gr.update(), gr.update()
    print(f"--- User Feedback ---")
    print(f"Updating index for image: {last_query_path}")
    print(f"Correct Plant Name: {correct_name}")
    new_doc = Document(
        page_content=last_query_path,
        metadata={"plant_name": correct_name.strip()}
    )
    vectorstore.add_documents([new_doc])
    print(f"Saving updated FAISS index to: {AppConfig.FAISS_INDEX_PATH}")
    vectorstore.save_local(AppConfig.FAISS_INDEX_PATH)
    print(f"Saved updated FAISS index to: {AppConfig.FAISS_INDEX_PATH}")
    success_message = f"✅ **Thank you!** The model has learned that this image is a **'{correct_name.strip()}'**. The change is now saved."
    # Return success message, clear textbox, and hide the accordion
    return success_message, gr.update(value=""), gr.update(visible=False)

# --- 8. Gradio User Interface ---
print("Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 Plants Search 🌿")
    gr.Markdown("Upload a leaf image to identify the plant. If you feel the prediction is wrong, you can correct it!")
    last_query_path_state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Plant Leaf Image")
            submit_btn = gr.Button("Identify Plant", variant="primary")
            gr.Examples(
                examples=[image_paths[i] for i in np.random.choice(len(image_paths), 5, replace=False)] if image_paths and len(image_paths) > 5 else None,
                inputs=image_input
            )
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=AppConfig.TOP_K, label="Top Predictions")
            # Status message is OUTSIDE the accordion ---
            feedback_status = gr.Markdown()
            with gr.Accordion("Is the prediction wrong? Click here to correct it.", open=False, visible=False) as feedback_accordion:
                correct_name_input = gr.Textbox(
                    label="Enter Correct Plant Name",
                    placeholder="e.g., Papaya, Bamboo, etc."
                )
                feedback_btn = gr.Button("Submit Correction", variant="secondary")
    # --- Event Handling ---
    submit_btn.click(
        fn=predict,
        inputs=image_input,
        #  Added feedback_status to the outputs to clear it ---
        outputs=[label_output, last_query_path_state, feedback_accordion, feedback_status]
    )
    feedback_btn.click(
        fn=handle_feedback,
        inputs=[last_query_path_state, correct_name_input],
        # The first output now correctly targets the visible feedback_status component ---
        outputs=[feedback_status, correct_name_input, feedback_accordion]
    )

if __name__ == "__main__":
    demo.launch()