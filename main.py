import os
from data_cleaner import load_and_clean, visualize_dataset
from train_knn import train_itemknn
from train_svd import train_svd_model
#from infer_svd import load_model, predict_svd, recommend_svd
#from infer_knn import load_model, recommend_knn, predict_knn
#from infer_svd import load_model, predict_svd, recommend_svd
from classe_inferencia import Inferencia, SVDPredictions, KNNPredictions



if __name__ == "__main__":

    # ----- 1. Paths -----
    ROOT = os.path.dirname(__file__)
    DATA_PATH = os.path.join(ROOT, "ratings_Electronics(1).csv")
    MODELS_DIR = os.path.join(ROOT, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_item_model.pkl")
    SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")

    # ----- 2. Load + clean -----
    print("Loading and cleaning dataset...")
    df = load_and_clean(DATA_PATH)
    print(f"Dataset shape after cleaning: {df.shape}")

    #  ----- 3. Visualize dataset -----
    visualize_dataset(df)

    # ----- 4. Train KNN -----
    print("\nTraining item-item KNN (top-k)...")
    knn_model = train_itemknn(df, k=20, model_path=KNN_MODEL_PATH, use_topk=True)
    print(f"KNN model saved at: {KNN_MODEL_PATH}")

    # ----- 5. Train FunkSVD -----
    print("\nTraining FunkSVD...")
    svd_model = train_svd_model(df, k=20, epochs=20, model_path=SVD_MODEL_PATH)
    print(f"SVD model saved at: {SVD_MODEL_PATH}")

    # ----- 6. Load KNN model for inference -----
    knn_model_loaded = load_model(KNN_MODEL_PATH) #revisar

    # ----- 7. Load FunkSVD model for inference -----
    svd_model, users, items = load_model(SVD_MODEL_PATH)
    sample_user = users[0]
    print("\nUser seleccionat:", sample_user)

    # ----- 8. Predicció SVD per un ítem -----
    example_item = items[0]
    rating_pred = predict(sample_user, example_item, svd_model, users, items)
    print(f"\nPredicció rating per l'item {example_item}: {rating_pred:.3f}")

    # ----- 9. Recomanacions SVD -----
    recs = recommend(sample_user, svd_model, users, items, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score:.3f}")

    # Exemple amb un usuari real per KNN
    sample_user_knn = list(knn_model_loaded.user_index.keys())[0]
    print("\n[KNN] User seleccionat:", sample_user_knn)

    # Predicció per un ítem concret amb KNN
    example_item_knn = list(knn_model_loaded.item_index.keys())[0]
    rating_pred_knn = predict(sample_user_knn, example_item_knn, knn_model_loaded)
    print(f"[KNN] Predicció rating per l'item {example_item_knn}: {rating_pred_knn:.3f}")

    # Recomanacions top 10 KNN
    recs_knn = recommend(sample_user_knn, knn_model_loaded, top_n=10)
    print("\n[KNN] Top 10 recomanacions:")
    for item, score in recs_knn:
        print(f"{item}: {score:.3f}")

  
