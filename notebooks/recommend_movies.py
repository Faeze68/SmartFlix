from recommender.Itembase_CF import show_recommendations_for_user

if __name__ == "__main__":
    print(" Generating personalized movie recommendations...\n")
    show_recommendations_for_user(user_id=10, k=50, top_n=5)
    print("\n Recommendations generated successfully.")
