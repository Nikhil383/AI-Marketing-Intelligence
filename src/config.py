
DATA_PATH = "data/marketing_data.csv"
MODEL_PATH = "data/recommendation_model.pkl"

INDUSTRIES = ["Fashion", "SaaS", "E-commerce", "FinTech", "Health", "Real Estate"]
PLATFORMS = ["Instagram", "LinkedIn", "YouTube", "Facebook", "X"]
CONTENT_TYPES = ["Reel", "Carousel", "Case Study", "Infographic", "Blog Post"]
TOPICS = ["Education", "Behind the Scenes", "Customer Success", "Trends", "Promo"]

CAT_FEATURES = ["Industry", "Platform", "Content_Type", "Content_Topic", "Posting_Day"]
NUM_FEATURES = ["Posting_Hour", "Ad_Spend", "Reach"]


TEST_SIZE = 0.20
RANDOM_STATE = 42

N_ESTIMATORS = 100
LEARNING_RATE = 0.05

TOP_K_RECOMMENDATIONS = 3