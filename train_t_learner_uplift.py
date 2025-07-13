import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftRandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)

def train_uplift_and_simulate(input_path="uplift_dashboard_raw.csv", output_path="uplift_dashboard_data.csv"):
    df = pd.read_csv(input_path)
    logging.info(f"Loaded raw dataset: {df.shape}")

    df_encoded = df.copy()
    categorical = ['user_segment', 'geo', 'device_type', 'signup_source']
    for col in categorical:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    features = ['user_segment', 'geo', 'device_type', 'signup_source', 'days_since_signup', 'model_score']
    X = df_encoded[features].values
    treatment = df_encoded['treatment'].values

    # Simulate base CTR for later use
    base_ctr = df['predicted_ctr'].values
    y_sim = np.random.binomial(1, base_ctr)

    # Fit uplift model
    uplift_model = UpliftRandomForestClassifier(control_name=0, n_estimators=100, random_state=42)
    uplift_model.fit(X=X, treatment=treatment, y=y_sim)
    uplift = uplift_model.predict(X)

    # Use uplift to simulate conversion probability
    actual_ctr = np.clip(base_ctr + uplift * treatment, 0.01, 0.4)
    converted = np.random.binomial(1, actual_ctr)

    # Simulate business outcomes
    promo_cost = np.where(treatment == 1, np.random.uniform(0.5, 2.0, size=len(df)), 0)
    conversion_revenue = np.where(converted == 1, np.random.uniform(3.0, 6.0, size=len(df)), 0)
    roi = conversion_revenue - promo_cost

    # Add uplift simulation outcomes
    df['uplift_score_model'] = uplift
    df['actual_ctr'] = actual_ctr
    df['converted'] = converted
    df['conversion_revenue'] = conversion_revenue
    df['promo_cost'] = promo_cost
    df['roi'] = roi

    df.to_csv(output_path, index=False)
    logging.info(f"Full dataset with simulated outcomes saved to {output_path}")
    return df

if __name__ == "__main__":
    train_uplift_and_simulate()
