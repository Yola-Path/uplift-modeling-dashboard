import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import qini_score, auuc_score
import logging

logging.basicConfig(level=logging.INFO)

def train_and_simulate_all_models(input_path="uplift_dashboard_raw.csv", output_path="uplift_dashboard_data.csv"):
    df = pd.read_csv(input_path)
    logging.info(f"Loaded raw dataset: {df.shape}")

    df_encoded = df.copy()
    categorical = ['user_segment', 'geo', 'device_type', 'signup_source']
    for col in categorical:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    features = ['user_segment', 'geo', 'device_type', 'signup_source', 'days_since_signup', 'model_score']
    X = df_encoded[features].values
    treatment = df_encoded['treatment'].values
    base_ctr = df['predicted_ctr'].values
    y_sim = np.random.binomial(1, base_ctr)

    uplift_scores = {}

    # 1. T-Learner
    t_clf_t = RandomForestClassifier(n_estimators=100, random_state=42)
    t_clf_c = RandomForestClassifier(n_estimators=100, random_state=42)
    t_clf_t.fit(X[treatment == 1], y_sim[treatment == 1])
    t_clf_c.fit(X[treatment == 0], y_sim[treatment == 0])
    uplift_scores['t_learner'] = t_clf_t.predict_proba(X)[:, 1] - t_clf_c.predict_proba(X)[:, 1]

    # 2. S-Learner
    X_s = np.concatenate([X, treatment.reshape(-1, 1)], axis=1)
    s_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    s_clf.fit(X_s, y_sim)
    treat_1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    treat_0 = np.concatenate([X, np.zeros((len(X), 1))], axis=1)
    uplift_scores['s_learner'] = s_clf.predict_proba(treat_1)[:, 1] - s_clf.predict_proba(treat_0)[:, 1]

    # 3. CausalML Forest
    causal_model = UpliftRandomForestClassifier(control_name=0, n_estimators=100, random_state=42)
    causal_model.fit(X=X, treatment=treatment, y=y_sim)
    uplift_scores['causalml'] = causal_model.predict(X)

    metrics = []
    for name, uplift in uplift_scores.items():
        df[f'uplift_score_{name}'] = uplift
        # Simulate outcomes
        ctr = np.clip(base_ctr + uplift * treatment, 0.01, 0.4)
        converted = np.random.binomial(1, ctr)
        revenue = np.where(converted == 1, np.random.uniform(3.0, 6.0, size=len(df)), 0)
        promo = np.where(treatment == 1, np.random.uniform(0.5, 2.0, size=len(df)), 0)
        roi = revenue - promo

        df[f'actual_ctr_{name}'] = ctr
        df[f'converted_{name}'] = converted
        df[f'conversion_revenue_{name}'] = revenue
        df[f'promo_cost_{name}'] = promo
        df[f'roi_{name}'] = roi

        score_qini = qini_score(conversion=converted, treatment=treatment, uplift=uplift)
        score_auuc = auuc_score(conversion=converted, treatment=treatment, uplift=uplift)
        metrics.append({'model': name, 'qini': score_qini, 'auuc': score_auuc})

    metrics_df = pd.DataFrame(metrics)
    best_model = metrics_df.sort_values(by='qini', ascending=False).iloc[0]['model']
    df['best_model'] = best_model
    metrics_df.to_csv("uplift_model_metrics.csv", index=False)
    logging.info("Model comparison results saved.")
    logging.info(metrics_df)

    df.to_csv(output_path, index=False)
    logging.info(f"Final dataset with multi-model simulation saved to {output_path}")
    return df

if __name__ == "__main__":
    train_and_simulate_all_models()
