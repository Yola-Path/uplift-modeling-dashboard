import pandas as pd
import numpy as np

def generate_raw_user_data(output_path="uplift_dashboard_raw.csv", seed=42, n=10000):
    np.random.seed(seed)

    segments = ['Dormant User', 'Returning User', 'Power User']
    geo_regions = ['North America', 'Europe', 'Asia']
    device_types = ['Mobile', 'Desktop']
    signup_sources = ['Organic', 'Paid Ads', 'Referral']
    days_since_signup = np.random.randint(1, 365*3, size=n)

    treatment = np.random.binomial(1, 0.5, size=n)
    user_segment = np.random.choice(segments, size=n, p=[0.5, 0.3, 0.2])
    geo = np.random.choice(geo_regions, size=n, p=[0.5, 0.3, 0.2])
    device = np.random.choice(device_types, size=n)
    source = np.random.choice(signup_sources, size=n, p=[0.6, 0.3, 0.1])
    model_score = np.clip(np.random.normal(loc=0.5, scale=0.15, size=n), 0, 1)

    base_ctr = (
        model_score
        + 0.02 * (np.array(user_segment) == 'Returning User')
        + 0.01 * (np.array(geo) == 'Europe')
        - 0.01 * (np.array(source) == 'Paid Ads')
    )
    base_ctr = np.clip(base_ctr * np.random.uniform(0.6, 1.2, size=n), 0.01, 0.3)

    df = pd.DataFrame({
        'user_id': np.arange(1, n + 1),
        'user_segment': user_segment,
        'geo': geo,
        'device_type': device,
        'signup_source': source,
        'days_since_signup': days_since_signup,
        'treatment': treatment,
        'model_score': model_score,
        'predicted_ctr': base_ctr
    })

    return df

if __name__ == "__main__":
    generate_raw_user_data()