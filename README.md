# Uplift Modeling Dashboard

This Streamlit app is designed for communicating uplift model results to business stakeholders. It helps product managers and marketing teams evaluate model performance, visualize user segment behavior, and make data-driven campaign decisions.

## Features
- Interactive filters: segment, geography, and device
- KPI cards: uplift score, predicted CTR, actual CTR
- Visualizations:
  - Segment-level uplift analysis
  - Model score distribution
  - Predicted vs actual CTR
- Downloadable filtered data
- Top uplift users list

## Files
- `main.py`: Streamlit app entry point
- `utils.py`: Modular functions (data load, render, filter)
- `uplift_dashboard_data.csv`: Sample dataset
- `requirements.txt`: Python packages

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy `main.py`
4. Your live app will be ready to share with stakeholders

## For Private Deployment

### Option A: Localhost via VPN or Browser Tunnel (Simple)
```bash
pip install streamlit
streamlit run main.py
# Then in another terminal:
ngrok http 8501
```

### Option B: Host on AWS EC2
1. Launch Ubuntu EC2 instance and SSH in
2. Install Python dependencies
3. Upload files or clone repo
4. Run:
```bash
streamlit run main.py --server.port 8501 --server.enableCORS false
```
5. Open port 8501 in security group
6. Access via `http://<EC2-IP>:8501`
