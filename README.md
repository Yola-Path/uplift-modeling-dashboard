# Uplift Modeling Dashboard

This project simulates a real-world decision-making scenario where a growth or marketing team must decide who to target with a promotional treatment based on uplift modeling results.

### Features
- KPI Cards: Avg uplift, conversion rate, ROI
- Filters: Segment, region, device, uplift threshold
- Charts:
  - Uplift by Segment
  - ROI by Segment (treated)
  - ROI by Segment & Treatment (cohort analysis)
- Exportable user table
- Strategic recommendation panel

### Dataset
Includes 100,000 simulated users with features like:
- `user_segment`, `geo`, `device_type`, `signup_source`, `days_since_signup`
- Treatment flag, conversion, ROI, uplift score, model score

To regenerate the data:
```bash
python generate_uplift_dataset.py
```

### Run the Dashboard
```bash
streamlit run main.py
```

---

Developed for instructional use in modeling data science courses.