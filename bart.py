from transformers import pipeline
import pandas as pd

def classify_events(csv_file, public_access_threshold=0.5):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    events_df = pd.read_csv(csv_file)
    
    # Define labels for public access classification
    public_access_labels = ["open to the general public", "restricted access"]
    
    results = []

    for _, row in events_df.iterrows():
        location_match = (row['virtual'] or "Canada" in str(row['location_city']))
        location_score = 1.0 if location_match else 0.0
        
        event_text = f"{row['title']} {row['description']}"
        public_access_classification = classifier(event_text, public_access_labels)
        public_access_score = public_access_classification['scores'][public_access_classification['labels'].index("open to the general public")]

        if public_access_score >= public_access_threshold and location_score == 1.0:
            result = {
                "title": row['title'],
                "location_score": location_score,
                "public_access_score": public_access_score
            }

            # debug
            result.update({
                "description": row['description'],
                "virtual": row['virtual'],
                "location_city": row['location_city']
            })

            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('bart_events_results.csv', index=False)

if __name__ == "__main__":
    csv_file = 'events.csv'
    public_access_threshold = 0.5
    classify_events(csv_file, public_access_threshold)