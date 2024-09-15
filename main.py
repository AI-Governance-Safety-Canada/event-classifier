from transformers import pipeline
import pandas as pd

def classify_events(csv_file):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    events_df = pd.read_csv(csv_file)
    candidate_labels = ["available to Canadians", "open to the general public"]
    results = []

    for _, row in events_df.iterrows():
        event_text = f"{row['title']} {row['description']} {row['location_country']} {row['virtual']}"

        classification = classifier(event_text, candidate_labels)
        results.append({
            "title": row['title'],
            "labels": classification['labels'],
            "scores": classification['scores']
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('events_results.csv', index=False)

if __name__ == "__main__":
    csv_file = 'events.csv'
    classify_events(csv_file)