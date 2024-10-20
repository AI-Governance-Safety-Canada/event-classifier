import argparse

from transformers import pipeline
import pandas as pd

def classify_events(input_path, output_path, public_access_threshold=0.5):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    events_df = pd.read_csv(input_path)
    
    # Define labels for public access classification
    public_access_labels = ["open to the general public", "restricted access"]

    for index, row in events_df.iterrows():
        location_score = pd.NA
        if row["virtual"]:
            location_score = 1.0
        else:
            for column in row.index:
                if "location" not in column.lower():
                    # This isn't a location field
                    continue
                if not isinstance(row[column], str):
                    # This field isn't a string
                    continue
                if "Canada" in row[column]:
                    location_score = 1.0
                else:
                    # This event isn't in Canada
                    location_score = 0.0
        
        event_text = f"{row['title']} {row['description']}"
        public_access_classification = classifier(event_text, public_access_labels)
        public_access_score = public_access_classification['scores'][public_access_classification['labels'].index("open to the general public")]

        events_df.at[index, "accessible_to_canadians"] = location_score
        events_df.at[index, "open_to_public"] = public_access_score
        print(row)

    events_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify events using ChatGPT")
    parser.add_argument(
        "input_path",
        help="Path to input CSV file containing events",
    )
    parser.add_argument(
        "output_path",
        help="""
            Path to output CSV file where classified events will be saved. If it already
            exists, it will be overwritten.
        """,
    )
    args = parser.parse_args()

    public_access_threshold = 0.5
    classify_events(args.input_path, args.output_path, public_access_threshold)
