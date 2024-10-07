import pandas as pd
from openai import OpenAI
from apikey import OPENAI_API_KEY

# Function to check if the event is virtual or in Canada using ChatGPT
def check_event_in_canada(client, city, virtual):
    if virtual:
        return 'True'
    else:
        prompt = f"Is the following event being held in Canada based on this location: {city}? Answer True or False."
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant who knows about cities in Canada."},
                  {"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3)
        return response.choices[0].message.content.strip()

# Function to check if the event is open to the public using ChatGPT
def check_open_to_public(client, description):
    prompt = f"Event description: {description}? Is this event open to the general public? Answer True if yes, False if no. Answer True if you're not sure. Explain how you came to the conclusion."
    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt}],
    max_tokens=100,
    temperature=0.4)
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

if __name__ == "__main__":

    # TODO: find out how many tokens I use per transaction.
    client = OpenAI(api_key=OPENAI_API_KEY)

    file_path = 'events_with_private.csv'
    events_df = pd.read_csv(file_path)

    events_df['in_canada'] = events_df.apply(lambda row: check_event_in_canada(client, row['location_city'], row['virtual']), axis=1)
    events_df['open_to_public'] = events_df['description'].apply(lambda desc: check_open_to_public(client, desc) if pd.notnull(desc) else 'Unknown')

    #debug
    intermediary_csv = 'intermediary_private_events_df.csv'
    events_df[['in_canada', 'open_to_public']].to_csv(intermediary_csv, index=False)

    # Filter for events that are either virtual or held in Canada, and open to the public
    filtered_events = events_df[
        (events_df['in_canada'].str.contains('True', case=False)) & 
        (events_df['open_to_public'].str.strip().str.casefold() == 'true')]

    output_csv = 'chatgpt_private_events_results.csv'
    filtered_events.to_csv(output_csv, index=False)