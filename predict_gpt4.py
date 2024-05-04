from openai import OpenAI
import pandas as pd
import json


def sample(train, n=5):
  return train.sample(n=n)[['source', 'label']]


def generate_prompt(examples):
    prompt = "\n".join([f"{i+1}. \"{pair[0]} -> {pair[1]}\"" for i, pair in enumerate(examples)])
    prompt += f"\n\nGiven a name of web-source and a some examples of mapping from 'web-source' to its 'labels',\nYou should generate the list of labels for a web-source using from the best of your knowledge. If you do not know, give approximate answer.\nStrictly follow the aforementioned format"
    return prompt


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return [{'source': i['source'], 'label': i['large_label']} for i in data]


if __name__ == "__main__":
    client = OpenAI()

    # Load and preprocess test and train data
    test_data = load_data('test.json')
    train_data = load_data('train.json')

    test_df = pd.DataFrame(test_data).dropna()
    train_df = pd.DataFrame(train_data).dropna()

    results = []
    for source in test_df['source'].tolist():
        samples = train_df.sample(n=len(train_df))
        examples = list(zip(samples['source'].tolist(), samples['label'].tolist()))
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": generate_prompt(examples)},
                {"role": "user", "content": "is it clear?"},
                {"role": "assistant",
                 "content": "Sure, I will follow the give structure of output and answer from the best of my knowledge"},
                {"role": "user", "content": f"Generate labels for {source}"}
            ],
        )
        results.append(response.choices[0].message.content)

    test_df['preds'] = results
    test_df.to_csv('gpt_predictions.csv')

