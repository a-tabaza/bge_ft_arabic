# Finetuning BGE for Arabic Embeddings

## WARNING!
This repo is a work in progress and I don't mean the model, just the code, it's horrible trust me I know, once I'm done with the dataset, I'll push it to huggingface, and clean up the code.
Model, same thing.

## Introduction
Hi there, this is the first article I ever write, and it will be a combination of documentation, explanation and steps for everything I did with this project. I will try to explain everything in detail.
## Dataset Curation
These steps will explain the process of curating the dataset for the finetuning process. The dataset is a filtered version of Wikimatrix, which is a dataset that contains pairs of Arabic and English sentences from Wikipedia, it was used in some quite large-scale projects such as NLLB from meta, and the dataset is available on [this link](https://opus.nlpl.eu/WikiMatrix.php).
The dataset is available in many formats, one of them is TMX, which is an XML-based format for translation memories, it can be parsed using BeautifulSoup.

An example of what a pair looks like in the TMX format:

```xml
    <tu>
        <tuv xml:lang="ar">
            <seg>2) #40, على الرغم من أنه يمكن أن ينظر إلى إصلاحه بعد ذلك. </seg>
        </tuv>
        <tuv xml:lang="en">
            <seg>2) #40, though he can be seen reforming afterward. </seg>
        </tuv>
    </tu>
    <!-- keep this example in mind because we will use it later when I talk about the quality of the Data -->
```

Code:

```python
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

file = open("ar-en.tmx", "r", encoding="UTF-8")
contents = file.read()

soup = BeautifulSoup(contents, 'xml')
tu = soup.find_all('tu')

# get pairs
pairs = []
for t in tqdm(tu):
    ar = t.find('tuv', {'xml:lang': 'ar'}).find('seg').text
    en = t.find('tuv', {'xml:lang': 'en'}).find('seg').text
    pairs.append((ar, en))

df = pd.DataFrame(pairs, columns=['ar', 'en'])
```

## Dataset Cleaning

The dataset has around 1M pairs, which sounds too good to be true, which it was sadly, it had a lot of noise, which I will categorize into two types:

- Duplicated pairs, but this wasn't just duplicated text, it would have some sort of phrase about the date it was retrieved or something similar, which would make it a different pair, but it would be the same text.
- Bad quality translation, check the example above, especially in religious text.

To deduplicated the data, I used some primitive methods, then fuzzy matching, which I will show below:

```python
from fuzzywuzzy import fuzz
df = df[df['ar'] != df['en']]
duplicates = []

for pair in df.itertuples():
    if pair.ar == pair.en:
        duplicates.append(pair.Index)
    if pair.en == pair.ar:
        duplicates.append(pair.Index)
    if pair.ar in pair.en:
        duplicates.append(pair.Index)
    if pair.en in pair.ar:
        duplicates.append(pair.Index)

duplicates = list(set(duplicates))

df = df.drop(duplicates)

dedup = df.drop_duplicates(subset=['ar', 'en'])

def get_similarity_ratio(str1, str2):
    return fuzz.ratio(str1, str2)

dedup['similarity_ratio'] = dedup.progress_apply(lambda x: get_similarity_ratio(x['en'], x['ar']), axis=1)
df_deduplicated = df[df['similarity_ratio'] <= 75].drop('similarity_ratio', axis=1)
df_deduplicated.to_parquet('data/ar-en-final-fuzzy-deduplicated.parquet')
```

This took off about 10k examples, which is not a lot, but it's something.

Now the fun part, actually filter the data based on quality of pairs, I devised a simple method that leverages sentence embeddings, since BGE provides very high quality embeddings, I can use it to calculate the cosine similarity between the Arabic and English sentences, and if the similarity was below a certain threshold, I would drop the pair.

But another problem arose, BGE is trained for English embeddings, which is why this whole project started, I can't garuntee good results when embedding Arabic text, so I thought of inserting an intermediary step, where I'd translate the Arabic text to English (en_translated), and then calculate the similarity (cosine_similatiry(en_translated, en)), and if it was below the threshold, I would drop the pair.

To get the intermediary step, I used Google Translate API, which is a paid service, but for a data of this volume it gets expensive fast, this dataset cost $2,013.42, which we won't talk about, you can get $300 free credits on a new account.

To run the API, you have to set up the Google Cloud SDK, which you can connect to VS Code, to process the data, I first tried to use the API directly, but it was very slow, so I used multithreading to speed up the process, which was still slow, I was going to move to async programming, but I found out that the API has a batch processing feature, which I used, and it was very fast, it took about 20 minutes to process the whole dataset.

To use the batch processing feature, I had to split the dataset into chunks containing 10k pairs each, in .tsv format with two columns, an index, and the text I want to translate, then uploaded them to a storage bucket.

The code to process the data and translate it:

```python
from google.cloud import translate
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# process and split data

# the two hundred here is the len of token, I got it from the mbart tokenizer, which I was going to use for translation, but I ended up using Google Translate API, mbart takes a max seq len of 200, but I filtered the data based on it anyways.

pairs_200 = pd.read_parquet('ar-en-200.parquet')
pairs_200.reset_index(inplace=True)
pairs_200.drop(columns=['index'], inplace=True)

n = 10_000
pairs_split = [pairs_200[i:i+n] for i in range(0,pairs_200.shape[0],n)]

for idx, split in tqdm(enumerate(pairs_split)):
    split['ar'].to_csv(f'batches/ar_{idx}.tsv', index=True, header=False, sep='\t')

# translate
def batch_translate_text(
    input_uris,
    output_uri: str = "gs://wiki_matrix_translated/",
    project_id: str = "neurofy-403605"
) -> translate.TranslateTextResponse:
    """Translates a batch of texts on GCS and stores the result in a GCS location.

    Args:
        input_uri: The input URI of the texts to be translated.
        output_uri: The output URI of the translated texts.
        project_id: The ID of the project that owns the destination bucket.
        timeout: The timeout for this batch translation operation.

    Returns:
        The translated texts.
    """

    client = translate.TranslationServiceClient()

    location = "us-central1"
    # Supported file types: https://cloud.google.com/translate/docs/supported-formats

    input_configs_elements = [{
        "gcs_source": {"input_uri": input_uri},
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    } for input_uri in input_uris]

    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "ar",
            "target_language_codes": ["en"],  # Up to 10 language codes here.
            "input_configs": input_configs_elements,
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout=None) # Infinite timeout.

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    return response

# very primitive split since uploading all the data at once exceeds the limit
done = [f"gs://wiki_matrix_batches/ar_{idx}.tsv" for idx in range(50)]
all = [f"gs://wiki_matrix_batches/ar_{idx}.tsv" for idx in range(99)]
remaining = [x for x in all if x not in done]

batch_translate_text(input_uris = done)
batch_translate_text(input_uris = remaining)

# upload and download translated data
# I used the gcloud CLI to upload and download the data, you can check it out in the cloud documentation.

# process translated data (join them back and append as new column)
from glob import glob
paths = glob('translated/*.tsv')
dfs = pd.concat([pd.read_csv(path, sep='\t', index_col=0, header=None) for path in paths])
dfs.sort_index(inplace=True)
dfs.columns = ['ar', 'en_translated']
pairs_200 = pd.read_parquet('ar-en-200.parquet')
pairs_200['en_translated'] = dfs['en_translated'].values
pairs_200.drop(columns=['tokens_ar'], inplace=True)

pairs_200.to_parquet('ar-en-200-translated.parquet')
```

To filter further, I checked the original English sentences for language since they had a lot of noise, I used the langdetect library.

```python
from langdetect import detect
df = pd.read_parquet('ar-en-200-translated.parquet')

# very primitive error handling but it works.
def detect_lang(text):
    try:
        return detect(text)
    except:
        return 'unknown'

langs = [detect_lang(str(x)) for x in tqdm(df['en'].values)]
df['lang'] = langs

# filter original pairs to only include English
df = df[df['lang'] == 'en']

df.reset_index(drop=True, inplace=True)

# this is the dataset I pushed to the huggingface.
df.to_parquet('ar-en-final.parquet')
```



