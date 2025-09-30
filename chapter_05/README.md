# Chapter 5. Exploring, Tokenizing, and Visualizing Face Datasets

## What is Hugging Face Datasets?

Alongside providing a hub for trained models, Hugging Face also hosts a wide array
of datasets (available at [https://huggingface.co/datasets](https://huggingface.co/datasets)), which you can leverage for your own projects.

Hugging Face datasets come pre-split in training and testing datasets, as well as
the unlabelled datasets for unsupervised or semi-supervised learning tasks.

Datasets can be streamed using `datasets` and `huggingface_hub` libraries, and validated
either with the same or using CLI (your Hugging Face bearer token from their web-site will be required). Datasets can be downloaded too using Python SDK and CLI.

### Streaming the Dataset

[An example](dataset_streaming.py) of how to stream training dataset for the IMDB dataset:

```bash
uv run python chapter_05/dataset_streaming.py
IterableDataset({
    features: ['text', 'label'],
    num_shards: 1
})
First two rows of the fetched dataset:
{'text': "or anyone who was praying for the sight of Al Cliver wrestling a naked, 7ft tall black guy into a full nelson, your film has arrived! Film starlet Laura Crawford (Ursula Buchfellner) is kidnapped by a group who demand the ransom of $6 million to be delivered to their island hideaway. What they don't count on is rugged Vietnam vet Peter Weston (Cliver) being hired by a film producer to save the girl. And what they really didn't count on was a local tribe that likes to offer up young women to their monster cannibal god with bloodshot bug eyes.<br /><br />Pretty much the same filming set up as CANNIBALS, this one fares a bit better when it comes to entertainment value, thanks mostly a hilarious dub track and the impossibly goofy monster with the bulging eyes (Franco confirms they were split ping pong balls on the disc's interview). Franco gets a strong EuroCult supporting cast including Gisela Hahn (CONTAMINATION) and Werner Pochath (whose death is one of the most head-scratching things I ever seen as a guy who is totally not him is shown - in close up - trying to be him). The film features tons of nudity and the gore (Tempra paint variety) is there. The highlight for me was the world's slowly fistfight between Cliver and Antonio de Cabo in the splashing waves. Sadly, ol' Jess pads this one out to an astonishing (and, at times, agonizing) 1 hour and 40 minutes when it should have run 80 minutes tops. <br /><br />For the most part, the Severin DVD looks pretty nice but there are some odd ghosting images going on during some of the darker scenes. Also, one long section of dialog is in Spanish with no subs (they are an option, but only when you listen to the French track). Franco gives a nice 16- minute interview about the film and has much more pleasant things to say about Buchfellner than his CANNIBALS star Sabrina Siani.", 'label': 0}
{'text': "I saw this regurgitated pile of vignettes tonight at a preview screening and I was straight up blown away by how bad it was. <br /><br />First off, the film practically flaunted its gaping blind spots. There are no black or gay New Yorkers in love? Or who, say, know the self-involved white people in love? I know it's not the love Crash of anvil-tastic inclusiveness but you can't pretend to have a cinematic New York with out these fairly prevalent members of society. Plus, you know the people who produced this ish thought Crash deserved that ham-handed Oscar, so where is everyone? <br /><br />Possibly worse than the bizarre and willful socioeconomic ignorance were the down right offensive chapters (remember when you were in high school and people were openly disgusted with pretty young women in wheelchairs? Me either). This movie ran the gamut of ways to be the worst. Bad acting, bad writing, bad directing -- all spanning every possible genre ever to concern wealthy white people who smoke cigarettes outside fancy restaurants. <br /><br />But thank god they finally got powerhouses Hayden Christensen and Rachel Bilson back together for that Jumper reunion. And, side note, Uma dodged a bullet; Ethan Hawke looks ravaged. This, of course, is one thing in terms of his looks, but added an incredibly creepy extra vibe of horribleness to his terrifyingly scripted scene opposite poor, lovely Maggie Q.<br /><br />I had a terrible time choosing my least favorite scene for the end of film questionnaire, but it has to be the Anton Yelchin/ Olivia Thirlby bit for the sheer lack of taste, which saddens me because I really like those two actors. I don't consider myself easily offended, but all I could do was scoff and look around with disgust like someone's 50 year old aunt. <br /><br />A close second place in this incredibly tight contest of terrible things is Shia LaBeouf's tone deaf portrayal of what it means for a former Disney Channel star to act against Julie Christie. I don't mean opposite, I mean against. Against is the only explanation. I realize now that the early sequence with Orlando Bloom is a relative highlight. HIGHLIGHT. Please keep that in mind when your brain begins to leak out your ear soon after the opening credits, which seem to be a nod to the first New York Real World. This film is embarrassing, strangely dated, inarticulate, ineffective, pretentious and, in the end, completely divorced from any real idea of New York at all. <br /><br />(The extra star is for the Cloris Leachman/ Eli Wallach sequence, as it is actually quite sweet, but it is only one bright spot in what feels like hours of pointless, masturbatory torment.)", 'label': 0}
```

### Getting the Parquet files of a dataset

For efficient data storage and processing there may be times when you prefer to directly download the dataset in Parquet format.

This format is optimized for querying and analyzing large datasets, offering significant compression and performance
improvements over row-based formats like CSV.

> Parquet is schema-based, meaning it stores both the data and its schema, enabling better data organization and faster access.
> Its columnar structure allows for efficient read and write operations, especially when only a subset of columns is needed, and
> it supports complex nested data structures. Parquet is widely used with data processing frameworks like Apache Spark, Hive,
> and Hadoop due to its compatibility with big data tools and systems.

An example of how to get the Parquet files associated with the `stanfordnlp/imdb` dataset:

```bash
curl -XGET "https://datasets-server.huggingface.co/parquet?dataset=stanfordnlp/imdb" | jq .
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   765  100   765    0     0   1875      0 --:--:-- --:--:-- --:--:--  1875
{
  "parquet_files": [
    {
      "dataset": "stanfordnlp/imdb",
      "config": "plain_text",
      "split": "test",
      "url": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/test/0000.parquet",
      "filename": "0000.parquet",
      "size": 20470363
    },
    {
      "dataset": "stanfordnlp/imdb",
      "config": "plain_text",
      "split": "train",
      "url": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet",
      "filename": "0000.parquet",
      "size": 20979968
    },
    {
      "dataset": "stanfordnlp/imdb",
      "config": "plain_text",
      "split": "unsupervised",
      "url": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/unsupervised/0000.parquet",
      "filename": "0000.parquet",
      "size": 41996509
    }
  ],
  "pending": [],
  "failed": [],
  "partial": false
}
```

The output contains the URLs of the Paquet file for each of the splits. In [this example](reading_parquet_file.py) we are
using these URLs in Python to access the Parquet file of the split(s) directly:

```bash
uv run python chapter_05/reading_parquet_file.py
Reading Parquet file from URL...
https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/unsupervised/0000.parquet

Successfully loaded DataFrame. Here are the first 5 rows:
                                                text  label
0  This is just a precious little diamond. The pl...     -1
1  When I say this is my favourite film of all ti...     -1
2  I saw this movie because I am a huge fan of th...     -1
3  Being that the only foreign films I usually li...     -1
4  After seeing Point of No Return (a great movie...     -1

Here is some information about the DataFrame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   text    50000 non-null  object
 1   label   50000 non-null  int64
dtypes: int64(1), object(1)
memory usage: 781.4+ KB
```

## Tokenization in NLP

### Types of Tokenization Methods

- **Word-level tokenization** - splits text into individual words
  - Examples: `Word2Vec`, and `GloVe`
- **Subword-level tokenization** - smaller meaningful units or subwords
- **Character-level tokenization** - commonly used for languages like Chinese or Japanese, where word boundaries are less obvious

Most newer models, especially transformer-based models like **BERT** and **GPT**, prefer subword or byte-pair encoding (BPE) tokenization
to overcome issues of out-of-vocabulary words, large vocabulary for diverse languages, not capturing internal structure of words.
They provide better flexibility and generalization across languages and word forms.

Subword-level tokenization handles out-of-vocabulary (OOV) words by breaking them into smaller, recognizable sub-units.

> **Token ID**
> A token ID is a unique numerical identifier assigned to a token (a word, subword, punctuation mark, or even whitespace, depending
> on the tokenizer used) in the context of NLP. Token IDs are generated during the tokenization process when text data is processed
> into input that a large language model (e.g. BERT, GPT) can understand and use.

In the [tokenization_in_nlp.py](tokenization_in_nlp.py) example we are using the same
IDMB dataset with BERT tokenizer to tokenize a review, get its token ids, and then
convert the ids back to tokens and back to the original text:

```bash
uv run python chapter_05/tokenization_i
n_nlp.py
Loading IMDB dataset...
Skipping the first two reviews...
Loading BERT tokenizer...
Tokenizing reviews...

--- First example after skipping ---
Raw review text: If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />
Sentiment label: 0
...
```
