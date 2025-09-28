import pandas as pd

# For long strings, wrapping them in parentheses is an idiomatic way to
# format them for better readability, as recommended by Python's style guide (PEP 8).
url = (
    "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/"
    "plain_text/unsupervised/0000.parquet"
)

print("Reading Parquet file from URL...")
print(url)

# The 'pyarrow' engine is specified for reading the Parquet file.
df = pd.read_parquet(url, engine="pyarrow")

print("\nSuccessfully loaded DataFrame. Here are the first 5 rows:")
# to return the first 5 rows of the DataFrame.
print(df.head())

print("\nHere is some information about the DataFrame:")
df.info()

