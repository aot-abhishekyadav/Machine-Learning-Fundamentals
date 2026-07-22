# 🐼 Pandas Fundamentals and Data Handling

A beginner-friendly repository containing my practical learning and implementation of **Pandas**, one of the most important Python libraries for data analysis, data cleaning and Machine Learning.

This repository covers importing data from different sources, working with CSV and JSON files, handling nested JSON data, using APIs, generating automatic EDA reports and exporting processed data.

---

## 📌 About Pandas

Pandas is an open-source Python library used for:

* Data manipulation
* Data cleaning
* Data analysis
* Reading and writing files
* Handling missing values
* Working with tabular data
* Preparing datasets for Machine Learning

The two main data structures in Pandas are:

* `Series` – one-dimensional labelled data
* `DataFrame` – two-dimensional tabular data

---

# 📚 Topics Covered

## 1. Reading CSV Files

The notebooks demonstrate how to load CSV files from:

* Local storage
* Google Drive
* Online URLs
* GitHub raw URLs
* Large datasets

Example:

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
```

---

## 2. Reading a CSV File from a URL

The `requests` library and `StringIO` are used to fetch and read a CSV file directly from an online URL.

```python
import requests
import pandas as pd
from io import StringIO

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

data = StringIO(response.text)

df = pd.read_csv(data)
```

---

# ⚙️ Important `read_csv()` Parameters

## 3. `sep` Parameter

The `sep` parameter is used when the columns in a file are separated by a character other than a comma.

For example, TSV files use tabs:

```python
df = pd.read_csv(
    "movie_titles_metadata.tsv",
    sep="\t",
    names=[
        "sno",
        "name",
        "release_year",
        "rating",
        "votes",
        "genres"
    ]
)
```

---

## 4. `index_col` Parameter

The `index_col` parameter converts an existing column into the DataFrame index.

```python
df = pd.read_csv(
    "aug_train.csv",
    index_col="enrollee_id"
)
```

---

## 5. `header` Parameter

The `header` parameter specifies which row should be treated as the column names.

```python
df = pd.read_csv(
    "test.csv",
    header=1
)
```

---

## 6. `usecols` Parameter

The `usecols` parameter loads only the required columns from a dataset.

```python
df = pd.read_csv(
    "aug_train.csv",
    usecols=[
        "enrollee_id",
        "gender",
        "education_level"
    ]
)
```

This can reduce memory usage when working with large datasets.

---

## 7. `nrows` Parameter

The `nrows` parameter loads only a specified number of rows.

```python
df = pd.read_csv(
    "aug_train.csv",
    nrows=100
)
```

This is useful for previewing or testing a large dataset.

---

## 8. `skiprows` Parameter

The `skiprows` parameter skips unnecessary rows while loading a dataset.

```python
df = pd.read_csv(
    "aug_train.csv",
    skiprows=[0, 2]
)
```

---

## 9. `dtype` Parameter

The `dtype` parameter assigns a specific data type to one or more columns.

```python
df = pd.read_csv(
    "aug_train.csv",
    dtype={
        "target": int
    }
)
```

---

## 10. Handling Dates with `parse_dates`

The `parse_dates` parameter converts a date column from the `object` datatype into Pandas datetime format.

```python
df = pd.read_csv(
    "IPL Matches 2008-2020.csv",
    parse_dates=["date"]
)
```

After conversion, the column datatype becomes:

```text
datetime64[ns]
```

This makes it easier to extract:

* Day
* Month
* Year
* Weekday
* Date differences

---

## 11. Using `converters`

The `converters` parameter applies a custom function while reading a file.

```python
def rename_team(name):

    if name == "Royal Challengers Bangalore":
        return "RCB"

    return name
```

```python
df = pd.read_csv(
    "IPL Matches 2008-2020.csv",
    converters={
        "team1": rename_team
    }
)
```

---

## 12. Using `na_values`

The `na_values` parameter converts selected values into missing values or `NaN`.

```python
df = pd.read_csv(
    "aug_train.csv",
    na_values=["Male"]
)
```

---

## 13. Loading Large Datasets in Chunks

Large datasets can be loaded in smaller parts using `chunksize`.

```python
chunks = pd.read_csv(
    "large_dataset.csv",
    chunksize=1000
)

for chunk in chunks:
    print(chunk.shape)
```

This helps when the complete dataset cannot fit into memory.

---

# 📄 Working with JSON Data

## 14. Reading JSON Data

Pandas provides the `read_json()` function for converting JSON data into a DataFrame.

```python
import pandas as pd
from io import StringIO

data = """
{
    "name": "Abhi",
    "email": "abhi@gmail.com",
    "job_profile": [
        {
            "title1": "Team Lead",
            "title2": "Junior Developer"
        }
    ]
}
"""

df = pd.read_json(StringIO(data))
```

`StringIO` is used because passing literal JSON strings directly to `read_json()` is deprecated in newer Pandas versions.

---

# 🧭 JSON Orientations

Pandas supports different JSON orientations that control how rows, columns and indexes are represented.

## 15. `records` Orientation

```python
df.to_json(orient="records")
```

Example output:

```json
[
    {
        "col1": "a",
        "col2": "b"
    },
    {
        "col1": "c",
        "col2": "d"
    }
]
```

---

## 16. `index` Orientation

```python
df.to_json(orient="index")
```

The index values become the main JSON keys.

---

## 17. `columns` Orientation

```python
df.to_json(orient="columns")
```

The column names become the main JSON keys.

This is also the default orientation.

---

## 18. `split` Orientation

```python
df.to_json(orient="split")
```

It stores data separately as:

* Columns
* Index
* Values

---

## 19. `table` Orientation

```python
df.to_json(orient="table")
```

This orientation stores both:

* Dataset values
* Dataset schema

The JSON data can be converted back into a DataFrame using:

```python
df = pd.read_json(
    StringIO(json_data),
    orient="table"
)
```

---

# 🔄 Converting a DataFrame to JSON

A Pandas DataFrame can be converted into JSON using `to_json()`.

```python
df = pd.DataFrame(
    [
        ["a", "b"],
        ["c", "d"]
    ],
    index=["row1", "row2"],
    columns=["col1", "col2"]
)
```

```python
json_data = df.to_json(
    orient="records"
)
```

---

# 🧩 Normalizing Nested JSON Data

## 20. Basic JSON Normalization

Nested JSON data can be converted into a flat DataFrame using `pd.json_normalize()`.

```python
data = [
    {
        "name": "Abhi",
        "email": "abhi@gmail.com",
        "job_profile": {
            "title1": "Team Lead",
            "title2": "Junior Developer"
        }
    }
]
```

```python
df = pd.json_normalize(data)
```

Resulting columns:

```text
name
email
job_profile.title1
job_profile.title2
```

---

## 21. Using `max_level`

The `max_level` parameter controls how deeply nested data is flattened.

```python
df = pd.json_normalize(
    data,
    max_level=0
)
```

With `max_level=0`, nested dictionaries remain inside a single column.

```python
df = pd.json_normalize(
    data,
    max_level=1
)
```

With `max_level=1`, the first nested level is converted into separate columns.

---

## 22. Using `record_path`

The `record_path` parameter extracts records from a nested list.

```python
data = [
    {
        "state": "Florida",
        "shortname": "FL",
        "counties": [
            {
                "name": "Dade",
                "population": 12345
            },
            {
                "name": "Broward",
                "population": 40000
            }
        ]
    }
]
```

```python
df = pd.json_normalize(
    data,
    record_path="counties"
)
```

---

## 23. Using `meta`

The `meta` parameter adds related parent information to every extracted record.

```python
df = pd.json_normalize(
    data,
    record_path="counties",
    meta=[
        "state",
        "shortname"
    ]
)
```

This produces a clean table containing:

* County name
* Population
* State
* State short name

---

# 🌐 Reading JSON from an API

Pandas can directly read JSON data from an API URL.

```python
df = pd.read_json(
    "https://api.exchangerate-api.com/v4/latest/INR"
)
```

This can be used to load:

* Currency exchange data
* Weather data
* Movie data
* Public API responses
* Real-time datasets

---

# 🎬 TMDB Top-Rated Movies API Project

This repository also contains a mini data collection project using the **TMDB API**.

The project:

1. Sends requests to the TMDB API
2. Receives movie data in JSON format
3. Converts JSON results into DataFrames
4. Fetches multiple pages using a loop
5. Combines the results using `pd.concat()`
6. Exports the final dataset into a CSV file

---

## Movie Dataset Columns

| Column         | Description               |
| -------------- | ------------------------- |
| `id`           | Unique TMDB movie ID      |
| `title`        | Movie title               |
| `overview`     | Short movie description   |
| `release_date` | Release date of the movie |
| `popularity`   | TMDB popularity score     |
| `vote_average` | Average user rating       |
| `vote_count`   | Total number of votes     |

---

## Fetching One Page

```python
import requests
import pandas as pd

url = "https://api.themoviedb.org/3/movie/top_rated"

params = {
    "api_key": API_KEY,
    "language": "en-US",
    "page": 1
}

response = requests.get(
    url,
    params=params
)

response.raise_for_status()

temp_df = pd.DataFrame(
    response.json()["results"]
)[
    [
        "id",
        "title",
        "overview",
        "release_date",
        "popularity",
        "vote_average",
        "vote_count"
    ]
]
```

---

## Fetching Multiple Pages

```python
df = pd.DataFrame()

for page in range(1, 429):

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": page
    }

    response = requests.get(
        url,
        params=params
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        []
    )

    if not results:
        break

    temp_df = pd.DataFrame(results)[
        [
            "id",
            "title",
            "overview",
            "release_date",
            "popularity",
            "vote_average",
            "vote_count"
        ]
    ]

    df = pd.concat(
        [df, temp_df],
        ignore_index=True
    )
```

---

## Exporting the Dataset

```python
df.to_csv(
    "Top_Rated_movies.csv",
    index=False
)
```

The collected dataset contains approximately:

```text
8,560 rows
7 columns
```

---

# 📊 Automatic EDA with YData Profiling

The repository also demonstrates automatic Exploratory Data Analysis using the Titanic dataset.

## Installing the Library

```python
!pip install ydata-profiling
```

## Creating a Profile Report

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title="Titanic Dataset Report"
)

profile.to_file(
    "output.html"
)
```

YData Profiling automatically provides:

* Dataset shape
* Column data types
* Missing values
* Duplicate rows
* Statistical summaries
* Mean, median and standard deviation
* Minimum and maximum values
* Percentiles
* Correlations
* Distributions
* Outlier information
* Data quality warnings

---

## Saving the Profiling Report to Google Drive

```python
import os
import shutil

drive_path = "/content/drive/MyDrive/output.html"

shutil.move(
    "output.html",
    drive_path
)

print(
    f"output.html saved to {drive_path}"
)
```

---

# 🗂️ Datasets Used

The notebooks use different datasets for learning and practice:

* Countries dataset
* Movie titles metadata
* HR analytics dataset
* IPL Matches dataset
* Titanic dataset
* Coffee sales dataset
* TMDB top-rated movies
* Currency exchange-rate API data

---



# 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
requests
python-dotenv
ydata-profiling
```

Install all required libraries using:

```bash
pip install -r requirements.txt
```

---



# 🎯 What I Learned

Through these notebooks, I learned how to:

* Import Pandas
* Create and work with DataFrames
* Read CSV files
* Load online CSV files
* Work with TSV files
* Use important `read_csv()` parameters
* Select specific columns and rows
* Assign column data types
* Handle date columns
* Apply converter functions
* Define custom missing values
* Load large datasets in chunks
* Read and write JSON data
* Understand JSON orientations
* Flatten nested JSON data
* Use `record_path` and `meta`
* Read data from APIs
* Send HTTP requests using `requests`
* Handle API pagination
* Combine DataFrames using `pd.concat()`
* Export DataFrames into CSV files
* Generate automatic EDA reports
* Save generated reports to Google Drive
* Protect API keys using environment variables

---

# 🚀 Future Improvements

The following topics will be added in the future:

* DataFrame indexing with `loc` and `iloc`
* Filtering and sorting data
* Missing-value handling
* Duplicate-value handling
* GroupBy operations
* Merging, joining and concatenation
* Pivot tables
* String operations
* Datetime operations
* Exploratory Data Analysis
* Data visualization
* Feature engineering
* Pandas interview questions
* Practice exercises

---

# 👨‍💻 Author

**Abhishek Yadav**

* GitHub: [aot-abhishekyadav](https://github.com/aot-abhishekyadav)


---

# 🤝 Contributing

Suggestions, corrections and improvements are welcome.

You can:

1. Fork this repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

---

# ⭐ Support

If this repository helped you understand Pandas, consider giving it a ⭐.
