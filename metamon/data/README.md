# Smogon Data

This module allows you to download or scrape pre-calculated competitive Pokémon moveset statistics from [Smogon](https://www.smogon.com/) for use in your own projects.

## Option 1: Download Pre-Processed Movesets

To quickly get started with pre-calculated movesets, simply run:

```bash
python metamon/data/download_stats.py
```

This will download the latest processed moveset data directly.

---

## Option 2: Scrape and Parse Smogon Data Yourself

If you prefer to scrape the data manually and have more control over the dataset, follow the steps below.

### Prerequisites

- Clone and install the [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server locally.

---

### Step 1: Scrape Data from Smogon's Stats Page

Run the following command to scrape Smogon's usage stats:

```bash
python stat_scraper.py
```

By default, data is saved to the `./stats` directory.

#### Optional: Specify a Date Range

You can set a custom date range using the `--start_date` and `--end_date` flags:

```bash
python stat_scraper.py --start_date 2021 --end_date 2022
```

Default range: **2015 to 2024**

---

### Step 2: Parse the Raw Data

After scraping, you need to parse two components: **movesets** and **checks**.

#### 1. Parse Movesets

```bash
cd metamon/data
python create_movesets_jsons.py --smogon_stat_dir ./stats --ps_path LOCAL_POKEMONSHOWDOWN_SERVER_PATH
```

#### 2. Parse Checks

```bash
python create_checks_jsons.py --smogon_stat_dir ./stats
```

By default, the parsed JSON files will be saved in the `metamon/data` directory.  
To customize the output path, modify the `DATA_PATH` variable in `metamon/data/__init__.py`.

---

### Step 3: Load the Data

Use the `PreloadedSmogonStat` class to load the processed data:

```python
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat

stats = PreloadedSmogonStat("gen8ou", inclusive=False)
print(stats.movesets['Mimikyu'])
```
