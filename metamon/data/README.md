# Smogon Data

You can download the pre-calculated movesets data by simply running the following command:

``` bash
python metamon/data/download_stats.py
```

If you want to scrape the data by yourself, you can follow the steps below.

## Prerequisites

Install pokemon showdown server from [https://github.com/smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown)

## Step 1: Scrape Data from Smogon Stats Page

Run `python stat_scraper.py` to scrape data from smogon stat page. The default smogon data saving dir is `./stats`

You can specify the date range of the data. Just add `--start_date` and `--end_date` to the command. Default date is from 2015 to 2024. Example:

``` bash
python stat_scraper.py --start_date 2021 --end_date 2022
```

## Step 2: Parse the raw data movesets

There are two things needed, movesets data and checks data, you can run the following command to parse the data:

``` bash
# parse the movesets data
cd metamon/data
python create_movesets_jsons.py --smogon_stat_dir ./stats --ps_path LOCAL_POKEMONSHOWDOWN_SERVER_PATH
python create_checks_jsons.py --smogon_stat_dir ./stats 
```

The default save directory will be under `metamon/data`, you can change the directory by changing `DATA_PATH` in `metamon/data/__init__.py`.

## Step 3: Load the data

To load data from Smogon Stat, you can use the `metamon.data.team_builder.stat_reader.PreloadedSmogonStat`

### Example

```python
stats = PreloadedSmogonStat("gen8ou", inclusive=False)
print(stats.movesets['Mimikyu'])
```
