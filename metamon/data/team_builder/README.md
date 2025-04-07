# Smogon Data

## Data scraper

Run `python stat_scraper.py` to scrape data from smogon stat page. The default data saving dir is `./stats`

You can specify the date of data you want by adjusting `date` on line 73. The default script will download data after 2021.

## Smogon Stat Reader
To load data from Smogon Stat, you can use the `SmogonStat` in `stat_reader.py`

### `__init__(format, date=None, rank=None)`
`format` is the format you want to know, e.g. `gen8ou`.

`date` can be a list of str or a str specifying the date range of the data. e.g. `"2023-12-DLC2"`

`rank` is a str, you can specify what rank of data you want. e.g. `"1500"` means data over elo 1500.

### `self.movesets`
The data will stored in a dict, index by pokemon name. It will contain:
```
{
    'count': total usage count of this pokemon,
    'abilities': ability choice in percent,
    'items': item choice in percent,
    'spreads': Nature and Ev choice in percent,
    'moves': Move choices in percent,
    'teammates': Teammates in percent,
    'checks': Check and counters in percent, # some pokemon is empty
}
```

### Example
```
stats = SmogonStat("gen8ou", date="2022-07", rank="1500")
stats.load()
print(stats.movesets['Mimikyu'])
```