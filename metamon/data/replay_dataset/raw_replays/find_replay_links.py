import asyncio
import os
import signal
from itertools import product
from datetime import datetime
import time as time_module

import aiohttp
from tqdm import tqdm

REPLAY_DOMAIN = "https://replay.pokemonshowdown.com"

N_REQUESTS_PER_SECOND = 1


def parse_date(date_str):
    """Convert date string to Unix timestamp"""
    if not date_str:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())


class WriteProtection:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print("SIGINT received, waiting for file writing to finish...")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


async def scrape_format(
    format: str, out_path: str, session, tqdm_bar, start_date=None, end_date=None
) -> None:
    # Convert end_date to before_time if provided, otherwise use most recent
    if end_date:
        before_time = end_date
    else:
        most_recent_time_path = os.path.join(
            os.path.dirname(out_path), ".most_recent_time"
        )
        if os.path.exists(most_recent_time_path):
            with open(most_recent_time_path, "r") as f:
                try:
                    before_time = int(f.read().strip())
                except:
                    before_time = int(time_module.time())
        else:
            before_time = int(time_module.time())

    time = before_time
    new_stop_time = before_time

    if os.path.isfile(out_path):
        with open(out_path, "r") as f:
            found_links = set(f.read().splitlines())
    else:
        found_links = set()

    with open(out_path, "a") as f:
        while True:
            await request_sem.acquire()

            # Debug: Print current search parameters
            print(f"\nSearching with parameters:")
            print(f"Format: {format}")
            print(f"Before time: {time}")

            url = f"https://replay.pokemonshowdown.com/search.json?format={format}&before={time}"
            print(f"URL: {url}")

            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Error fetching {url}: {response.status}")
                    break

                j = await response.json()
                if not j:  # No more results
                    print("No results returned from API")
                    break

                # Debug: Print information about results
                print(f"Got {len(j)} results")
                if j:
                    print("Sample IDs:")
                    for entry in j[:5]:
                        print(f"  {entry['id']} (time: {entry['uploadtime']})")

            # Filter entries within the date range
            filtered_entries = [
                entry
                for entry in j
                if (start_date is None or entry["uploadtime"] >= start_date)
                and (end_date is None or entry["uploadtime"] <= end_date)
            ]
            filtered_links = [
                f"{REPLAY_DOMAIN}/{entry['id'].replace('?p2', '')}"
                for entry in filtered_entries
            ]

            with WriteProtection():
                for link in filtered_links:
                    if link not in found_links:
                        f.write(f"{link}\n")
                        found_links.add(link)
            tqdm_bar.update(len(filtered_links))

            # If the last entry is before the start_date, stop paginating
            if start_date is not None and j[-1]["uploadtime"] <= start_date:
                break

            # Update progress bar with actual number of new results
            tqdm_bar.update(len(j))

            # Check if we have a complete page (indicating more results)
            if len(j) < 51:
                print("Less than 51 results, ending pagination")
                break

            # Use the uploadtime of the last entry for pagination
            time = j[-1]["uploadtime"]
            print(f"Moving to next page with time: {time}")

    # Only update most recent time file if we're not using a custom date range
    if not end_date:
        with open(most_recent_time_path, "w") as f_time, WriteProtection():
            f_time.seek(0)
            f_time.write(str(new_stop_time))
            f_time.truncate()


async def request_throttle(args):
    global request_sem
    while True:
        await asyncio.sleep(1 / args.n_requests_per_second)
        request_sem.release()


async def main(args):
    global request_sem
    request_sem = asyncio.Semaphore(0)
    asyncio.create_task(request_throttle(args))

    # Convert date strings to timestamps
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    assert end_date > start_date

    tqdm_bar = tqdm()
    futures = []
    async with aiohttp.ClientSession() as session:
        for format in args.formats:
            out_game_link_file = os.path.join(
                args.output_dir, f"{format}-{args.start_date}-{args.end_date}.txt"
            )
            print(f"Writing to {out_game_link_file}")
            os.makedirs(os.path.dirname(out_game_link_file), exist_ok=True)

            future = scrape_format(
                format=format,
                out_path=out_game_link_file,
                session=session,
                tqdm_bar=tqdm_bar,
                start_date=start_date,
                end_date=end_date,
            )
            futures.append(future)
        await asyncio.gather(*futures)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=[
            f"gen{gen}{f}" for gen in range(1, 10) for f in ["ou", "uu", "nu", "ubers"]
        ],
    )
    parser.add_argument("--n_requests_per_second", type=float, default=5.0)
    parser.add_argument("--start_date", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", help="Output directory")
    args = parser.parse_args()
    asyncio.run(main(args))
