import os
import argparse
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://www.smogon.com/stats/"

parser = argparse.ArgumentParser(description="Scrape Smogon stats")
parser.add_argument(
    "--start_date",
    type=int,
    default=2015,
    help="Start date for scraping (YYYY)",
)
parser.add_argument(
    "--end_date",
    type=int,
    default=2024,
    help="End date for scraping (YYYY)",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./stats",
    help="Local directory to save the scraped files",
)
args = parser.parse_args()


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


async def save_text_file(session, url, local_path):
    # Check if the file already exists
    if os.path.isfile(local_path):
        print(f"File already exists: {local_path}")
        return
    async with session.get(url) as response:
        if response.status == 200:
            text = await response.text()
            async with aiofiles.open(local_path, "w", encoding="utf-8") as file:
                await file.write(text)


async def scrape_base(session, url, local_dir, start_date, end_date):
    async with session.get(url) as response:
        text = await response.text()
        soup = BeautifulSoup(text, "html.parser")

        tasks = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and not href.startswith("?") and href != "../":
                href_date = int(href[:4])
                href_full = urljoin(url, href)
                local_path = os.path.join(local_dir, href)

                if (
                    href.endswith("/")
                    and href_date >= start_date
                    and href_date < end_date
                ):  # It's a directory
                    ensure_dir(local_path)
                    task = asyncio.create_task(scrape(session, href_full, local_path))
                    tasks.append(task)

        await asyncio.gather(*tasks)


async def scrape(session, url, local_dir):
    try:
        async with session.get(url) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            tasks = []
            for link in soup.find_all("a"):
                href = link.get("href")
                if "chaos" in href or "monotype" in href or "metagame" in href:
                    continue
                if href and not href.startswith("?"):
                    href_full = urljoin(url, href)
                    local_path = os.path.join(local_dir, href)

                    if href.endswith("/") and href != "../":  # It's a directory
                        ensure_dir(local_path)
                        task = asyncio.create_task(
                            scrape(session, href_full, local_path)
                        )
                        tasks.append(task)
                    elif href.endswith(".txt") or href.endswith(
                        ".json"
                    ):  # It's a txt file
                        print(f"Downloading {href_full} to {local_path}")
                        task = asyncio.create_task(
                            save_text_file(session, href_full, local_path)
                        )
                        tasks.append(task)

            await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error on url {url}: {e}")


ensure_dir(args.save_dir)


async def main():
    async with aiohttp.ClientSession() as session:
        await scrape_base(
            session,
            base_url,
            args.save_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )


asyncio.run(main())
