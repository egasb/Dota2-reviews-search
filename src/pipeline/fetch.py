import asyncio
import json
from pathlib import Path

import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.config import settings

APP_ID = settings.app_id
LANG = settings.language
NUM_PER_PAGE = settings.num_per_page
OUTPUT_FILE = settings.raw_file
CURSORS_DIR = settings.cursors_dir

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
CURSORS_DIR.mkdir(parents=True, exist_ok=True)

COMBINATIONS = [
    {"review_type": "negative", "min_hrs": 0, "max_hrs": 300},
    {"review_type": "negative", "min_hrs": 300, "max_hrs": 0},
    {"review_type": "positive", "min_hrs": 0, "max_hrs": 100},
    {"review_type": "positive", "min_hrs": 100, "max_hrs": 300},
    {"review_type": "positive", "min_hrs": 300, "max_hrs": 800},
    {"review_type": "positive", "min_hrs": 800, "max_hrs": 1500},
    {"review_type": "positive", "min_hrs": 1500, "max_hrs": 3000},
    {"review_type": "positive", "min_hrs": 3000, "max_hrs": 6000},
    {"review_type": "positive", "min_hrs": 6000, "max_hrs": 0},
]

console = Console()
file_lock = asyncio.Lock()


def append_lines(file_path, lines):
    with Path.open(file_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def save_cursor_sync(file_path, cursor_text):
    file_path.write_text(cursor_text, encoding="utf-8")


def read_cursor_sync(file_path):
    return file_path.read_text(encoding="utf-8").strip()


async def fetch_stream(session, combo, progress, task_id):
    review_type = combo["review_type"]
    min_hrs = combo["min_hrs"]
    max_hrs = combo["max_hrs"]

    stream_name = f"{review_type[:3].upper()}|{min_hrs}-{max_hrs or 'MAX'}h"
    cursor_file = CURSORS_DIR / f"cursor_{review_type}_{min_hrs}_{max_hrs}.txt"
    cursor = "*"

    if cursor_file.exists():
        cursor = await asyncio.to_thread(read_cursor_sync, cursor_file)
        if cursor == "FINISHED":
            progress.update(
                task_id,
                description=f"[bold green]✔ {stream_name} (Already done)[/bold green]",
            )
            return

    url = f"https://store.steampowered.com/appreviews/{APP_ID}?json=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    backoff = 1
    total_saved = 0
    empty_streak = 0

    while True:
        params = {
            "filter": "recent",
            "language": LANG,
            "num_per_page": NUM_PER_PAGE,
            "cursor": cursor,
            "review_type": review_type,
            "purchase_type": "all",
            "filter_offtopic_activity": 0,
        }

        if min_hrs > 0:
            params["playtime_filter_min"] = min_hrs
        if max_hrs > 0:
            params["playtime_filter_max"] = max_hrs

        try:
            async with session.get(
                url, params=params, headers=headers, timeout=15
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                if response.status != 200:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                data = await response.json()
                if data.get("success") != 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue

                backoff = 1
                reviews = data.get("reviews", [])

                if not reviews:
                    empty_streak += 1
                    if empty_streak > 50:
                        await asyncio.to_thread(
                            save_cursor_sync, cursor_file, "FINISHED"
                        )
                        progress.update(
                            task_id,
                            description=f"[bold green]✔ {stream_name} (Done: {total_saved})[/bold green]",
                        )
                        break

                parsed_lines = []
                for r in reviews:
                    author = r.get("author", {})
                    parsed = {
                        "id": r.get("recommendationid"),
                        "text": r.get("review"),
                        "voted_up": r.get("voted_up"),
                        "votes_up": r.get("votes_up"),
                        "votes_funny": r.get("votes_funny"),
                        "comment_count": r.get("comment_count", 0),
                        "weighted_vote_score": float(
                            r.get("weighted_vote_score") or 0.0
                        ),
                        "timestamp_created": r.get("timestamp_created"),
                        "timestamp_updated": r.get("timestamp_updated"),
                        "playtime_hours": round(
                            (author.get("playtime_forever") or 0) / 60, 1
                        ),
                        "playtime_at_review_hours": round(
                            (author.get("playtime_at_review") or 0) / 60, 1
                        ),
                        "num_games_owned": author.get("num_games_owned", 0),
                        "num_reviews": author.get("num_reviews", 0),
                    }
                    parsed_lines.append(json.dumps(parsed, ensure_ascii=False) + "\n")

                async with file_lock:
                    await asyncio.to_thread(append_lines, OUTPUT_FILE, parsed_lines)

                total_saved += len(reviews)
                progress.update(
                    task_id,
                    description=f"[bold yellow]↻ {stream_name}[/bold yellow] DL: {total_saved}",
                )

                next_cursor = data.get("cursor")
                if not next_cursor or next_cursor == cursor:
                    await asyncio.to_thread(save_cursor_sync, cursor_file, "FINISHED")
                    progress.update(
                        task_id,
                        description=f"[bold green]✔ {stream_name} (Done: {total_saved})[/bold green]",
                    )
                    break

                cursor = next_cursor
                await asyncio.to_thread(save_cursor_sync, cursor_file, cursor)

                await asyncio.sleep(1.0)

        except Exception as _:
            await asyncio.sleep(5)


async def main():
    console.print(
        f"[bold cyan]Starting Async Downloader for AppID {APP_ID}[/bold cyan]"
    )

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for combo in COMBINATIONS:
                task_id = progress.add_task(
                    f"[bold gray]⏳ {combo['review_type'][:3].upper()}|{combo['min_hrs']}-{combo['max_hrs'] or 'MAX'}h (Starting...)[/bold gray]"
                )
                tasks.append(fetch_stream(session, combo, progress, task_id))

            await asyncio.gather(*tasks)

    console.print(
        "[bold green]All parallel streams finished successfully![/bold green]"
    )


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
