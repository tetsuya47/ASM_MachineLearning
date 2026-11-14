
from playwright.sync_api import sync_playwright
from pathlib import Path
import json

OUTPUT = Path("data/raw/api_samples")
OUTPUT.mkdir(parents=True, exist_ok=True)

captured = []


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def capture(response):
            url = response.url.lower()

            if not any(x in url for x in [
                "trading", "history", "price",
                "stock", "chart"
            ]):
                return

            item = {
                "url": response.url,
                "method": response.request.method,
                "status": response.status,
                "content_type": response.headers.get(
                    "content-type", ""
                )
            }

            try:
                if "json" in item["content_type"].lower():
                    data = response.json()

                    item["response_keys"] = (
                        list(data.keys())
                        if isinstance(data, dict) else []
                    )

                    item["data_preview"] = str(data)[:1000]

                    filename = f"response_{len(captured)}.json"

                    (OUTPUT / filename).write_text(
                        json.dumps(
                            data,
                            ensure_ascii=False,
                            indent=2
                        ),
                        encoding="utf-8"
                    )

                captured.append(item)

                print("\nREQUEST FOUND:")
                print(json.dumps(
                    item,
                    ensure_ascii=False,
                    indent=2
                ))

            except Exception as error:
                print("Capture error:", error)

        page.on("response", capture)

        print("Opening Vietstock...")

        page.goto(
            "https://finance.vietstock.vn/",
            wait_until="domcontentloaded",
            timeout=60000
        )

        print("\nBrowser is ready.")
        print("Open VIC historical trading page.")
        print("Select date range and search.")
        print("Press Ctrl+C in terminal when finished.")

        try:
            while True:
                page.wait_for_timeout(1000)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            (OUTPUT / "capture_report.json").write_text(
                json.dumps(
                    captured,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            browser.close()

            print("Captured:", len(captured))
            print("Report:", OUTPUT / "capture_report.json")


if __name__ == "__main__":
    main()
