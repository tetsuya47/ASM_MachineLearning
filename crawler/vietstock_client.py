
import re
import time
from datetime import datetime

import pandas as pd
from playwright.sync_api import sync_playwright


BASE_URL = "https://finance.vietstock.vn"
API_URL = f"{BASE_URL}/data/gettradingresult"


class VietstockClient:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def initialize(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

        print("Dang mo Vietstock...")
        self.page.goto(
            BASE_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )
        self.page.wait_for_timeout(3000)

    def _get_token(self):
        try:
            token = self.page.locator(
                'input[name="__RequestVerificationToken"]'
            ).first

            if token.count():
                value = token.get_attribute("value")
                if value:
                    return value
        except Exception:
            pass

        for cookie in self.context.cookies():
            if cookie["name"] == "__RequestVerificationToken":
                return cookie["value"]

        return ""

    @staticmethod
    def _parse_date(value):
        if value is None:
            return None

        text = str(value)
        match = re.search(r"/Date\((-?\d+)", text)

        if match:
            timestamp = int(match.group(1)) / 1000
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            return text

    def _request_page(self, payload, symbol, page_index):
        for attempt in range(1, 4):
            try:
                if self.page.is_closed():
                    self.page = self.context.new_page()
                    self.page.goto(
                        BASE_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                result = self.page.evaluate(
                    """async ({url, payload}) => {
                        const body = new URLSearchParams(payload);

                        const response = await fetch(url, {
                            method: "POST",
                            credentials: "same-origin",
                            headers: {
                                "Content-Type":
                                    "application/x-www-form-urlencoded; charset=UTF-8",
                                "X-Requested-With": "XMLHttpRequest"
                            },
                            body: body.toString()
                        });

                        const text = await response.text();

                        let data;
                        try {
                            data = JSON.parse(text);
                        } catch {
                            throw new Error(
                                "API khong tra JSON: " + text.slice(0, 200)
                            );
                        }

                        return {
                            status: response.status,
                            data: data
                        };
                    }""",
                    {"url": API_URL, "payload": payload}
                )

                if result["status"] != 200:
                    raise RuntimeError(
                        f"HTTP {result['status']}"
                    )

                return result["data"]

            except Exception as error:
                print(
                    f"{symbol}: thu lai trang {page_index} "
                    f"({attempt}/3): {error}"
                )

                if attempt == 3:
                    raise RuntimeError(
                        f"{symbol}: khong lay duoc trang {page_index}"
                    ) from error

                try:
                    self.page.goto(
                        BASE_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )
                    self.page.wait_for_timeout(2000)
                except Exception:
                    pass

        return None

    def get_trading_data(self, symbol, start_date, end_date):
        token = self._get_token()

        all_rows = []
        page_index = 1
        page_size = 100

        while True:
            payload = {
                "Code": symbol,
                "OrderBy": "",
                "OrderDirection": "desc",
                "PageIndex": str(page_index),
                "PageSize": str(page_size),
                "FromDate": start_date.strftime("%d/%m/%Y"),
                "ToDate": end_date.strftime("%d/%m/%Y"),
                "ExportType": "default",
                "Cols": "TKLGD,TGTGD,VHTT,TGG,DC,TGPTG,KLGDKL,GTGDKL",
                "ExchangeID": "1",
            }

            if token:
                payload["__RequestVerificationToken"] = token

            body = self._request_page(
                payload,
                symbol,
                page_index
            )

            if isinstance(body, dict):
                rows = body.get("Data", [])
            elif isinstance(body, list):
                rows = body
            else:
                rows = []

            if not rows:
                break

            all_rows.extend(rows)

            print(
                f"{symbol}: Trang {page_index}, "
                f"da lay {len(all_rows)} dong"
            )

            if len(rows) < page_size:
                break

            page_index += 1
            time.sleep(0.5)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # API co the da tra san StockCode, khong chen cot trung lap
        df["StockCode"] = symbol

        if "TradingDate" in df.columns:
            df["Date"] = df["TradingDate"].apply(self._parse_date)

        elif "Date" in df.columns:
            df["Date"] = df["Date"].apply(self._parse_date)

        if "Date" in df.columns:
            df = df.drop_duplicates(
                subset=["StockCode", "Date"]
            )
            df = df.sort_values("Date")

        df = df.reset_index(drop=True)

        return df

    def close(self):
        if self.browser:
            self.browser.close()

        if self.playwright:
            self.playwright.stop()
