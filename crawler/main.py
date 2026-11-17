
from datetime import datetime
from pathlib import Path

import pandas as pd
from vietstock_client import VietstockClient


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = [
    "VIC", "VHM", "VCB", "BID", "CTG",
    "TCB", "VPB", "MBB", "HPG", "GAS"
]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime.now()


def main():
    client = VietstockClient()
    all_data = []
    failed = []

    print("=" * 60)
    print("CRAWL DU LIEU CO PHIEU VIETSTOCK")
    print(f"Tu ngay: {START_DATE:%d/%m/%Y}")
    print(f"Den ngay: {END_DATE:%d/%m/%Y}")
    print("=" * 60)

    try:
        client.initialize()

        for index, symbol in enumerate(STOCKS, start=1):
            print(f"\n[{index}/{len(STOCKS)}] Dang lay {symbol}...")

            try:
                df = client.get_trading_data(
                    symbol,
                    START_DATE,
                    END_DATE
                )

                if df.empty:
                    print(f"KHONG CO DU LIEU: {symbol}")
                    failed.append(symbol)
                    continue

                if "Date" not in df.columns:
                    raise ValueError("Du lieu khong co cot Date")

                df["Date"] = pd.to_datetime(
                    df["Date"], errors="coerce"
                )
                df = df.dropna(subset=["Date"])

                df = df[
                    (df["Date"] >= START_DATE) &
                    (df["Date"] <= END_DATE)
                ].copy()

                if df.empty:
                    raise ValueError("Khong co ban ghi trong khoang ngay yeu cau")

                df = df.sort_values("Date").reset_index(drop=True)

                min_date = df["Date"].min()
                max_date = df["Date"].max()

                output_file = OUTPUT_DIR / f"{symbol}_from_2024.csv"
                df.to_csv(
                    output_file,
                    index=False,
                    encoding="utf-8-sig"
                )

                all_data.append(df)

                print(f"Da luu: {output_file}")
                print(f"So ban ghi: {len(df)}")
                print(f"Ngay dau: {min_date:%d/%m/%Y}")
                print(f"Ngay cuoi: {max_date:%d/%m/%Y}")

                if min_date > START_DATE.replace(day=10):
                    print(
                        "CANH BAO: Du lieu bat dau muon hon moc 01/01/2024. "
                        "Can kiem tra API Vietstock."
                    )

            except Exception as error:
                print(f"LOI {symbol}: {error}")
                failed.append(symbol)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values(
                ["StockCode", "Date"]
            ).reset_index(drop=True)

            combined_file = OUTPUT_DIR / "all_stocks_from_2024.csv"
            combined.to_csv(
                combined_file,
                index=False,
                encoding="utf-8-sig"
            )

            print("\nFILE TONG HOP:")
            print(combined_file)
            print("Tong so ban ghi:", len(combined))

        print("\n========== KET QUA ==========")
        print(f"Thanh cong: {len(all_data)}/{len(STOCKS)}")
        print("That bai:", failed)

    finally:
        client.close()


if __name__ == "__main__":
    main()
