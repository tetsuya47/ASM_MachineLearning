
from datetime import datetime, timedelta
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


def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365)

    client = VietstockClient()
    all_data = []
    failed = []

    try:
        client.initialize()

        for index, symbol in enumerate(STOCKS, start=1):
            print(f"\n[{index}/{len(STOCKS)}] Đang lấy {symbol}...")

            try:
                df = client.get_trading_data(
                    symbol,
                    start_date,
                    end_date
                )

                if df.empty:
                    print(f"KHÔNG CÓ DỮ LIỆU: {symbol}")
                    failed.append(symbol)
                    continue

                output_file = OUTPUT_DIR / f"{symbol}_2years.csv"
                df.to_csv(
                    output_file,
                    index=False,
                    encoding="utf-8-sig"
                )

                all_data.append(df)

                print(f"Đã lưu: {output_file}")
                print(f"Số bản ghi: {len(df)}")

            except Exception as error:
                print(f"LỖI {symbol}: {error}")
                failed.append(symbol)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_file = OUTPUT_DIR / "all_stocks_2years.csv"

            combined.to_csv(
                combined_file,
                index=False,
                encoding="utf-8-sig"
            )

            print("\nĐÃ XUẤT FILE TỔNG HỢP:")
            print(combined_file)
            print("Tổng số bản ghi:", len(combined))

        print("\n========== KẾT QUẢ ==========")
        print("Thành công:", len(all_data), "/", len(STOCKS))
        print("Thất bại:", failed)

    finally:
        client.close()


if __name__ == "__main__":
    main()
