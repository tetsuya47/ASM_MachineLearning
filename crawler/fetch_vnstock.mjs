
import { vn } from "vnstock-js";
import fs from "node:fs";
import path from "node:path";

const STOCKS = [
    "VIC", "VHM", "VCB", "BID", "CTG",
    "TCB", "VPB", "MBB", "HPG", "GAS"
];

const FROM = "2024-11-13";
const TO = "2026-11-13";

const OUTPUT = path.resolve("data/raw");
fs.mkdirSync(OUTPUT, { recursive: true });

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function csvEscape(value) {
    const text = String(value ?? "");
    return `"${text.replaceAll('"', '""')}"`;
}

function makeCsv(rows) {
    const columns = [
        "StockCode", "Date", "Open", "High", "Low", "Close", "Volume"
    ];

    return "\uFEFF" + [
        columns.join(","),
        ...rows.map(row =>
            columns.map(key => csvEscape(row[key])).join(",")
        )
    ].join("\n");
}

async function crawlStock(symbol) {
    console.log(`\nDang lay ${symbol}...`);

    const bars = await vn.stock(symbol).history({
        from: FROM,
        to: TO,
        interval: "1d"
    });

    if (!Array.isArray(bars) || bars.length === 0) {
        throw new Error("Nguon khong tra ve du lieu");
    }

    const unique = new Map();

    for (const bar of bars) {
        const date = String(bar.time ?? "").slice(0, 10);

        if (!date || date < FROM || date > TO) continue;

        const values = [
            bar.open, bar.high, bar.low, bar.close, bar.volume
        ];

        if (!values.every(value =>
            value !== null &&
            value !== undefined &&
            Number.isFinite(Number(value))
        )) {
            continue;
        }

        unique.set(date, {
            StockCode: symbol,
            Date: date,
            Open: Number(bar.open),
            High: Number(bar.high),
            Low: Number(bar.low),
            Close: Number(bar.close),
            Volume: Number(bar.volume)
        });
    }

    const rows = [...unique.values()]
        .sort((a, b) => a.Date.localeCompare(b.Date));

    if (rows.length === 0) {
        throw new Error("Khong co ban ghi OHLCV hop le");
    }

    if (rows[0].Date > "2024-11-20") {
        throw new Error(
            `Du lieu bat dau qua muon: ${rows[0].Date}`
        );
    }

    if (rows.at(-1).Date < "2026-11-10") {
        throw new Error(
            `Du lieu ket thuc qua som: ${rows.at(-1).Date}`
        );
    }

    return rows;
}

async function main() {
    const allRows = [];
    const failed = [];

    console.log("======================================");
    console.log("STOCK DATA CRAWLER - VIETCAP");
    console.log(`Period: ${FROM} -> ${TO}`);
    console.log("======================================");

    for (const [index, symbol] of STOCKS.entries()) {
        console.log(`\n[${index + 1}/${STOCKS.length}] ${symbol}`);

        try {
            const rows = await crawlStock(symbol);

            const file = path.join(
                OUTPUT,
                `${symbol}_20241113_20261113.csv`
            );

            fs.writeFileSync(file, makeCsv(rows), "utf8");
            allRows.push(...rows);

            console.log("Thanh cong");
            console.log("So phien:", rows.length);
            console.log("Ngay dau:", rows[0].Date);
            console.log("Ngay cuoi:", rows.at(-1).Date);
            console.log("File:", file);

        } catch (error) {
            console.error("That bai:", error.message);
            failed.push(symbol);
        }

        await sleep(500);
    }

    console.log("\n========== TONG KET ==========");

    if (failed.length === 0) {
        allRows.sort((a, b) =>
            a.StockCode.localeCompare(b.StockCode) ||
            a.Date.localeCompare(b.Date)
        );

        const combined = path.join(
            OUTPUT,
            "all_stocks_20241113_20261113.csv"
        );

        fs.writeFileSync(combined, makeCsv(allRows), "utf8");

        console.log("Da tao file tong hop:", combined);
        console.log("Tong so dong:", allRows.length);
    } else {
        console.log("Khong tao file tong hop vi con ma that bai.");
    }

    console.log(`Thanh cong: ${STOCKS.length - failed.length}/${STOCKS.length}`);
    console.log("Ma that bai:", failed);
}

main().catch(error => {
    console.error("CRAWLER ERROR:", error);
    process.exitCode = 1;
});

