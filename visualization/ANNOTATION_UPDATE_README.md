# Visualization annotation update

Updated scripts 01-10 and shared common.py.

- 01: VND axis and latest-close labels.
- 02: indexed-price baseline at 100 and endpoint labels.
- 03: average daily volume in shares and bar-value labels.
- 04: daily-return (%) histogram with zero, mean and median reference lines.
- 05: Pearson return-correlation heatmap with coefficient scale.
- 06: candlestick hover details for OHLC (VND) and Volume (shares).
- 07: console data dictionary for summary CSV fields.
- 08: boxplot explanation of quartiles, median, IQR and potential outliers.
- 09: console guide for quality metrics and outlier detection.
- 10: console guide for quarantine, duplicates, holidays and winsorization.
- common.py: shared source, actual date-range and stock-coverage footer.

The source CSV path, core data-quality/data-treatment calculations, IQR thresholds and existing CSV schemas are preserved. No source values are imputed. Scripts 07, 09 and 10 generate reports rather than chart images.

Run from the repository root with py -3.12 and the project dependencies installed.
