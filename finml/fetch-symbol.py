import argparse
import json
from pathlib import Path

import yfinance as yf

data_folder = Path('market-data')


def download_daily_history(symbol: str):
    print(f'Downloading daily history of symbol: {symbol}')
    data = yf.download(symbol, period='max', interval='1d', auto_adjust=True, repair=True)
    # column index level 1 contains the symbol name for multiple queries
    data.columns = data.columns.droplevel(1)
    # some symbol characters from YF cannot be used for filenames: remove spaces, then special chars
    symbol_stripped = ''.join(map(
        lambda x: x if x.isalnum() else '', map(
        lambda x: '_' if x.isspace() else x, 
        symbol)))
    data.to_csv(data_folder / f"{symbol_stripped}-1d.csv", encoding='utf-8')
    with open(data_folder / f"{symbol_stripped}-meta.txt", mode='w', encoding='utf-8') as meta:
        json.dump({
            'symbol_name': symbol,
        }, meta)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("symbols", nargs='+')

    args = argparser.parse_args()
    symbols: list[str] = args.symbols

    if any(s.upper().startswith('DEFAULT') for s in symbols):
        DEFAULT_SET = ["^GSPC", "VT", "QQQ", "GLD", "SPY", "EUR=X"]
        symbols = DEFAULT_SET

    for s in symbols:
        download_daily_history(s.upper())