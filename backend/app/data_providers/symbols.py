"""Static PSX symbol reference.

This is a small, hand-maintained reference used for symbol resolution and display
names. It is not an authoritative listing: active/suspended status is unknown
(``active=None``) unless a configured data provider reports it.
"""
from __future__ import annotations

from app.schemas.market import SymbolInfo

_REGISTRY: dict[str, tuple[str, str]] = {
    "KSE100": ("KSE-100 Index", "Index"),
    "OGDC": ("Oil & Gas Development Company Ltd", "Oil & Gas Exploration"),
    "PPL": ("Pakistan Petroleum Ltd", "Oil & Gas Exploration"),
    "PSO": ("Pakistan State Oil Company Ltd", "Oil & Gas Marketing"),
    "SNGP": ("Sui Northern Gas Pipelines Ltd", "Oil & Gas Marketing"),
    "NRL": ("National Refinery Ltd", "Refinery"),
    "PRL": ("Pakistan Refinery Ltd", "Refinery"),
    "ATRL": ("Attock Refinery Ltd", "Refinery"),
    "LUCK": ("Lucky Cement Ltd", "Cement"),
    "DGKC": ("D.G. Khan Cement Company Ltd", "Cement"),
    "MLCF": ("Maple Leaf Cement Factory Ltd", "Cement"),
    "FCCL": ("Fauji Cement Company Ltd", "Cement"),
    "SYS": ("Systems Ltd", "Technology"),
    "TRG": ("TRG Pakistan Ltd", "Technology"),
    "MUGHAL": ("Mughal Iron & Steel Industries Ltd", "Engineering"),
    "SAZEW": ("Sazgar Engineering Works Ltd", "Automobile Assembler"),
    "GAL": ("Ghandhara Automobiles Ltd", "Automobile Assembler"),
    "AGP": ("AGP Ltd", "Pharmaceuticals"),
    "HUBC": ("Hub Power Company Ltd", "Power"),
    "ENGRO": ("Engro Corporation Ltd", "Fertilizer"),
    "FFC": ("Fauji Fertilizer Company Ltd", "Fertilizer"),
    "EFERT": ("Engro Fertilizers Ltd", "Fertilizer"),
    "MCB": ("MCB Bank Ltd", "Commercial Banks"),
    "UBL": ("United Bank Ltd", "Commercial Banks"),
    "HBL": ("Habib Bank Ltd", "Commercial Banks"),
    "MEBL": ("Meezan Bank Ltd", "Commercial Banks"),
}

_ALIASES = {"KSE-100": "KSE100", "KSE 100": "KSE100", "^KSE": "KSE100"}


def resolve_symbol(query: str) -> SymbolInfo:
    q = query.strip().upper()
    q = _ALIASES.get(q, q)
    if q in _REGISTRY:
        name, sector = _REGISTRY[q]
        return SymbolInfo(symbol=q, name=name, sector=sector, is_index=sector == "Index")
    for sym, (name, sector) in _REGISTRY.items():
        if q and q in name.upper():
            return SymbolInfo(symbol=sym, name=name, sector=sector, is_index=sector == "Index")
    return SymbolInfo(symbol=q, name=None, sector=None, source="unresolved")


def known_symbols() -> list[SymbolInfo]:
    return [SymbolInfo(symbol=s, name=n, sector=sec, is_index=sec == "Index") for s, (n, sec) in _REGISTRY.items()]
