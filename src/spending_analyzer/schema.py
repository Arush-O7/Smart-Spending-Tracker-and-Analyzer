"""Map whatever headers a CSV happens to use onto our canonical names."""

import re

from .config import CANON_ORDER, OPTIONAL_CANON, REQUIRED_CANON

# Compared after lowercasing and stripping non-alphanumerics.
ALIASES = {
    "timestamp": {
        "transdatetranstime", "transactiontime", "transactiondatetime",
        "datetime", "date", "time", "timestamp", "transtime", "transdatetime",
        "transdate", "transtimestamp",
    },
    "merchant": {
        "merchant", "merchantname", "vendor", "payee", "description", "narration",
        "merchantdesc", "merchantdescription", "merchanttext",
    },
    "amount": {
        "amount", "amt", "amnt", "transactionamount", "txnamount", "value",
        "debit", "credit", "amountinr", "amtinr", "totalamount", "amountrs",
        "amountinrs", "txnamt",
    },
    "city": {"city", "merchantcity", "billingcity", "txncity"},
    "state": {"state", "merchantstate", "region", "province", "txnstate"},
    "category": {"category", "label", "class", "txncategory", "merchantcategory"},
}

# Substring fallbacks, only tried when nothing in ALIASES matched.
FUZZY = {
    "amount": ["amount", "amt"],
    "timestamp": ["timestamp", "datetime", "date", "time"],
    "merchant": ["merchant", "desc", "narrat", "vendor", "payee"],
}


class SchemaError(ValueError):
    pass


def _normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def guess_schema_columns(df):
    """Return (canonical -> source column, list of missing required columns)."""
    pairs = [(_normalize(c), c) for c in df.columns]
    found = {}
    taken = set()

    for canon, aliases in ALIASES.items():
        found[canon] = None
        for norm, original in pairs:
            if original in taken:
                continue
            if norm == canon or norm in aliases:
                found[canon] = original
                taken.add(original)
                break

    for canon, hints in FUZZY.items():
        if found[canon] is not None:
            continue
        for hint in hints:
            match = next((o for n, o in pairs if hint in n and o not in taken), None)
            if match:
                found[canon] = match
                taken.add(match)
                break

    return found, [c for c in REQUIRED_CANON if found.get(c) is None]


def align_to_canonical(df):
    mapping, missing = guess_schema_columns(df)
    if missing:
        seen = ", ".join(str(c) for c in df.columns[:20])
        raise SchemaError(
            f"Could not detect required column(s): {missing}. "
            f"Columns found in the file: [{seen}]. "
            "Rename the headers or use a recognised alias (see data/README.md)."
        )

    rename = {mapping[c]: c for c in REQUIRED_CANON}
    rename.update({mapping[c]: c for c in OPTIONAL_CANON if mapping.get(c)})

    out = df.rename(columns=rename)
    # A file can hold both "amt" and "amount"; renaming then leaves duplicates.
    out = out.loc[:, ~out.columns.duplicated()]
    return out[[c for c in CANON_ORDER if c in out.columns]].copy()
