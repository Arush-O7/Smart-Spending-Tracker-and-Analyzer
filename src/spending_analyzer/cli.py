"""Command line entry point.

    spending-analyzer predict data/for_testing.csv
    spending-analyzer risk data/for_testing.csv --percentile 99.5
    spending-analyzer fit-bins data/credit_card_transactions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import AMOUNT_BIN_COUNT, load_amount_binning, save_amount_binning
from .features import build_features, fit_amount_bin_edges
from .models import ModelLoadError, load_categorization_pipeline, load_risk_bundle
from .risk import RiskThresholds, apply_rules, build_risk_features, score_risk
from .schema import SchemaError, align_to_canonical


def read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"error: input file not found: {path}") from None
    except Exception as exc:
        raise SystemExit(f"error: could not read '{path}': {exc}") from exc


def emit(df, output, limit):
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Wrote {len(df):,} rows to {output}")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.head(limit).to_string(index=False))
    if len(df) > limit:
        print(f"... {len(df) - limit:,} more rows (use --output to write them all)")


def cmd_predict(args):
    aligned = align_to_canonical(read_csv(args.input))
    feats, dropped, reasons = build_features(aligned)
    if dropped:
        print(
            f"Dropped {dropped:,} unusable row(s): "
            f"{reasons['bad_timestamp']} bad timestamps, {reasons['bad_amount']} bad amounts",
            file=sys.stderr,
        )
    if feats.empty:
        raise SystemExit("error: nothing usable left after cleaning.")

    pipe = load_categorization_pipeline()
    keep = [c for c in ["timestamp", "merchant", "amount", "city", "state"] if c in feats]
    out = feats[keep].copy()
    out["pred_category"] = pipe.predict(feats)
    if "category" in feats.columns:
        out["category"] = feats["category"]
        print(f"Agreement with provided labels: {(out.pred_category == out.category).mean():.2%}",
              file=sys.stderr)

    emit(out, args.output, args.limit)
    return 0


def cmd_risk(args):
    feats = build_risk_features(read_csv(args.input))
    feats["risk_score"] = score_risk(feats, load_risk_bundle())
    scored = apply_rules(feats, RiskThresholds(
        percentile=args.percentile,
        require_two_rules=not args.single_rule,
        z_cut=args.z_cut,
        geo_km=args.geo_km,
        velocity_multiplier=args.velocity,
    ))

    alerts = scored.loc[scored["is_alert"]].sort_values("risk_score", ascending=False)
    rate = 100 * len(alerts) / max(len(scored), 1)
    print(f"{len(alerts):,} alert(s) out of {len(scored):,} transactions ({rate:.2f}%), "
          f"threshold={scored.attrs['risk_threshold']:.4f}", file=sys.stderr)

    cols = [c for c in ["timestamp", "merchant", "amount", "city", "state", "risk_score", "reasons"]
            if c in alerts.columns]
    emit(alerts[cols], args.output, args.limit)
    return 0


def cmd_fit_bins(args):
    raw = read_csv(args.input)
    col = args.column or next((c for c in ["amt", "amount"] if c in raw.columns), None)
    if col is None:
        raise SystemExit("error: no amount column found, pass --column.")
    edges = fit_amount_bin_edges(raw[col], args.bins)
    print(f"Wrote {len(edges) + 1} bins to {save_amount_binning(edges)}: {edges}")
    return 0


def cmd_show_config(args):
    binning = load_amount_binning()
    print(f"amount bins : {binning.n_bins} (edges={binning.edges})")
    print(f"bin source  : {binning.source}")
    for name, loader in [("categorization", load_categorization_pipeline),
                         ("risk engine", load_risk_bundle)]:
        try:
            loader()
            print(f"{name:<14}: loaded OK")
        except ModelLoadError as exc:
            print(f"{name:<14}: UNAVAILABLE - {exc}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="spending-analyzer",
        description="Transaction categorization and risk scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", type=Path, help="transactions CSV")
    common.add_argument("-o", "--output", type=Path, help="write results here instead of stdout")
    common.add_argument("-n", "--limit", type=int, default=50, help="rows to print (default 50)")

    p = sub.add_parser("predict", parents=[common], help="categorize transactions")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("risk", parents=[common], help="score transactions for risk")
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--z-cut", type=float, default=4.0)
    p.add_argument("--geo-km", type=float, default=1000.0)
    p.add_argument("--velocity", type=float, default=4.5)
    p.add_argument("--single-rule", action="store_true", help="alert on one rule hit, not two")
    p.set_defaults(func=cmd_risk)

    p = sub.add_parser("fit-bins", help="persist amount bin edges from training data")
    p.add_argument("input", type=Path, help="training CSV")
    p.add_argument("--bins", type=int, default=AMOUNT_BIN_COUNT)
    p.add_argument("--column", help="amount column name (auto-detected by default)")
    p.set_defaults(func=cmd_fit_bins)

    p = sub.add_parser("show-config", help="print resolved paths and check the models load")
    p.set_defaults(func=cmd_show_config)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (SchemaError, ModelLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
