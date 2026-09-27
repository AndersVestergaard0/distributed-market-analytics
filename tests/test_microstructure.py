"""Unit tests for src/features/microstructure.py.

Uses small synthetic order-book/trade DataFrames with hand-computed expected
values (rather than the real Optiver parquet data) so these run fast and
don't depend on data/ being present.
"""

import math

import pytest

from microstructure import (
    add_flow_features,
    add_price_features,
    aggregate_book_features,
    aggregate_trade_features,
)
from regression import FEATURE_COLS

BOOK_COLS = ["stock_id", "time_id", "seconds_in_bucket", "bid_price1", "ask_price1", "bid_size1", "ask_size1"]


def test_add_price_features_micro_price_and_spread(spark):
    # bid=100 (size 10), ask=101 (size 30) -> mid=100.5, spread=1,
    # micro_price weights each side by the *opposite* side's size:
    # (100*30 + 101*10) / 40 = 100.25
    book = spark.createDataFrame(
        [(1, 100, 0, 100.0, 101.0, 10, 30)], BOOK_COLS
    )
    out = add_price_features(book).collect()[0]

    assert out["mid_price"] == pytest.approx(100.5)
    assert out["spread"] == pytest.approx(1.0)
    assert out["micro_price"] == pytest.approx(100.25)
    assert out["spread_bps"] == pytest.approx(1.0 / 100.5 * 10000)


def test_add_flow_features_ofi_sign_and_magnitude(spark):
    # One (stock_id, time_id) window, 5 snapshots, each transition designed
    # to hit a different branch of the OFI formula:
    #   row1: bid size up at same bid price      -> bid_flow=+5, ask_flow=0   -> ofi=+5
    #   row2: bid price drops                     -> bid_flow=-15, ask_flow=+2 -> ofi=-17
    #   row3: ask price rises                     -> bid_flow=0, ask_flow=-12 -> ofi=+12
    #   row4: ask price drops                     -> bid_flow=0, ask_flow=+9  -> ofi=-9
    rows = [
        (1, 100, 0, 100.0, 101.0, 10, 10),
        (1, 100, 1, 100.0, 101.0, 15, 10),
        (1, 100, 2, 99.0, 101.0, 8, 12),
        (1, 100, 3, 99.0, 102.0, 8, 5),
        (1, 100, 4, 99.0, 101.5, 8, 9),
    ]
    book = spark.createDataFrame(rows, BOOK_COLS)
    out = add_flow_features(add_price_features(book)).orderBy("seconds_in_bucket").collect()

    ofi = [row["ofi"] for row in out]
    assert ofi[0] == pytest.approx(0.0)  # no previous snapshot -> coalesced to 0
    assert ofi[1] == pytest.approx(5.0)
    assert ofi[2] == pytest.approx(-17.0)
    assert ofi[3] == pytest.approx(12.0)
    assert ofi[4] == pytest.approx(-9.0)


def test_aggregate_book_features_no_nan_or_inf(spark):
    rows = [
        (1, 100, 0, 100.0, 101.0, 10, 10),
        (1, 100, 1, 100.0, 101.0, 15, 10),
        (1, 100, 2, 99.0, 101.0, 8, 12),
        (1, 100, 3, 99.0, 102.0, 8, 5),
        (1, 100, 4, 99.0, 101.5, 8, 9),
    ]
    book = spark.createDataFrame(rows, BOOK_COLS)
    agg = aggregate_book_features(add_flow_features(add_price_features(book))).collect()[0]

    assert agg["total_ofi"] == pytest.approx(-9.0)
    assert agg["avg_ofi"] == pytest.approx(-1.8)
    assert agg["book_update_count"] == 5

    for col in ["avg_spread_bps", "avg_micro_price", "mid_price_stddev", "realized_vol_book"]:
        value = agg[col]
        assert value is not None
        assert not math.isnan(value)
        assert not math.isinf(value)
    assert agg["realized_vol_book"] > 0


def test_aggregate_trade_features(spark):
    trade_cols = ["stock_id", "time_id", "seconds_in_bucket", "price", "size", "order_count"]
    rows = [
        (1, 100, 0, 100.0, 5, 1),
        (1, 100, 1, 101.0, 10, 2),
        (1, 100, 2, 99.0, 15, 3),
    ]
    trade = spark.createDataFrame(rows, trade_cols)
    agg = aggregate_trade_features(trade).collect()[0]

    assert agg["trade_volume"] == 30
    assert agg["trade_count"] == 3
    assert agg["avg_trade_size"] == pytest.approx(10.0)
    assert agg["avg_order_count"] == pytest.approx(2.0)
    assert agg["trade_price_stddev"] == pytest.approx(1.0)


def test_feature_table_schema_matches_feature_cols(spark):
    rows = [
        (1, 100, 0, 100.0, 101.0, 10, 10),
        (1, 100, 1, 100.0, 101.0, 15, 10),
    ]
    book = spark.createDataFrame(rows, BOOK_COLS)
    book_features = aggregate_book_features(add_flow_features(add_price_features(book)))

    trade_cols = ["stock_id", "time_id", "seconds_in_bucket", "price", "size", "order_count"]
    trade = spark.createDataFrame([(1, 100, 0, 100.0, 5, 1)], trade_cols)
    trade_features = aggregate_trade_features(trade)

    combined_cols = set(book_features.columns) | set(trade_features.columns)
    missing = [c for c in FEATURE_COLS if c not in combined_cols]
    assert not missing, f"FEATURE_COLS references columns the feature table never produces: {missing}"
