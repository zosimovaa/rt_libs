WITH
    %(pair)s AS p_pair,
    toUInt32(%(period)s) AS p_period,
    toUInt32(%(start)s) AS ts_min,
    toUInt32(%(end)s+p_period) AS ts_max

SELECT ts_main as ts, lowest_ask, highest_bid
FROM (SELECT ts_r, AVG(lowest_ask) AS lowest_ask, AVG(highest_bid) AS highest_bid
    FROM (SELECT FLOOR(ts / p_period) * p_period AS ts_r, lowest_ask, highest_bid
        FROM rt.orderbook
        WHERE pair = p_pair AND ts BETWEEN ts_min AND ts_max
        ORDER BY ts DESC) AS TB
    GROUP BY ts_r) AS tb1
    RIGHT JOIN (SELECT arrayJoin (range(ts_min, ts_max, p_period)) AS ts_main) AS tb2 ON tb1.ts_r = tb2.ts_main
    ORDER BY ts ASC