BASIC_QUERY = """
WITH
    toUInt32(%(period)s) AS p_period,
    toUInt32(%(start)s) AS ts_min,
    toUInt32(%(end)s+p_period) AS ts_max,
    toUInt32(%(step)s) AS step

	SELECT 
		period_gr + ts_min as ts,
		pair_gr as pair,
		--max(ts_local_gr)+ts_min as ts,	

		--anyLast(highest_bid_gr) as open_price,
		--any(highest_bid_gr) as close_price, 

		--min(highest_bid_gr) as min_price,
		--max(highest_bid_gr) as max_price,
		--anyLast(highest_bid_gr) as avg_price,

		anyLast(highest_bid_gr) as highest_bid,
		anyLast(lowest_ask_gr) as lowest_ask



	FROM (
        SELECT 
            pair as pair_gr, 
            ts-ts_min as ts_local_gr, 
            CEIL((ts-ts_min) / p_period) * p_period AS period_gr, 
            lowest_ask as lowest_ask_gr, 
            highest_bid as highest_bid_gr
        FROM orderbook
        WHERE ts BETWEEN ts_min+1 AND ts_max
        ORDER BY ts DESC
        ) AS TB1

        RIGHT JOIN (SELECT arrayJoin (range(p_period, toUInt32(ts_max - ts_min + p_period), p_period)) AS ts_main) AS tb2 ON TB1.period_gr = tb2.ts_main

    GROUP BY pair_gr, period_gr
	ORDER BY pair_gr, period_gr

"""

QUERY_WITH_OB = """
WITH
    toUInt32(%(period)s) AS p_period,
    toUInt32(%(start)s) AS ts_min,
    toUInt32(%(end)s+p_period) AS ts_max,
    toUInt32(%(step)s) AS step,
    %(pair)s AS p_pair

SELECT 
	ts, 
	highest_bid,
	lowest_ask,

	(asks_vol_1 - bids_vol_1) / if(asks_vol_1>bids_vol_1, asks_vol_1,  bids_vol_1) as vol_1,
	(asks_vol_2 - bids_vol_2) / if(asks_vol_2>bids_vol_2, asks_vol_2,  bids_vol_2) as vol_2,
	(asks_vol_3 - bids_vol_3) / if(asks_vol_3>bids_vol_3, asks_vol_3,  bids_vol_3) as vol_3,
	(asks_vol_4 - bids_vol_4) / if(asks_vol_4>bids_vol_4, asks_vol_4,  bids_vol_4) as vol_4,
	(asks_vol_5 - bids_vol_5) / if(asks_vol_5>bids_vol_5, asks_vol_5,  bids_vol_5) as vol_5,
	(asks_vol_6 - bids_vol_6) / if(asks_vol_6>bids_vol_6, asks_vol_6,  bids_vol_6) as vol_6,

    asks_vol_1,
    asks_vol_2,
    asks_vol_3,
    asks_vol_4,
    asks_vol_5,
    asks_vol_6,

    bids_vol_1,
    bids_vol_2,
    bids_vol_3,
    bids_vol_4,
    bids_vol_5,
    bids_vol_6


FROM
	(
	SELECT 
		period_gr + ts_min as ts,
		--pair_gr as pair,

        --max(ts_local_gr)+ts_min as ts,	

		--anyLast(highest_bid_gr) as open_price,
		--any(highest_bid_gr) as close_price, 

		--min(highest_bid_gr) as min_price,
		--max(highest_bid_gr) as max_price,
		--anyLast(highest_bid_gr) as avg_price,

		avg(highest_bid_gr) as highest_bid,
		avg(lowest_ask_gr) as lowest_ask,


        AVG(asks_vol_1_gr) AS asks_vol_1,
        AVG(asks_vol_2_gr) AS asks_vol_2,
        AVG(asks_vol_3_gr) AS asks_vol_3,
        AVG(asks_vol_4_gr) AS asks_vol_4,
        AVG(asks_vol_5_gr) AS asks_vol_5,
        AVG(asks_vol_6_gr) AS asks_vol_6,

        AVG(bids_vol_1_gr) AS bids_vol_1,
        AVG(bids_vol_2_gr) AS bids_vol_2,
        AVG(bids_vol_3_gr) AS bids_vol_3,
        AVG(bids_vol_4_gr) AS bids_vol_4,
        AVG(bids_vol_5_gr) AS bids_vol_5,
        AVG(bids_vol_6_gr) AS bids_vol_6

	FROM (
        SELECT 
            pair as pair_gr, 
            ts-ts_min as ts_local_gr, 
            CEIL((ts-ts_min) / p_period) * p_period AS period_gr, 
            lowest_ask as lowest_ask_gr, 
            highest_bid as highest_bid_gr, 

            tupleElement(asks_vol, 1) as asks_vol_1_gr,
            tupleElement(asks_vol, 2) as asks_vol_2_gr,
            tupleElement(asks_vol, 3) as asks_vol_3_gr,
            tupleElement(asks_vol, 4) as asks_vol_4_gr,
            tupleElement(asks_vol, 5) as asks_vol_5_gr,
            tupleElement(asks_vol, 6) as asks_vol_6_gr,

            tupleElement(bids_vol, 1) as bids_vol_1_gr,
            tupleElement(bids_vol, 2) as bids_vol_2_gr,
            tupleElement(bids_vol, 3) as bids_vol_3_gr,
            tupleElement(bids_vol, 4) as bids_vol_4_gr,
            tupleElement(bids_vol, 5) as bids_vol_5_gr,
            tupleElement(bids_vol, 6) as bids_vol_6_gr

        FROM rt.orderbook
        WHERE ts BETWEEN ts_min AND ts_max and pair = p_pair
        ORDER BY ts DESC
        ) AS TB1

        RIGHT JOIN (SELECT arrayJoin (range(p_period, toUInt32(ts_max -  CEIL(ts_min/ p_period) * p_period), p_period)) AS ts_main) AS tb2 ON TB1.period_gr = tb2.ts_main

    GROUP BY pair_gr, period_gr
	ORDER BY pair_gr, period_gr

) AS TB3

"""
