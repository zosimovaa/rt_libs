with
    toUInt32(%(period)s) as p_period,
    toUInt32(%(n_steps)s) as p_steps,
    (select toUInt32(max(ts)) from rt.orderbook) as max_ts_tb,
    toUInt32(floor(max_ts_tb/60)*60+p_period) as ts_max,
    toUInt32(ts_max-p_period*p_steps) as ts_min

select
  pair_r as pair,
  ttt as ts,
  lowest_ask_r as lowest_ask,
  highest_bid_r as highest_bid

 from

(select
  pair_r,
  ts_r,
  AVG(lowest_ask) as lowest_ask_r,
  AVG(highest_bid) as highest_bid_r
FROM
  (select
     pair as pair_r,
     floor(ts/60)*60 AS ts_r,
     lowest_ask,
     highest_bid
  from rt.orderbook
  where ts between ts_min and ts_max

ORDER BY ts desc) AS TB
GROUP BY pair_r, ts_r) as tb1
right join (select arrayJoin(range(ts_min, ts_max, p_period)) as ttt) as tb2 on tb1.ts_r=tb2.ttt

order by pair asc, ts asc