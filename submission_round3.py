import json
import math
from typing import Dict, List, Optional, Tuple, Any

from datamodel import OrderDepth, TradingState, Order, Symbol


HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

VOUCHERS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]


class Trader:
    def __init__(self):
        self.POSITION_LIMITS: Dict[str, int] = {
            HYDROGEL: 200,
            VELVET: 200,
            "VEV_4000": 300, "VEV_4500": 300,
            "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
            "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
            "VEV_6000": 300, "VEV_6500": 300,
        }

        self.VOUCHER_STRIKES: Dict[str, int] = {
            "VEV_4000": 4000, "VEV_4500": 4500,
            "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
            "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
            "VEV_6000": 6000, "VEV_6500": 6500,
        }

        # Active vouchers for IV scalping (5000-5500 from data exploration).
        # VEV_5400 EXCLUDED: it is structurally priced -2.87 ticks below ANY smooth
        # smile interpolation. When excluded from the fit, the residual persists
        # permanently (ACF1=0.977). Our model always says "buy cheap" but the market
        # correctly prices it lower. Including it causes persistent long positions
        # that bleed as it never reverts. Also excluded from smile fit to avoid
        # dragging the quadratic curve down for all other strikes.
        self.ACTIVE_VOUCHERS = {
            "VEV_5000", "VEV_5100", "VEV_5200",
            "VEV_5300", "VEV_5500",
        }
        # Strikes used for smile fitting - same as active (5400 biases the fit).
        self.SMILE_FIT_VOUCHERS = {
            "VEV_5000", "VEV_5100", "VEV_5200",
            "VEV_5300", "VEV_5500",
        }

        # Smile-fit fallback (from historical day 2). Used only if dynamic
        # refit fails. Dynamic refit is the primary path.
        self.SMILE_A_FALLBACK = 2.038
        self.SMILE_B_FALLBACK = -0.041
        self.SMILE_C_FALLBACK = 0.229

        # Smoothing on dynamically-fit smile coefficients.
        # Lower window -> adapts faster to underlying moves.
        self.SMILE_FIT_WINDOW = 40

        # Partial residual absorption.
        # Lowered from 0.30: high values amplify EMA noise during warm-up ticks.
        self.MEAN_DIFF_SHRINK = 0.15
        self.THEO_DIFF_WINDOW = 50

        # Warmup: don't open option positions until this timestamp.
        # Reason: smile EMA + mean_diff EMA both need ~30 ticks to converge.
        # Bad fills in the first 3K timestamps caused the early -300 dip in v2.
        self.WARMUP_TIMESTAMPS = 5000

        # Option execution: tighter and smaller than before.
        # ----- Open thresholds widened by ~0.15 across the board -----
        self.BASE_OPTION_OPEN_EDGE = {
            "VEV_5000": 1.20,
            "VEV_5100": 1.05,
            "VEV_5200": 0.90,
            "VEV_5300": 0.80,
            "VEV_5400": 0.60,
            "VEV_5500": 0.60,
        }

        self.OPTION_PASSIVE_EDGE_ADD = 0.35
        # Hard cap on per-voucher absolute position (was 220).
        self.MAX_OPTION_TARGET = 100
        # Per-tick clip per voucher (was 80).
        self.MAX_OPTION_TRADE_SIZE = 35
        self.MAX_OPTION_PASSIVE_SIZE = 12

        # Faster flattening when edge disappears.
        self.CLOSE_SLACK = 0.05

        # Stop-loss: force flatten if position has been open too long.
        self.MAX_HOLD_TICKS = 35

        # Low vega means IV signal is unreliable.
        self.LOW_VEGA_CUTOFF = 5.0
        self.LOW_VEGA_THR_ADJ = 0.50

        # Delta-1 market making caps (smaller than before).
        self.HYDROGEL_BASE_SIZE = 30
        self.HYDROGEL_MAX_SIZE = 45
        self.VELVET_BASE_SIZE = 22
        self.VELVET_MAX_SIZE = 35

        # Delta hedge: how aggressively to neutralize option delta on Velvet.
        # Increased from 0.50 to 0.65: the mid-period drawdown was partly delta-driven
        # (short VEV_5300 = short underlying exposure; VFR rally = loss).
        self.DELTA_HEDGE_RATIO = 0.65

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        if not od.buy_orders or not od.sell_orders:
            return None, None
        return max(od.buy_orders.keys()), min(od.sell_orders.keys())

    def get_mid(self, od: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def ema_update(self, data: Dict[str, Any], key: str, value: float, window: int) -> float:
        alpha = 2.0 / (window + 1.0)
        old = data.get(key)
        new = value if old is None else alpha * value + (1.0 - alpha) * float(old)
        data[key] = new
        return new

    def buy_capacity(self, product: str, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] - position)

    def sell_capacity(self, product: str, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] + position)

    # ------------------------------------------------------------------
    # Option model
    # ------------------------------------------------------------------

    def tte_years(self, timestamp: int) -> float:
        # Round 3 final starts with 5 days to expiry.
        # NOTE: when backtesting on the 3-day historical pack, change 5.0 -> 8.0.
        tte_days = 5.0 - (timestamp / 1_000_000.0)
        return max(tte_days / 365.0, 1e-6)

    def smile_iv_from_coefs(self, S: float, K: float, a: float, b: float, c: float) -> float:
        m = math.log(S / K)
        iv = a * m * m + b * m + c
        return max(0.05, min(1.00, iv))

    def bs_call(
        self, S: float, K: float, T: float, sigma: float, r: float = 0.0,
    ) -> Tuple[Optional[float], float, float]:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None, 0.0, 0.0
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        vega = S * sqrt_t * self.norm_pdf(d1)
        return price, delta, vega

    def implied_vol(
        self, market_price: float, S: float, K: float, T: float, r: float = 0.0,
    ) -> Optional[float]:
        """Newton-Raphson IV solver. Returns None if outside no-arb bounds."""
        if T <= 0 or S <= 0 or K <= 0:
            return None
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        if market_price <= intrinsic + 1e-3 or market_price >= S:
            return None

        sigma = 0.23  # data exploration mean
        for _ in range(20):
            price, _, vega = self.bs_call(S, K, T, sigma, r)
            if price is None or vega < 1e-8:
                return None
            diff = price - market_price
            if abs(diff) < 1e-4:
                break
            sigma -= diff / vega
            sigma = max(0.05, min(1.0, sigma))
        return sigma

    def fit_smile(
        self, S: float, T: float, market_ivs: List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fit IV(m) = a*m^2 + b*m + c via least squares on (log-moneyness, iv) pairs.
        Needs >=3 distinct points (we have 6 active strikes).
        """
        if len(market_ivs) < 4:
            return None

        s0 = float(len(market_ivs))
        s1 = sum(m for m, _ in market_ivs)
        s2 = sum(m * m for m, _ in market_ivs)
        s3 = sum(m ** 3 for m, _ in market_ivs)
        s4 = sum(m ** 4 for m, _ in market_ivs)
        sy = sum(iv for _, iv in market_ivs)
        sxy = sum(m * iv for m, iv in market_ivs)
        sx2y = sum(m * m * iv for m, iv in market_ivs)

        # Normal equations: [s4 s3 s2; s3 s2 s1; s2 s1 s0] @ [a;b;c] = [sx2y; sxy; sy]
        M = [[s4, s3, s2, sx2y],
             [s3, s2, s1, sxy],
             [s2, s1, s0, sy]]
        for i in range(3):
            piv = i
            for k in range(i + 1, 3):
                if abs(M[k][i]) > abs(M[piv][i]):
                    piv = k
            M[i], M[piv] = M[piv], M[i]
            if abs(M[i][i]) < 1e-12:
                return None
            for k in range(3):
                if k == i:
                    continue
                f = M[k][i] / M[i][i]
                for j in range(i, 4):
                    M[k][j] -= f * M[i][j]
        return (M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2])

    def get_smile_coefs(
        self,
        S: float,
        T: float,
        order_depths: Dict[str, OrderDepth],
        data: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """Refit the smile each tick from observed mids; smooth with EMA.
        Uses SMILE_FIT_VOUCHERS (excludes VEV_5400 which biases the fit downward)."""
        market_ivs: List[Tuple[float, float]] = []
        for v in self.SMILE_FIT_VOUCHERS:
            if v not in order_depths:
                continue
            mid = self.get_mid(order_depths[v])
            if mid is None:
                continue
            K = self.VOUCHER_STRIKES[v]
            iv = self.implied_vol(mid, S, K, T)
            if iv is None:
                continue
            m = math.log(S / K)
            market_ivs.append((m, iv))

        fit = self.fit_smile(S, T, market_ivs)
        if fit is None:
            return (
                data.get("smile_a", self.SMILE_A_FALLBACK),
                data.get("smile_b", self.SMILE_B_FALLBACK),
                data.get("smile_c", self.SMILE_C_FALLBACK),
            )

        a_new, b_new, c_new = fit
        a = self.ema_update(data, "smile_a", a_new, self.SMILE_FIT_WINDOW)
        b = self.ema_update(data, "smile_b", b_new, self.SMILE_FIT_WINDOW)
        c = self.ema_update(data, "smile_c", c_new, self.SMILE_FIT_WINDOW)
        return a, b, c

    # ------------------------------------------------------------------
    # Delta-1 products
    # ------------------------------------------------------------------

    def trade_hydrogel(
        self, product: str, od: OrderDepth, position: int, data: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return orders

        mid = (bid + ask) / 2.0
        spread = ask - bid

        ema = self.ema_update(data, f"{product}_ema_mid", mid, window=20)
        fair = 0.65 * mid + 0.35 * ema

        # Inventory skew.
        inv_skew = 2.5 * (position / self.POSITION_LIMITS[product])
        fair -= inv_skew

        if spread >= 12:
            min_edge = 4.0
            size = self.HYDROGEL_MAX_SIZE
        elif spread >= 8:
            min_edge = 3.0
            size = self.HYDROGEL_BASE_SIZE
        else:
            min_edge = 2.0
            size = 18

        if abs(position) > 120:
            size = min(size, 18)
        if abs(position) > 165:
            size = min(size, 8)

        buy_cap = self.buy_capacity(product, position)
        sell_cap = self.sell_capacity(product, position)

        bid_px = min(bid + 1, int(math.floor(fair - min_edge)))
        ask_px = max(ask - 1, int(math.ceil(fair + min_edge)))

        if bid_px >= ask:
            bid_px = ask - 1
        if ask_px <= bid:
            ask_px = bid + 1
        if bid_px >= ask_px:
            return orders

        buy_qty = min(buy_cap, size)
        sell_qty = min(sell_cap, size)

        if position > 100:
            buy_qty = min(buy_qty, 6)
            sell_qty = min(sell_qty, size + 12)
        elif position < -100:
            buy_qty = min(buy_qty, size + 12)
            sell_qty = min(sell_qty, 6)

        if buy_qty > 0:
            orders.append(Order(product, int(bid_px), int(buy_qty)))
        if sell_qty > 0:
            orders.append(Order(product, int(ask_px), -int(sell_qty)))
        return orders

    def trade_velvetfruit(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        data: Dict[str, Any],
        delta_hedge_target: float,
    ) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return orders

        mid = (bid + ask) / 2.0
        spread = ask - bid

        last_mid_key = f"{product}_last_mid"
        last_mid = data.get(last_mid_key)
        data[last_mid_key] = mid
        recent_move = 0.0 if last_mid is None else mid - float(last_mid)

        # Lag-1 mean reversion.
        fair = mid - 0.16 * recent_move
        ema = self.ema_update(data, f"{product}_ema_mid", mid, window=12)
        fair = 0.80 * fair + 0.20 * ema

        # ---- Effective position incorporates delta hedge target ----
        # If options gave us +50 delta, we want -50 of velvet -> effective_pos
        # treats us as if we're already at +50 (so we'll happily sell).
        effective_pos = position - delta_hedge_target
        inv_skew = 1.8 * (effective_pos / self.POSITION_LIMITS[product])
        fair -= inv_skew

        if spread >= 4:
            min_edge = 1.25
            size = self.VELVET_MAX_SIZE
        elif spread >= 2:
            min_edge = 1.0
            size = self.VELVET_BASE_SIZE
        else:
            min_edge = 1.0
            size = 12

        if abs(position) > 120:
            size = min(size, 14)
        if abs(position) > 165:
            size = min(size, 6)

        buy_cap = self.buy_capacity(product, position)
        sell_cap = self.sell_capacity(product, position)

        bid_px = min(bid + 1, int(math.floor(fair - min_edge)))
        ask_px = max(ask - 1, int(math.ceil(fair + min_edge)))

        if bid_px >= ask:
            bid_px = ask - 1
        if ask_px <= bid:
            ask_px = bid + 1
        if bid_px >= ask_px:
            return orders

        buy_qty = min(buy_cap, size)
        sell_qty = min(sell_cap, size)

        # Hedge-aware sizing: if we need to short to hedge option deltas,
        # bias the sell quote up and cut buy size.
        if effective_pos > 60:
            buy_qty = min(buy_qty, 5)
            sell_qty = min(sell_qty, size + 10)
        elif effective_pos < -60:
            buy_qty = min(buy_qty, size + 10)
            sell_qty = min(sell_qty, 5)

        # Hard inventory caps.
        if position > 100:
            buy_qty = min(buy_qty, 5)
        elif position < -100:
            sell_qty = min(sell_qty, 5)

        if buy_qty > 0:
            orders.append(Order(product, int(bid_px), int(buy_qty)))
        if sell_qty > 0:
            orders.append(Order(product, int(ask_px), -int(sell_qty)))
        return orders

    # ------------------------------------------------------------------
    # Voucher strategy
    # ------------------------------------------------------------------

    def option_open_edge(self, product: str, spread: int, vega: float) -> float:
        edge = self.BASE_OPTION_OPEN_EDGE.get(product, 0.90)
        edge += 0.20 * max(0, spread - 2)
        if vega <= self.LOW_VEGA_CUTOFF:
            edge += self.LOW_VEGA_THR_ADJ
        return edge

    def option_target_size(self, edge: float, open_edge: float, vega: float) -> int:
        ratio = max(0.0, edge / max(open_edge, 1e-6))
        # Tighter ramp: linear up to ratio=2, capped.
        raw = 12 + int(20 * min(2.0, ratio) / 2.0)
        if vega <= self.LOW_VEGA_CUTOFF:
            raw = int(raw * 0.55)
        return min(max(1, raw), self.MAX_OPTION_TRADE_SIZE)

    def trade_voucher_iv_scalp(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        S: float,
        T: float,
        smile: Tuple[float, float, float],
        timestamp: int,
        data: Dict[str, Any],
    ) -> Tuple[List[Order], float]:
        """Returns (orders, signed_delta_after_orders)."""
        orders: List[Order] = []

        if product not in self.ACTIVE_VOUCHERS:
            return orders, 0.0

        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return orders, position * 0.5  # rough fallback delta

        top_mid = (bid + ask) / 2.0
        spread = ask - bid

        K = self.VOUCHER_STRIKES[product]
        a, b, c = smile
        sigma = self.smile_iv_from_coefs(S, K, a, b, c)
        theo, delta, vega = self.bs_call(S, K, T, sigma)
        if theo is None:
            return orders, position * 0.5

        raw_diff = top_mid - theo
        mean_diff = self.ema_update(data, f"{product}_mean_theo_diff",
                                    raw_diff, self.THEO_DIFF_WINDOW)
        fair = theo + self.MEAN_DIFF_SHRINK * mean_diff
        model_diff = top_mid - fair

        open_edge = self.option_open_edge(product, spread, vega)

        # Track when this position was opened (for max-hold stop).
        pos_age_key = f"{product}_pos_age"
        prev_pos = data.get(f"{product}_prev_pos", 0)
        if (position > 0) != (prev_pos > 0) or (position < 0) != (prev_pos < 0) or position == 0:
            data[pos_age_key] = timestamp
        data[f"{product}_prev_pos"] = position
        age = timestamp - data.get(pos_age_key, timestamp)
        force_close = abs(position) > 0 and age > self.MAX_HOLD_TICKS * 100  # ticks->ts

        buy_cap = self.buy_capacity(product, position)
        sell_cap = self.sell_capacity(product, position)

        sell_edge = bid - fair  # >0 means market bid above fair -> sell into it
        buy_edge = fair - ask   # >0 means market ask below fair -> lift it

        new_position_after = position  # to compute signed delta at end

        # ------------------------------------------------------------------
        # 1. Opening trades only when edge clearly exceeds threshold.
        #    Warmup guard: no new opens until models have converged.
        # ------------------------------------------------------------------
        warmed_up = timestamp >= self.WARMUP_TIMESTAMPS

        if sell_edge > open_edge and sell_cap > 0 and not force_close and warmed_up:
            target = -int(min(
                self.MAX_OPTION_TARGET,
                self.MAX_OPTION_TARGET * min(2.0, sell_edge / open_edge) / 2.0,
            ))
            desired = max(0, position - target)
            size = self.option_target_size(sell_edge, open_edge, vega)
            book_qty = abs(od.buy_orders[bid])
            qty = min(sell_cap, desired, size, book_qty)
            if qty > 0:
                orders.append(Order(product, int(bid), -int(qty)))
                new_position_after = position - qty
                return orders, new_position_after * delta

        if buy_edge > open_edge and buy_cap > 0 and not force_close and warmed_up:
            target = int(min(
                self.MAX_OPTION_TARGET,
                self.MAX_OPTION_TARGET * min(2.0, buy_edge / open_edge) / 2.0,
            ))
            desired = max(0, target - position)
            size = self.option_target_size(buy_edge, open_edge, vega)
            book_qty = abs(od.sell_orders[ask])
            qty = min(buy_cap, desired, size, book_qty)
            if qty > 0:
                orders.append(Order(product, int(ask), int(qty)))
                new_position_after = position + qty
                return orders, new_position_after * delta

        # ------------------------------------------------------------------
        # 2. Faster flatten when edge has disappeared OR hold time exceeded.
        # ------------------------------------------------------------------
        if position > 0:
            should_close = (sell_edge >= -self.CLOSE_SLACK) or (model_diff >= 0) or force_close
            if should_close:
                book_qty = abs(od.buy_orders[bid])
                # Bigger closing chunk than we use to open -- accept paying spread
                # to get out before a bigger move hits us.
                qty = min(position, book_qty, self.MAX_OPTION_TRADE_SIZE * 2)
                # Allow a slightly worse fill when force_close.
                tolerable_loss = -0.5 if force_close else -0.20
                if qty > 0 and sell_edge >= tolerable_loss:
                    orders.append(Order(product, int(bid), -int(qty)))
                    new_position_after = position - qty
                    return orders, new_position_after * delta

        elif position < 0:
            should_close = (buy_edge >= -self.CLOSE_SLACK) or (model_diff <= 0) or force_close
            if should_close:
                book_qty = abs(od.sell_orders[ask])
                qty = min(-position, book_qty, self.MAX_OPTION_TRADE_SIZE * 2)
                tolerable_loss = -0.5 if force_close else -0.20
                if qty > 0 and buy_edge >= tolerable_loss:
                    orders.append(Order(product, int(ask), int(qty)))
                    new_position_after = position + qty
                    return orders, new_position_after * delta

        # ------------------------------------------------------------------
        # 3. Small passive quotes inside spread.
        # ------------------------------------------------------------------
        passive_edge = open_edge + self.OPTION_PASSIVE_EDGE_ADD
        passive_size = self.MAX_OPTION_PASSIVE_SIZE

        # Inventory-aware passive quoting.
        allow_buy = position < 80
        allow_sell = position > -80

        bid_quote = int(math.floor(fair - passive_edge))
        ask_quote = int(math.ceil(fair + passive_edge))

        if allow_buy and buy_cap > 0 and bid < bid_quote < ask:
            qty = min(buy_cap, passive_size)
            if qty > 0:
                orders.append(Order(product, int(bid_quote), int(qty)))
        if allow_sell and sell_cap > 0 and bid < ask_quote < ask:
            qty = min(sell_cap, passive_size)
            if qty > 0:
                orders.append(Order(product, int(ask_quote), -int(qty)))

        return orders, new_position_after * delta

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        # ----- Underlying mid -----
        underlying_mid: Optional[float] = None
        if VELVET in state.order_depths:
            underlying_mid = self.get_mid(state.order_depths[VELVET])

        # ----- Refit smile dynamically -----
        smile: Optional[Tuple[float, float, float]] = None
        if underlying_mid is not None:
            T = self.tte_years(state.timestamp)
            smile = self.get_smile_coefs(underlying_mid, T,
                                         state.order_depths, data)

        # ----- Voucher orders FIRST (so we know option delta exposure) -----
        total_option_delta = 0.0
        if underlying_mid is not None and smile is not None:
            T = self.tte_years(state.timestamp)
            for v in self.ACTIVE_VOUCHERS:
                if v not in state.order_depths:
                    continue
                pos = state.position.get(v, 0)
                v_orders, v_delta = self.trade_voucher_iv_scalp(
                    product=v, od=state.order_depths[v], position=pos,
                    S=underlying_mid, T=T, smile=smile,
                    timestamp=state.timestamp, data=data,
                )
                if v_orders:
                    result[v] = v_orders
                total_option_delta += v_delta

        # ----- HYDROGEL_PACK (independent) -----
        if HYDROGEL in state.order_depths:
            pos = state.position.get(HYDROGEL, 0)
            orders = self.trade_hydrogel(HYDROGEL, state.order_depths[HYDROGEL],
                                         pos, data)
            if orders:
                result[HYDROGEL] = orders

        # ----- VELVETFRUIT_EXTRACT (with delta-hedge bias) -----
        if VELVET in state.order_depths:
            pos = state.position.get(VELVET, 0)
            # We want net underlying exposure ~ 0, so target velvet position
            # = -DELTA_HEDGE_RATIO * total_option_delta.
            hedge_target = -self.DELTA_HEDGE_RATIO * total_option_delta
            # Cap the hedge target so it doesn't override velvet's own MM logic.
            hedge_target = max(-150.0, min(150.0, hedge_target))
            orders = self.trade_velvetfruit(
                VELVET, state.order_depths[VELVET], pos, data, hedge_target,
            )
            if orders:
                result[VELVET] = orders

        traderData = json.dumps(data, separators=(",", ":"))
        return result, conversions, traderData