from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json


ASH = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
OLIVIA = "Olivia"

POSITION_LIMITS: Dict[str, int] = {
    ASH: 80,
    PEPPER: 80,
}

ASH_FAIR_VALUE = 10000

# ---------- ASH tuning ----------
ASH_SOFT = 45
ASH_HARD = 79
ASH_BASE_SIZE = 70
ASH_WIDE_SIZE = 70
ASH_LAYER2_SIZE = 24

# ---------- PEPPER tuning ----------
PEPPER_HISTORY = 40            # extended for slope regression
PEPPER_OLIVIA_MEMORY = 900

PEPPER_BASE_SIZE = 20
PEPPER_FAVORED_SIZE = 55
PEPPER_UNFAVORED_SIZE = 4
PEPPER_LAYER2_SIZE = 15

PEPPER_TARGET_SCALE = 22.0
PEPPER_TARGET_CAP = 72

PEPPER_STRONG_SIGNAL = 1.0
PEPPER_VERY_STRONG_SIGNAL = 1.8

# NEW: slope thresholds (ticks per timestep)
PEPPER_SLOPE_CONFIRM = 0.05    # trend confirmed -> commit to max target
PEPPER_SLOPE_STRONG  = 0.10    # extreme trend -> pay-the-premium taking


def clamp(value: float, low: int, high: int) -> int:
    if high < low:
        return 0
    return max(low, min(high, int(round(value))))


def best_bid(order_depth: OrderDepth) -> Optional[int]:
    return max(order_depth.buy_orders) if order_depth.buy_orders else None


def best_ask(order_depth: OrderDepth) -> Optional[int]:
    return min(order_depth.sell_orders) if order_depth.sell_orders else None


def sorted_bids(order_depth: OrderDepth) -> List[Tuple[int, int]]:
    return sorted(
        [(p, abs(v)) for p, v in order_depth.buy_orders.items()],
        key=lambda x: x[0], reverse=True,
    )


def sorted_asks(order_depth: OrderDepth) -> List[Tuple[int, int]]:
    return sorted(
        [(p, abs(v)) for p, v in order_depth.sell_orders.items()],
        key=lambda x: x[0],
    )


def filtered_best_bid(order_depth: OrderDepth, min_size: int = 10) -> Optional[int]:
    valid = [p for p, v in order_depth.buy_orders.items() if abs(v) >= min_size]
    return max(valid) if valid else best_bid(order_depth)


def filtered_best_ask(order_depth: OrderDepth, min_size: int = 10) -> Optional[int]:
    valid = [p for p, v in order_depth.sell_orders.items() if abs(v) >= min_size]
    return min(valid) if valid else best_ask(order_depth)


def mid_price(order_depth: OrderDepth) -> Optional[float]:
    b, a = best_bid(order_depth), best_ask(order_depth)
    return None if b is None or a is None else (b + a) / 2.0


def filtered_mid_price(order_depth: OrderDepth, min_size: int = 10) -> Optional[float]:
    b = filtered_best_bid(order_depth, min_size)
    a = filtered_best_ask(order_depth, min_size)
    return None if b is None or a is None else (b + a) / 2.0


def microprice(order_depth: OrderDepth) -> Optional[float]:
    b, a = best_bid(order_depth), best_ask(order_depth)
    if b is None or a is None:
        return None
    bv = abs(order_depth.buy_orders[b])
    av = abs(order_depth.sell_orders[a])
    total = bv + av
    if total == 0:
        return (b + a) / 2.0
    return (b * av + a * bv) / total


def book_imbalance(order_depth: OrderDepth, levels: int = 3) -> float:
    bv = sum(v for _, v in sorted_bids(order_depth)[:levels])
    av = sum(v for _, v in sorted_asks(order_depth)[:levels])
    total = bv + av
    return 0.0 if total == 0 else (bv - av) / total


def linear_slope(history: List[float], window: int = 30) -> float:
    """Linear regression slope in price-ticks per timestep."""
    if len(history) < 10:
        return 0.0
    n = min(window, len(history))
    recent = history[-n:]
    mean_x = (n - 1) / 2.0
    mean_y = sum(recent) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = i - mean_x
        num += dx * (recent[i] - mean_y)
        den += dx * dx
    return num / den if den > 0 else 0.0


class StaticTrader:
    NAME = ASH

    def get_orders(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = POSITION_LIMITS[self.NAME]

        buy_capacity = limit - position
        sell_capacity = limit + position

        for ask_price, ask_volume in sorted_asks(order_depth):
            if ask_price < ASH_FAIR_VALUE:
                qty = min(ask_volume, buy_capacity)
                if qty > 0:
                    orders.append(Order(self.NAME, int(ask_price), int(qty)))
                    buy_capacity -= qty
            else:
                break

        for bid_price, bid_volume in sorted_bids(order_depth):
            if bid_price > ASH_FAIR_VALUE:
                qty = min(bid_volume, sell_capacity)
                if qty > 0:
                    orders.append(Order(self.NAME, int(bid_price), -int(qty)))
                    sell_capacity -= qty
            else:
                break

        bid = best_bid(order_depth)
        ask = best_ask(order_depth)
        if bid is None or ask is None:
            return orders

        spread = ask - bid

        if spread >= 3:
            bid_quote = min(bid + 1, ASH_FAIR_VALUE - 1)
            ask_quote = max(ask - 1, ASH_FAIR_VALUE + 1)
        else:
            bid_quote = min(bid, ASH_FAIR_VALUE - 1)
            ask_quote = max(ask, ASH_FAIR_VALUE + 1)

        if position >= ASH_HARD:
            bid_quote = ASH_FAIR_VALUE - 3
            ask_quote = ASH_FAIR_VALUE + 1
        elif position >= ASH_SOFT:
            bid_quote = min(bid_quote, ASH_FAIR_VALUE - 2)
            ask_quote = min(ask_quote, ASH_FAIR_VALUE)
        elif position <= -ASH_HARD:
            bid_quote = ASH_FAIR_VALUE
            ask_quote = ASH_FAIR_VALUE + 3
        elif position <= -ASH_SOFT:
            bid_quote = max(bid_quote, ASH_FAIR_VALUE)
            ask_quote = max(ask_quote, ASH_FAIR_VALUE + 2)

        bid_quote = int(min(bid_quote, ASH_FAIR_VALUE))
        ask_quote = int(max(ask_quote, ASH_FAIR_VALUE))

        if bid_quote >= ask_quote:
            bid_quote = ASH_FAIR_VALUE - 1
            ask_quote = ASH_FAIR_VALUE + 1

        base_size = ASH_BASE_SIZE
        if spread >= 5 and abs(position) <= 20:
            base_size = 42
        if spread >= 7 and abs(position) <= 12:
            base_size = ASH_WIDE_SIZE

        buy_size = min(buy_capacity, base_size)
        sell_size = min(sell_capacity, base_size)

        if position > ASH_SOFT:
            buy_size = min(buy_size, 10)
            sell_size = min(sell_capacity, max(base_size, 38))
        elif position < -ASH_SOFT:
            buy_size = min(buy_capacity, max(base_size, 38))
            sell_size = min(sell_size, 10)

        if buy_size > 0:
            orders.append(Order(self.NAME, bid_quote, int(buy_size)))
        if sell_size > 0:
            orders.append(Order(self.NAME, ask_quote, -int(sell_size)))

        # Second passive layer when book is wide and inventory is controlled
        if spread >= 5 and abs(position) <= 18:
            remaining_buy = max(0, buy_capacity - buy_size)
            remaining_sell = max(0, sell_capacity - sell_size)
            layer2_buy = min(remaining_buy, ASH_LAYER2_SIZE)
            layer2_sell = min(remaining_sell, ASH_LAYER2_SIZE)
            if layer2_buy > 0:
                orders.append(Order(self.NAME, ASH_FAIR_VALUE - 2, int(layer2_buy)))
            if layer2_sell > 0:
                orders.append(Order(self.NAME, ASH_FAIR_VALUE + 2, -int(layer2_sell)))

        # NEW: Third layer — catches book sweeps at deeper edge
        if spread >= 7 and abs(position) <= 15:
            remaining_buy3 = max(0, remaining_buy - layer2_buy)
            remaining_sell3 = max(0, remaining_sell - layer2_sell)
            layer3_buy = min(remaining_buy3, 12)
            layer3_sell = min(remaining_sell3, 12)
            if layer3_buy > 0:
                orders.append(Order(self.NAME, ASH_FAIR_VALUE - 3, int(layer3_buy)))
            if layer3_sell > 0:
                orders.append(Order(self.NAME, ASH_FAIR_VALUE + 3, -int(layer3_sell)))

        return orders


class DynamicTrader:
    NAME = PEPPER

    def update_olivia_signal(self, state: TradingState, timestamp: int, trader_data: dict) -> None:
        last_buy_ts = trader_data.get(f"{self.NAME}_olivia_buy_ts")
        last_sell_ts = trader_data.get(f"{self.NAME}_olivia_sell_ts")

        trades = list(state.market_trades.get(self.NAME, [])) \
               + list(state.own_trades.get(self.NAME, []))
        for t in trades:
            if t.buyer == OLIVIA:
                last_buy_ts = t.timestamp
            if t.seller == OLIVIA:
                last_sell_ts = t.timestamp

        trader_data[f"{self.NAME}_olivia_buy_ts"] = last_buy_ts
        trader_data[f"{self.NAME}_olivia_sell_ts"] = last_sell_ts

        buy_s = sell_s = 0.0
        if last_buy_ts is not None and timestamp >= last_buy_ts:
            age = timestamp - last_buy_ts
            if age <= PEPPER_OLIVIA_MEMORY:
                buy_s = max(0.0, 1.0 - age / PEPPER_OLIVIA_MEMORY)
        if last_sell_ts is not None and timestamp >= last_sell_ts:
            age = timestamp - last_sell_ts
            if age <= PEPPER_OLIVIA_MEMORY:
                sell_s = max(0.0, 1.0 - age / PEPPER_OLIVIA_MEMORY)

        trader_data[f"{self.NAME}_olivia_bias"] = 2.5 * (buy_s - sell_s)

    def estimate_fair_and_slope(
        self, order_depth: OrderDepth, trader_data: dict
    ) -> Tuple[Optional[float], float]:
        raw_mid = mid_price(order_depth)
        filt_mid = filtered_mid_price(order_depth, min_size=10)
        micro = microprice(order_depth)

        if raw_mid is None and filt_mid is None and micro is None:
            return None, 0.0

        reference = filt_mid if filt_mid is not None else (raw_mid if raw_mid is not None else micro)

        history = trader_data.setdefault(f"{self.NAME}_mid_history", [])
        history.append(reference)
        if len(history) > PEPPER_HISTORY:
            del history[:-PEPPER_HISTORY]

        # Linear regression slope - the new trend detector
        slope = linear_slope(history, window=30)
        trader_data[f"{self.NAME}_slope"] = slope

        # Legacy EMA trend as secondary signal
        trend = 0.0
        if len(history) >= 4:
            fast = sum(history[-4:]) / 4.0
            slow_window = min(16, len(history))
            slow = sum(history[-slow_window:]) / slow_window
            trend = fast - slow

        accel = 0.0
        if len(history) >= 8:
            prev_fast = sum(history[-8:-4]) / 4.0
            fast_now = sum(history[-4:]) / 4.0
            accel = fast_now - prev_fast

        # Imbalance weight dropped to 0.3 (was 1.8) - contrarian in trends
        imbalance = book_imbalance(order_depth) * 0.3
        olivia_bias = trader_data.get(f"{self.NAME}_olivia_bias", 0.0)

        components: List[Tuple[float, float]] = []
        if filt_mid is not None:
            components.append((filt_mid, 0.45))
        if micro is not None:
            components.append((micro, 0.35))
        if raw_mid is not None:
            components.append((raw_mid, 0.20))

        fair = sum(v * w for v, w in components)
        fair += 0.85 * trend
        fair += 0.35 * accel
        fair += imbalance
        fair += olivia_bias
        fair += slope * 8.0   # project 8 ticks forward

        trader_data[f"{self.NAME}_trend"] = trend
        trader_data[f"{self.NAME}_accel"] = accel
        trader_data[f"{self.NAME}_signal_strength"] = \
            abs(0.85 * trend + 0.35 * accel + imbalance + olivia_bias)

        return fair, slope

    def get_orders(
        self,
        order_depth: OrderDepth,
        position: int,
        timestamp: int,
        state: TradingState,
        trader_data: dict,
    ) -> List[Order]:
        self.update_olivia_signal(state, timestamp, trader_data)
        fair_value, slope = self.estimate_fair_and_slope(order_depth, trader_data)
        if fair_value is None:
            return []

        orders: List[Order] = []
        limit = POSITION_LIMITS[self.NAME]

        bid = best_bid(order_depth)
        ask = best_ask(order_depth)
        if bid is None or ask is None:
            return orders

        spread = ask - bid
        mid = (bid + ask) / 2.0

        olivia_bias = float(trader_data.get(f"{self.NAME}_olivia_bias", 0.0))
        signal_strength = float(trader_data.get(f"{self.NAME}_signal_strength", 0.0))

        signal_dir = fair_value - mid
        target_position = clamp(
            signal_dir * PEPPER_TARGET_SCALE, -PEPPER_TARGET_CAP, PEPPER_TARGET_CAP
        )

        # Slope-based target override with hysteresis
        # Once committed to a trend, require clearer reversal to exit
        was_strong_up = trader_data.get(f"{self.NAME}_sticky_up", False)
        was_strong_dn = trader_data.get(f"{self.NAME}_sticky_dn", False)

        strong_up = slope >= PEPPER_SLOPE_CONFIRM or (was_strong_up and slope >= 0.02)
        strong_dn = slope <= -PEPPER_SLOPE_CONFIRM or (was_strong_dn and slope <= -0.02)
        # Mutual exclusion — can't be both
        if strong_up and strong_dn:
            strong_up = slope > 0
            strong_dn = not strong_up

        extreme_up = slope >= PEPPER_SLOPE_STRONG
        extreme_dn = slope <= -PEPPER_SLOPE_STRONG

        trader_data[f"{self.NAME}_sticky_up"] = strong_up
        trader_data[f"{self.NAME}_sticky_dn"] = strong_dn

        if strong_up:
            target_position = max(target_position, PEPPER_TARGET_CAP)
        elif strong_dn:
            target_position = min(target_position, -PEPPER_TARGET_CAP)

        effective_position = position - target_position
        buy_capacity = limit - position
        sell_capacity = limit + position

        # Taking widths - pay premium in confirmed trend
        if extreme_up:
            buy_take_width = -1.0
            sell_take_width = 3.0
        elif strong_up:
            buy_take_width = 0.25
            sell_take_width = 2.25
        elif extreme_dn:
            buy_take_width = 3.0
            sell_take_width = -1.0
        elif strong_dn:
            buy_take_width = 2.25
            sell_take_width = 0.25
        elif signal_dir > 0:
            buy_take_width = 0.5 if signal_strength >= PEPPER_STRONG_SIGNAL else 0.75
            sell_take_width = 1.75
        elif signal_dir < 0:
            buy_take_width = 1.75
            sell_take_width = 0.5 if signal_strength >= PEPPER_STRONG_SIGNAL else 0.75
        else:
            buy_take_width = 1.0
            sell_take_width = 1.0

        clear_width = 0.0
        if abs(effective_position) >= 25:
            clear_width = 0.25
        if abs(effective_position) >= 45:
            clear_width = 0.5

        # Take asks
        for ap, av in sorted_asks(order_depth):
            favored = ap <= fair_value - buy_take_width
            clear_buy = effective_position < 0 and ap <= fair_value - clear_width
            if favored or clear_buy:
                qty = min(av, buy_capacity)
                if qty > 0:
                    orders.append(Order(self.NAME, int(ap), int(qty)))
                    buy_capacity -= qty
            else:
                break

        # Take bids
        for bp, bv in sorted_bids(order_depth):
            favored = bp >= fair_value + sell_take_width
            clear_sell = effective_position > 0 and bp >= fair_value + clear_width
            if favored or clear_sell:
                qty = min(bv, sell_capacity)
                if qty > 0:
                    orders.append(Order(self.NAME, int(bp), -int(qty)))
                    sell_capacity -= qty
            else:
                break

        # Passive making
        inv_skew = int(round(effective_position / 10.0))
        base_bid = int(round(fair_value)) - 1 - inv_skew
        base_ask = int(round(fair_value)) + 1 - inv_skew

        if spread >= 2:
            quote_bid = min(bid + 1, base_bid)
            quote_ask = max(ask - 1, base_ask)
        else:
            quote_bid = min(bid, base_bid)
            quote_ask = max(ask, base_ask)

        if signal_dir > 0.6:
            quote_bid += 1
        elif signal_dir < -0.6:
            quote_ask -= 1

        if olivia_bias > 1.0:
            quote_bid += 1
        elif olivia_bias < -1.0:
            quote_ask -= 1

        if position >= 70:
            quote_bid -= 2
            quote_ask -= 1
        elif position <= -70:
            quote_bid += 1
            quote_ask += 2

        quote_bid = int(min(quote_bid, int(fair_value)))
        quote_ask = int(max(quote_ask, int(fair_value)))

        if quote_bid >= quote_ask:
            quote_bid = int(fair_value) - 1
            quote_ask = int(fair_value) + 1

        quote_buy_size = PEPPER_BASE_SIZE
        quote_sell_size = PEPPER_BASE_SIZE

        if signal_dir > 0:
            quote_buy_size = PEPPER_FAVORED_SIZE
            quote_sell_size = PEPPER_UNFAVORED_SIZE
        elif signal_dir < 0:
            quote_buy_size = PEPPER_UNFAVORED_SIZE
            quote_sell_size = PEPPER_FAVORED_SIZE

        if signal_strength >= PEPPER_VERY_STRONG_SIGNAL:
            if signal_dir > 0:
                quote_buy_size += 8
            elif signal_dir < 0:
                quote_sell_size += 8

        if olivia_bias > 0.75:
            quote_buy_size += 5
        elif olivia_bias < -0.75:
            quote_sell_size += 5

        # Let position run in confirmed trends
        if strong_up and position < 65:
            quote_buy_size = max(quote_buy_size, 60)
            quote_sell_size = 0
        elif strong_dn and position > -65:
            quote_sell_size = max(quote_sell_size, 60)
            quote_buy_size = 0

        # Inventory control - only if NOT in confirmed trend
        if effective_position > 35 and not strong_up:
            quote_buy_size = min(quote_buy_size, 8)
            quote_sell_size = max(quote_sell_size, 28)
        elif effective_position < -35 and not strong_dn:
            quote_buy_size = max(quote_buy_size, 28)
            quote_sell_size = min(quote_sell_size, 8)

        if position > 72 and not strong_up:
            quote_buy_size = min(quote_buy_size, 4)
        if position < -72 and not strong_dn:
            quote_sell_size = min(quote_sell_size, 4)

        quote_buy_size = clamp(quote_buy_size, 0, buy_capacity)
        quote_sell_size = clamp(quote_sell_size, 0, sell_capacity)

        if quote_buy_size > 0:
            orders.append(Order(self.NAME, quote_bid, int(quote_buy_size)))
        if quote_sell_size > 0:
            orders.append(Order(self.NAME, quote_ask, -int(quote_sell_size)))

        # Backup layer
        rem_buy = max(0, buy_capacity - quote_buy_size)
        rem_sell = max(0, sell_capacity - quote_sell_size)

        if (signal_dir > 0.5 or strong_up) and rem_buy > 0:
            backup_buy = min(rem_buy, PEPPER_LAYER2_SIZE)
            backup_bid = int(min(quote_bid - 2, int(fair_value) - 1))
            if backup_buy > 0 and backup_bid < quote_bid:
                orders.append(Order(self.NAME, backup_bid, int(backup_buy)))
        elif (signal_dir < -0.5 or strong_dn) and rem_sell > 0:
            backup_sell = min(rem_sell, PEPPER_LAYER2_SIZE)
            backup_ask = int(max(quote_ask + 2, int(fair_value) + 1))
            if backup_sell > 0 and backup_ask > quote_ask:
                orders.append(Order(self.NAME, backup_ask, -int(backup_sell)))

        return orders


class Trader:
    def __init__(self):
        self.static_trader = StaticTrader()
        self.dynamic_trader = DynamicTrader()

    def run(self, state: TradingState):
        trader_data: dict = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}

        result: Dict[str, List[Order]] = {}

        if ASH in state.order_depths:
            position = state.position.get(ASH, 0)
            result[ASH] = self.static_trader.get_orders(
                order_depth=state.order_depths[ASH],
                position=position,
            )

        if PEPPER in state.order_depths:
            position = state.position.get(PEPPER, 0)
            result[PEPPER] = self.dynamic_trader.get_orders(
                order_depth=state.order_depths[PEPPER],
                position=position,
                timestamp=state.timestamp,
                state=state,
                trader_data=trader_data,
            )

        return result, 0, json.dumps(trader_data)