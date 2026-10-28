import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from jito_searcher_client import get_async_searcher_client
from jito_searcher_client.convert import versioned_tx_to_protobuf_packet
from jito_searcher_client.generated.bundle_pb2 import Bundle
from jito_searcher_client.generated.searcher_pb2 import (
    GetTipAccountsRequest,
    SendBundleRequest,
    SubscribeBundleResultsRequest,
)
from solana.rpc.async_api import AsyncClient
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

MEMO_PROGRAM = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
LANDED_STATES = {"processed", "finalized"}

@dataclass
class Outcome:
    uuid: str
    round_idx: int
    sent_at: float
    state: Optional[str] = None
    resolved_at: Optional[float] = None
    reason: str = ""

    @property
    def landed(self) -> bool:
        return self.state in LANDED_STATES

    @property
    def latency(self) -> float:
        if self.resolved_at is None or not self.landed:
            return float("nan")
        return self.resolved_at - self.sent_at


@dataclass
class Ledger:
    rows: Dict[str, Outcome] = field(default_factory=dict)
    early: Dict[str, tuple] = field(default_factory=dict)

    def register(self, outcome: Outcome) -> None:
        self.rows[outcome.uuid] = outcome
        if outcome.uuid in self.early:
            state, ts, reason = self.early.pop(outcome.uuid)
            self.resolve(outcome.uuid, state, ts, reason)

    def resolve(self, uuid: str, state: str, ts: float, reason: str = "") -> None:
        row = self.rows.get(uuid)
        if row is None:
            self.early[uuid] = (state, ts, reason)
            return
        if row.state in LANDED_STATES:
            return
        row.state, row.resolved_at, row.reason = state, ts, reason

    def frame(self) -> pd.DataFrame:
        records = [
            {
                "uuid": r.uuid,
                "round": r.round_idx,
                "sent_at": r.sent_at,
                "state": r.state or "unseen",
                "landed": r.landed,
                "time_to_land_s": r.latency,
                "reason": r.reason,
            }
            for r in self.rows.values()
        ]
        return pd.DataFrame.from_records(records)


async def consume_results(client, ledger: Ledger) -> None:
    stream = client.SubscribeBundleResults(SubscribeBundleResultsRequest())
    async for msg in stream:
        now = time.time()
        kind = msg.WhichOneof("result")
        if kind is not None:
            reason = str(getattr(msg, kind)).replace("\n", " ")[:120] if kind in ("rejected", "dropped") else ""
            ledger.resolve(msg.bundle_id, kind, now, reason)


def build_tx(kp: Keypair, tip_to: Pubkey, tip_lamports: int, blockhash, round_idx: int) -> VersionedTransaction:
    memo = Instruction(
        MEMO_PROGRAM,
        f"landrate-{round_idx}-{random.getrandbits(32):08x}".encode(),
        [AccountMeta(kp.pubkey(), True, False)],
    )
    tip = transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=tip_to, lamports=tip_lamports))
    msg = MessageV0.try_compile(kp.pubkey(), [memo, tip], [], blockhash)
    return VersionedTransaction(msg, [kp])


async def run(args: argparse.Namespace) -> pd.DataFrame:
    kp = Keypair.from_json(Path(args.keypair).read_text())
    client = await get_async_searcher_client(args.block_engine, kp)
    rpc = AsyncClient(args.rpc)
    ledger = Ledger()
    listener = asyncio.create_task(consume_results(client, ledger))

    tip_resp = await client.GetTipAccounts(GetTipAccountsRequest())
    tip_accounts = [Pubkey.from_string(a) for a in tip_resp.accounts]

    try:
        for i in range(args.rounds):
            bh = (await rpc.get_latest_blockhash()).value.blockhash
            tx = build_tx(kp, random.choice(tip_accounts), args.tip, bh, i)
            req = SendBundleRequest(bundle=Bundle(header=None, packets=[versioned_tx_to_protobuf_packet(tx)]))
            sent_at = time.time()
            resp = await client.SendBundle(req)
            ledger.register(Outcome(uuid=resp.uuid, round_idx=i, sent_at=sent_at))
            await asyncio.sleep(args.interval)
        await asyncio.sleep(args.settle)
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
        await rpc.close()
    return ledger.frame()


def report(df: pd.DataFrame, out: Path) -> None:
    sent, landed = len(df), int(df["landed"].sum()) if len(df) else 0
    ttl = df["time_to_land_s"].dropna()
    summary = ttl.describe() if len(ttl) else pd.Series({"count": 0.0})
    print(summary.round(3).to_frame("time_to_land_s"))
    rate = landed / sent if sent else float("nan")
    print(f"landed/sent: {landed}/{sent} ({np.round(rate * 100, 1)}%)")
    df.to_csv(out, index=False)
    print(f"wrote {out}")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--block-engine", default="frankfurt.mainnet.block-engine.jito.wtf")
    p.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    p.add_argument("--keypair", required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--tip", type=int, default=10_000)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--settle", type=float, default=20.0)
    p.add_argument("--out", default="land_rate.csv")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse()
    report(asyncio.run(run(ns)), Path(ns.out))
