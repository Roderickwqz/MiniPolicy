from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.build import build_workflow


RUN_META_DIR = Path("app/runs/meta")


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_meta(run_id: str, meta: Dict[str, Any]) -> str:
    RUN_META_DIR.mkdir(parents=True, exist_ok=True)
    p = RUN_META_DIR / f"{run_id}.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def load_meta(run_id: str) -> Optional[Dict[str, Any]]:
    p = RUN_META_DIR / f"{run_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get_latest_checkpoint_id(checkpointer, thread_id: str) -> Optional[str]:
    """
    取该 thread 最新 checkpoint 的 checkpoint_id（用于 replay/resume）
    """
    config = {"configurable": {"thread_id": thread_id}}
    items = list(checkpointer.list(config, limit=1))
    if not items:
        return None
    # CheckpointTuple: (config, checkpoint, metadata, parent_config) 结构在不同版本略有差异
    # 这里最稳是读取 tuple[0] 里的 config
    cfg = items[0][0]
    return (cfg.get("configurable") or {}).get("checkpoint_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to sample_run.json")
    parser.add_argument("--run_id", required=False, help="Optional run_id for resume")
    args = parser.parse_args()

    user_input = load_json(args.input)
    run_id = args.run_id or user_input.get("run_id") or f"run_{uuid.uuid4().hex[:10]}"

    # 固定 thread_id = run_id（Phase0 简化）
    meta = load_meta(run_id) or {}
    thread_id = meta.get("thread_id") or run_id

    sqlite_path = "app/runs/checkpoints.sqlite"

    with SqliteSaver.from_conn_string(sqlite_path) as checkpointer:
        """
        只要这个对象是“资源型（需要打开/关闭连接或句柄）”的 checkpointer/tool client，就应该用 with。
        你现在看到的 _GeneratorContextManager 报错，本质就是：
        from_conn_string() 返回的是一个 context manager（延迟打开连接）
        你不 with，就拿不到真正的 saver 实例（所以没有 .list() 等方法）

        LangGraph 的 persistence 机制里，一个 thread_id 会对应一串 checkpoints（每次状态推进后写入一个）。
        get_latest_checkpoint_id(checkpointer, thread_id) 的作用就是：
        查询这个 thread_id 下的 checkpoint 列表
        拿到“最新的那个” checkpoint 的 checkpoint_id
        在下一次 graph.invoke(..., config=...) 时告诉 LangGraph：
        从这个 checkpoint 开始 replay / resume

        thread_id：一本书的名字
        checkpoint_id：书签位置（第几页/哪个段落）
        """
        workflow = build_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        """
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver() # 不需要with
        graph = workflow.compile(checkpointer=checkpointer)
        """

        # 如果存在历史 checkpoint，则使用 checkpoint_id 做 replay/resume
        checkpoint_id = get_latest_checkpoint_id(checkpointer, thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        meta_path = save_meta(run_id, {"run_id": run_id, "thread_id": thread_id})
        print(f"[meta] {meta_path}")
        try:
            # 初次执行需要 input；如果你想做“严格 resume”，也可以在 interrupt 场景下传 None。
            # Phase0 用 replay + 原始 input 的方式即可：之前的步骤会 replay，之后继续跑。:contentReference[oaicite:2]{index=2}
            initial_state = {
                "run_id": run_id,
                "thread_id": thread_id,
                "user_input": user_input,
                "envelopes": [],
            }
            out = graph.invoke(initial_state, config=config)
            # 完成后输出验收信息
            print("\n[done]")
            print(f"run_id: {run_id}")
            print(f"thread_id: {thread_id}")
            print(f"report_path: {out.get('report_path')}")
            print(f"artifacts_path: {out.get('artifacts_path')}")
            print(f"envelopes: {len(out.get('envelopes', []))}")
        except KeyboardInterrupt:
            # 中断也要把关键信息打印出来
            print("\n[interrupt] KeyboardInterrupt")
            print(f"[resume] Re-run with: python -m app.cli.run --input {args.input} --run_id {run_id}")
            return


if __name__ == "__main__":
    # python -m app.cli.run --input sample_run.json
    main()
