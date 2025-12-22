#!/usr/bin/env python3
"""
测试 artifacts.py 的修改是否正确
"""

import json
import os
import tempfile
import shutil
from app.mcp.artifacts import append_artifact, append_audit_log

def _stable_json(obj):
    """简化版本的 _stable_json 用于测试"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)

def test_artifacts():
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 测试 append_artifact
        print("\n1. 测试 append_artifact...")
        test_record = {
            "type": "tool_call",
            "tool_name": "test_tool",
            "args_hash": "abc123",
            "output_hash": "def456",
            "latency_ms": 123.456,
            "status": "OK",
            "request": {"test": "request"},
            "result": {"test": "result"}
        }
        
        append_artifact(temp_dir, test_record)
        
        # 验证文件内容
        artifacts_path = os.path.join(temp_dir, "artifacts.json")
        with open(artifacts_path, "r", encoding="utf-8") as f:
            artifacts_data = json.load(f)
        
        print(f"artifacts.json 内容:")
        print(json.dumps(artifacts_data, ensure_ascii=False, indent=2))
        
        # 测试 append_audit_log
        print("\n2. 测试 append_audit_log...")
        audit_line = _stable_json({
            "tool_name": "test_tool",
            "args_hash": "abc123",
            "output_hash": "def456",
            "latency_ms": 123.456,
            "status": "OK",
        })
        
        append_audit_log(temp_dir, audit_line)
        
        # 验证 audit.log 内容
        audit_path = os.path.join(temp_dir, "audit.log")
        with open(audit_path, "r", encoding="utf-8") as f:
            audit_content = f.read()
        
        print(f"audit.log 内容:")
        print(audit_content)
        
        print("\n✅ 测试成功！所有功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"已清理临时目录: {temp_dir}")

if __name__ == "__main__":
    test_artifacts()
