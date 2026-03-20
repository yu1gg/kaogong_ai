"""
题库数据加载与预处理工具模块。
负责隔离底层文件系统与上层业务逻辑。
"""
import json
from pathlib import Path
from typing import Dict, Any

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    加载并解析本地 JSON 数据文件。

    Args:
        file_path (str): JSON 文件的绝对或相对路径。

    Returns:
        Dict[str, Any]: 解析后的字典格式数据。

    Raises:
        FileNotFoundError: 文件路径不存在时抛出。
        ValueError: 文件内容不是合法的 JSON 格式时抛出。
    """
    path = Path(file_path)
    
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path.absolute()}")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式解析失败 [{file_path}]: {str(e)}")