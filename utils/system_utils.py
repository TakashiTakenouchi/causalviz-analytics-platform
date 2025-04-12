import os
import psutil
import logging
import pandas as pd
import numpy as np
import json
import gc
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

# ロガーの設定
logger = logging.getLogger(__name__)

def monitor_memory() -> float:
    """
    現在のメモリ使用率を取得する
    
    Returns:
    --------
    float
        メモリ使用率（パーセント）
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # ログにメモリ使用量の詳細を記録
        logger.debug(f"メモリ使用量: {memory_info.rss / (1024 * 1024):.2f} MB, 使用率: {memory_percent:.2f}%")
        
        return memory_percent
    except Exception as e:
        logger.error(f"メモリ使用率の取得中にエラーが発生しました: {str(e)}")
        return -1.0

def cleanup_memory(threshold: float = 80.0) -> float:
    """
    メモリ使用率が閾値を超えた場合にガベージコレクションを実行する
    
    Parameters:
    -----------
    threshold : float
        メモリ使用率の閾値（パーセント）
        
    Returns:
    --------
    float
        クリーンアップ後のメモリ使用率
    """
    current_usage = monitor_memory()
    
    if current_usage > threshold:
        logger.warning(f"メモリ使用率が閾値を超えています: {current_usage:.2f}% > {threshold:.2f}%")
        
        # 未使用オブジェクトの解放を試みる
        before_gc = current_usage
        gc.collect()
        after_gc = monitor_memory()
        
        logger.info(f"ガベージコレクション実行: {before_gc:.2f}% -> {after_gc:.2f}% ({before_gc - after_gc:.2f}% 削減)")
        return after_gc
    
    return current_usage

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    様々な形式のデータファイルを読み込む
    
    Parameters:
    -----------
    file_path : str
        読み込むファイルのパス
    **kwargs : dict
        読み込み関数に渡す追加パラメータ
        
    Returns:
    --------
    pd.DataFrame
        読み込まれたデータフレーム
    """
    start_time = time.time()
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # ファイル形式に応じた読み込み
        if file_extension == '.csv':
            # CSVファイルの読み込み
            df = pd.read_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            # Excelファイルの読み込み
            df = pd.read_excel(file_path, **kwargs)
        elif file_extension == '.json':
            # JSONファイルの読み込み
            df = pd.read_json(file_path, **kwargs)
        elif file_extension == '.parquet':
            # Parquetファイルの読み込み
            df = pd.read_parquet(file_path, **kwargs)
        elif file_extension == '.feather':
            # Featherファイルの読み込み
            df = pd.read_feather(file_path, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            # Pickleファイルの読み込み
            df = pd.read_pickle(file_path, **kwargs)
        elif file_extension == '.h5':
            # HDF5ファイルの読み込み
            key = kwargs.pop('key', None)
            df = pd.read_hdf(file_path, key=key, **kwargs)
        elif file_extension == '.dta':
            # Stataファイルの読み込み
            df = pd.read_stata(file_path, **kwargs)
        elif file_extension == '.sas7bdat':
            # SASファイルの読み込み
            df = pd.read_sas(file_path, **kwargs)
        else:
            raise ValueError(f"サポートされていないファイル形式です: {file_extension}")
        
        # 読み込み時間とデータサイズのログ
        elapsed_time = time.time() - start_time
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB単位
        
        logger.info(f"データ読み込み完了: {file_path}")
        logger.info(f"  行数: {len(df)}, 列数: {len(df.columns)}")
        logger.info(f"  メモリ使用量: {memory_usage:.2f} MB")
        logger.info(f"  読み込み時間: {elapsed_time:.2f} 秒")
        
        return df
        
    except Exception as e:
        logger.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
        raise

def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    データフレームを様々な形式で保存する
    
    Parameters:
    -----------
    df : pd.DataFrame
        保存するデータフレーム
    file_path : str
        保存先のファイルパス
    **kwargs : dict
        保存関数に渡す追加パラメータ
    """
    start_time = time.time()
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # ファイル形式に応じた保存
        if file_extension == '.csv':
            # CSVファイルの保存
            index = kwargs.pop('index', False)
            df.to_csv(file_path, index=index, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            # Excelファイルの保存
            index = kwargs.pop('index', False)
            df.to_excel(file_path, index=index, **kwargs)
        elif file_extension == '.json':
            # JSONファイルの保存
            orient = kwargs.pop('orient', 'records')
            df.to_json(file_path, orient=orient, **kwargs)
        elif file_extension == '.parquet':
            # Parquetファイルの保存
            index = kwargs.pop('index', False)
            df.to_parquet(file_path, index=index, **kwargs)
        elif file_extension == '.feather':
            # Featherファイルの保存
            df.to_feather(file_path, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            # Pickleファイルの保存
            df.to_pickle(file_path, **kwargs)
        elif file_extension == '.h5':
            # HDF5ファイルの保存
            key = kwargs.pop('key', 'data')
            df.to_hdf(file_path, key=key, **kwargs)
        else:
            raise ValueError(f"サポートされていないファイル形式です: {file_extension}")
        
        # 保存時間のログ
        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB単位
        
        logger.info(f"データ保存完了: {file_path}")
        logger.info(f"  ファイルサイズ: {file_size:.2f} MB")
        logger.info(f"  保存時間: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"データ保存中にエラーが発生しました: {str(e)}")
        raise

def setup_logging(log_file: Optional[str] = None, 
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    ロギングの設定を行う
    
    Parameters:
    -----------
    log_file : Optional[str]
        ログファイルのパス（Noneの場合はファイル出力なし）
    console_level : int
        コンソール出力のログレベル
    file_level : int
        ファイル出力のログレベル
        
    Returns:
    --------
    logging.Logger
        設定されたロガー
    """
    # ルートロガーの取得
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 最も低いレベルに設定
    
    # 既存のハンドラをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # コンソール出力の設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # ファイル出力の設定（指定された場合）
    if log_file:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    logger.info("ロギングの設定が完了しました")
    if log_file:
        logger.info(f"ログファイル: {log_file}")
    
    return logger

def get_file_info(file_path: str) -> Dict:
    """
    ファイルの情報を取得する
    
    Parameters:
    -----------
    file_path : str
        情報を取得するファイルのパス
        
    Returns:
    --------
    Dict
        ファイル情報を含む辞書
    """
    try:
        # ファイルが存在するか確認
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        # ファイル情報の取得
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        modified_time = datetime.fromtimestamp(file_stat.st_mtime)
        created_time = datetime.fromtimestamp(file_stat.st_ctime)
        
        # ファイル拡張子
        _, file_extension = os.path.splitext(file_path)
        
        # 結果の辞書
        file_info = {
            'path': os.path.abspath(file_path),
            'name': os.path.basename(file_path),
            'directory': os.path.dirname(os.path.abspath(file_path)),
            'extension': file_extension,
            'size_bytes': file_size,
            'size_kb': file_size / 1024,
            'size_mb': file_size / (1024 * 1024),
            'created': created_time,
            'modified': modified_time,
            'is_file': os.path.isfile(file_path),
            'is_directory': os.path.isdir(file_path)
        }
        
        return file_info
        
    except Exception as e:
        logger.error(f"ファイル情報の取得中にエラーが発生しました: {str(e)}")
        raise

def list_directory(directory: str, pattern: Optional[str] = None) -> List[Dict]:
    """
    ディレクトリ内のファイル一覧を取得する
    
    Parameters:
    -----------
    directory : str
        一覧を取得するディレクトリのパス
    pattern : Optional[str]
        ファイル名のパターン（例: '*.csv'）
        
    Returns:
    --------
    List[Dict]
        ファイル情報のリスト
    """
    try:
        # ディレクトリが存在するか確認
        if not os.path.exists(directory):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"指定されたパスはディレクトリではありません: {directory}")
        
        # ファイル一覧の取得
        file_list = []
        
        import glob
        if pattern:
            # パターンに一致するファイルを取得
            pattern_path = os.path.join(directory, pattern)
            files = glob.glob(pattern_path)
        else:
            # すべてのファイルを取得
            files = [os.path.join(directory, f) for f in os.listdir(directory)]
        
        # ファイル情報の取得
        for file_path in files:
            try:
                file_info = get_file_info(file_path)
                file_list.append(file_info)
            except Exception as e:
                logger.warning(f"ファイル情報の取得中にエラーが発生しました: {file_path}, {str(e)}")
        
        # 名前でソート
        file_list.sort(key=lambda x: x['name'])
        
        return file_list
        
    except Exception as e:
        logger.error(f"ディレクトリ一覧の取得中にエラーが発生しました: {str(e)}")
        raise

def save_config(config: Dict, file_path: str) -> None:
    """
    設定をJSONファイルに保存する
    
    Parameters:
    -----------
    config : Dict
        保存する設定
    file_path : str
        保存先のファイルパス
    """
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # JSONファイルに保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"設定を保存しました: {file_path}")
        
    except Exception as e:
        logger.error(f"設定の保存中にエラーが発生しました: {str(e)}")
        raise

def load_config(file_path: str) -> Dict:
    """
    設定をJSONファイルから読み込む
    
    Parameters:
    -----------
    file_path : str
        読み込むファイルのパス
        
    Returns:
    --------
    Dict
        読み込まれた設定
    """
    try:
        # ファイルが存在するか確認
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {file_path}")
        
        # JSONファイルから読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"設定を読み込みました: {file_path}")
        
        return config
        
    except Exception as e:
        logger.error(f"設定の読み込み中にエラーが発生しました: {str(e)}")
        raise

def get_system_info() -> Dict:
    """
    システム情報を取得する
    
    Returns:
    --------
    Dict
        システム情報を含む辞書
    """
    try:
        # CPUの情報
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # メモリの情報
        memory = psutil.virtual_memory()
        
        # ディスクの情報
        disk = psutil.disk_usage('/')
        
        # プロセスの情報
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        # システム情報の辞書
        system_info = {
            'cpu': {
                'physical_cores': cpu_count,
                'logical_cores': cpu_count_logical,
                'usage_percent': cpu_percent
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent
            },
            'process': {
                'pid': os.getpid(),
                'memory_rss_mb': process_memory.rss / (1024**2),
                'memory_vms_mb': process_memory.vms / (1024**2),
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(interval=1)
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"システム情報の取得中にエラーが発生しました: {str(e)}")
        # 最低限の情報を返す
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def limit_data_size(df: pd.DataFrame, max_rows: int = 10000, method: str = 'random') -> pd.DataFrame:
    """
    データフレームのサイズを制限する
    
    Parameters:
    -----------
    df : pd.DataFrame
        制限するデータフレーム
    max_rows : int
        最大行数
    method : str
        サンプリング方法 ('random', 'head', 'tail')
        
    Returns:
    --------
    pd.DataFrame
        サイズが制限されたデータフレーム
    """
    if len(df) <= max_rows:
        return df
    
    logger.info(f"データサイズを制限します: {len(df)} -> {max_rows} 行")
    
    if method == 'random':
        # ランダムサンプリング
        return df.sample(n=max_rows, random_state=42)
    elif method == 'head':
        # 先頭から取得
        return df.head(max_rows)
    elif method == 'tail':
        # 末尾から取得
        return df.tail(max_rows)
    else:
        raise ValueError(f"サポートされていないサンプリング方法です: {method}")

def timer(func):
    """
    関数の実行時間を計測するデコレータ
    
    Parameters:
    -----------
    func : callable
        計測対象の関数
        
    Returns:
    --------
    callable
        ラップされた関数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger.debug(f"関数 {func.__name__} の実行時間: {elapsed_time:.4f} 秒")
        return result
    
    return wrapper

# プラットフォーム情報のインポート（get_system_info関数で使用）
import platform
