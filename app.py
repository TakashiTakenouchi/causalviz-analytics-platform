import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64

# 自作モジュールのインポート
from modules.data_analyzer.data_analyzer import DataAnalyzer
from modules.data_analyzer.interactive_data_editor import InteractiveDataEditor
from modules.data_analyzer.pivot_table_builder import PivotTableBuilder
from modules.data_analyzer.visualization import Visualization
from modules.causal_inference.causal_inference import CausalInference
from modules.llm_explainer.llm_explainer import LLMExplainer
from utils.system_utils import (
    monitor_memory, cleanup_memory, load_data, save_data, 
    setup_logging, get_file_info, list_directory, 
    save_config, load_config, get_system_info
)

# ロギングの設定
logger = setup_logging(
    log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    console_level=logging.INFO,
    file_level=logging.DEBUG
)

# アプリケーションの設定
APP_TITLE = "CausalViz Analytics Platform"
APP_ICON = "📊"
APP_DESCRIPTION = "Streamlit + AutoGluon + LLMベースのデータ分析・因果推論プラットフォーム"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Manus AI"

# サンプルデータのパス
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'samples')

# セッション状態の初期化
def initialize_session_state():
    """セッション状態を初期化する"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_analyzer' not in st.session_state:
        st.session_state.data_analyzer = DataAnalyzer()
    if 'interactive_data_editor' not in st.session_state:
        st.session_state.interactive_data_editor = InteractiveDataEditor()
    if 'pivot_table_builder' not in st.session_state:
        st.session_state.pivot_table_builder = PivotTableBuilder()
    if 'visualization' not in st.session_state:
        st.session_state.visualization = Visualization()
    if 'causal_inference' not in st.session_state:
        st.session_state.causal_inference = CausalInference()
    if 'llm_explainer' not in st.session_state:
        st.session_state.llm_explainer = LLMExplainer()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "ホーム"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_info' not in st.session_state:
        st.session_state.system_info = get_system_info()
    if 'last_memory_check' not in st.session_state:
        st.session_state.last_memory_check = time.time()
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'theme': 'light',
            'language': 'ja',
            'max_memory_percent': 80,
            'auto_cleanup': True,
            'show_memory_warning': True,
            'default_chart_height': 500,
            'default_chart_width': 800,
            'default_chart_template': 'plotly_white',
            'default_color_scheme': 'viridis',
            'api_keys': {
                'openai': '',
                'anthropic': ''
            }
        }

# ナビゲーションメニュー
def render_navigation():
    """ナビゲーションメニューを描画する"""
    st.sidebar.title(f"{APP_ICON} {APP_TITLE}")
    st.sidebar.caption(f"バージョン: {APP_VERSION}")
    
    # メインメニュー
    menu_options = [
        "ホーム",
        "データ読み込み",
        "データエディタ",
        "ピボットテーブル",
        "データ可視化",
        "因果推論分析",
        "AIアシスタント",
        "設定"
    ]
    
    selected_menu = st.sidebar.radio("メニュー", menu_options)
    
    # メニュー選択時の処理
    if selected_menu != st.session_state.current_page:
        st.session_state.current_page = selected_menu
        # ページ切り替え時にメモリ使用状況をチェック
        if st.session_state.settings['auto_cleanup']:
            cleanup_memory(st.session_state.settings['max_memory_percent'])
    
    # メモリ使用状況の表示
    current_time = time.time()
    if current_time - st.session_state.last_memory_check > 30:  # 30秒ごとに更新
        memory_percent = monitor_memory()
        st.session_state.last_memory_check = current_time
        
        # メモリ使用率の警告表示
        if st.session_state.settings['show_memory_warning'] and memory_percent > st.session_state.settings['max_memory_percent']:
            st.sidebar.warning(f"メモリ使用率が高いです: {memory_percent:.1f}%")
        else:
            st.sidebar.info(f"メモリ使用率: {memory_percent:.1f}%")
    
    # データ情報の表示（データが読み込まれている場合）
    if st.session_state.data is not None:
        st.sidebar.subheader("データ情報")
        st.sidebar.text(f"行数: {len(st.session_state.data)}")
        st.sidebar.text(f"列数: {len(st.session_state.data.columns)}")
        
        # データプレビューボタン
        if st.sidebar.button("データプレビュー"):
            st.session_state.current_page = "データエディタ"
    
    return selected_menu

# ホームページ
def render_home_page():
    """ホームページを描画する"""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.subheader(APP_DESCRIPTION)
    
    # アプリケーションの概要
    st.markdown("""
    ## CausalViz Analytics Platformへようこそ！
    
    このプラットフォームは、データ分析、因果推論、AIアシスタントを統合した総合的な分析ツールです。
    Streamlit、AutoGluon、LLMを組み合わせることで、高度なデータ分析と因果関係の探索を直感的に行うことができます。
    
    ### 主な機能
    
    1. **インタラクティブなデータ分析**
       - データの読み込み、編集、フィルタリング
       - ピボットテーブルによる多次元分析
       - 多様なグラフによるデータ可視化
    
    2. **因果推論分析**
       - 処理効果の推定
       - 共変量バランスの評価
       - 因果グラフの可視化
       - 特徴量重要度の分析
    
    3. **AIアシスタント**
       - 自然言語によるデータ分析の質問応答
       - 因果関係に関する説明生成
       - データインサイトの提案
    
    ### 使い方
    
    1. 左側のメニューから「データ読み込み」を選択し、分析したいデータをアップロードします
    2. 「データエディタ」でデータの確認や編集を行います
    3. 「ピボットテーブル」や「データ可視化」で様々な角度からデータを分析します
    4. 「因果推論分析」で変数間の因果関係を探索します
    5. 「AIアシスタント」に質問して、データに関する洞察を得ます
    
    さあ、データ分析を始めましょう！
    """)
    
    # クイックスタートボタン
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 データを読み込む"):
            st.session_state.current_page = "データ読み込み"
            st.experimental_rerun()
    
    with col2:
        if st.button("🔍 サンプルデータを試す"):
            # サンプルデータの読み込み
            try:
                sample_data_path = os.path.join(SAMPLE_DATA_DIR, 'boston_housing.csv')
                if os.path.exists(sample_data_path):
                    st.session_state.data = load_data(sample_data_path)
                    st.session_state.data_analyzer.load_data(st.session_state.data)
                    st.session_state.interactive_data_editor.load_data(st.session_state.data)
                    st.session_state.pivot_table_builder.load_data(st.session_state.data)
                    st.session_state.visualization.load_data(st.session_state.data)
                    st.session_state.causal_inference.load_data(st.session_state.data)
                    st.session_state.current_page = "データエディタ"
                    st.experimental_rerun()
                else:
                    st.error("サンプルデータが見つかりません。")
            except Exception as e:
                st.error(f"サンプルデータの読み込みエラー: {str(e)}")
    
    with col3:
        if st.button("💬 AIアシスタントに質問"):
            st.session_state.current_page = "AIアシスタント"
            st.experimental_rerun()
    
    # システム情報
    with st.expander("システム情報", expanded=False):
        system_info = st.session_state.system_info
        
        st.subheader("システムリソース")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CPU使用率", f"{system_info['cpu']['usage_percent']:.1f}%")
            st.metric("物理コア数", system_info['cpu']['physical_cores'])
            st.metric("論理コア数", system_info['cpu']['logical_cores'])
        
        with col2:
            st.metric("メモリ使用率", f"{system_info['memory']['percent']:.1f}%")
            st.metric("合計メモリ", f"{system_info['memory']['total_gb']:.1f} GB")
            st.metric("利用可能メモリ", f"{system_info['memory']['available_gb']:.1f} GB")
        
        st.subheader("ディスク情報")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ディスク使用率", f"{system_info['disk']['percent']:.1f}%")
            st.metric("合計容量", f"{system_info['disk']['total_gb']:.1f} GB")
        
        with col2:
            st.metric("使用容量", f"{system_info['disk']['used_gb']:.1f} GB")
            st.metric("空き容量", f"{system_info['disk']['free_gb']:.1f} GB")
        
        st.subheader("Python情報")
        st.text(f"Python バージョン: {system_info['python']['version']}")
        st.text(f"実装: {system_info['python']['implementation']}")

# データ読み込みページ
def render_data_load_page():
    """データ読み込みページを描画する"""
    st.title("データ読み込み")
    st.subheader("分析するデータを読み込みます")
    
    # タブでUIを分割
    tab1, tab2, tab3 = st.tabs(["ファイルアップロード", "サンプルデータ", "URLからデータ取得"])
    
    with tab1:
        st.subheader("ファイルをアップロード")
        
        # ファイルアップローダー
        uploaded_file = st.file_uploader(
            "CSVまたはExcelファイルを選択してください",
            type=["csv", "xlsx", "xls", "parquet", "json", "feather", "pickle", "pkl"],
            help="サポートされているファイル形式: CSV, Excel, Parquet, JSON, Feather, Pickle"
        )
        
        if uploaded_file is not None:
            try:
                # ファイル形式に応じた読み込み
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                with st.spinner("データを読み込んでいます..."):
                    if file_extension == '.csv':
                        # CSVファイルの読み込み設定
                        col1, col2 = st.columns(2)
                        with col1:
                            encoding = st.selectbox("エンコーディング", ["utf-8", "shift-jis", "cp932", "euc-jp", "iso-2022-jp"])
                        with col2:
                            separator = st.selectbox("区切り文字", [",", "\t", ";", "|"])
                            separator = "\t" if separator == "\\t" else separator
                        
                        # ヘッダー行の設定
                        header_row = st.number_input("ヘッダー行", min_value=0, value=0)
                        
                        if st.button("CSVファイルを読み込む"):
                            df = pd.read_csv(uploaded_file, encoding=encoding, sep=separator, header=header_row)
                            st.session_state.data = df
                            
                            # 各モジュールにデータを読み込む
                            st.session_state.data_analyzer.load_data(df)
                            st.session_state.interactive_data_editor.load_data(df)
                            st.session_state.pivot_table_builder.load_data(df)
                            st.session_state.visualization.load_data(df)
                            st.session_state.causal_inference.load_data(df)
                            
                            st.success(f"データを読み込みました: {len(df)}行 x {len(df.columns)}列")
                            
                    elif file_extension in ['.xlsx', '.xls']:
                        # Excelファイルの読み込み設定
                        # シート名の取得
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names
                        
                        selected_sheet = st.selectbox("シートを選択", sheet_names)
                        
                        # ヘッダー行の設定
                        header_row = st.number_input("ヘッダー行", min_value=0, value=0)
                        
                        if st.button("Excelファイルを読み込む"):
                            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=header_row)
                            st.session_state.data = df
                            
                            # 各モジュールにデータを読み込む
                            st.session_state.data_analyzer.load_data(df)
                            st.session_state.interactive_data_editor.load_data(df)
                            st.session_state.pivot_table_builder.load_data(df)
                            st.session_state.visualization.load_data(df)
                            st.session_state.causal_inference.load_data(df)
                            
                            st.success(f"データを読み込みました: {len(df)}行 x {len(df.columns)}列")
                            
                    else:
                        # その他のファイル形式
                        if st.button(f"{file_extension[1:].upper()}ファイルを読み込む"):
                            if file_extension == '.parquet':
                                df = pd.read_parquet(uploaded_file)
                            elif file_extension == '.json':
                                df = pd.read_json(uploaded_file)
                            elif file_extension == '.feather':
                                df = pd.read_feather(uploaded_file)
                            elif file_extension in ['.pickle', '.pkl']:
                                df = pd.read_pickle(uploaded_file)
                            else:
                                st.error(f"サポートされていないファイル形式です: {file_extension}")
                                return
                            
                            st.session_state.data = df
                            
                            # 各モジュールにデータを読み込む
                            st.session_state.data_analyzer.load_data(df)
                            st.session_state.interactive_data_editor.load_data(df)
                            st.session_state.pivot_table_builder.load_data(df)
                            st.session_state.visualization.load_data(df)
                            st.session_state.causal_inference.load_data(df)
                            
                            st.success(f"データを読み込みました: {len(df)}行 x {len(df.columns)}列")
            
            except Exception as e:
                st.error(f"データの読み込み中にエラーが発生しました: {str(e)}")
                logger.error(f"データ読み込みエラー: {str(e)}", exc_info=True)
    
    with tab2:
        st.subheader("サンプルデータを使用")
        
        # サンプルデータの一覧
        try:
            sample_files = list_directory(SAMPLE_DATA_DIR, pattern="*.csv")
            
            if not sample_files:
                st.warning("サンプルデータが見つかりません。")
            else:
                # サンプルデータの選択
                sample_options = [file['name'] for file in sample_files]
                selected_sample = st.selectbox("サンプルデータを選択", sample_options)
                
                # サンプルデータの説明
                sample_descriptions = {
                    "boston_housing.csv": "ボストンの住宅価格データセット。住宅価格と関連する様々な特徴量を含みます。",
                    "iris.csv": "アイリスの花のデータセット。3種類のアイリスの花の特徴量を含みます。",
                    "titanic.csv": "タイタニック号の乗客データ。生存者と犠牲者の情報を含みます。",
                    "wine.csv": "ワインの品質データセット。化学的特性と品質評価を含みます。"
                }
                
                if selected_sample in sample_descriptions:
                    st.info(sample_descriptions[selected_sample])
                
                # サンプルデータのプレビュー
                sample_path = os.path.join(SAMPLE_DATA_DIR, selected_sample)
                sample_preview = pd.read_csv(sample_path, nrows=5)
                st.dataframe(sample_preview)
                
                # 読み込みボタン
                if st.button("このサンプルデータを使用"):
                    with st.spinner("サンプルデータを読み込んでいます..."):
                        df = pd.read_csv(sample_path)
                        st.session_state.data = df
                        
                        # 各モジュールにデータを読み込む
                        st.session_state.data_analyzer.load_data(df)
                        st.session_state.interactive_data_editor.load_data(df)
                        st.session_state.pivot_table_builder.load_data(df)
                        st.session_state.visualization.load_data(df)
                        st.session_state.causal_inference.load_data(df)
                        
                        st.success(f"サンプルデータを読み込みました: {len(df)}行 x {len(df.columns)}列")
        
        except Exception as e:
            st.error(f"サンプルデータの読み込み中にエラーが発生しました: {str(e)}")
            logger.error(f"サンプルデータ読み込みエラー: {str(e)}", exc_info=True)
    
    with tab3:
        st.subheader("URLからデータを取得")
        
        # URL入力
        data_url = st.text_input("データのURL", placeholder="https://example.com/data.csv")
        
        # ファイル形式の選択
        url_file_format = st.selectbox("ファイル形式", ["CSV", "Excel", "Parquet", "JSON", "Feather"])
        
        # 読み込み設定
        if url_file_format == "CSV":
            col1, col2 = st.columns(2)
            with col1:
                url_encoding = st.selectbox("エンコーディング (URL)", ["utf-8", "shift-jis", "cp932", "euc-jp", "iso-2022-jp"])
            with col2:
                url_separator = st.selectbox("区切り文字 (URL)", [",", "\t", ";", "|"])
                url_separator = "\t" if url_separator == "\\t" else url_separator
        
        # 読み込みボタン
        if st.button("URLからデータを読み込む"):
            if not data_url:
                st.warning("URLを入力してください。")
            else:
                try:
                    with st.spinner("URLからデータを読み込んでいます..."):
                        if url_file_format == "CSV":
                            df = pd.read_csv(data_url, encoding=url_encoding, sep=url_separator)
                        elif url_file_format == "Excel":
                            df = pd.read_excel(data_url)
                        elif url_file_format == "Parquet":
                            df = pd.read_parquet(data_url)
                        elif url_file_format == "JSON":
                            df = pd.read_json(data_url)
                        elif url_file_format == "Feather":
                            df = pd.read_feather(data_url)
                        
                        st.session_state.data = df
                        
                        # 各モジュールにデータを読み込む
                        st.session_state.data_analyzer.load_data(df)
                        st.session_state.interactive_data_editor.load_data(df)
                        st.session_state.pivot_table_builder.load_data(df)
                        st.session_state.visualization.load_data(df)
                        st.session_state.causal_inference.load_data(df)
                        
                        st.success(f"URLからデータを読み込みました: {len(df)}行 x {len(df.columns)}列")
                
                except Exception as e:
                    st.error(f"URLからのデータ読み込み中にエラーが発生しました: {str(e)}")
                    logger.error(f"URL読み込みエラー: {str(e)}", exc_info=True)
    
    # データが読み込まれている場合、データ情報を表示
    if st.session_state.data is not None:
        st.subheader("現在のデータ")
        
        # データ情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("行数", len(st.session_state.data))
        with col2:
            st.metric("列数", len(st.session_state.data.columns))
        with col3:
            memory_usage = st.session_state.data.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("メモリ使用量", f"{memory_usage:.2f} MB")
        
        # データプレビュー
        st.subheader("データプレビュー")
        st.dataframe(st.session_state.data.head(10))
        
        # データの保存
        st.subheader("データの保存")
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_format = st.selectbox("保存形式", ["CSV", "Excel", "Parquet", "JSON", "Feather", "Pickle"])
        
        with col2:
            save_filename = st.text_input("ファイル名", f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if st.button("データを保存"):
            try:
                # 保存ディレクトリの作成
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'exports')
                os.makedirs(save_dir, exist_ok=True)
                
                # ファイル形式に応じた保存
                if save_format == "CSV":
                    file_path = os.path.join(save_dir, f"{save_filename}.csv")
                    st.session_state.data.to_csv(file_path, index=False)
                elif save_format == "Excel":
                    file_path = os.path.join(save_dir, f"{save_filename}.xlsx")
                    st.session_state.data.to_excel(file_path, index=False)
                elif save_format == "Parquet":
                    file_path = os.path.join(save_dir, f"{save_filename}.parquet")
                    st.session_state.data.to_parquet(file_path, index=False)
                elif save_format == "JSON":
                    file_path = os.path.join(save_dir, f"{save_filename}.json")
                    st.session_state.data.to_json(file_path, orient="records")
                elif save_format == "Feather":
                    file_path = os.path.join(save_dir, f"{save_filename}.feather")
                    st.session_state.data.to_feather(file_path)
                elif save_format == "Pickle":
                    file_path = os.path.join(save_dir, f"{save_filename}.pkl")
                    st.session_state.data.to_pickle(file_path)
                
                st.success(f"データを保存しました: {file_path}")
                
                # ダウンロードリンクの作成
                if save_format == "CSV":
                    csv_data = st.session_state.data.to_csv(index=False)
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{save_filename}.csv">ダウンロード {save_filename}.csv</a>'
                    st.markdown(href, unsafe_allow_html=True)
                elif save_format == "Excel":
                    buffer = io.BytesIO()
                    st.session_state.data.to_excel(buffer, index=False)
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{save_filename}.xlsx">ダウンロード {save_filename}.xlsx</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"データの保存中にエラーが発生しました: {str(e)}")
                logger.error(f"データ保存エラー: {str(e)}", exc_info=True)

# データエディタページ
def render_data_editor_page():
    """データエディタページを描画する"""
    st.title("データエディタ")
    st.subheader("データの表示、編集、フィルタリングを行います")
    
    if st.session_state.data is None:
        st.warning("データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
        if st.button("データ読み込みページへ"):
            st.session_state.current_page = "データ読み込み"
            st.experimental_rerun()
        return
    
    # インタラクティブデータエディタの表示
    edited_data = st.session_state.interactive_data_editor.render_ui()
    
    # データが編集された場合
    if edited_data is not None:
        # 編集されたデータを他のモジュールにも反映
        st.session_state.data = st.session_state.interactive_data_editor.get_data()
        st.session_state.data_analyzer.load_data(st.session_state.data, make_copy=False)
        st.session_state.pivot_table_builder.load_data(st.session_state.data, make_copy=False)
        st.session_state.visualization.load_data(st.session_state.data, make_copy=False)
        st.session_state.causal_inference.load_data(st.session_state.data, make_copy=False)
        
        st.success("データが更新されました")

# ピボットテーブルページ
def render_pivot_table_page():
    """ピボットテーブルページを描画する"""
    st.title("ピボットテーブル")
    st.subheader("データのピボット分析を行います")
    
    if st.session_state.data is None:
        st.warning("データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
        if st.button("データ読み込みページへ"):
            st.session_state.current_page = "データ読み込み"
            st.experimental_rerun()
        return
    
    # ピボットテーブルビルダーの表示
    st.session_state.pivot_table_builder.render_ui()

# データ可視化ページ
def render_visualization_page():
    """データ可視化ページを描画する"""
    st.title("データ可視化")
    st.subheader("様々なグラフでデータを可視化します")
    
    if st.session_state.data is None:
        st.warning("データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
        if st.button("データ読み込みページへ"):
            st.session_state.current_page = "データ読み込み"
            st.experimental_rerun()
        return
    
    # 可視化コンポーネントの表示
    st.session_state.visualization.render_ui()

# 因果推論分析ページ
def render_causal_inference_page():
    """因果推論分析ページを描画する"""
    st.title("因果推論分析")
    st.subheader("変数間の因果関係を分析します")
    
    if st.session_state.data is None:
        st.warning("データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
        if st.button("データ読み込みページへ"):
            st.session_state.current_page = "データ読み込み"
            st.experimental_rerun()
        return
    
    # 因果推論モジュールのUI表示
    st.session_state.causal_inference.render_ui()

# AIアシスタントページ
def render_ai_assistant_page():
    """AIアシスタントページを描画する"""
    st.title("AIアシスタント")
    st.subheader("データに関する質問に回答します")
    
    # APIキーの確認
    api_keys = st.session_state.settings['api_keys']
    
    if not api_keys['openai'] and not api_keys['anthropic']:
        st.warning("APIキーが設定されていません。「設定」ページでAPIキーを設定してください。")
        if st.button("設定ページへ"):
            st.session_state.current_page = "設定"
            st.experimental_rerun()
        return
    
    # LLMエクスプレイナーの設定
    if not st.session_state.llm_explainer.is_initialized():
        if api_keys['anthropic']:
            st.session_state.llm_explainer.set_api_key('anthropic', api_keys['anthropic'])
        if api_keys['openai']:
            st.session_state.llm_explainer.set_api_key('openai', api_keys['openai'])
    
    # データの確認
    if st.session_state.data is None:
        st.info("データが読み込まれていませんが、一般的な質問には回答できます。データに関する具体的な質問をするには、「データ読み込み」ページからデータを読み込んでください。")
    else:
        # データコンテキストの設定
        st.session_state.llm_explainer.set_data_context(st.session_state.data)
        
        # 因果モデルが存在する場合はそのコンテキストも設定
        if hasattr(st.session_state.causal_inference, 'model') and st.session_state.causal_inference.model is not None:
            st.session_state.llm_explainer.set_causal_context(st.session_state.causal_inference.model)
    
    # チャットインターフェースの表示
    st.session_state.llm_explainer.render_chat_ui()

# 設定ページ
def render_settings_page():
    """設定ページを描画する"""
    st.title("設定")
    st.subheader("アプリケーションの設定を行います")
    
    # タブでUIを分割
    tab1, tab2, tab3 = st.tabs(["一般設定", "APIキー設定", "システム情報"])
    
    with tab1:
        st.subheader("一般設定")
        
        # テーマ設定
        theme = st.selectbox(
            "テーマ",
            options=["light", "dark"],
            index=0 if st.session_state.settings['theme'] == "light" else 1
        )
        
        # 言語設定
        language = st.selectbox(
            "言語",
            options=["ja", "en"],
            index=0 if st.session_state.settings['language'] == "ja" else 1
        )
        
        # メモリ設定
        st.subheader("メモリ設定")
        
        max_memory_percent = st.slider(
            "メモリ使用率の上限",
            min_value=50,
            max_value=95,
            value=st.session_state.settings['max_memory_percent']
        )
        
        auto_cleanup = st.checkbox(
            "メモリ使用率が上限を超えた場合に自動クリーンアップする",
            value=st.session_state.settings['auto_cleanup']
        )
        
        show_memory_warning = st.checkbox(
            "メモリ使用率が高い場合に警告を表示する",
            value=st.session_state.settings['show_memory_warning']
        )
        
        # グラフ設定
        st.subheader("グラフ設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_chart_height = st.number_input(
                "デフォルトのグラフの高さ",
                min_value=300,
                max_value=1200,
                value=st.session_state.settings['default_chart_height']
            )
            
        with col2:
            default_chart_width = st.number_input(
                "デフォルトのグラフの幅",
                min_value=300,
                max_value=1500,
                value=st.session_state.settings['default_chart_width']
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_chart_template = st.selectbox(
                "デフォルトのグラフテンプレート",
                options=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"],
                index=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"].index(st.session_state.settings['default_chart_template'])
            )
            
        with col2:
            default_color_scheme = st.selectbox(
                "デフォルトのカラースキーム",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Blues", "Greens", "Oranges", "Reds"],
                index=["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Blues", "Greens", "Oranges", "Reds"].index(st.session_state.settings['default_color_scheme']) if st.session_state.settings['default_color_scheme'] in ["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Blues", "Greens", "Oranges", "Reds"] else 0
            )
        
        # 設定の保存ボタン
        if st.button("一般設定を保存"):
            # 設定の更新
            st.session_state.settings.update({
                'theme': theme,
                'language': language,
                'max_memory_percent': max_memory_percent,
                'auto_cleanup': auto_cleanup,
                'show_memory_warning': show_memory_warning,
                'default_chart_height': default_chart_height,
                'default_chart_width': default_chart_width,
                'default_chart_template': default_chart_template,
                'default_color_scheme': default_color_scheme
            })
            
            # 可視化コンポーネントの設定更新
            st.session_state.visualization.set_figure_settings({
                'height': default_chart_height,
                'width': default_chart_width,
                'template': default_chart_template,
                'color_scheme': default_color_scheme
            })
            
            st.success("設定を保存しました")
            
            # 設定ファイルへの保存
            try:
                config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
                os.makedirs(config_dir, exist_ok=True)
                save_config(st.session_state.settings, os.path.join(config_dir, 'settings.json'))
            except Exception as e:
                st.error(f"設定ファイルの保存中にエラーが発生しました: {str(e)}")
                logger.error(f"設定保存エラー: {str(e)}", exc_info=True)
    
    with tab2:
        st.subheader("APIキー設定")
        
        # APIキー入力
        st.info("AIアシスタント機能を使用するには、OpenAIまたはAnthropic（Claude）のAPIキーが必要です。")
        
        openai_api_key = st.text_input(
            "OpenAI APIキー",
            value=st.session_state.settings['api_keys']['openai'],
            type="password",
            help="OpenAIのAPIキーを入力してください。入力されたキーは暗号化されて保存されます。"
        )
        
        anthropic_api_key = st.text_input(
            "Anthropic APIキー（Claude）",
            value=st.session_state.settings['api_keys']['anthropic'],
            type="password",
            help="AnthropicのAPIキーを入力してください。入力されたキーは暗号化されて保存されます。"
        )
        
        # APIキーの保存ボタン
        if st.button("APIキーを保存"):
            # APIキーの更新
            st.session_state.settings['api_keys'].update({
                'openai': openai_api_key,
                'anthropic': anthropic_api_key
            })
            
            # LLMエクスプレイナーの設定更新
            if openai_api_key:
                st.session_state.llm_explainer.set_api_key('openai', openai_api_key)
            if anthropic_api_key:
                st.session_state.llm_explainer.set_api_key('anthropic', anthropic_api_key)
            
            st.success("APIキーを保存しました")
            
            # 設定ファイルへの保存
            try:
                config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
                os.makedirs(config_dir, exist_ok=True)
                save_config(st.session_state.settings, os.path.join(config_dir, 'settings.json'))
            except Exception as e:
                st.error(f"設定ファイルの保存中にエラーが発生しました: {str(e)}")
                logger.error(f"設定保存エラー: {str(e)}", exc_info=True)
    
    with tab3:
        st.subheader("システム情報")
        
        # システム情報の更新ボタン
        if st.button("システム情報を更新"):
            st.session_state.system_info = get_system_info()
            st.success("システム情報を更新しました")
        
        # システム情報の表示
        system_info = st.session_state.system_info
        
        st.subheader("CPU情報")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("物理コア数", system_info['cpu']['physical_cores'])
        with col2:
            st.metric("論理コア数", system_info['cpu']['logical_cores'])
        with col3:
            st.metric("CPU使用率", f"{system_info['cpu']['usage_percent']:.1f}%")
        
        st.subheader("メモリ情報")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("合計メモリ", f"{system_info['memory']['total_gb']:.1f} GB")
        with col2:
            st.metric("使用メモリ", f"{system_info['memory']['used_gb']:.1f} GB")
        with col3:
            st.metric("利用可能メモリ", f"{system_info['memory']['available_gb']:.1f} GB")
        with col4:
            st.metric("メモリ使用率", f"{system_info['memory']['percent']:.1f}%")
        
        st.subheader("ディスク情報")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("合計容量", f"{system_info['disk']['total_gb']:.1f} GB")
        with col2:
            st.metric("使用容量", f"{system_info['disk']['used_gb']:.1f} GB")
        with col3:
            st.metric("空き容量", f"{system_info['disk']['free_gb']:.1f} GB")
        with col4:
            st.metric("ディスク使用率", f"{system_info['disk']['percent']:.1f}%")
        
        st.subheader("プロセス情報")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("プロセスID", system_info['process']['pid'])
        with col2:
            st.metric("プロセスメモリ", f"{system_info['process']['memory_rss_mb']:.1f} MB")
        with col3:
            st.metric("プロセスCPU", f"{system_info['process']['cpu_percent']:.1f}%")
        
        st.subheader("Python情報")
        st.text(f"Python バージョン: {system_info['python']['version']}")
        st.text(f"実装: {system_info['python']['implementation']}")
        st.text(f"タイムスタンプ: {system_info['timestamp']}")
        
        # 環境変数情報
        with st.expander("環境変数", expanded=False):
            env_vars = {key: value for key, value in os.environ.items() if not key.startswith('OPENAI_') and not key.startswith('ANTHROPIC_')}
            st.json(env_vars)

# メイン関数
def main():
    """メイン関数"""
    # ページ設定
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # セッション状態の初期化
    initialize_session_state()
    
    # 設定ファイルの読み込み
    try:
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        config_file = os.path.join(config_dir, 'settings.json')
        if os.path.exists(config_file):
            settings = load_config(config_file)
            st.session_state.settings.update(settings)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みエラー: {str(e)}", exc_info=True)
    
    # ナビゲーションメニューの描画
    selected_menu = render_navigation()
    
    # 選択されたページの描画
    if selected_menu == "ホーム":
        render_home_page()
    elif selected_menu == "データ読み込み":
        render_data_load_page()
    elif selected_menu == "データエディタ":
        render_data_editor_page()
    elif selected_menu == "ピボットテーブル":
        render_pivot_table_page()
    elif selected_menu == "データ可視化":
        render_visualization_page()
    elif selected_menu == "因果推論分析":
        render_causal_inference_page()
    elif selected_menu == "AIアシスタント":
        render_ai_assistant_page()
    elif selected_menu == "設定":
        render_settings_page()

# エントリーポイント
if __name__ == "__main__":
    main()
