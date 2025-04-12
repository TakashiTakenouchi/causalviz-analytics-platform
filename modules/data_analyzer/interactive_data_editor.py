import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
import time
import gc
from datetime import datetime

logger = logging.getLogger(__name__)

class InteractiveDataEditor:
    """
    Streamlitを使用したインタラクティブなデータ編集機能を提供するクラス
    データのフィルタリング、ソート、検索、編集機能を実装
    """
    
    def __init__(self, key_prefix: str = "data_editor"):
        """
        InteractiveDataEditorクラスの初期化
        
        Parameters:
        -----------
        key_prefix : str
            Streamlitウィジェットのキープレフィックス
        """
        self.data = None
        self.original_data = None
        self.edited_data = None
        self.column_config = {}
        self.key_prefix = key_prefix
        self.filter_conditions = {}
        self.sort_columns = []
        self.sort_ascending = []
        self.search_text = ""
        self.search_columns = []
        self.selected_rows = []
        self.edit_history = []
        self.max_history = 10
        self.display_settings = {
            'page_size': 10,
            'current_page': 0,
            'show_index': True,
            'hide_columns': [],
            'column_order': None
        }
    
    def load_data(self, data: pd.DataFrame, make_copy: bool = True) -> None:
        """
        データを読み込む
        
        Parameters:
        -----------
        data : pd.DataFrame
            読み込むデータフレーム
        make_copy : bool
            データのコピーを作成するかどうか
        """
        if make_copy:
            self.data = data.copy()
            self.original_data = data.copy()
        else:
            self.data = data
            self.original_data = data
            
        self.edited_data = None
        self.filter_conditions = {}
        self.sort_columns = []
        self.sort_ascending = []
        self.search_text = ""
        self.search_columns = []
        self.selected_rows = []
        self.edit_history = []
        self.display_settings['current_page'] = 0
        
        # 列の設定を初期化
        self._initialize_column_config()
        
        logger.info(f"データを読み込みました。行数: {len(self.data)}, 列数: {len(self.data.columns)}")
    
    def _initialize_column_config(self) -> None:
        """
        列の設定を初期化する
        """
        if self.data is None:
            return
            
        self.column_config = {}
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            
            # データ型に基づいて列の設定を作成
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    self.column_config[col] = st.column_config.NumberColumn(
                        col,
                        help=f"数値列: {col}",
                        format="%d",
                        step=1
                    )
                else:
                    self.column_config[col] = st.column_config.NumberColumn(
                        col,
                        help=f"数値列: {col}",
                        format="%.2f"
                    )
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.column_config[col] = st.column_config.DatetimeColumn(
                    col,
                    help=f"日時列: {col}",
                    format="YYYY-MM-DD HH:mm:ss"
                )
            elif pd.api.types.is_bool_dtype(dtype):
                self.column_config[col] = st.column_config.CheckboxColumn(
                    col,
                    help=f"ブール列: {col}"
                )
            elif pd.api.types.is_categorical_dtype(dtype) or self.data[col].nunique() < 20:
                # カテゴリ列または少数のユニーク値を持つ列
                unique_values = self.data[col].dropna().unique().tolist()
                self.column_config[col] = st.column_config.SelectboxColumn(
                    col,
                    help=f"カテゴリ列: {col}",
                    options=unique_values,
                    required=False
                )
            else:
                # テキスト列
                self.column_config[col] = st.column_config.TextColumn(
                    col,
                    help=f"テキスト列: {col}",
                    max_chars=100
                )
    
    def set_column_config(self, column_config: Dict[str, Any]) -> None:
        """
        列の設定をカスタマイズする
        
        Parameters:
        -----------
        column_config : Dict[str, Any]
            列名をキーとし、st.column_configオブジェクトを値とする辞書
        """
        if self.column_config is None:
            self._initialize_column_config()
            
        # 新しい設定で更新
        self.column_config.update(column_config)
        
        logger.info(f"列の設定を更新しました: {list(column_config.keys())}")
    
    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        フィルタ条件を適用する
        
        Parameters:
        -----------
        data : pd.DataFrame
            フィルタを適用するデータフレーム
            
        Returns:
        --------
        pd.DataFrame
            フィルタリングされたデータフレーム
        """
        if not self.filter_conditions:
            return data
            
        filtered_data = data.copy()
        
        for column, condition in self.filter_conditions.items():
            if column not in filtered_data.columns:
                continue
                
            if 'min' in condition and condition['min'] is not None:
                filtered_data = filtered_data[filtered_data[column] >= condition['min']]
                
            if 'max' in condition and condition['max'] is not None:
                filtered_data = filtered_data[filtered_data[column] <= condition['max']]
                
            if 'values' in condition and condition['values']:
                filtered_data = filtered_data[filtered_data[column].isin(condition['values'])]
                
            if 'text' in condition and condition['text']:
                if pd.api.types.is_string_dtype(filtered_data[column]) or pd.api.types.is_object_dtype(filtered_data[column]):
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(condition['text'], case=False, na=False)]
        
        return filtered_data
    
    def _apply_search(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        検索条件を適用する
        
        Parameters:
        -----------
        data : pd.DataFrame
            検索を適用するデータフレーム
            
        Returns:
        --------
        pd.DataFrame
            検索結果のデータフレーム
        """
        if not self.search_text or not self.search_columns:
            return data
            
        search_result = pd.DataFrame(index=data.index, columns=['match'])
        search_result['match'] = False
        
        for column in self.search_columns:
            if column not in data.columns:
                continue
                
            # 列のデータ型に応じた検索
            if pd.api.types.is_string_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
                # 文字列列
                match = data[column].astype(str).str.contains(self.search_text, case=False, na=False)
            elif pd.api.types.is_numeric_dtype(data[column]):
                # 数値列（数値として検索可能な場合）
                try:
                    search_value = float(self.search_text)
                    match = data[column] == search_value
                except ValueError:
                    match = pd.Series(False, index=data.index)
            else:
                # その他の列（文字列に変換して検索）
                match = data[column].astype(str).str.contains(self.search_text, case=False, na=False)
                
            search_result['match'] = search_result['match'] | match
        
        return data[search_result['match']]
    
    def _apply_sorting(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ソート条件を適用する
        
        Parameters:
        -----------
        data : pd.DataFrame
            ソートを適用するデータフレーム
            
        Returns:
        --------
        pd.DataFrame
            ソートされたデータフレーム
        """
        if not self.sort_columns:
            return data
            
        # 有効な列のみを使用
        valid_sort_columns = [col for col in self.sort_columns if col in data.columns]
        valid_sort_ascending = self.sort_ascending[:len(valid_sort_columns)]
        
        if not valid_sort_columns:
            return data
            
        return data.sort_values(by=valid_sort_columns, ascending=valid_sort_ascending)
    
    def _get_paginated_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ページネーションを適用する
        
        Parameters:
        -----------
        data : pd.DataFrame
            ページネーションを適用するデータフレーム
            
        Returns:
        --------
        pd.DataFrame
            現在のページのデータフレーム
        """
        if data.empty:
            return data
            
        page_size = self.display_settings['page_size']
        current_page = self.display_settings['current_page']
        
        # ページ数の計算
        total_pages = max(1, (len(data) + page_size - 1) // page_size)
        
        # 現在のページが範囲外の場合は調整
        if current_page >= total_pages:
            current_page = total_pages - 1
            self.display_settings['current_page'] = current_page
        
        # ページのデータを取得
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(data))
        
        return data.iloc[start_idx:end_idx]
    
    def _prepare_display_data(self) -> pd.DataFrame:
        """
        表示用のデータを準備する
        
        Returns:
        --------
        pd.DataFrame
            表示用のデータフレーム
        """
        if self.data is None:
            return pd.DataFrame()
            
        # フィルタリング
        display_data = self._apply_filters(self.data)
        
        # 検索
        display_data = self._apply_search(display_data)
        
        # ソート
        display_data = self._apply_sorting(display_data)
        
        # 列の順序を適用
        if self.display_settings['column_order'] is not None:
            # 有効な列のみを使用
            valid_columns = [col for col in self.display_settings['column_order'] if col in display_data.columns]
            # 指定されていない列を追加
            remaining_columns = [col for col in display_data.columns if col not in valid_columns]
            # 列の順序を適用
            display_data = display_data[valid_columns + remaining_columns]
        
        # 非表示の列を除外
        if self.display_settings['hide_columns']:
            display_data = display_data[[col for col in display_data.columns if col not in self.display_settings['hide_columns']]]
        
        return display_data
    
    def add_filter(self, column: str, condition: Dict) -> None:
        """
        フィルタ条件を追加する
        
        Parameters:
        -----------
        column : str
            フィルタを適用する列名
        condition : Dict
            フィルタ条件（'min', 'max', 'values', 'text'のいずれかを含む辞書）
        """
        if self.data is None or column not in self.data.columns:
            logger.warning(f"列 '{column}' はデータフレームに存在しません")
            return
            
        self.filter_conditions[column] = condition
        logger.info(f"フィルタを追加しました: {column} - {condition}")
    
    def remove_filter(self, column: str) -> None:
        """
        フィルタ条件を削除する
        
        Parameters:
        -----------
        column : str
            削除するフィルタの列名
        """
        if column in self.filter_conditions:
            del self.filter_conditions[column]
            logger.info(f"フィルタを削除しました: {column}")
    
    def clear_filters(self) -> None:
        """
        すべてのフィルタ条件をクリアする
        """
        self.filter_conditions = {}
        logger.info("すべてのフィルタをクリアしました")
    
    def set_sort(self, columns: List[str], ascending: List[bool]) -> None:
        """
        ソート条件を設定する
        
        Parameters:
        -----------
        columns : List[str]
            ソートする列名のリスト
        ascending : List[bool]
            各列の昇順/降順を指定するブールのリスト
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        # 有効な列のみを使用
        valid_columns = [col for col in columns if col in self.data.columns]
        valid_ascending = ascending[:len(valid_columns)]
        
        self.sort_columns = valid_columns
        self.sort_ascending = valid_ascending
        
        logger.info(f"ソート条件を設定しました: {valid_columns}, 昇順={valid_ascending}")
    
    def set_search(self, text: str, columns: Optional[List[str]] = None) -> None:
        """
        検索条件を設定する
        
        Parameters:
        -----------
        text : str
            検索テキスト
        columns : Optional[List[str]]
            検索対象の列名リスト（Noneの場合はすべての列）
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        self.search_text = text
        
        if columns is None:
            # すべての列を検索対象にする
            self.search_columns = self.data.columns.tolist()
        else:
            # 有効な列のみを使用
            self.search_columns = [col for col in columns if col in self.data.columns]
        
        logger.info(f"検索条件を設定しました: '{text}', 対象列={self.search_columns}")
    
    def clear_search(self) -> None:
        """
        検索条件をクリアする
        """
        self.search_text = ""
        self.search_columns = []
        logger.info("検索条件をクリアしました")
    
    def set_page_size(self, page_size: int) -> None:
        """
        ページサイズを設定する
        
        Parameters:
        -----------
        page_size : int
            1ページあたりの行数
        """
        if page_size < 1:
            logger.warning(f"無効なページサイズです: {page_size}")
            return
            
        self.display_settings['page_size'] = page_size
        # ページ番号をリセット
        self.display_settings['current_page'] = 0
        
        logger.info(f"ページサイズを設定しました: {page_size}")
    
    def set_page(self, page: int) -> None:
        """
        表示するページを設定する
        
        Parameters:
        -----------
        page : int
            ページ番号（0から始まる）
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        # 有効なページ番号に調整
        page_size = self.display_settings['page_size']
        total_pages = max(1, (len(self.data) + page_size - 1) // page_size)
        
        if page < 0:
            page = 0
        elif page >= total_pages:
            page = total_pages - 1
            
        self.display_settings['current_page'] = page
        
        logger.info(f"ページを設定しました: {page + 1}/{total_pages}")
    
    def next_page(self) -> None:
        """
        次のページに移動する
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        current_page = self.display_settings['current_page']
        page_size = self.display_settings['page_size']
        total_pages = max(1, (len(self.data) + page_size - 1) // page_size)
        
        if current_page < total_pages - 1:
            self.display_settings['current_page'] = current_page + 1
            logger.info(f"次のページに移動しました: {current_page + 2}/{total_pages}")
    
    def prev_page(self) -> None:
        """
        前のページに移動する
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        current_page = self.display_settings['current_page']
        page_size = self.display_settings['page_size']
        total_pages = max(1, (len(self.data) + page_size - 1) // page_size)
        
        if current_page > 0:
            self.display_settings['current_page'] = current_page - 1
            logger.info(f"前のページに移動しました: {current_page}/{total_pages}")
    
    def set_column_order(self, columns: List[str]) -> None:
        """
        列の表示順序を設定する
        
        Parameters:
        -----------
        columns : List[str]
            表示順序の列名リスト
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        # 有効な列のみを使用
        valid_columns = [col for col in columns if col in self.data.columns]
        
        self.display_settings['column_order'] = valid_columns
        
        logger.info(f"列の表示順序を設定しました: {valid_columns}")
    
    def hide_columns(self, columns: List[str]) -> None:
        """
        列を非表示にする
        
        Parameters:
        -----------
        columns : List[str]
            非表示にする列名のリスト
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return
            
        # 有効な列のみを使用
        valid_columns = [col for col in columns if col in self.data.columns]
        
        self.display_settings['hide_columns'] = valid_columns
        
        logger.info(f"列を非表示にしました: {valid_columns}")
    
    def show_all_columns(self) -> None:
        """
        すべての列を表示する
        """
        self.display_settings['hide_columns'] = []
        logger.info("すべての列を表示します")
    
    def reset_display_settings(self) -> None:
        """
        表示設定をリセットする
        """
        self.display_settings = {
            'page_size': 10,
            'current_page': 0,
            'show_index': True,
            'hide_columns': [],
            'column_order': None
        }
        logger.info("表示設定をリセットしました")
    
    def _save_edit_history(self, edited_data: pd.DataFrame) -> None:
        """
        編集履歴を保存する
        
        Parameters:
        -----------
        edited_data : pd.DataFrame
            編集後のデータフレーム
        """
        # 履歴が最大数に達している場合、古いものを削除
        if len(self.edit_history) >= self.max_history:
            self.edit_history.pop(0)
            
        # 現在のデータのコピーを履歴に追加
        self.edit_history.append({
            'data': self.data.copy() if self.data is not None else None,
            'timestamp': datetime.now()
        })
    
    def undo_edit(self) -> bool:
        """
        最後の編集を元に戻す
        
        Returns:
        --------
        bool
            成功したかどうか
        """
        if not self.edit_history:
            logger.warning("編集履歴がありません")
            return False
            
        # 最後の履歴を取得
        last_state = self.edit_history.pop()
        
        # データを復元
        self.data = last_state['data'].copy() if last_state['data'] is not None else None
        
        logger.info(f"編集を元に戻しました: {last_state['timestamp']}")
        return True
    
    def reset_data(self) -> None:
        """
        データを元の状態に戻す
        """
        if self.original_data is None:
            logger.warning("元のデータがありません")
            return
            
        self.data = self.original_data.copy()
        self.edited_data = None
        self.filter_conditions = {}
        self.sort_columns = []
        self.sort_ascending = []
        self.search_text = ""
        self.search_columns = []
        self.selected_rows = []
        self.edit_history = []
        self.display_settings['current_page'] = 0
        
        logger.info("データを元の状態にリセットしました")
    
    def render_filter_ui(self) -> None:
        """
        フィルタUIを描画する
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return
            
        st.subheader("データフィルタ")
        
        # フィルタUIの表示
        with st.expander("フィルタ条件", expanded=bool(self.filter_conditions)):
            # 列の選択
            columns = self.data.columns.tolist()
            selected_column = st.selectbox("フィルタする列を選択", [""] + columns, key=f"{self.key_prefix}_filter_column")
            
            if selected_column:
                # 列のデータ型に応じたフィルタUI
                col_dtype = self.data[selected_column].dtype
                
                if pd.api.types.is_numeric_dtype(col_dtype):
                    # 数値列
                    min_val = self.data[selected_column].min()
                    max_val = self.data[selected_column].max()
                    
                    # 現在のフィルタ値を取得
                    current_min = self.filter_conditions.get(selected_column, {}).get('min', min_val)
                    current_max = self.filter_conditions.get(selected_column, {}).get('max', max_val)
                    
                    # スライダーでフィルタ範囲を設定
                    filter_range = st.slider(
                        f"{selected_column}の範囲",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(current_min), float(current_max)),
                        key=f"{self.key_prefix}_filter_range_{selected_column}"
                    )
                    
                    # フィルタボタン
                    if st.button("このフィルタを適用", key=f"{self.key_prefix}_apply_filter_{selected_column}"):
                        self.add_filter(selected_column, {'min': filter_range[0], 'max': filter_range[1]})
                        
                elif pd.api.types.is_categorical_dtype(col_dtype) or self.data[selected_column].nunique() < 20:
                    # カテゴリ列または少数のユニーク値を持つ列
                    unique_values = self.data[selected_column].dropna().unique().tolist()
                    
                    # 現在のフィルタ値を取得
                    current_values = self.filter_conditions.get(selected_column, {}).get('values', unique_values)
                    
                    # マルチセレクトでフィルタ値を設定
                    selected_values = st.multiselect(
                        f"{selected_column}の値",
                        options=unique_values,
                        default=current_values,
                        key=f"{self.key_prefix}_filter_values_{selected_column}"
                    )
                    
                    # フィルタボタン
                    if st.button("このフィルタを適用", key=f"{self.key_prefix}_apply_filter_{selected_column}"):
                        self.add_filter(selected_column, {'values': selected_values})
                        
                else:
                    # テキスト列
                    # 現在のフィルタテキストを取得
                    current_text = self.filter_conditions.get(selected_column, {}).get('text', "")
                    
                    # テキスト入力でフィルタを設定
                    filter_text = st.text_input(
                        f"{selected_column}に含まれるテキスト",
                        value=current_text,
                        key=f"{self.key_prefix}_filter_text_{selected_column}"
                    )
                    
                    # フィルタボタン
                    if st.button("このフィルタを適用", key=f"{self.key_prefix}_apply_filter_{selected_column}"):
                        self.add_filter(selected_column, {'text': filter_text})
            
            # 現在のフィルタの表示
            if self.filter_conditions:
                st.subheader("現在のフィルタ")
                
                for column, condition in self.filter_conditions.items():
                    filter_desc = []
                    
                    if 'min' in condition and 'max' in condition:
                        filter_desc.append(f"{condition['min']} ≤ {column} ≤ {condition['max']}")
                    elif 'min' in condition:
                        filter_desc.append(f"{column} ≥ {condition['min']}")
                    elif 'max' in condition:
                        filter_desc.append(f"{column} ≤ {condition['max']}")
                        
                    if 'values' in condition and condition['values']:
                        filter_desc.append(f"{column} ∈ {condition['values']}")
                        
                    if 'text' in condition and condition['text']:
                        filter_desc.append(f"{column} contains '{condition['text']}'")
                        
                    # フィルタの説明と削除ボタン
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(", ".join(filter_desc))
                    with col2:
                        if st.button("削除", key=f"{self.key_prefix}_remove_filter_{column}"):
                            self.remove_filter(column)
                
                # すべてのフィルタをクリアするボタン
                if st.button("すべてのフィルタをクリア", key=f"{self.key_prefix}_clear_filters"):
                    self.clear_filters()
    
    def render_search_ui(self) -> None:
        """
        検索UIを描画する
        """
        if self.data is None:
            return
            
        st.subheader("データ検索")
        
        # 検索UIの表示
        with st.expander("検索条件", expanded=bool(self.search_text)):
            # 検索テキスト
            search_text = st.text_input(
                "検索テキスト",
                value=self.search_text,
                key=f"{self.key_prefix}_search_text"
            )
            
            # 検索対象の列
            columns = self.data.columns.tolist()
            search_columns = st.multiselect(
                "検索対象の列",
                options=columns,
                default=self.search_columns,
                key=f"{self.key_prefix}_search_columns"
            )
            
            # 検索ボタン
            col1, col2 = st.columns(2)
            with col1:
                if st.button("検索", key=f"{self.key_prefix}_apply_search"):
                    self.set_search(search_text, search_columns)
            with col2:
                if st.button("検索クリア", key=f"{self.key_prefix}_clear_search"):
                    self.clear_search()
    
    def render_sort_ui(self) -> None:
        """
        ソートUIを描画する
        """
        if self.data is None:
            return
            
        st.subheader("データソート")
        
        # ソートUIの表示
        with st.expander("ソート条件", expanded=bool(self.sort_columns)):
            # ソート列の選択
            columns = self.data.columns.tolist()
            sort_column = st.selectbox(
                "ソートする列を選択",
                [""] + columns,
                key=f"{self.key_prefix}_sort_column"
            )
            
            if sort_column:
                # ソート順の選択
                sort_order = st.radio(
                    "ソート順",
                    ["昇順", "降順"],
                    key=f"{self.key_prefix}_sort_order"
                )
                
                # ソートボタン
                if st.button("このソートを適用", key=f"{self.key_prefix}_apply_sort"):
                    ascending = sort_order == "昇順"
                    self.set_sort([sort_column], [ascending])
            
            # 現在のソートの表示
            if self.sort_columns:
                st.subheader("現在のソート")
                
                for i, (column, ascending) in enumerate(zip(self.sort_columns, self.sort_ascending)):
                    order = "昇順" if ascending else "降順"
                    st.text(f"{i+1}. {column} ({order})")
                
                # ソートをクリアするボタン
                if st.button("ソートをクリア", key=f"{self.key_prefix}_clear_sort"):
                    self.sort_columns = []
                    self.sort_ascending = []
    
    def render_pagination_ui(self, total_rows: int) -> None:
        """
        ページネーションUIを描画する
        
        Parameters:
        -----------
        total_rows : int
            総行数
        """
        if total_rows == 0:
            return
            
        page_size = self.display_settings['page_size']
        current_page = self.display_settings['current_page']
        
        # ページ数の計算
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        
        # ページネーションUI
        st.subheader("ページネーション")
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("<<", key=f"{self.key_prefix}_first_page", disabled=current_page == 0):
                self.set_page(0)
                
        with col2:
            if st.button("<", key=f"{self.key_prefix}_prev_page", disabled=current_page == 0):
                self.prev_page()
                
        with col3:
            st.text(f"ページ {current_page + 1} / {total_pages} (全{total_rows}行)")
            
        with col4:
            if st.button(">", key=f"{self.key_prefix}_next_page", disabled=current_page >= total_pages - 1):
                self.next_page()
                
        with col5:
            if st.button(">>", key=f"{self.key_prefix}_last_page", disabled=current_page >= total_pages - 1):
                self.set_page(total_pages - 1)
        
        # ページサイズの選択
        page_size_options = [5, 10, 20, 50, 100]
        selected_page_size = st.selectbox(
            "1ページあたりの行数",
            options=page_size_options,
            index=page_size_options.index(page_size) if page_size in page_size_options else 1,
            key=f"{self.key_prefix}_page_size"
        )
        
        if selected_page_size != page_size:
            self.set_page_size(selected_page_size)
    
    def render_display_options_ui(self) -> None:
        """
        表示オプションUIを描画する
        """
        if self.data is None:
            return
            
        st.subheader("表示オプション")
        
        with st.expander("表示設定", expanded=False):
            # インデックス表示の切り替え
            show_index = st.checkbox(
                "インデックスを表示",
                value=self.display_settings['show_index'],
                key=f"{self.key_prefix}_show_index"
            )
            
            if show_index != self.display_settings['show_index']:
                self.display_settings['show_index'] = show_index
            
            # 非表示列の選択
            columns = self.data.columns.tolist()
            hide_columns = st.multiselect(
                "非表示にする列",
                options=columns,
                default=self.display_settings['hide_columns'],
                key=f"{self.key_prefix}_hide_columns"
            )
            
            if hide_columns != self.display_settings['hide_columns']:
                self.hide_columns(hide_columns)
            
            # 列の順序
            st.text("列の表示順序（ドラッグで並べ替え）")
            
            if self.display_settings['column_order'] is not None:
                # 現在の順序を取得
                current_order = self.display_settings['column_order']
                # 表示されていない列を追加
                remaining_columns = [col for col in columns if col not in current_order]
                # 順序付きリスト
                ordered_columns = current_order + remaining_columns
            else:
                ordered_columns = columns
            
            # 非表示の列を除外
            ordered_columns = [col for col in ordered_columns if col not in hide_columns]
            
            # 列の順序を設定
            if ordered_columns:
                column_order = st.multiselect(
                    "列の順序",
                    options=ordered_columns,
                    default=ordered_columns,
                    key=f"{self.key_prefix}_column_order"
                )
                
                if column_order != self.display_settings['column_order']:
                    self.set_column_order(column_order)
            
            # 表示設定のリセット
            if st.button("表示設定をリセット", key=f"{self.key_prefix}_reset_display"):
                self.reset_display_settings()
    
    def render_data_editor(self, key: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        データエディタを描画する
        
        Parameters:
        -----------
        key : Optional[str]
            Streamlitウィジェットのキー
            
        Returns:
        --------
        Optional[pd.DataFrame]
            編集されたデータフレーム（変更がない場合はNone）
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        # 表示用のデータを準備
        display_data = self._prepare_display_data()
        
        # ページネーションを適用
        paginated_data = self._get_paginated_data(display_data)
        
        # データの統計情報
        st.subheader("データ情報")
        st.text(f"全体: {len(self.data)}行 x {len(self.data.columns)}列")
        st.text(f"フィルタ後: {len(display_data)}行")
        st.text(f"現在のページ: {len(paginated_data)}行")
        
        # データが空の場合
        if paginated_data.empty:
            st.warning("表示するデータがありません")
            return None
        
        # データエディタの表示
        st.subheader("データエディタ")
        
        editor_key = key if key else f"{self.key_prefix}_editor"
        
        edited_data = st.data_editor(
            paginated_data,
            column_config=self.column_config,
            hide_index=not self.display_settings['show_index'],
            use_container_width=True,
            num_rows="fixed",
            key=editor_key
        )
        
        # 編集の検出
        if not paginated_data.equals(edited_data):
            # 編集履歴の保存
            self._save_edit_history(edited_data)
            
            # 元のデータフレームの対応する行を更新
            for idx, row in edited_data.iterrows():
                self.data.loc[idx] = row
            
            self.edited_data = edited_data
            logger.info(f"データが編集されました: {len(edited_data)}行")
            return edited_data
        
        return None
    
    def render_ui(self, key_suffix: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        完全なUIを描画する
        
        Parameters:
        -----------
        key_suffix : Optional[str]
            キーのサフィックス
            
        Returns:
        --------
        Optional[pd.DataFrame]
            編集されたデータフレーム（変更がない場合はNone）
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        key = f"{self.key_prefix}_{key_suffix}" if key_suffix else self.key_prefix
        
        # タブでUIを分割
        tab1, tab2, tab3, tab4 = st.tabs(["データエディタ", "フィルタ・検索", "ソート・ページネーション", "表示オプション"])
        
        with tab1:
            edited_data = self.render_data_editor(key=f"{key}_editor")
            
            # 編集関連のボタン
            col1, col2 = st.columns(2)
            with col1:
                if st.button("元に戻す", key=f"{key}_undo", disabled=len(self.edit_history) == 0):
                    self.undo_edit()
            with col2:
                if st.button("すべてリセット", key=f"{key}_reset"):
                    self.reset_data()
        
        with tab2:
            # フィルタと検索UI
            self.render_filter_ui()
            st.divider()
            self.render_search_ui()
        
        with tab3:
            # ソートとページネーションUI
            self.render_sort_ui()
            st.divider()
            
            # 表示用のデータを準備（ページネーションUIのため）
            display_data = self._prepare_display_data()
            self.render_pagination_ui(len(display_data))
        
        with tab4:
            # 表示オプションUI
            self.render_display_options_ui()
        
        return edited_data
    
    def get_data(self) -> pd.DataFrame:
        """
        現在のデータを取得する
        
        Returns:
        --------
        pd.DataFrame
            現在のデータフレーム
        """
        return self.data.copy() if self.data is not None else pd.DataFrame()
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        フィルタリングされたデータを取得する
        
        Returns:
        --------
        pd.DataFrame
            フィルタリングされたデータフレーム
        """
        if self.data is None:
            return pd.DataFrame()
            
        # フィルタリングと検索を適用
        filtered_data = self._apply_filters(self.data)
        filtered_data = self._apply_search(filtered_data)
        
        return filtered_data
    
    def get_display_data(self) -> pd.DataFrame:
        """
        表示用のデータを取得する（フィルタ、検索、ソートを適用）
        
        Returns:
        --------
        pd.DataFrame
            表示用のデータフレーム
        """
        if self.data is None:
            return pd.DataFrame()
            
        return self._prepare_display_data()
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.data = None
        self.original_data = None
        self.edited_data = None
        self.edit_history = []
        gc.collect()
