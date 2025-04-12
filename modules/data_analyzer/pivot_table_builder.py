import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
import gc
from datetime import datetime

logger = logging.getLogger(__name__)

class PivotTableBuilder:
    """
    ピボットテーブル機能を提供するクラス
    pandasのpivot_table関数を活用し、インタラクティブなピボットテーブル分析を実装
    """
    
    def __init__(self, key_prefix: str = "pivot_table"):
        """
        PivotTableBuilderクラスの初期化
        
        Parameters:
        -----------
        key_prefix : str
            Streamlitウィジェットのキープレフィックス
        """
        self.data = None
        self.pivot_table = None
        self.key_prefix = key_prefix
        self.settings = {
            'index': None,
            'columns': None,
            'values': None,
            'aggfunc': 'mean',
            'fill_value': None,
            'margins': False,
            'margins_name': 'All',
            'dropna': True,
            'multi_index': False,
            'multi_columns': False
        }
        self.available_agg_funcs = {
            'mean': np.mean,
            'sum': np.sum,
            'count': len,
            'min': np.min,
            'max': np.max,
            'median': np.median,
            'std': np.std,
            'var': np.var,
            'first': lambda x: x.iloc[0] if len(x) > 0 else None,
            'last': lambda x: x.iloc[-1] if len(x) > 0 else None
        }
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.display_settings = {
            'show_heatmap': False,
            'color_scale': 'RdBu_r',
            'show_chart': False,
            'chart_type': 'bar',
            'transpose': False,
            'normalize': False,
            'precision': 2
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
        else:
            self.data = data
            
        self.pivot_table = None
        
        # 列の種類を分析
        self._analyze_columns()
        
        logger.info(f"データを読み込みました。行数: {len(self.data)}, 列数: {len(self.data.columns)}")
    
    def _analyze_columns(self) -> None:
        """
        データフレームの列を分析し、種類ごとに分類する
        """
        if self.data is None:
            return
            
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.datetime_columns.append(col)
            else:
                # カテゴリ列または文字列列
                self.categorical_columns.append(col)
        
        logger.info(f"列の分析: 数値列={len(self.numeric_columns)}, カテゴリ列={len(self.categorical_columns)}, 日時列={len(self.datetime_columns)}")
    
    def set_pivot_settings(self, settings: Dict[str, Any]) -> None:
        """
        ピボットテーブルの設定を更新する
        
        Parameters:
        -----------
        settings : Dict[str, Any]
            更新する設定の辞書
        """
        # 設定を更新
        self.settings.update(settings)
        
        # ピボットテーブルをリセット
        self.pivot_table = None
        
        logger.info(f"ピボットテーブルの設定を更新しました: {settings.keys()}")
    
    def set_display_settings(self, settings: Dict[str, Any]) -> None:
        """
        表示設定を更新する
        
        Parameters:
        -----------
        settings : Dict[str, Any]
            更新する設定の辞書
        """
        # 設定を更新
        self.display_settings.update(settings)
        
        logger.info(f"表示設定を更新しました: {settings.keys()}")
    
    def build_pivot_table(self) -> Optional[pd.DataFrame]:
        """
        ピボットテーブルを構築する
        
        Returns:
        --------
        Optional[pd.DataFrame]
            構築されたピボットテーブル
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 必須パラメータの確認
        if self.settings['index'] is None:
            logger.warning("インデックス列が設定されていません")
            return None
            
        if self.settings['values'] is None:
            logger.warning("値の列が設定されていません")
            return None
            
        try:
            # インデックスの準備
            index = self.settings['index']
            if self.settings['multi_index'] and isinstance(index, list) and len(index) > 1:
                index_param = index
            else:
                index_param = index[0] if isinstance(index, list) else index
            
            # 列の準備
            columns = self.settings['columns']
            if self.settings['multi_columns'] and isinstance(columns, list) and len(columns) > 1:
                columns_param = columns
            else:
                columns_param = columns[0] if isinstance(columns, list) and columns else None
            
            # 値の準備
            values = self.settings['values']
            values_param = values if isinstance(values, list) else [values]
            
            # 集計関数の準備
            aggfunc = self.settings['aggfunc']
            if isinstance(aggfunc, str):
                if aggfunc in self.available_agg_funcs:
                    aggfunc_param = self.available_agg_funcs[aggfunc]
                else:
                    logger.warning(f"無効な集計関数です: {aggfunc}")
                    aggfunc_param = np.mean
            else:
                aggfunc_param = aggfunc
            
            # ピボットテーブルの構築
            start_time = time.time()
            
            pivot_table = pd.pivot_table(
                self.data,
                values=values_param,
                index=index_param,
                columns=columns_param,
                aggfunc=aggfunc_param,
                fill_value=self.settings['fill_value'],
                margins=self.settings['margins'],
                margins_name=self.settings['margins_name'],
                dropna=self.settings['dropna']
            )
            
            elapsed_time = time.time() - start_time
            
            # 正規化（オプション）
            if self.display_settings['normalize']:
                if self.display_settings['transpose']:
                    # 列方向に正規化
                    pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)
                else:
                    # 行方向に正規化
                    pivot_table = pivot_table.div(pivot_table.sum(axis=0), axis=1)
            
            # 転置（オプション）
            if self.display_settings['transpose']:
                pivot_table = pivot_table.transpose()
            
            self.pivot_table = pivot_table
            
            logger.info(f"ピボットテーブルを構築しました。形状: {pivot_table.shape}, 所要時間: {elapsed_time:.2f}秒")
            
            return pivot_table
            
        except Exception as e:
            logger.error(f"ピボットテーブルの構築中にエラーが発生しました: {str(e)}")
            return None
    
    def get_pivot_table(self) -> Optional[pd.DataFrame]:
        """
        構築済みのピボットテーブルを取得する
        
        Returns:
        --------
        Optional[pd.DataFrame]
            ピボットテーブル
        """
        if self.pivot_table is None:
            return self.build_pivot_table()
        return self.pivot_table
    
    def render_pivot_settings_ui(self) -> None:
        """
        ピボットテーブル設定UIを描画する
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return
            
        st.subheader("ピボットテーブル設定")
        
        with st.form(key=f"{self.key_prefix}_settings_form"):
            # インデックス（行）の選択
            all_columns = self.data.columns.tolist()
            
            # インデックス列の選択
            st.subheader("行（インデックス）")
            
            index_multi = st.checkbox(
                "複数の行インデックスを使用",
                value=self.settings['multi_index'],
                key=f"{self.key_prefix}_multi_index"
            )
            
            if index_multi:
                index = st.multiselect(
                    "行インデックスとして使用する列",
                    options=all_columns,
                    default=self.settings['index'] if isinstance(self.settings['index'], list) else [self.settings['index']] if self.settings['index'] else [],
                    key=f"{self.key_prefix}_index_multi"
                )
            else:
                index = st.selectbox(
                    "行インデックスとして使用する列",
                    options=[""] + all_columns,
                    index=0 if self.settings['index'] is None else all_columns.index(self.settings['index'][0] if isinstance(self.settings['index'], list) else self.settings['index']) + 1,
                    key=f"{self.key_prefix}_index_single"
                )
                index = index if index else None
            
            # 列の選択
            st.subheader("列")
            
            columns_multi = st.checkbox(
                "複数の列インデックスを使用",
                value=self.settings['multi_columns'],
                key=f"{self.key_prefix}_multi_columns"
            )
            
            if columns_multi:
                columns = st.multiselect(
                    "列インデックスとして使用する列",
                    options=all_columns,
                    default=self.settings['columns'] if isinstance(self.settings['columns'], list) else [self.settings['columns']] if self.settings['columns'] else [],
                    key=f"{self.key_prefix}_columns_multi"
                )
            else:
                columns = st.selectbox(
                    "列インデックスとして使用する列",
                    options=[""] + all_columns,
                    index=0 if self.settings['columns'] is None else all_columns.index(self.settings['columns'][0] if isinstance(self.settings['columns'], list) else self.settings['columns']) + 1,
                    key=f"{self.key_prefix}_columns_single"
                )
                columns = columns if columns else None
            
            # 値の選択
            st.subheader("値")
            
            values_multi = st.checkbox(
                "複数の値列を使用",
                value=isinstance(self.settings['values'], list) and len(self.settings['values']) > 1,
                key=f"{self.key_prefix}_values_multi"
            )
            
            if values_multi:
                values = st.multiselect(
                    "集計する値の列",
                    options=self.numeric_columns,
                    default=self.settings['values'] if isinstance(self.settings['values'], list) else [self.settings['values']] if self.settings['values'] else [],
                    key=f"{self.key_prefix}_values_multi"
                )
            else:
                values = st.selectbox(
                    "集計する値の列",
                    options=[""] + self.numeric_columns,
                    index=0 if self.settings['values'] is None else self.numeric_columns.index(self.settings['values'][0] if isinstance(self.settings['values'], list) else self.settings['values']) + 1 if self.settings['values'] in self.numeric_columns or (isinstance(self.settings['values'], list) and self.settings['values'][0] in self.numeric_columns) else 0,
                    key=f"{self.key_prefix}_values_single"
                )
                values = values if values else None
            
            # 集計関数の選択
            st.subheader("集計関数")
            
            aggfunc = st.selectbox(
                "集計関数",
                options=list(self.available_agg_funcs.keys()),
                index=list(self.available_agg_funcs.keys()).index(self.settings['aggfunc']) if self.settings['aggfunc'] in self.available_agg_funcs else 0,
                key=f"{self.key_prefix}_aggfunc"
            )
            
            # 追加オプション
            st.subheader("追加オプション")
            
            col1, col2 = st.columns(2)
            
            with col1:
                margins = st.checkbox(
                    "合計行と列を表示",
                    value=self.settings['margins'],
                    key=f"{self.key_prefix}_margins"
                )
                
                margins_name = st.text_input(
                    "合計の表示名",
                    value=self.settings['margins_name'],
                    key=f"{self.key_prefix}_margins_name"
                )
            
            with col2:
                dropna = st.checkbox(
                    "欠損値を除外",
                    value=self.settings['dropna'],
                    key=f"{self.key_prefix}_dropna"
                )
                
                fill_value = st.text_input(
                    "欠損値の代替値",
                    value=str(self.settings['fill_value']) if self.settings['fill_value'] is not None else "",
                    key=f"{self.key_prefix}_fill_value"
                )
                fill_value = float(fill_value) if fill_value and fill_value.replace('.', '', 1).isdigit() else None
            
            # 表示オプション
            st.subheader("表示オプション")
            
            col1, col2 = st.columns(2)
            
            with col1:
                transpose = st.checkbox(
                    "行と列を入れ替える",
                    value=self.settings['transpose'],
                    key=f"{self.key_prefix}_transpose"
                )
                
                normalize = st.checkbox(
                    "正規化（割合表示）",
                    value=self.settings['normalize'],
                    key=f"{self.key_prefix}_normalize"
                )
            
            with col2:
                precision = st.number_input(
                    "小数点以下の桁数",
                    min_value=0,
                    max_value=6,
                    value=self.display_settings['precision'],
                    key=f"{self.key_prefix}_precision"
                )
                
                show_heatmap = st.checkbox(
                    "ヒートマップ表示",
                    value=self.display_settings['show_heatmap'],
                    key=f"{self.key_prefix}_show_heatmap"
                )
            
            # ヒートマップのカラースケール
            if show_heatmap:
                color_scale = st.selectbox(
                    "カラースケール",
                    options=["RdBu_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Greys", "YlGnBu", "YlOrRd"],
                    index=0 if self.display_settings['color_scale'] == "RdBu_r" else ["RdBu_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Greys", "YlGnBu", "YlOrRd"].index(self.display_settings['color_scale']),
                    key=f"{self.key_prefix}_color_scale"
                )
            else:
                color_scale = self.display_settings['color_scale']
            
            # チャート表示オプション
            show_chart = st.checkbox(
                "チャート表示",
                value=self.display_settings['show_chart'],
                key=f"{self.key_prefix}_show_chart"
            )
            
            if show_chart:
                chart_type = st.selectbox(
                    "チャートタイプ",
                    options=["bar", "line", "area", "heatmap", "scatter"],
                    index=["bar", "line", "area", "heatmap", "scatter"].index(self.display_settings['chart_type']),
                    key=f"{self.key_prefix}_chart_type"
                )
            else:
                chart_type = self.display_settings['chart_type']
            
            # 設定の適用ボタン
            submitted = st.form_submit_button("ピボットテーブルを作成")
            
            if submitted:
                # 設定の更新
                new_settings = {
                    'index': index,
                    'columns': columns,
                    'values': values,
                    'aggfunc': aggfunc,
                    'fill_value': fill_value,
                    'margins': margins,
                    'margins_name': margins_name,
                    'dropna': dropna,
                    'multi_index': index_multi,
                    'multi_columns': columns_multi
                }
                
                new_display_settings = {
                    'show_heatmap': show_heatmap,
                    'color_scale': color_scale,
                    'show_chart': show_chart,
                    'chart_type': chart_type,
                    'transpose': transpose,
                    'normalize': normalize,
                    'precision': precision
                }
                
                self.set_pivot_settings(new_settings)
                self.set_display_settings(new_display_settings)
                
                # ピボットテーブルの構築
                self.build_pivot_table()
    
    def render_pivot_table(self) -> None:
        """
        ピボットテーブルを描画する
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return
            
        # ピボットテーブルの取得
        pivot_table = self.get_pivot_table()
        
        if pivot_table is None or pivot_table.empty:
            st.warning("ピボットテーブルが空です。設定を確認してください。")
            return
        
        st.subheader("ピボットテーブル")
        
        # ピボットテーブルの表示
        if self.display_settings['show_heatmap']:
            # ヒートマップとしてスタイル付きで表示
            cm = px.colors.sequential.Viridis if self.display_settings['color_scale'] == "Viridis" else self.display_settings['color_scale']
            
            # データフレームのスタイル設定
            styled_pivot = pivot_table.style.background_gradient(cmap=cm)
            
            # 小数点以下の桁数を設定
            styled_pivot = styled_pivot.format(precision=self.display_settings['precision'])
            
            # スタイル付きデータフレームの表示
            st.dataframe(styled_pivot, use_container_width=True)
        else:
            # 通常のデータフレームとして表示
            # 小数点以下の桁数を設定
            formatted_pivot = pivot_table.round(self.display_settings['precision'])
            st.dataframe(formatted_pivot, use_container_width=True)
        
        # チャートの表示
        if self.display_settings['show_chart']:
            self.render_pivot_chart(pivot_table)
        
        # ダウンロードボタン
        self._add_download_button(pivot_table)
    
    def render_pivot_chart(self, pivot_table: pd.DataFrame) -> None:
        """
        ピボットテーブルのチャートを描画する
        
        Parameters:
        -----------
        pivot_table : pd.DataFrame
            描画するピボットテーブル
        """
        st.subheader("ピボットチャート")
        
        # マルチインデックスの処理
        if isinstance(pivot_table.index, pd.MultiIndex):
            pivot_table = pivot_table.reset_index()
        
        if isinstance(pivot_table.columns, pd.MultiIndex):
            # 列名を結合
            pivot_table.columns = [' - '.join(map(str, col)).strip() for col in pivot_table.columns.values]
        
        # チャートタイプに応じた描画
        chart_type = self.display_settings['chart_type']
        
        try:
            if chart_type == 'bar':
                # 棒グラフ
                fig = px.bar(
                    pivot_table.reset_index(),
                    x=pivot_table.index.name if pivot_table.index.name else 'index',
                    y=pivot_table.columns.tolist(),
                    title="ピボットテーブル - 棒グラフ",
                    barmode='group'
                )
                
            elif chart_type == 'line':
                # 折れ線グラフ
                fig = px.line(
                    pivot_table.reset_index(),
                    x=pivot_table.index.name if pivot_table.index.name else 'index',
                    y=pivot_table.columns.tolist(),
                    title="ピボットテーブル - 折れ線グラフ",
                    markers=True
                )
                
            elif chart_type == 'area':
                # エリアチャート
                fig = px.area(
                    pivot_table.reset_index(),
                    x=pivot_table.index.name if pivot_table.index.name else 'index',
                    y=pivot_table.columns.tolist(),
                    title="ピボットテーブル - エリアチャート"
                )
                
            elif chart_type == 'heatmap':
                # ヒートマップ
                fig = px.imshow(
                    pivot_table,
                    title="ピボットテーブル - ヒートマップ",
                    color_continuous_scale=self.display_settings['color_scale']
                )
                
            elif chart_type == 'scatter':
                # 散布図（最初の2列のみ）
                if len(pivot_table.columns) >= 2:
                    fig = px.scatter(
                        pivot_table.reset_index(),
                        x=pivot_table.columns[0],
                        y=pivot_table.columns[1],
                        color=pivot_table.index.name if pivot_table.index.name else None,
                        title="ピボットテーブル - 散布図"
                    )
                else:
                    st.warning("散布図には少なくとも2つの数値列が必要です")
                    return
            else:
                st.warning(f"サポートされていないチャートタイプです: {chart_type}")
                return
            
            # チャートの表示
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"チャートの描画中にエラーが発生しました: {str(e)}")
    
    def _add_download_button(self, pivot_table: pd.DataFrame) -> None:
        """
        ピボットテーブルのダウンロードボタンを追加する
        
        Parameters:
        -----------
        pivot_table : pd.DataFrame
            ダウンロード対象のピボットテーブル
        """
        st.subheader("ピボットテーブルのダウンロード")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSVダウンロード
            csv = pivot_table.to_csv()
            st.download_button(
                label="CSVダウンロード",
                data=csv,
                file_name="pivot_table.csv",
                mime="text/csv",
                key=f"{self.key_prefix}_download_csv"
            )
            
        with col2:
            # Excelダウンロード
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                pivot_table.to_excel(writer, sheet_name='PivotTable')
                # ワークシートの取得
                workbook = writer.book
                worksheet = writer.sheets['PivotTable']
                
                # フォーマットの設定
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # ヘッダー行に適用
                for col_num, value in enumerate(pivot_table.columns.values):
                    worksheet.write(0, col_num + 1, value, header_format)
            
            st.download_button(
                label="Excelダウンロード",
                data=buffer.getvalue(),
                file_name="pivot_table.xlsx",
                mime="application/vnd.ms-excel",
                key=f"{self.key_prefix}_download_excel"
            )
            
        with col3:
            # HTMLダウンロード
            html = pivot_table.to_html()
            st.download_button(
                label="HTMLダウンロード",
                data=html,
                file_name="pivot_table.html",
                mime="text/html",
                key=f"{self.key_prefix}_download_html"
            )
    
    def render_ui(self) -> None:
        """
        完全なUIを描画する
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return
            
        # タブでUIを分割
        tab1, tab2 = st.tabs(["ピボットテーブル設定", "ピボットテーブル表示"])
        
        with tab1:
            self.render_pivot_settings_ui()
            
        with tab2:
            self.render_pivot_table()
    
    def get_summary_stats(self, pivot_table: Optional[pd.DataFrame] = None) -> Dict:
        """
        ピボットテーブルの要約統計を取得する
        
        Parameters:
        -----------
        pivot_table : Optional[pd.DataFrame]
            要約統計を計算するピボットテーブル（Noneの場合は現在のピボットテーブルを使用）
            
        Returns:
        --------
        Dict
            要約統計の辞書
        """
        if pivot_table is None:
            pivot_table = self.get_pivot_table()
            
        if pivot_table is None or pivot_table.empty:
            return {}
            
        # 要約統計の計算
        summary = {
            'shape': pivot_table.shape,
            'total_cells': pivot_table.size,
            'non_null_cells': pivot_table.count().sum(),
            'null_cells': pivot_table.isnull().sum().sum(),
            'min': pivot_table.min().min(),
            'max': pivot_table.max().max(),
            'mean': pivot_table.mean().mean(),
            'sum': pivot_table.sum().sum()
        }
        
        return summary
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.data = None
        self.pivot_table = None
        gc.collect()

# BytesIOのインポート（Excelダウンロード用）
import io
