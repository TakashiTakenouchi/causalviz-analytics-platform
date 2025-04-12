import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
import logging
import time
import io
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class Visualization:
    """
    データ可視化機能を提供するクラス
    MatplotlibとPlotlyを活用した多様なグラフタイプを実装
    """
    
    def __init__(self, key_prefix: str = "visualization"):
        """
        Visualizationクラスの初期化
        
        Parameters:
        -----------
        key_prefix : str
            Streamlitウィジェットのキープレフィックス
        """
        self.data = None
        self.key_prefix = key_prefix
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.figure_settings = {
            'title': '',
            'x_label': '',
            'y_label': '',
            'color_column': None,
            'size_column': None,
            'facet_column': None,
            'facet_row': None,
            'height': 500,
            'width': 800,
            'color_scheme': 'viridis',
            'template': 'plotly_white',
            'show_legend': True,
            'interactive': True
        }
        self.available_color_schemes = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
            'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
            'BuGn', 'YlGn'
        ]
        self.available_templates = [
            'plotly', 'plotly_white', 'plotly_dark', 'ggplot2',
            'seaborn', 'simple_white', 'none'
        ]
        self.last_figure = None
        self.last_figure_type = None
    
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
    
    def set_figure_settings(self, settings: Dict[str, Any]) -> None:
        """
        図の設定を更新する
        
        Parameters:
        -----------
        settings : Dict[str, Any]
            更新する設定の辞書
        """
        # 設定を更新
        self.figure_settings.update(settings)
        
        logger.info(f"図の設定を更新しました: {settings.keys()}")
    
    def _apply_common_settings(self, fig) -> None:
        """
        共通の図の設定を適用する
        
        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            設定を適用する図
        """
        # タイトルと軸ラベルの設定
        fig.update_layout(
            title=self.figure_settings['title'],
            xaxis_title=self.figure_settings['x_label'],
            yaxis_title=self.figure_settings['y_label'],
            height=self.figure_settings['height'],
            width=self.figure_settings['width'],
            template=self.figure_settings['template'],
            showlegend=self.figure_settings['show_legend']
        )
    
    def line_chart(self, x_column: str, y_columns: Union[str, List[str]], 
                  group_by: Optional[str] = None, 
                  agg_func: str = 'mean',
                  markers: bool = True,
                  line_shape: str = 'linear',
                  show_area: bool = False) -> go.Figure:
        """
        折れ線グラフを作成する
        
        Parameters:
        -----------
        x_column : str
            X軸の列名
        y_columns : Union[str, List[str]]
            Y軸の列名または列名のリスト
        group_by : Optional[str]
            グループ化する列名
        agg_func : str
            集計関数（'mean', 'sum', 'count', 'min', 'max'）
        markers : bool
            マーカーを表示するかどうか
        line_shape : str
            線の形状（'linear', 'spline', 'hv', 'vh', 'hvh', 'vhv'）
        show_area : bool
            エリアを表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成された折れ線グラフ
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # Y軸の列をリストに変換
        if isinstance(y_columns, str):
            y_columns = [y_columns]
            
        # 有効な列の確認
        if x_column not in self.data.columns:
            logger.warning(f"列 '{x_column}' はデータフレームに存在しません")
            return None
            
        for col in y_columns:
            if col not in self.data.columns:
                logger.warning(f"列 '{col}' はデータフレームに存在しません")
                return None
        
        try:
            # グループ化が指定されている場合
            if group_by is not None and group_by in self.data.columns:
                # グループごとのデータを準備
                grouped_data = self.data.groupby([group_by, x_column])[y_columns].agg(agg_func).reset_index()
                
                # 折れ線グラフの作成
                if show_area:
                    fig = px.area(
                        grouped_data,
                        x=x_column,
                        y=y_columns,
                        color=group_by,
                        line_shape=line_shape,
                        markers=markers,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    fig = px.line(
                        grouped_data,
                        x=x_column,
                        y=y_columns,
                        color=group_by,
                        line_shape=line_shape,
                        markers=markers,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
            else:
                # 通常の折れ線グラフ
                if show_area:
                    fig = px.area(
                        self.data,
                        x=x_column,
                        y=y_columns,
                        line_shape=line_shape,
                        markers=markers,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    fig = px.line(
                        self.data,
                        x=x_column,
                        y=y_columns,
                        line_shape=line_shape,
                        markers=markers,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'line'
            
            return fig
            
        except Exception as e:
            logger.error(f"折れ線グラフの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def scatter_plot(self, x_column: str, y_column: str, 
                    color_column: Optional[str] = None,
                    size_column: Optional[str] = None,
                    text_column: Optional[str] = None,
                    opacity: float = 0.7,
                    trendline: Optional[str] = None) -> go.Figure:
        """
        散布図を作成する
        
        Parameters:
        -----------
        x_column : str
            X軸の列名
        y_column : str
            Y軸の列名
        color_column : Optional[str]
            色分けする列名
        size_column : Optional[str]
            サイズを変える列名
        text_column : Optional[str]
            ホバーテキストに表示する列名
        opacity : float
            不透明度（0.0〜1.0）
        trendline : Optional[str]
            トレンドラインの種類（'ols', 'lowess', None）
            
        Returns:
        --------
        go.Figure
            作成された散布図
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if x_column not in self.data.columns:
            logger.warning(f"列 '{x_column}' はデータフレームに存在しません")
            return None
            
        if y_column not in self.data.columns:
            logger.warning(f"列 '{y_column}' はデータフレームに存在しません")
            return None
        
        try:
            # 散布図の作成
            fig = px.scatter(
                self.data,
                x=x_column,
                y=y_column,
                color=color_column,
                size=size_column,
                text=text_column,
                opacity=opacity,
                trendline=trendline,
                color_continuous_scale=self.figure_settings['color_scheme'] if color_column and color_column in self.numeric_columns else None,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column and color_column in self.categorical_columns else None
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'scatter'
            
            return fig
            
        except Exception as e:
            logger.error(f"散布図の作成中にエラーが発生しました: {str(e)}")
            return None
    
    def bar_chart(self, x_column: str, y_column: str, 
                 color_column: Optional[str] = None,
                 orientation: str = 'v',
                 barmode: str = 'group',
                 text_auto: bool = False,
                 sort_values: bool = False) -> go.Figure:
        """
        棒グラフを作成する
        
        Parameters:
        -----------
        x_column : str
            X軸の列名
        y_column : str
            Y軸の列名
        color_column : Optional[str]
            色分けする列名
        orientation : str
            棒の向き（'v'=垂直, 'h'=水平）
        barmode : str
            棒の表示モード（'group', 'stack', 'relative'）
        text_auto : bool
            棒の上に値を表示するかどうか
        sort_values : bool
            値でソートするかどうか
            
        Returns:
        --------
        go.Figure
            作成された棒グラフ
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if x_column not in self.data.columns:
            logger.warning(f"列 '{x_column}' はデータフレームに存在しません")
            return None
            
        if y_column not in self.data.columns:
            logger.warning(f"列 '{y_column}' はデータフレームに存在しません")
            return None
        
        try:
            # データの準備
            plot_data = self.data.copy()
            
            # ソートが指定されている場合
            if sort_values:
                if orientation == 'v':
                    plot_data = plot_data.sort_values(by=y_column)
                else:
                    plot_data = plot_data.sort_values(by=x_column)
            
            # 棒グラフの作成
            fig = px.bar(
                plot_data,
                x=x_column if orientation == 'v' else y_column,
                y=y_column if orientation == 'v' else x_column,
                color=color_column,
                orientation=orientation,
                barmode=barmode,
                text_auto=text_auto,
                color_continuous_scale=self.figure_settings['color_scheme'] if color_column and color_column in self.numeric_columns else None,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column and color_column in self.categorical_columns else None
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'bar'
            
            return fig
            
        except Exception as e:
            logger.error(f"棒グラフの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def pareto_chart(self, category_column: str, value_column: str, 
                    top_n: Optional[int] = None,
                    cumulative_line: bool = True) -> go.Figure:
        """
        パレート図を作成する
        
        Parameters:
        -----------
        category_column : str
            カテゴリの列名
        value_column : str
            値の列名
        top_n : Optional[int]
            表示するカテゴリの数（上位N個）
        cumulative_line : bool
            累積線を表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成されたパレート図
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if category_column not in self.data.columns:
            logger.warning(f"列 '{category_column}' はデータフレームに存在しません")
            return None
            
        if value_column not in self.data.columns:
            logger.warning(f"列 '{value_column}' はデータフレームに存在しません")
            return None
        
        try:
            # データの準備
            # カテゴリごとに集計
            pareto_data = self.data.groupby(category_column)[value_column].sum().reset_index()
            
            # 値の降順でソート
            pareto_data = pareto_data.sort_values(by=value_column, ascending=False)
            
            # 上位N個に制限（指定されている場合）
            if top_n is not None and top_n > 0 and top_n < len(pareto_data):
                pareto_data = pareto_data.head(top_n)
            
            # 累積パーセントの計算
            total = pareto_data[value_column].sum()
            pareto_data['cumulative'] = pareto_data[value_column].cumsum() / total * 100
            
            # パレート図の作成
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 棒グラフの追加
            fig.add_trace(
                go.Bar(
                    x=pareto_data[category_column],
                    y=pareto_data[value_column],
                    name=value_column,
                    marker_color='royalblue'
                ),
                secondary_y=False
            )
            
            # 累積線の追加（オプション）
            if cumulative_line:
                fig.add_trace(
                    go.Scatter(
                        x=pareto_data[category_column],
                        y=pareto_data['cumulative'],
                        name='累積 %',
                        marker_color='red',
                        mode='lines+markers'
                    ),
                    secondary_y=True
                )
            
            # 軸の設定
            fig.update_layout(
                title=self.figure_settings['title'] or 'パレート図',
                xaxis_title=self.figure_settings['x_label'] or category_column,
                yaxis_title=self.figure_settings['y_label'] or value_column,
                height=self.figure_settings['height'],
                width=self.figure_settings['width'],
                template=self.figure_settings['template'],
                showlegend=self.figure_settings['show_legend']
            )
            
            # 第2軸の設定
            if cumulative_line:
                fig.update_yaxes(title_text='累積 %', secondary_y=True, range=[0, 100])
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'pareto'
            
            return fig
            
        except Exception as e:
            logger.error(f"パレート図の作成中にエラーが発生しました: {str(e)}")
            return None
    
    def histogram(self, column: str, bins: int = 20, 
                 color_column: Optional[str] = None,
                 marginal: Optional[str] = None,
                 kde: bool = False) -> go.Figure:
        """
        ヒストグラムを作成する
        
        Parameters:
        -----------
        column : str
            ヒストグラムを作成する列名
        bins : int
            ビンの数
        color_column : Optional[str]
            色分けする列名
        marginal : Optional[str]
            マージナル分布の種類（'box', 'violin', 'rug', None）
        kde : bool
            KDE（カーネル密度推定）を表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成されたヒストグラム
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if column not in self.data.columns:
            logger.warning(f"列 '{column}' はデータフレームに存在しません")
            return None
        
        try:
            # ヒストグラムの作成
            fig = px.histogram(
                self.data,
                x=column,
                color=color_column,
                nbins=bins,
                marginal=marginal,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column else None
            )
            
            # KDEの追加（オプション）
            if kde and color_column is None:
                # KDEの計算
                kde_data = self.data[column].dropna()
                kde_x = np.linspace(kde_data.min(), kde_data.max(), 1000)
                kde_y = sns.kdeplot(kde_data, bw_adjust=0.5).get_lines()[0].get_data()[1]
                
                # KDEの正規化
                hist_max = fig.data[0].y.max()
                kde_y = kde_y * (hist_max / kde_y.max())
                
                # KDEの追加
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2)
                    )
                )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'histogram'
            
            return fig
            
        except Exception as e:
            logger.error(f"ヒストグラムの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def box_plot(self, x_column: Optional[str], y_column: str, 
                color_column: Optional[str] = None,
                notched: bool = False,
                points: str = 'outliers') -> go.Figure:
        """
        箱ひげ図を作成する
        
        Parameters:
        -----------
        x_column : Optional[str]
            X軸の列名（カテゴリ）
        y_column : str
            Y軸の列名（数値）
        color_column : Optional[str]
            色分けする列名
        notched : bool
            ノッチ付きの箱ひげ図にするかどうか
        points : str
            点の表示方法（'outliers', 'suspectedoutliers', 'all', False）
            
        Returns:
        --------
        go.Figure
            作成された箱ひげ図
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if x_column is not None and x_column not in self.data.columns:
            logger.warning(f"列 '{x_column}' はデータフレームに存在しません")
            return None
            
        if y_column not in self.data.columns:
            logger.warning(f"列 '{y_column}' はデータフレームに存在しません")
            return None
        
        try:
            # 箱ひげ図の作成
            fig = px.box(
                self.data,
                x=x_column,
                y=y_column,
                color=color_column,
                notched=notched,
                points=points,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column else None
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'box'
            
            return fig
            
        except Exception as e:
            logger.error(f"箱ひげ図の作成中にエラーが発生しました: {str(e)}")
            return None
    
    def heatmap(self, columns: Optional[List[str]] = None, 
               correlation: bool = True,
               z_min: Optional[float] = None,
               z_max: Optional[float] = None,
               text_auto: bool = True) -> go.Figure:
        """
        ヒートマップを作成する
        
        Parameters:
        -----------
        columns : Optional[List[str]]
            ヒートマップに含める列名のリスト（Noneの場合は数値列すべて）
        correlation : bool
            相関行列を表示するかどうか（Falseの場合は生データ）
        z_min : Optional[float]
            カラースケールの最小値
        z_max : Optional[float]
            カラースケールの最大値
        text_auto : bool
            セルに値を表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成されたヒートマップ
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
        
        try:
            # 列の選択
            if columns is None:
                # デフォルトでは数値列すべて
                selected_columns = self.numeric_columns
            else:
                # 指定された列のうち、データフレームに存在するものだけを使用
                selected_columns = [col for col in columns if col in self.data.columns]
            
            if not selected_columns:
                logger.warning("有効な列がありません")
                return None
            
            # データの準備
            if correlation:
                # 相関行列の計算
                heatmap_data = self.data[selected_columns].corr()
            else:
                # 生データの使用
                heatmap_data = self.data[selected_columns]
            
            # ヒートマップの作成
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale=self.figure_settings['color_scheme'],
                zmin=z_min,
                zmax=z_max,
                text_auto=text_auto
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'heatmap'
            
            return fig
            
        except Exception as e:
            logger.error(f"ヒートマップの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def pie_chart(self, names_column: str, values_column: str, 
                 hole: float = 0.0,
                 pull: Optional[List[float]] = None) -> go.Figure:
        """
        円グラフを作成する
        
        Parameters:
        -----------
        names_column : str
            カテゴリの列名
        values_column : str
            値の列名
        hole : float
            中央の穴のサイズ（0.0〜1.0）
        pull : Optional[List[float]]
            各セグメントの引き出し量のリスト
            
        Returns:
        --------
        go.Figure
            作成された円グラフ
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if names_column not in self.data.columns:
            logger.warning(f"列 '{names_column}' はデータフレームに存在しません")
            return None
            
        if values_column not in self.data.columns:
            logger.warning(f"列 '{values_column}' はデータフレームに存在しません")
            return None
        
        try:
            # データの準備
            # カテゴリごとに集計
            pie_data = self.data.groupby(names_column)[values_column].sum().reset_index()
            
            # 円グラフの作成
            fig = px.pie(
                pie_data,
                names=names_column,
                values=values_column,
                hole=hole,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            # セグメントの引き出し（オプション）
            if pull is not None:
                fig.update_traces(pull=pull)
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'pie'
            
            return fig
            
        except Exception as e:
            logger.error(f"円グラフの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def violin_plot(self, x_column: Optional[str], y_column: str, 
                   color_column: Optional[str] = None,
                   box: bool = True,
                   points: str = 'outliers') -> go.Figure:
        """
        バイオリンプロットを作成する
        
        Parameters:
        -----------
        x_column : Optional[str]
            X軸の列名（カテゴリ）
        y_column : str
            Y軸の列名（数値）
        color_column : Optional[str]
            色分けする列名
        box : bool
            内部に箱ひげ図を表示するかどうか
        points : str
            点の表示方法（'outliers', 'suspectedoutliers', 'all', False）
            
        Returns:
        --------
        go.Figure
            作成されたバイオリンプロット
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if x_column is not None and x_column not in self.data.columns:
            logger.warning(f"列 '{x_column}' はデータフレームに存在しません")
            return None
            
        if y_column not in self.data.columns:
            logger.warning(f"列 '{y_column}' はデータフレームに存在しません")
            return None
        
        try:
            # バイオリンプロットの作成
            fig = px.violin(
                self.data,
                x=x_column,
                y=y_column,
                color=color_column,
                box=box,
                points=points,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column else None
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'violin'
            
            return fig
            
        except Exception as e:
            logger.error(f"バイオリンプロットの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def scatter_matrix(self, columns: Optional[List[str]] = None, 
                      color_column: Optional[str] = None,
                      diagonal_visible: bool = True) -> go.Figure:
        """
        散布図行列を作成する
        
        Parameters:
        -----------
        columns : Optional[List[str]]
            散布図行列に含める列名のリスト（Noneの場合は数値列すべて）
        color_column : Optional[str]
            色分けする列名
        diagonal_visible : bool
            対角線上にヒストグラムを表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成された散布図行列
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
        
        try:
            # 列の選択
            if columns is None:
                # デフォルトでは数値列すべて（最大6列まで）
                selected_columns = self.numeric_columns[:6]
            else:
                # 指定された列のうち、データフレームに存在するものだけを使用
                selected_columns = [col for col in columns if col in self.data.columns]
            
            if len(selected_columns) < 2:
                logger.warning("散布図行列には少なくとも2つの列が必要です")
                return None
            
            # 散布図行列の作成
            fig = px.scatter_matrix(
                self.data,
                dimensions=selected_columns,
                color=color_column,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_column and color_column in self.categorical_columns else None,
                color_continuous_scale=self.figure_settings['color_scheme'] if color_column and color_column in self.numeric_columns else None
            )
            
            # 対角線上の表示設定
            if not diagonal_visible:
                fig.update_traces(showupperhalf=False, diagonal_visible=False)
            
            # 共通設定の適用（高さと幅は調整）
            fig.update_layout(
                title=self.figure_settings['title'],
                height=max(500, self.figure_settings['height']),
                width=max(600, self.figure_settings['width']),
                template=self.figure_settings['template']
            )
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'scatter_matrix'
            
            return fig
            
        except Exception as e:
            logger.error(f"散布図行列の作成中にエラーが発生しました: {str(e)}")
            return None
    
    def time_series(self, date_column: str, value_columns: Union[str, List[str]],
                   group_by: Optional[str] = None,
                   resample: Optional[str] = None,
                   agg_func: str = 'mean',
                   markers: bool = True) -> go.Figure:
        """
        時系列グラフを作成する
        
        Parameters:
        -----------
        date_column : str
            日付の列名
        value_columns : Union[str, List[str]]
            値の列名または列名のリスト
        group_by : Optional[str]
            グループ化する列名
        resample : Optional[str]
            リサンプリング間隔（'D', 'W', 'M', 'Q', 'Y'など）
        agg_func : str
            集計関数（'mean', 'sum', 'count', 'min', 'max'）
        markers : bool
            マーカーを表示するかどうか
            
        Returns:
        --------
        go.Figure
            作成された時系列グラフ
        """
        if self.data is None:
            logger.warning("データが読み込まれていません")
            return None
            
        # 有効な列の確認
        if date_column not in self.data.columns:
            logger.warning(f"列 '{date_column}' はデータフレームに存在しません")
            return None
            
        # 値の列をリストに変換
        if isinstance(value_columns, str):
            value_columns = [value_columns]
            
        for col in value_columns:
            if col not in self.data.columns:
                logger.warning(f"列 '{col}' はデータフレームに存在しません")
                return None
        
        try:
            # データの準備
            plot_data = self.data.copy()
            
            # 日付列の確認と変換
            if not pd.api.types.is_datetime64_any_dtype(plot_data[date_column]):
                try:
                    plot_data[date_column] = pd.to_datetime(plot_data[date_column])
                except:
                    logger.warning(f"列 '{date_column}' を日付型に変換できません")
                    return None
            
            # リサンプリング（オプション）
            if resample is not None:
                if group_by is not None:
                    # グループごとにリサンプリング
                    resampled_data = []
                    for group, group_data in plot_data.groupby(group_by):
                        group_data = group_data.set_index(date_column)
                        group_resampled = group_data[value_columns].resample(resample).agg(agg_func)
                        group_resampled[group_by] = group
                        group_resampled = group_resampled.reset_index()
                        resampled_data.append(group_resampled)
                    
                    if resampled_data:
                        plot_data = pd.concat(resampled_data)
                    else:
                        logger.warning("リサンプリング後のデータが空です")
                        return None
                else:
                    # 全体をリサンプリング
                    plot_data = plot_data.set_index(date_column)
                    plot_data = plot_data[value_columns].resample(resample).agg(agg_func).reset_index()
            
            # 時系列グラフの作成
            fig = px.line(
                plot_data,
                x=date_column,
                y=value_columns,
                color=group_by,
                markers=markers,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            # 共通設定の適用
            self._apply_common_settings(fig)
            
            # 結果の保存
            self.last_figure = fig
            self.last_figure_type = 'time_series'
            
            return fig
            
        except Exception as e:
            logger.error(f"時系列グラフの作成中にエラーが発生しました: {str(e)}")
            return None
    
    def render_figure_settings_ui(self) -> None:
        """
        図の設定UIを描画する
        """
        st.subheader("グラフ設定")
        
        with st.expander("グラフの基本設定", expanded=False):
            # タイトルと軸ラベル
            title = st.text_input(
                "グラフタイトル",
                value=self.figure_settings['title'],
                key=f"{self.key_prefix}_title"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_label = st.text_input(
                    "X軸ラベル",
                    value=self.figure_settings['x_label'],
                    key=f"{self.key_prefix}_x_label"
                )
                
            with col2:
                y_label = st.text_input(
                    "Y軸ラベル",
                    value=self.figure_settings['y_label'],
                    key=f"{self.key_prefix}_y_label"
                )
            
            # サイズ設定
            col1, col2 = st.columns(2)
            
            with col1:
                width = st.number_input(
                    "幅（ピクセル）",
                    min_value=300,
                    max_value=1500,
                    value=self.figure_settings['width'],
                    step=50,
                    key=f"{self.key_prefix}_width"
                )
                
            with col2:
                height = st.number_input(
                    "高さ（ピクセル）",
                    min_value=300,
                    max_value=1200,
                    value=self.figure_settings['height'],
                    step=50,
                    key=f"{self.key_prefix}_height"
                )
            
            # テンプレートとカラースキーム
            col1, col2 = st.columns(2)
            
            with col1:
                template = st.selectbox(
                    "テンプレート",
                    options=self.available_templates,
                    index=self.available_templates.index(self.figure_settings['template']) if self.figure_settings['template'] in self.available_templates else 0,
                    key=f"{self.key_prefix}_template"
                )
                
            with col2:
                color_scheme = st.selectbox(
                    "カラースキーム",
                    options=self.available_color_schemes,
                    index=self.available_color_schemes.index(self.figure_settings['color_scheme']) if self.figure_settings['color_scheme'] in self.available_color_schemes else 0,
                    key=f"{self.key_prefix}_color_scheme"
                )
            
            # その他の設定
            show_legend = st.checkbox(
                "凡例を表示",
                value=self.figure_settings['show_legend'],
                key=f"{self.key_prefix}_show_legend"
            )
            
            # 設定の適用ボタン
            if st.button("設定を適用", key=f"{self.key_prefix}_apply_settings"):
                new_settings = {
                    'title': title,
                    'x_label': x_label,
                    'y_label': y_label,
                    'width': width,
                    'height': height,
                    'template': template,
                    'color_scheme': color_scheme,
                    'show_legend': show_legend
                }
                
                self.set_figure_settings(new_settings)
                
                # 最後の図を再描画（存在する場合）
                if self.last_figure is not None and self.last_figure_type is not None:
                    st.success("設定を適用しました。グラフを再描画してください。")
    
    def render_line_chart_ui(self) -> Optional[go.Figure]:
        """
        折れ線グラフ作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された折れ線グラフ
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("折れ線グラフ")
        
        with st.form(key=f"{self.key_prefix}_line_chart_form"):
            # X軸の列選択
            x_options = self.datetime_columns + self.numeric_columns + self.categorical_columns
            x_column = st.selectbox(
                "X軸の列",
                options=[""] + x_options,
                key=f"{self.key_prefix}_line_x"
            )
            
            # Y軸の列選択（複数可）
            y_columns = st.multiselect(
                "Y軸の列（複数選択可）",
                options=self.numeric_columns,
                key=f"{self.key_prefix}_line_y"
            )
            
            # グループ化の列選択
            group_by = st.selectbox(
                "グループ化する列（オプション）",
                options=[""] + self.categorical_columns,
                key=f"{self.key_prefix}_line_group"
            )
            group_by = group_by if group_by else None
            
            # 集計関数の選択
            agg_func = st.selectbox(
                "集計関数",
                options=['mean', 'sum', 'count', 'min', 'max', 'median'],
                key=f"{self.key_prefix}_line_agg"
            )
            
            # 追加オプション
            col1, col2, col3 = st.columns(3)
            
            with col1:
                markers = st.checkbox(
                    "マーカーを表示",
                    value=True,
                    key=f"{self.key_prefix}_line_markers"
                )
                
            with col2:
                line_shape = st.selectbox(
                    "線の形状",
                    options=['linear', 'spline', 'hv', 'vh', 'hvh', 'vhv'],
                    key=f"{self.key_prefix}_line_shape"
                )
                
            with col3:
                show_area = st.checkbox(
                    "エリアを表示",
                    value=False,
                    key=f"{self.key_prefix}_line_area"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not x_column:
                    st.warning("X軸の列を選択してください")
                    return None
                    
                if not y_columns:
                    st.warning("Y軸の列を選択してください")
                    return None
                    
                # 折れ線グラフの作成
                fig = self.line_chart(
                    x_column=x_column,
                    y_columns=y_columns,
                    group_by=group_by,
                    agg_func=agg_func,
                    markers=markers,
                    line_shape=line_shape,
                    show_area=show_area
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_scatter_plot_ui(self) -> Optional[go.Figure]:
        """
        散布図作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された散布図
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("散布図")
        
        with st.form(key=f"{self.key_prefix}_scatter_plot_form"):
            # X軸とY軸の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                x_column = st.selectbox(
                    "X軸の列",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_scatter_x"
                )
                
            with col2:
                y_column = st.selectbox(
                    "Y軸の列",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_scatter_y"
                )
            
            # 色とサイズの列選択
            col1, col2 = st.columns(2)
            
            with col1:
                color_column = st.selectbox(
                    "色分けする列（オプション）",
                    options=[""] + self.numeric_columns + self.categorical_columns,
                    key=f"{self.key_prefix}_scatter_color"
                )
                color_column = color_column if color_column else None
                
            with col2:
                size_column = st.selectbox(
                    "サイズを変える列（オプション）",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_scatter_size"
                )
                size_column = size_column if size_column else None
            
            # テキストとトレンドラインの設定
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "ホバーテキストに表示する列（オプション）",
                    options=[""] + self.data.columns.tolist(),
                    key=f"{self.key_prefix}_scatter_text"
                )
                text_column = text_column if text_column else None
                
            with col2:
                trendline = st.selectbox(
                    "トレンドライン",
                    options=["なし", "OLS回帰", "LOWESS"],
                    key=f"{self.key_prefix}_scatter_trendline"
                )
                trendline_map = {"なし": None, "OLS回帰": "ols", "LOWESS": "lowess"}
                trendline = trendline_map[trendline]
            
            # 不透明度の設定
            opacity = st.slider(
                "不透明度",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key=f"{self.key_prefix}_scatter_opacity"
            )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not x_column:
                    st.warning("X軸の列を選択してください")
                    return None
                    
                if not y_column:
                    st.warning("Y軸の列を選択してください")
                    return None
                    
                # 散布図の作成
                fig = self.scatter_plot(
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    size_column=size_column,
                    text_column=text_column,
                    opacity=opacity,
                    trendline=trendline
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_bar_chart_ui(self) -> Optional[go.Figure]:
        """
        棒グラフ作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された棒グラフ
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("棒グラフ")
        
        with st.form(key=f"{self.key_prefix}_bar_chart_form"):
            # X軸とY軸の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                x_column = st.selectbox(
                    "X軸の列",
                    options=[""] + self.categorical_columns + self.datetime_columns,
                    key=f"{self.key_prefix}_bar_x"
                )
                
            with col2:
                y_column = st.selectbox(
                    "Y軸の列",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_bar_y"
                )
            
            # 色分けとバーモードの設定
            col1, col2 = st.columns(2)
            
            with col1:
                color_column = st.selectbox(
                    "色分けする列（オプション）",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_bar_color"
                )
                color_column = color_column if color_column else None
                
            with col2:
                barmode = st.selectbox(
                    "バーの表示モード",
                    options=['group', 'stack', 'relative'],
                    key=f"{self.key_prefix}_bar_mode"
                )
            
            # 追加オプション
            col1, col2, col3 = st.columns(3)
            
            with col1:
                orientation = st.selectbox(
                    "棒の向き",
                    options=['垂直 (v)', '水平 (h)'],
                    key=f"{self.key_prefix}_bar_orientation"
                )
                orientation = 'v' if orientation == '垂直 (v)' else 'h'
                
            with col2:
                text_auto = st.checkbox(
                    "値を表示",
                    value=False,
                    key=f"{self.key_prefix}_bar_text"
                )
                
            with col3:
                sort_values = st.checkbox(
                    "値でソート",
                    value=False,
                    key=f"{self.key_prefix}_bar_sort"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not x_column:
                    st.warning("X軸の列を選択してください")
                    return None
                    
                if not y_column:
                    st.warning("Y軸の列を選択してください")
                    return None
                    
                # 棒グラフの作成
                fig = self.bar_chart(
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    orientation=orientation,
                    barmode=barmode,
                    text_auto=text_auto,
                    sort_values=sort_values
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_pareto_chart_ui(self) -> Optional[go.Figure]:
        """
        パレート図作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成されたパレート図
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("パレート図")
        
        with st.form(key=f"{self.key_prefix}_pareto_chart_form"):
            # カテゴリと値の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                category_column = st.selectbox(
                    "カテゴリの列",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_pareto_category"
                )
                
            with col2:
                value_column = st.selectbox(
                    "値の列",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_pareto_value"
                )
            
            # 追加オプション
            col1, col2 = st.columns(2)
            
            with col1:
                top_n = st.number_input(
                    "表示するカテゴリ数（0=すべて）",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    key=f"{self.key_prefix}_pareto_top_n"
                )
                top_n = top_n if top_n > 0 else None
                
            with col2:
                cumulative_line = st.checkbox(
                    "累積線を表示",
                    value=True,
                    key=f"{self.key_prefix}_pareto_cumulative"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not category_column:
                    st.warning("カテゴリの列を選択してください")
                    return None
                    
                if not value_column:
                    st.warning("値の列を選択してください")
                    return None
                    
                # パレート図の作成
                fig = self.pareto_chart(
                    category_column=category_column,
                    value_column=value_column,
                    top_n=top_n,
                    cumulative_line=cumulative_line
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_histogram_ui(self) -> Optional[go.Figure]:
        """
        ヒストグラム作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成されたヒストグラム
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("ヒストグラム")
        
        with st.form(key=f"{self.key_prefix}_histogram_form"):
            # 列選択
            column = st.selectbox(
                "ヒストグラムを作成する列",
                options=[""] + self.numeric_columns,
                key=f"{self.key_prefix}_histogram_column"
            )
            
            # ビン数と色分けの設定
            col1, col2 = st.columns(2)
            
            with col1:
                bins = st.slider(
                    "ビンの数",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    key=f"{self.key_prefix}_histogram_bins"
                )
                
            with col2:
                color_column = st.selectbox(
                    "色分けする列（オプション）",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_histogram_color"
                )
                color_column = color_column if color_column else None
            
            # 追加オプション
            col1, col2 = st.columns(2)
            
            with col1:
                marginal = st.selectbox(
                    "マージナル分布",
                    options=["なし", "box", "violin", "rug"],
                    key=f"{self.key_prefix}_histogram_marginal"
                )
                marginal_map = {"なし": None, "box": "box", "violin": "violin", "rug": "rug"}
                marginal = marginal_map[marginal]
                
            with col2:
                kde = st.checkbox(
                    "KDEを表示（色分けなしの場合のみ）",
                    value=False,
                    key=f"{self.key_prefix}_histogram_kde"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not column:
                    st.warning("列を選択してください")
                    return None
                    
                # ヒストグラムの作成
                fig = self.histogram(
                    column=column,
                    bins=bins,
                    color_column=color_column,
                    marginal=marginal,
                    kde=kde
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_box_plot_ui(self) -> Optional[go.Figure]:
        """
        箱ひげ図作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された箱ひげ図
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("箱ひげ図")
        
        with st.form(key=f"{self.key_prefix}_box_plot_form"):
            # X軸とY軸の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                x_column = st.selectbox(
                    "X軸の列（カテゴリ、オプション）",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_box_x"
                )
                x_column = x_column if x_column else None
                
            with col2:
                y_column = st.selectbox(
                    "Y軸の列（数値）",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_box_y"
                )
            
            # 色分けと追加オプション
            col1, col2, col3 = st.columns(3)
            
            with col1:
                color_column = st.selectbox(
                    "色分けする列（オプション）",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_box_color"
                )
                color_column = color_column if color_column else None
                
            with col2:
                notched = st.checkbox(
                    "ノッチ付き",
                    value=False,
                    key=f"{self.key_prefix}_box_notched"
                )
                
            with col3:
                points = st.selectbox(
                    "点の表示",
                    options=['外れ値のみ', '疑わしい外れ値', 'すべて', '表示しない'],
                    key=f"{self.key_prefix}_box_points"
                )
                points_map = {'外れ値のみ': 'outliers', '疑わしい外れ値': 'suspectedoutliers', 'すべて': 'all', '表示しない': False}
                points = points_map[points]
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not y_column:
                    st.warning("Y軸の列を選択してください")
                    return None
                    
                # 箱ひげ図の作成
                fig = self.box_plot(
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color_column,
                    notched=notched,
                    points=points
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_heatmap_ui(self) -> Optional[go.Figure]:
        """
        ヒートマップ作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成されたヒートマップ
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("ヒートマップ")
        
        with st.form(key=f"{self.key_prefix}_heatmap_form"):
            # 列の選択
            columns = st.multiselect(
                "ヒートマップに含める列（数値列のみ、空の場合はすべての数値列）",
                options=self.numeric_columns,
                key=f"{self.key_prefix}_heatmap_columns"
            )
            columns = columns if columns else None
            
            # 相関行列とカラースケールの設定
            col1, col2 = st.columns(2)
            
            with col1:
                correlation = st.checkbox(
                    "相関行列を表示",
                    value=True,
                    key=f"{self.key_prefix}_heatmap_correlation"
                )
                
            with col2:
                text_auto = st.checkbox(
                    "セルに値を表示",
                    value=True,
                    key=f"{self.key_prefix}_heatmap_text"
                )
            
            # カラースケールの範囲設定
            col1, col2 = st.columns(2)
            
            with col1:
                z_min = st.number_input(
                    "カラースケールの最小値（空の場合は自動）",
                    value=None,
                    key=f"{self.key_prefix}_heatmap_z_min"
                )
                
            with col2:
                z_max = st.number_input(
                    "カラースケールの最大値（空の場合は自動）",
                    value=None,
                    key=f"{self.key_prefix}_heatmap_z_max"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                # ヒートマップの作成
                fig = self.heatmap(
                    columns=columns,
                    correlation=correlation,
                    z_min=z_min,
                    z_max=z_max,
                    text_auto=text_auto
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_pie_chart_ui(self) -> Optional[go.Figure]:
        """
        円グラフ作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された円グラフ
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("円グラフ")
        
        with st.form(key=f"{self.key_prefix}_pie_chart_form"):
            # カテゴリと値の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                names_column = st.selectbox(
                    "カテゴリの列",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_pie_names"
                )
                
            with col2:
                values_column = st.selectbox(
                    "値の列",
                    options=[""] + self.numeric_columns,
                    key=f"{self.key_prefix}_pie_values"
                )
            
            # 追加オプション
            hole = st.slider(
                "中央の穴のサイズ",
                min_value=0.0,
                max_value=0.9,
                value=0.0,
                step=0.1,
                key=f"{self.key_prefix}_pie_hole"
            )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not names_column:
                    st.warning("カテゴリの列を選択してください")
                    return None
                    
                if not values_column:
                    st.warning("値の列を選択してください")
                    return None
                    
                # 円グラフの作成
                fig = self.pie_chart(
                    names_column=names_column,
                    values_column=values_column,
                    hole=hole
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_time_series_ui(self) -> Optional[go.Figure]:
        """
        時系列グラフ作成UIを描画する
        
        Returns:
        --------
        Optional[go.Figure]
            作成された時系列グラフ
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return None
            
        st.subheader("時系列グラフ")
        
        with st.form(key=f"{self.key_prefix}_time_series_form"):
            # 日付と値の列選択
            col1, col2 = st.columns(2)
            
            with col1:
                date_column = st.selectbox(
                    "日付の列",
                    options=[""] + self.datetime_columns + [col for col in self.data.columns if 'date' in col.lower() or 'time' in col.lower()],
                    key=f"{self.key_prefix}_time_date"
                )
                
            with col2:
                value_columns = st.multiselect(
                    "値の列（複数選択可）",
                    options=self.numeric_columns,
                    key=f"{self.key_prefix}_time_values"
                )
            
            # グループ化とリサンプリングの設定
            col1, col2 = st.columns(2)
            
            with col1:
                group_by = st.selectbox(
                    "グループ化する列（オプション）",
                    options=[""] + self.categorical_columns,
                    key=f"{self.key_prefix}_time_group"
                )
                group_by = group_by if group_by else None
                
            with col2:
                resample = st.selectbox(
                    "リサンプリング間隔",
                    options=["なし", "日 (D)", "週 (W)", "月 (M)", "四半期 (Q)", "年 (Y)"],
                    key=f"{self.key_prefix}_time_resample"
                )
                resample_map = {"なし": None, "日 (D)": "D", "週 (W)": "W", "月 (M)": "M", "四半期 (Q)": "Q", "年 (Y)": "Y"}
                resample = resample_map[resample]
            
            # 追加オプション
            col1, col2 = st.columns(2)
            
            with col1:
                agg_func = st.selectbox(
                    "集計関数",
                    options=['mean', 'sum', 'count', 'min', 'max', 'median'],
                    key=f"{self.key_prefix}_time_agg"
                )
                
            with col2:
                markers = st.checkbox(
                    "マーカーを表示",
                    value=True,
                    key=f"{self.key_prefix}_time_markers"
                )
            
            # グラフ作成ボタン
            submitted = st.form_submit_button("グラフを作成")
            
            if submitted:
                if not date_column:
                    st.warning("日付の列を選択してください")
                    return None
                    
                if not value_columns:
                    st.warning("値の列を選択してください")
                    return None
                    
                # 時系列グラフの作成
                fig = self.time_series(
                    date_column=date_column,
                    value_columns=value_columns,
                    group_by=group_by,
                    resample=resample,
                    agg_func=agg_func,
                    markers=markers
                )
                
                if fig is not None:
                    return fig
                else:
                    st.error("グラフの作成中にエラーが発生しました")
                    return None
        
        return None
    
    def render_ui(self) -> None:
        """
        完全なUIを描画する
        """
        if self.data is None:
            st.warning("データが読み込まれていません")
            return
            
        # グラフ設定UI
        self.render_figure_settings_ui()
        
        # グラフタイプの選択
        graph_type = st.selectbox(
            "グラフタイプを選択",
            options=[
                "折れ線グラフ", "散布図", "棒グラフ", "パレート図", "ヒストグラム",
                "箱ひげ図", "ヒートマップ", "円グラフ", "時系列グラフ"
            ],
            key=f"{self.key_prefix}_graph_type"
        )
        
        # 選択されたグラフタイプのUIを表示
        fig = None
        
        if graph_type == "折れ線グラフ":
            fig = self.render_line_chart_ui()
        elif graph_type == "散布図":
            fig = self.render_scatter_plot_ui()
        elif graph_type == "棒グラフ":
            fig = self.render_bar_chart_ui()
        elif graph_type == "パレート図":
            fig = self.render_pareto_chart_ui()
        elif graph_type == "ヒストグラム":
            fig = self.render_histogram_ui()
        elif graph_type == "箱ひげ図":
            fig = self.render_box_plot_ui()
        elif graph_type == "ヒートマップ":
            fig = self.render_heatmap_ui()
        elif graph_type == "円グラフ":
            fig = self.render_pie_chart_ui()
        elif graph_type == "時系列グラフ":
            fig = self.render_time_series_ui()
        
        # グラフの表示
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # ダウンロードオプション
            self._add_download_options(fig)
    
    def _add_download_options(self, fig: go.Figure) -> None:
        """
        グラフのダウンロードオプションを追加する
        
        Parameters:
        -----------
        fig : go.Figure
            ダウンロード対象のグラフ
        """
        st.subheader("グラフのダウンロード")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PNG形式でダウンロード
            img_bytes = fig.to_image(format="png", width=self.figure_settings['width'], height=self.figure_settings['height'])
            st.download_button(
                label="PNG形式でダウンロード",
                data=img_bytes,
                file_name="graph.png",
                mime="image/png",
                key=f"{self.key_prefix}_download_png"
            )
            
        with col2:
            # SVG形式でダウンロード
            img_bytes = fig.to_image(format="svg", width=self.figure_settings['width'], height=self.figure_settings['height'])
            st.download_button(
                label="SVG形式でダウンロード",
                data=img_bytes,
                file_name="graph.svg",
                mime="image/svg+xml",
                key=f"{self.key_prefix}_download_svg"
            )
            
        with col3:
            # HTML形式でダウンロード
            html_str = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                label="HTML形式でダウンロード",
                data=html_str,
                file_name="graph.html",
                mime="text/html",
                key=f"{self.key_prefix}_download_html"
            )
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.data = None
        self.last_figure = None
        gc.collect()
