import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gc
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    データ分析機能を提供するクラス
    インタラクティブなデータ表示、編集、基本的な統計分析、可視化機能を実装
    """
    
    def __init__(self):
        """
        DataAnalyzerクラスの初期化
        """
        self.data = None
        self.original_data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        self.column_dtypes = {}
        self.summary_stats = {}
        
    def load_data(self, data: Union[pd.DataFrame, str], **kwargs) -> pd.DataFrame:
        """
        データの読み込みとメタデータの初期化
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, str]
            読み込むデータフレームまたはファイルパス
        **kwargs : dict
            pd.read_csvなどに渡す追加パラメータ
            
        Returns:
        --------
        pd.DataFrame
            読み込まれたデータフレーム
        """
        try:
            if isinstance(data, pd.DataFrame):
                self.data = data.copy()
            elif isinstance(data, str):
                if data.endswith('.csv'):
                    self.data = pd.read_csv(data, **kwargs)
                elif data.endswith('.xlsx') or data.endswith('.xls'):
                    self.data = pd.read_excel(data, **kwargs)
                elif data.endswith('.json'):
                    self.data = pd.read_json(data, **kwargs)
                elif data.endswith('.parquet'):
                    self.data = pd.read_parquet(data, **kwargs)
                else:
                    raise ValueError(f"サポートされていないファイル形式です: {data}")
            else:
                raise TypeError("データはDataFrameまたはファイルパスである必要があります")
                
            # オリジナルデータのバックアップを作成
            self.original_data = self.data.copy()
            
            # データの初期分析
            self._analyze_data_types()
            self._calculate_summary_stats()
            
            logger.info(f"データを読み込みました。行数: {len(self.data)}, 列数: {len(self.data.columns)}")
            return self.data
            
        except Exception as e:
            logger.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def _analyze_data_types(self) -> None:
        """
        データフレームの列の型を分析し、カテゴリを特定する
        """
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        self.column_dtypes = {}
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            self.column_dtypes[col] = str(dtype)
            
            # 数値型の列
            if np.issubdtype(dtype, np.number):
                self.numeric_columns.append(col)
            
            # 日時型の列
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_columns.append(col)
            
            # カテゴリカル型または少数のユニーク値を持つ列
            elif pd.api.types.is_categorical_dtype(self.data[col]) or \
                 (self.data[col].nunique() < 0.1 * len(self.data) and self.data[col].nunique() < 100):
                self.categorical_columns.append(col)
            
            # テキスト型の列
            else:
                self.text_columns.append(col)
    
    def _calculate_summary_stats(self) -> Dict:
        """
        データフレームの基本的な要約統計量を計算する
        
        Returns:
        --------
        Dict
            列ごとの要約統計情報
        """
        summary = {}
        
        # 基本情報
        summary['basic_info'] = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'memory_usage': self.data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB単位
        }
        
        # 欠損値情報
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        summary['missing_values'] = {
            'count': missing_values.to_dict(),
            'percentage': missing_percentage.to_dict()
        }
        
        # 数値列の統計量
        if self.numeric_columns:
            summary['numeric_stats'] = self.data[self.numeric_columns].describe().to_dict()
        
        # カテゴリ列の統計量
        if self.categorical_columns:
            cat_stats = {}
            for col in self.categorical_columns:
                value_counts = self.data[col].value_counts()
                cat_stats[col] = {
                    'unique_values': self.data[col].nunique(),
                    'top_values': value_counts.head(5).to_dict(),
                    'top_percentage': (value_counts.head(5) / len(self.data) * 100).to_dict()
                }
            summary['categorical_stats'] = cat_stats
        
        # 日時列の統計量
        if self.datetime_columns:
            date_stats = {}
            for col in self.datetime_columns:
                date_stats[col] = {
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'range_days': (self.data[col].max() - self.data[col].min()).days \
                        if hasattr((self.data[col].max() - self.data[col].min()), 'days') else None
                }
            summary['datetime_stats'] = date_stats
        
        self.summary_stats = summary
        return summary
    
    def get_summary(self) -> Dict:
        """
        データの要約統計情報を取得する
        
        Returns:
        --------
        Dict
            データの要約統計情報
        """
        if not self.summary_stats:
            self._calculate_summary_stats()
        return self.summary_stats
    
    def filter_data(self, filters: Dict[str, Dict]) -> pd.DataFrame:
        """
        条件に基づいてデータをフィルタリングする
        
        Parameters:
        -----------
        filters : Dict[str, Dict]
            列名をキーとし、フィルタ条件を値とする辞書
            例: {'age': {'min': 20, 'max': 30}, 'category': {'values': ['A', 'B']}}
            
        Returns:
        --------
        pd.DataFrame
            フィルタリングされたデータフレーム
        """
        filtered_data = self.data.copy()
        
        for column, conditions in filters.items():
            if column not in filtered_data.columns:
                logger.warning(f"列 '{column}' はデータフレームに存在しません。スキップします。")
                continue
                
            if column in self.numeric_columns:
                if 'min' in conditions:
                    filtered_data = filtered_data[filtered_data[column] >= conditions['min']]
                if 'max' in conditions:
                    filtered_data = filtered_data[filtered_data[column] <= conditions['max']]
                    
            elif 'values' in conditions:
                filtered_data = filtered_data[filtered_data[column].isin(conditions['values'])]
                
            elif 'pattern' in conditions:
                filtered_data = filtered_data[filtered_data[column].str.contains(conditions['pattern'], na=False)]
        
        logger.info(f"フィルタリング後のデータ: 行数 {len(filtered_data)}, 元の {len(self.data)} 行から")
        return filtered_data
    
    def create_pivot_table(self, 
                          index: Union[str, List[str]], 
                          columns: Optional[Union[str, List[str]]] = None,
                          values: Optional[Union[str, List[str]]] = None,
                          aggfunc: str = 'mean') -> pd.DataFrame:
        """
        ピボットテーブルを作成する
        
        Parameters:
        -----------
        index : Union[str, List[str]]
            ピボットテーブルの行インデックスとして使用する列
        columns : Optional[Union[str, List[str]]]
            ピボットテーブルの列として使用する列
        values : Optional[Union[str, List[str]]]
            集計する値の列
        aggfunc : str
            集計関数 ('mean', 'sum', 'count', 'min', 'max', 'median', 'std')
            
        Returns:
        --------
        pd.DataFrame
            ピボットテーブル
        """
        # 集計関数の辞書
        agg_functions = {
            'mean': np.mean,
            'sum': np.sum,
            'count': len,
            'min': np.min,
            'max': np.max,
            'median': np.median,
            'std': np.std
        }
        
        # 集計関数の検証
        if aggfunc not in agg_functions:
            raise ValueError(f"サポートされていない集計関数です: {aggfunc}。サポートされている関数: {list(agg_functions.keys())}")
        
        # valuesが指定されていない場合、数値列を使用
        if values is None:
            values = self.numeric_columns
            if not values:
                raise ValueError("集計する数値列が見つかりません")
        
        try:
            pivot_table = pd.pivot_table(
                self.data,
                index=index,
                columns=columns,
                values=values,
                aggfunc=agg_functions[aggfunc]
            )
            
            logger.info(f"ピボットテーブルを作成しました: 行数 {len(pivot_table)}, 列数 {len(pivot_table.columns)}")
            return pivot_table
            
        except Exception as e:
            logger.error(f"ピボットテーブル作成中にエラーが発生しました: {str(e)}")
            raise
    
    def plot_histogram(self, column: str, bins: int = 30, kde: bool = True) -> go.Figure:
        """
        指定された列のヒストグラムを作成する
        
        Parameters:
        -----------
        column : str
            ヒストグラムを作成する列名
        bins : int
            ビンの数
        kde : bool
            カーネル密度推定を表示するかどうか
            
        Returns:
        --------
        go.Figure
            Plotlyのヒストグラム図
        """
        if column not in self.data.columns:
            raise ValueError(f"列 '{column}' はデータフレームに存在しません")
            
        if column not in self.numeric_columns:
            raise ValueError(f"列 '{column}' は数値型ではありません")
        
        fig = go.Figure()
        
        # ヒストグラムの追加
        fig.add_trace(go.Histogram(
            x=self.data[column],
            nbinsx=bins,
            name=column,
            opacity=0.7
        ))
        
        # KDEの追加（オプション）
        if kde:
            # KDE用のx軸の値を生成
            x_range = np.linspace(
                self.data[column].min(),
                self.data[column].max(),
                1000
            )
            
            # KDEの計算
            kde_values = sns.kdeplot(self.data[column].dropna(), bw_adjust=0.5).get_lines()[0].get_ydata()
            kde_x = sns.kdeplot(self.data[column].dropna(), bw_adjust=0.5).get_lines()[0].get_xdata()
            
            # スケーリング（ヒストグラムに合わせる）
            hist_max = np.histogram(self.data[column].dropna(), bins=bins)[0].max()
            kde_max = kde_values.max()
            scale_factor = hist_max / kde_max if kde_max > 0 else 1
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_values * scale_factor,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
        
        # レイアウトの設定
        fig.update_layout(
            title=f"{column}のヒストグラム",
            xaxis_title=column,
            yaxis_title="頻度",
            bargap=0.1,
            template="plotly_white"
        )
        
        return fig
    
    def plot_scatter(self, x: str, y: str, color: Optional[str] = None, 
                    size: Optional[str] = None, hover_data: Optional[List[str]] = None) -> go.Figure:
        """
        散布図を作成する
        
        Parameters:
        -----------
        x : str
            x軸の列名
        y : str
            y軸の列名
        color : Optional[str]
            点の色に使用する列名
        size : Optional[str]
            点のサイズに使用する列名
        hover_data : Optional[List[str]]
            ホバー時に表示する追加データの列名リスト
            
        Returns:
        --------
        go.Figure
            Plotlyの散布図
        """
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError(f"列 '{x}' または '{y}' はデータフレームに存在しません")
            
        if x not in self.numeric_columns or y not in self.numeric_columns:
            raise ValueError(f"列 '{x}' または '{y}' は数値型ではありません")
        
        # hover_dataの検証
        if hover_data:
            for col in hover_data:
                if col not in self.data.columns:
                    logger.warning(f"hover_dataの列 '{col}' はデータフレームに存在しません。スキップします。")
                    hover_data.remove(col)
        
        # 散布図の作成
        fig = px.scatter(
            self.data,
            x=x,
            y=y,
            color=color,
            size=size,
            hover_data=hover_data,
            title=f"{x} vs {y}の散布図",
            template="plotly_white"
        )
        
        # レイアウトの調整
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            legend_title=color if color else "",
            height=600,
            width=800
        )
        
        # マーカーの調整
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )
        
        return fig
    
    def plot_bar(self, x: str, y: Optional[str] = None, 
                color: Optional[str] = None, orientation: str = 'v',
                is_pareto: bool = False) -> go.Figure:
        """
        棒グラフまたはパレート図を作成する
        
        Parameters:
        -----------
        x : str
            x軸の列名（縦棒グラフの場合）またはカテゴリ列（横棒グラフの場合）
        y : Optional[str]
            y軸の列名（縦棒グラフの場合）または値列（横棒グラフの場合）
        color : Optional[str]
            棒の色に使用する列名
        orientation : str
            グラフの向き ('v'=縦棒, 'h'=横棒)
        is_pareto : bool
            パレート図として表示するかどうか
            
        Returns:
        --------
        go.Figure
            Plotlyの棒グラフまたはパレート図
        """
        if x not in self.data.columns:
            raise ValueError(f"列 '{x}' はデータフレームに存在しません")
            
        # yが指定されていない場合、カウント集計を行う
        if y is None:
            count_data = self.data[x].value_counts().reset_index()
            count_data.columns = [x, 'count']
            plot_data = count_data
            y = 'count'
        else:
            if y not in self.data.columns:
                raise ValueError(f"列 '{y}' はデータフレームに存在しません")
            plot_data = self.data
        
        # 通常の棒グラフ
        if not is_pareto:
            fig = px.bar(
                plot_data,
                x=x if orientation == 'v' else y,
                y=y if orientation == 'v' else x,
                color=color,
                orientation=orientation,
                title=f"{x}の棒グラフ",
                template="plotly_white"
            )
            
        # パレート図
        else:
            # データの準備
            if y is None:
                y = 'count'
                
            if isinstance(plot_data, pd.DataFrame) and y in plot_data.columns:
                # 集計が必要な場合
                if x in self.categorical_columns:
                    pareto_data = plot_data.groupby(x)[y].sum().reset_index()
                else:
                    pareto_data = plot_data
            else:
                pareto_data = plot_data
                
            # 降順にソート
            pareto_data = pareto_data.sort_values(y, ascending=False)
            
            # 累積パーセンテージの計算
            pareto_data['cumulative_percentage'] = (pareto_data[y].cumsum() / pareto_data[y].sum() * 100)
            
            # パレート図の作成
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 棒グラフの追加
            fig.add_trace(
                go.Bar(
                    x=pareto_data[x],
                    y=pareto_data[y],
                    name=y
                ),
                secondary_y=False
            )
            
            # 累積線の追加
            fig.add_trace(
                go.Scatter(
                    x=pareto_data[x],
                    y=pareto_data['cumulative_percentage'],
                    name="累積 %",
                    line=dict(color='red', width=2)
                ),
                secondary_y=True
            )
            
            # レイアウトの設定
            fig.update_layout(
                title=f"{x}のパレート図",
                template="plotly_white"
            )
            
            # 軸ラベルの設定
            fig.update_yaxes(title_text=y, secondary_y=False)
            fig.update_yaxes(title_text="累積パーセンテージ", secondary_y=True)
            
        return fig
    
    def plot_line(self, x: str, y: Union[str, List[str]], 
                 color: Optional[str] = None, line_dash: Optional[str] = None,
                 markers: bool = True) -> go.Figure:
        """
        折れ線グラフを作成する
        
        Parameters:
        -----------
        x : str
            x軸の列名（通常は時間や順序を表す列）
        y : Union[str, List[str]]
            y軸の列名または列名のリスト
        color : Optional[str]
            線の色に使用する列名
        line_dash : Optional[str]
            線のスタイルに使用する列名
        markers : bool
            マーカーを表示するかどうか
            
        Returns:
        --------
        go.Figure
            Plotlyの折れ線グラフ
        """
        if x not in self.data.columns:
            raise ValueError(f"列 '{x}' はデータフレームに存在しません")
            
        # yを常にリストとして扱う
        y_columns = [y] if isinstance(y, str) else y
        
        for col in y_columns:
            if col not in self.data.columns:
                raise ValueError(f"列 '{col}' はデータフレームに存在しません")
        
        # 時系列データの場合はソート
        plot_data = self.data.copy()
        if x in self.datetime_columns:
            plot_data = plot_data.sort_values(by=x)
        
        # 折れ線グラフの作成
        fig = px.line(
            plot_data,
            x=x,
            y=y_columns,
            color=color,
            line_dash=line_dash,
            markers=markers,
            title=f"{x}に対する{', '.join(y_columns)}の推移",
            template="plotly_white"
        )
        
        # レイアウトの調整
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y_columns[0] if len(y_columns) == 1 else "値",
            legend_title=color if color else "",
            height=500,
            width=800
        )
        
        return fig
    
    def plot_correlation_matrix(self, columns: Optional[List[str]] = None, 
                               method: str = 'pearson') -> go.Figure:
        """
        相関行列のヒートマップを作成する
        
        Parameters:
        -----------
        columns : Optional[List[str]]
            相関を計算する列のリスト。Noneの場合はすべての数値列を使用
        method : str
            相関係数の計算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
        --------
        go.Figure
            Plotlyのヒートマップ
        """
        # 列の検証
        if columns is None:
            columns = self.numeric_columns
        else:
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"列 '{col}' はデータフレームに存在しません")
                if col not in self.numeric_columns:
                    raise ValueError(f"列 '{col}' は数値型ではありません")
        
        # 相関行列の計算
        corr_matrix = self.data[columns].corr(method=method)
        
        # ヒートマップの作成
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title=f"相関行列 ({method})",
            template="plotly_white"
        )
        
        # レイアウトの調整
        fig.update_layout(
            height=600,
            width=700
        )
        
        return fig
    
    def plot_pca(self, n_components: int = 2, columns: Optional[List[str]] = None,
                color: Optional[str] = None, hover_data: Optional[List[str]] = None) -> go.Figure:
        """
        主成分分析（PCA）の結果を可視化する
        
        Parameters:
        -----------
        n_components : int
            主成分の数（2または3）
        columns : Optional[List[str]]
            PCAに使用する列のリスト。Noneの場合はすべての数値列を使用
        color : Optional[str]
            点の色に使用する列名
        hover_data : Optional[List[str]]
            ホバー時に表示する追加データの列名リスト
            
        Returns:
        --------
        go.Figure
            PCA結果のPlotly図
        """
        if n_components not in [2, 3]:
            raise ValueError("n_componentsは2または3である必要があります")
            
        # 列の検証
        if columns is None:
            columns = self.numeric_columns
        else:
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"列 '{col}' はデータフレームに存在しません")
                if col not in self.numeric_columns:
                    raise ValueError(f"列 '{col}' は数値型ではありません")
        
        # 欠損値の処理
        pca_data = self.data[columns].dropna()
        
        if len(pca_data) == 0:
            raise ValueError("PCAを実行するための有効なデータがありません")
            
        # データの標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # PCAの実行
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # 結果をデータフレームに変換
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # 元のインデックスを保持
        pca_df.index = pca_data.index
        
        # 色とホバーデータの追加
        plot_data = pca_df.copy()
        
        if color and color in self.data.columns:
            plot_data[color] = self.data.loc[pca_df.index, color]
            
        if hover_data:
            for col in hover_data:
                if col in self.data.columns:
                    plot_data[col] = self.data.loc[pca_df.index, col]
        
        # 寄与率の計算
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance_labels = [
            f'PC{i+1} ({var:.1%})' for i, var in enumerate(explained_variance_ratio)
        ]
        
        # 2次元または3次元の散布図
        if n_components == 2:
            fig = px.scatter(
                plot_data,
                x='PC1',
                y='PC2',
                color=color,
                hover_data=hover_data,
                title=f"PCA結果 (寄与率: PC1={explained_variance_ratio[0]:.1%}, PC2={explained_variance_ratio[1]:.1%})",
                template="plotly_white"
            )
            
            # 軸ラベルの更新
            fig.update_layout(
                xaxis_title=explained_variance_labels[0],
                yaxis_title=explained_variance_labels[1]
            )
            
        else:  # n_components == 3
            fig = px.scatter_3d(
                plot_data,
                x='PC1',
                y='PC2',
                z='PC3',
                color=color,
                hover_data=hover_data,
                title=f"PCA結果 (寄与率: PC1={explained_variance_ratio[0]:.1%}, PC2={explained_variance_ratio[1]:.1%}, PC3={explained_variance_ratio[2]:.1%})",
                template="plotly_white"
            )
            
            # 軸ラベルの更新
            fig.update_layout(
                scene=dict(
                    xaxis_title=explained_variance_labels[0],
                    yaxis_title=explained_variance_labels[1],
                    zaxis_title=explained_variance_labels[2]
                )
            )
        
        return fig
    
    def optimize_dataframe(self, inplace: bool = False) -> Optional[pd.DataFrame]:
        """
        データフレームのメモリ使用量を最適化する
        
        Parameters:
        -----------
        inplace : bool
            Trueの場合、現在のデータフレームを最適化されたものに置き換える
            
        Returns:
        --------
        Optional[pd.DataFrame]
            inplaceがFalseの場合、最適化されたデータフレームを返す
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません")
            
        df = self.data.copy()
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB単位
        logger.info(f"最適化前のメモリ使用量: {start_mem:.2f} MB")
        
        # 数値型の最適化
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # 整数型の最適化
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        
        # 浮動小数点型の最適化
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].astype(np.float32)
        
        # カテゴリ型への変換
        for col in self.categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        # 最適化後のメモリ使用量
        end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB単位
        reduction = 100 * (start_mem - end_mem) / start_mem
        
        logger.info(f"最適化後のメモリ使用量: {end_mem:.2f} MB ({reduction:.1f}% 削減)")
        
        if inplace:
            self.data = df
            # データ型の再分析
            self._analyze_data_types()
            return None
        else:
            return df
    
    def reset_data(self) -> None:
        """
        データを元の状態に戻す
        """
        if self.original_data is not None:
            self.data = self.original_data.copy()
            # データ型の再分析
            self._analyze_data_types()
            self._calculate_summary_stats()
            logger.info("データを元の状態にリセットしました")
        else:
            logger.warning("リセットするオリジナルデータがありません")
    
    def get_column_info(self, column: str) -> Dict:
        """
        特定の列の詳細情報を取得する
        
        Parameters:
        -----------
        column : str
            情報を取得する列名
            
        Returns:
        --------
        Dict
            列の詳細情報
        """
        if column not in self.data.columns:
            raise ValueError(f"列 '{column}' はデータフレームに存在しません")
            
        info = {
            'name': column,
            'dtype': str(self.data[column].dtype),
            'count': len(self.data[column]),
            'null_count': self.data[column].isnull().sum(),
            'null_percentage': (self.data[column].isnull().sum() / len(self.data)) * 100
        }
        
        # 数値列の場合
        if column in self.numeric_columns:
            info.update({
                'min': self.data[column].min(),
                'max': self.data[column].max(),
                'mean': self.data[column].mean(),
                'median': self.data[column].median(),
                'std': self.data[column].std(),
                'skew': self.data[column].skew(),
                'kurtosis': self.data[column].kurtosis()
            })
            
        # カテゴリ列の場合
        elif column in self.categorical_columns:
            value_counts = self.data[column].value_counts()
            info.update({
                'unique_values': self.data[column].nunique(),
                'top_values': value_counts.head(10).to_dict(),
                'top_percentage': (value_counts.head(10) / len(self.data) * 100).to_dict()
            })
            
        # 日時列の場合
        elif column in self.datetime_columns:
            info.update({
                'min': self.data[column].min(),
                'max': self.data[column].max(),
                'range_days': (self.data[column].max() - self.data[column].min()).days \
                    if hasattr((self.data[column].max() - self.data[column].min()), 'days') else None
            })
            
        # テキスト列の場合
        elif column in self.text_columns:
            if self.data[column].dtype == 'object':
                # 文字列長の統計
                str_lengths = self.data[column].str.len()
                info.update({
                    'unique_values': self.data[column].nunique(),
                    'min_length': str_lengths.min(),
                    'max_length': str_lengths.max(),
                    'mean_length': str_lengths.mean()
                })
        
        return info
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.data = None
        self.original_data = None
        gc.collect()
