import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
import logging
import warnings

# AutoGluon関連のインポート
try:
    from autogluon.tabular import TabularPredictor
    from autogluon.common.features.types import R_FLOAT, R_INT, S_BOOL
    from autogluon.tabular.models.knn.knn_utils import FAISSNeighborsClassifier
    from autogluon.core.utils.loaders import load_pkl
    from autogluon.core.utils.savers import save_pkl
except ImportError:
    warnings.warn("AutoGluonがインストールされていません。pip install autogluon")

# 因果推論関連のインポート
try:
    import econml
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import TLearner, SLearner, XLearner
    from econml.inference import BootstrapInference
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LassoCV
    import networkx as nx
except ImportError:
    warnings.warn("econmlがインストールされていません。pip install econml")

logger = logging.getLogger(__name__)

class CausalInference:
    """
    因果推論機能を提供するクラス
    AutoGluonとeconmlを活用した因果効果の推定と可視化を実装
    """
    
    def __init__(self):
        """
        CausalInferenceクラスの初期化
        """
        self.data = None
        self.treatment_col = None
        self.outcome_col = None
        self.covariate_cols = []
        self.categorical_cols = []
        self.is_trained = False
        self.causal_model = None
        self.model_type = None
        self.treatment_effect = None
        self.feature_importance = None
        self.balance_stats = None
        self.causal_graph = None
        self.model_params = {}
        self.evaluation_results = {}
        
    def load_data(self, data: pd.DataFrame, copy: bool = True) -> None:
        """
        データの読み込み
        
        Parameters:
        -----------
        data : pd.DataFrame
            分析対象のデータフレーム
        copy : bool
            データのコピーを作成するかどうか
        """
        if copy:
            self.data = data.copy()
        else:
            self.data = data
            
        logger.info(f"データを読み込みました。行数: {len(self.data)}, 列数: {len(self.data.columns)}")
    
    def set_causal_model(self, 
                        treatment_col: str, 
                        outcome_col: str, 
                        covariate_cols: List[str],
                        categorical_cols: Optional[List[str]] = None,
                        model_type: str = 'causal_forest',
                        binary_treatment: bool = True,
                        **model_params) -> None:
        """
        因果モデルの設定
        
        Parameters:
        -----------
        treatment_col : str
            処理変数の列名
        outcome_col : str
            結果変数の列名
        covariate_cols : List[str]
            共変量の列名リスト
        categorical_cols : Optional[List[str]]
            カテゴリ変数の列名リスト
        model_type : str
            使用する因果モデルのタイプ
            ('causal_forest', 'dr_learner', 't_learner', 's_learner', 'x_learner')
        binary_treatment : bool
            処理変数が二値（0/1）かどうか
        **model_params : dict
            モデルに渡す追加パラメータ
        """
        # 入力検証
        if treatment_col not in self.data.columns:
            raise ValueError(f"処理変数 '{treatment_col}' はデータフレームに存在しません")
            
        if outcome_col not in self.data.columns:
            raise ValueError(f"結果変数 '{outcome_col}' はデータフレームに存在しません")
            
        for col in covariate_cols:
            if col not in self.data.columns:
                raise ValueError(f"共変量 '{col}' はデータフレームに存在しません")
        
        # 処理変数が二値の場合、0/1に変換
        if binary_treatment:
            if not pd.api.types.is_numeric_dtype(self.data[treatment_col]):
                # カテゴリ変数の場合、二値に変換
                unique_values = self.data[treatment_col].unique()
                if len(unique_values) != 2:
                    raise ValueError(f"二値処理を指定しましたが、処理変数 '{treatment_col}' は2つの値を持っていません")
                
                # 0/1に変換
                treatment_map = {unique_values[0]: 0, unique_values[1]: 1}
                self.data[treatment_col] = self.data[treatment_col].map(treatment_map)
                logger.info(f"処理変数を二値に変換しました: {treatment_map}")
            
            # 数値変数の場合、閾値で二値に変換
            elif len(self.data[treatment_col].unique()) > 2:
                # 中央値を閾値として使用
                threshold = self.data[treatment_col].median()
                self.data[f"{treatment_col}_original"] = self.data[treatment_col].copy()
                self.data[treatment_col] = (self.data[treatment_col] > threshold).astype(int)
                logger.info(f"処理変数を閾値 {threshold} で二値に変換しました")
        
        # カテゴリ変数の処理
        if categorical_cols is None:
            categorical_cols = []
            for col in covariate_cols:
                if not pd.api.types.is_numeric_dtype(self.data[col]) or \
                   pd.api.types.is_categorical_dtype(self.data[col]) or \
                   self.data[col].nunique() < 10:
                    categorical_cols.append(col)
        
        # モデルタイプの検証
        valid_models = ['causal_forest', 'dr_learner', 't_learner', 's_learner', 'x_learner']
        if model_type not in valid_models:
            raise ValueError(f"無効なモデルタイプです: {model_type}。有効なタイプ: {valid_models}")
        
        # 変数の設定
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols
        self.categorical_cols = categorical_cols
        self.model_type = model_type
        self.model_params = model_params
        self.is_trained = False
        
        logger.info(f"因果モデルを設定しました: 処理変数={treatment_col}, 結果変数={outcome_col}, モデル={model_type}")
        logger.info(f"共変量の数: {len(covariate_cols)}, カテゴリ変数の数: {len(categorical_cols)}")
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        モデル学習用のデータを準備する
        
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]
            (X, T, Y) - 共変量、処理変数、結果変数
        """
        if self.data is None or self.treatment_col is None or self.outcome_col is None:
            raise ValueError("データと因果モデルの設定が必要です")
        
        # 欠損値を含む行を除外
        cols_to_check = [self.treatment_col, self.outcome_col] + self.covariate_cols
        data_clean = self.data[cols_to_check].dropna()
        
        if len(data_clean) < len(self.data):
            logger.warning(f"欠損値のため {len(self.data) - len(data_clean)} 行を除外しました")
        
        # カテゴリ変数のダミー変数化
        if self.categorical_cols:
            X = pd.get_dummies(data_clean[self.covariate_cols], columns=self.categorical_cols, drop_first=True)
        else:
            X = data_clean[self.covariate_cols].copy()
        
        # 処理変数と結果変数
        T = data_clean[self.treatment_col].values
        Y = data_clean[self.outcome_col].values
        
        return X, T, Y
    
    def _standardize_covariates(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        共変量を標準化する
        
        Parameters:
        -----------
        X : pd.DataFrame
            標準化する共変量データフレーム
            
        Returns:
        --------
        pd.DataFrame
            標準化された共変量データフレーム
        """
        # カテゴリ変数（ダミー変数）を除外して標準化
        numeric_cols = X.select_dtypes(include=['int', 'float']).columns
        
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            return X_scaled
        else:
            return X
    
    def train(self, random_state: int = 42, test_size: float = 0.3, standardize: bool = True) -> Any:
        """
        因果モデルを学習する
        
        Parameters:
        -----------
        random_state : int
            乱数シード
        test_size : float
            テストデータの割合
        standardize : bool
            共変量を標準化するかどうか
            
        Returns:
        --------
        Any
            学習済みの因果モデル
        """
        # データの準備
        X, T, Y = self._prepare_data()
        
        # 共変量の標準化（オプション）
        if standardize:
            X = self._standardize_covariates(X)
        
        # トレーニングデータとテストデータの分割
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, test_size=test_size, random_state=random_state
        )
        
        # モデルの初期化と学習
        if self.model_type == 'causal_forest':
            # デフォルトパラメータの設定
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 10,
                'max_samples': 0.5,
                'discrete_treatment': True,
                'verbose': 0
            }
            # ユーザー指定のパラメータで上書き
            params = {**default_params, **self.model_params}
            
            # モデルの初期化
            model = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                             min_samples_leaf=params['min_samples_leaf'], random_state=random_state),
                model_t=RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                              min_samples_leaf=params['min_samples_leaf'], random_state=random_state),
                discrete_treatment=params['discrete_treatment'],
                n_estimators=params['n_estimators'],
                verbose=params['verbose']
            )
            
        elif self.model_type == 'dr_learner':
            # デフォルトパラメータの設定
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 10
            }
            # ユーザー指定のパラメータで上書き
            params = {**default_params, **self.model_params}
            
            # モデルの初期化
            model = DRLearner(
                model_propensity=RandomForestClassifier(n_estimators=params['n_estimators'], 
                                                       max_depth=params['max_depth'],
                                                       min_samples_leaf=params['min_samples_leaf'], 
                                                       random_state=random_state),
                model_regression=RandomForestRegressor(n_estimators=params['n_estimators'], 
                                                      max_depth=params['max_depth'],
                                                      min_samples_leaf=params['min_samples_leaf'], 
                                                      random_state=random_state),
                model_final=RandomForestRegressor(n_estimators=params['n_estimators'], 
                                                 max_depth=params['max_depth'],
                                                 min_samples_leaf=params['min_samples_leaf'], 
                                                 random_state=random_state)
            )
            
        elif self.model_type == 't_learner':
            # デフォルトパラメータの設定
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 10
            }
            # ユーザー指定のパラメータで上書き
            params = {**default_params, **self.model_params}
            
            # モデルの初期化
            model = TLearner(
                models=RandomForestRegressor(n_estimators=params['n_estimators'], 
                                            max_depth=params['max_depth'],
                                            min_samples_leaf=params['min_samples_leaf'], 
                                            random_state=random_state)
            )
            
        elif self.model_type == 's_learner':
            # デフォルトパラメータの設定
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 10
            }
            # ユーザー指定のパラメータで上書き
            params = {**default_params, **self.model_params}
            
            # モデルの初期化
            model = SLearner(
                overall_model=RandomForestRegressor(n_estimators=params['n_estimators'], 
                                                   max_depth=params['max_depth'],
                                                   min_samples_leaf=params['min_samples_leaf'], 
                                                   random_state=random_state)
            )
            
        elif self.model_type == 'x_learner':
            # デフォルトパラメータの設定
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_leaf': 10
            }
            # ユーザー指定のパラメータで上書き
            params = {**default_params, **self.model_params}
            
            # モデルの初期化
            model = XLearner(
                models=RandomForestRegressor(n_estimators=params['n_estimators'], 
                                            max_depth=params['max_depth'],
                                            min_samples_leaf=params['min_samples_leaf'], 
                                            random_state=random_state),
                propensity_model=RandomForestClassifier(n_estimators=params['n_estimators'], 
                                                       max_depth=params['max_depth'],
                                                       min_samples_leaf=params['min_samples_leaf'], 
                                                       random_state=random_state)
            )
        
        # モデルの学習
        try:
            model.fit(Y_train, T_train, X=X_train)
            logger.info(f"{self.model_type}モデルの学習が完了しました")
            
            # モデルの評価
            self._evaluate_model(model, X_test, T_test, Y_test)
            
            # 特徴量重要度の計算
            self._calculate_feature_importance(model, X)
            
            # 共変量バランスの評価
            self._assess_covariate_balance(X, T)
            
            # 因果グラフの構築
            self._build_causal_graph()
            
            # モデルの保存
            self.causal_model = model
            self.is_trained = True
            
            return model
            
        except Exception as e:
            logger.error(f"モデル学習中にエラーが発生しました: {str(e)}")
            raise
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, T_test: np.ndarray, Y_test: np.ndarray) -> Dict:
        """
        モデルの評価を行う
        
        Parameters:
        -----------
        model : Any
            評価する因果モデル
        X_test : pd.DataFrame
            テスト用共変量
        T_test : np.ndarray
            テスト用処理変数
        Y_test : np.ndarray
            テスト用結果変数
            
        Returns:
        --------
        Dict
            評価結果
        """
        # 処理効果の予測
        treatment_effects = model.effect(X_test)
        
        # 結果の予測
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test, T_test)
            mse = np.mean((Y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(Y_test - y_pred))
        else:
            # 予測メソッドがない場合は代替アプローチ
            # 処理群と対照群に分割
            X_treated = X_test[T_test == 1]
            X_control = X_test[T_test == 0]
            Y_treated = Y_test[T_test == 1]
            Y_control = Y_test[T_test == 0]
            
            # 各群の予測
            if len(X_treated) > 0 and len(X_control) > 0:
                if self.model_type in ['causal_forest', 'dr_learner']:
                    y_pred_treated = model.model_regression_1.predict(X_treated)
                    y_pred_control = model.model_regression_0.predict(X_control)
                elif self.model_type == 't_learner':
                    y_pred_treated = model.models_[1].predict(X_treated)
                    y_pred_control = model.models_[0].predict(X_control)
                else:
                    # 他のモデルタイプの場合、評価指標は計算しない
                    y_pred_treated = np.zeros(len(X_treated))
                    y_pred_control = np.zeros(len(X_control))
                
                # 評価指標の計算
                mse_treated = np.mean((Y_treated - y_pred_treated) ** 2) if len(Y_treated) > 0 else np.nan
                mse_control = np.mean((Y_control - y_pred_control) ** 2) if len(Y_control) > 0 else np.nan
                mse = (mse_treated * len(Y_treated) + mse_control * len(Y_control)) / len(Y_test)
                rmse = np.sqrt(mse)
                mae_treated = np.mean(np.abs(Y_treated - y_pred_treated)) if len(Y_treated) > 0 else np.nan
                mae_control = np.mean(np.abs(Y_control - y_pred_control)) if len(Y_control) > 0 else np.nan
                mae = (mae_treated * len(Y_treated) + mae_control * len(Y_control)) / len(Y_test)
            else:
                mse = np.nan
                rmse = np.nan
                mae = np.nan
        
        # 処理効果の統計量
        ate = np.mean(treatment_effects)
        ate_std = np.std(treatment_effects)
        
        # 評価結果の保存
        evaluation = {
            'ATE': ate,
            'ATE_std': ate_std,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'min_effect': np.min(treatment_effects),
            'max_effect': np.max(treatment_effects),
            'median_effect': np.median(treatment_effects),
            'positive_effect_ratio': np.mean(treatment_effects > 0)
        }
        
        self.evaluation_results = evaluation
        self.treatment_effect = treatment_effects
        
        logger.info(f"モデル評価: ATE={ate:.4f}, RMSE={rmse:.4f}")
        return evaluation
    
    def _calculate_feature_importance(self, model: Any, X: pd.DataFrame) -> Dict:
        """
        特徴量重要度を計算する
        
        Parameters:
        -----------
        model : Any
            学習済みの因果モデル
        X : pd.DataFrame
            共変量データフレーム
            
        Returns:
        --------
        Dict
            特徴量重要度
        """
        feature_names = X.columns.tolist()
        importance = {}
        
        try:
            # モデルタイプに応じた特徴量重要度の計算
            if self.model_type == 'causal_forest' and hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    importance[feature] = importance_values[i]
                    
            elif self.model_type in ['t_learner', 's_learner', 'x_learner']:
                # T-Learnerの場合、各モデルの特徴量重要度を平均
                if hasattr(model, 'models_'):
                    importance_values = np.zeros(len(feature_names))
                    for m in model.models_:
                        if hasattr(m, 'feature_importances_'):
                            importance_values += m.feature_importances_
                    importance_values /= len(model.models_)
                    
                    for i, feature in enumerate(feature_names):
                        importance[feature] = importance_values[i]
                        
            elif self.model_type == 'dr_learner':
                # DR-Learnerの場合、回帰モデルの特徴量重要度を使用
                if hasattr(model, 'model_regression_0') and hasattr(model.model_regression_0, 'feature_importances_'):
                    importance_values_0 = model.model_regression_0.feature_importances_
                    importance_values_1 = model.model_regression_1.feature_importances_
                    importance_values = (importance_values_0 + importance_values_1) / 2
                    
                    for i, feature in enumerate(feature_names):
                        importance[feature] = importance_values[i]
            
            # 特徴量重要度が取得できない場合、代替アプローチ
            if not importance and hasattr(model, 'effect'):
                # 各特徴量の処理効果への影響を評価
                base_effect = model.effect(X)
                importance_values = []
                
                for i, feature in enumerate(feature_names):
                    # 特徴量の値をシャッフル
                    X_shuffled = X.copy()
                    X_shuffled[feature] = np.random.permutation(X_shuffled[feature].values)
                    
                    # シャッフル後の処理効果
                    shuffled_effect = model.effect(X_shuffled)
                    
                    # 重要度 = 元の効果との差の二乗平均
                    importance_value = np.mean((base_effect - shuffled_effect) ** 2)
                    importance_values.append(importance_value)
                
                # 正規化
                if np.sum(importance_values) > 0:
                    importance_values = importance_values / np.sum(importance_values)
                
                for i, feature in enumerate(feature_names):
                    importance[feature] = importance_values[i]
        
        except Exception as e:
            logger.warning(f"特徴量重要度の計算中にエラーが発生しました: {str(e)}")
            # エラーが発生した場合、均等な重要度を割り当て
            for feature in feature_names:
                importance[feature] = 1.0 / len(feature_names)
        
        # 重要度の降順でソート
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        self.feature_importance = importance
        logger.info(f"特徴量重要度を計算しました。上位3つ: {list(importance.items())[:3]}")
        
        return importance
    
    def _assess_covariate_balance(self, X: pd.DataFrame, T: np.ndarray) -> Dict:
        """
        共変量バランスを評価する
        
        Parameters:
        -----------
        X : pd.DataFrame
            共変量データフレーム
        T : np.ndarray
            処理変数
            
        Returns:
        --------
        Dict
            バランス評価結果
        """
        # 処理群と対照群に分割
        X_treated = X[T == 1]
        X_control = X[T == 0]
        
        if len(X_treated) == 0 or len(X_control) == 0:
            logger.warning("処理群または対照群のサンプルがありません")
            return {}
        
        # 標準化平均差（SMD）の計算
        balance_stats = {}
        
        for col in X.columns:
            # 各群の平均と標準偏差
            mean_treated = X_treated[col].mean()
            mean_control = X_control[col].mean()
            std_treated = X_treated[col].std()
            std_control = X_control[col].std()
            
            # プールされた標準偏差
            pooled_std = np.sqrt((std_treated**2 + std_control**2) / 2)
            
            # SMDの計算（プールされた標準偏差で正規化）
            if pooled_std > 0:
                smd = np.abs(mean_treated - mean_control) / pooled_std
            else:
                smd = np.nan
            
            balance_stats[col] = {
                'mean_treated': mean_treated,
                'mean_control': mean_control,
                'std_treated': std_treated,
                'std_control': std_control,
                'smd': smd,
                'is_balanced': smd < 0.1  # SMD < 0.1は一般的にバランスが取れていると見なされる
            }
        
        # 全体的なバランス評価
        overall_balance = {
            'mean_smd': np.mean([stats['smd'] for stats in balance_stats.values() if not np.isnan(stats['smd'])]),
            'max_smd': np.max([stats['smd'] for stats in balance_stats.values() if not np.isnan(stats['smd'])]),
            'balanced_covariates_ratio': np.mean([stats['is_balanced'] for stats in balance_stats.values() 
                                                if not np.isnan(stats['smd'])])
        }
        
        balance_stats['overall'] = overall_balance
        
        self.balance_stats = balance_stats
        logger.info(f"共変量バランスを評価しました。平均SMD: {overall_balance['mean_smd']:.4f}, "
                   f"バランスの取れた共変量の割合: {overall_balance['balanced_covariates_ratio']:.2%}")
        
        return balance_stats
    
    def _build_causal_graph(self) -> nx.DiGraph:
        """
        因果グラフを構築する
        
        Returns:
        --------
        nx.DiGraph
            因果グラフ
        """
        if not self.is_trained or self.feature_importance is None:
            logger.warning("モデルが学習されていないか、特徴量重要度が計算されていません")
            return None
        
        # グラフの初期化
        G = nx.DiGraph()
        
        # ノードの追加
        G.add_node(self.treatment_col, node_type='treatment')
        G.add_node(self.outcome_col, node_type='outcome')
        
        # 重要な共変量のみを追加（上位10個または重要度0.01以上）
        important_features = {}
        for feature, importance in self.feature_importance.items():
            if len(important_features) < 10 or importance >= 0.01:
                important_features[feature] = importance
        
        for feature, importance in important_features.items():
            G.add_node(feature, node_type='covariate', importance=importance)
            
            # 共変量から処理変数へのエッジ
            G.add_edge(feature, self.treatment_col, weight=importance)
            
            # 共変量から結果変数へのエッジ
            G.add_edge(feature, self.outcome_col, weight=importance)
        
        # 処理変数から結果変数へのエッジ（因果効果）
        ate = self.evaluation_results.get('ATE', 0)
        G.add_edge(self.treatment_col, self.outcome_col, weight=abs(ate), effect=ate)
        
        self.causal_graph = G
        logger.info(f"因果グラフを構築しました。ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
        
        return G
    
    def predict_individual_treatment_effect(self, X: pd.DataFrame) -> np.ndarray:
        """
        個別処理効果を予測する
        
        Parameters:
        -----------
        X : pd.DataFrame
            予測対象の共変量データフレーム
            
        Returns:
        --------
        np.ndarray
            予測された個別処理効果
        """
        if not self.is_trained or self.causal_model is None:
            raise ValueError("モデルが学習されていません")
        
        # カテゴリ変数のダミー変数化
        if self.categorical_cols:
            X_processed = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        else:
            X_processed = X.copy()
        
        # 学習時と同じ特徴量を持つことを確認
        missing_cols = set(self.causal_model.feature_names) - set(X_processed.columns)
        extra_cols = set(X_processed.columns) - set(self.causal_model.feature_names)
        
        if missing_cols:
            raise ValueError(f"予測データに以下の列が欠けています: {missing_cols}")
        
        if extra_cols:
            logger.warning(f"予測データに余分な列があります。無視されます: {extra_cols}")
            X_processed = X_processed[self.causal_model.feature_names]
        
        # 個別処理効果の予測
        ite = self.causal_model.effect(X_processed)
        
        return ite
    
    def get_evaluation(self) -> pd.DataFrame:
        """
        モデル評価結果を取得する
        
        Returns:
        --------
        pd.DataFrame
            評価結果のデータフレーム
        """
        if not self.is_trained or not self.evaluation_results:
            raise ValueError("モデルが学習されていないか、評価結果がありません")
        
        # 評価結果をデータフレームに変換
        eval_df = pd.DataFrame({
            '指標': list(self.evaluation_results.keys()),
            '値': list(self.evaluation_results.values())
        })
        
        return eval_df
    
    def get_balance_stats(self) -> pd.DataFrame:
        """
        共変量バランス評価結果を取得する
        
        Returns:
        --------
        pd.DataFrame
            バランス評価結果のデータフレーム
        """
        if not self.is_trained or not self.balance_stats:
            raise ValueError("モデルが学習されていないか、バランス評価結果がありません")
        
        # 全体評価を除外
        balance_data = {k: v for k, v in self.balance_stats.items() if k != 'overall'}
        
        # データフレームに変換
        balance_df = pd.DataFrame({
            '共変量': list(balance_data.keys()),
            '処理群平均': [v['mean_treated'] for v in balance_data.values()],
            '対照群平均': [v['mean_control'] for v in balance_data.values()],
            '処理群標準偏差': [v['std_treated'] for v in balance_data.values()],
            '対照群標準偏差': [v['std_control'] for v in balance_data.values()],
            '標準化平均差': [v['smd'] for v in balance_data.values()],
            'バランス状態': ['バランス良好' if v['is_balanced'] else 'バランス不良' for v in balance_data.values()]
        })
        
        # SMDでソート
        balance_df = balance_df.sort_values('標準化平均差', ascending=False)
        
        return balance_df
    
    def plot_feature_importance(self) -> go.Figure:
        """
        特徴量重要度を可視化する
        
        Returns:
        --------
        go.Figure
            特徴量重要度のPlotly図
        """
        if not self.is_trained or not self.feature_importance:
            raise ValueError("モデルが学習されていないか、特徴量重要度がありません")
        
        # 上位15個の特徴量を表示
        top_features = dict(list(self.feature_importance.items())[:15])
        
        # 棒グラフの作成
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=list(top_features.keys()),
            x=list(top_features.values()),
            orientation='h',
            marker=dict(
                color='rgba(50, 171, 96, 0.7)',
                line=dict(color='rgba(50, 171, 96, 1.0)', width=2)
            )
        ))
        
        # レイアウトの設定
        fig.update_layout(
            title='特徴量重要度（上位15）',
            xaxis_title='重要度',
            yaxis=dict(
                title='特徴量',
                categoryorder='total ascending'
            ),
            height=500,
            width=700,
            template='plotly_white'
        )
        
        return fig
    
    def plot_treatment_effect_distribution(self) -> go.Figure:
        """
        処理効果の分布を可視化する
        
        Returns:
        --------
        go.Figure
            処理効果分布のPlotly図
        """
        if not self.is_trained or self.treatment_effect is None:
            raise ValueError("モデルが学習されていないか、処理効果がありません")
        
        # ヒストグラムの作成
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.treatment_effect,
            nbinsx=30,
            marker=dict(
                color='rgba(0, 123, 255, 0.7)',
                line=dict(color='rgba(0, 123, 255, 1.0)', width=1)
            )
        ))
        
        # 平均処理効果（ATE）の垂直線
        ate = self.evaluation_results.get('ATE', 0)
        
        fig.add_shape(
            type='line',
            x0=ate,
            x1=ate,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # ATEのアノテーション
        fig.add_annotation(
            x=ate,
            y=0.95,
            yref='paper',
            text=f'ATE: {ate:.4f}',
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40
        )
        
        # レイアウトの設定
        fig.update_layout(
            title='処理効果の分布',
            xaxis_title='処理効果',
            yaxis_title='頻度',
            height=500,
            width=700,
            template='plotly_white'
        )
        
        return fig
    
    def plot_covariate_balance(self) -> go.Figure:
        """
        共変量バランスを可視化する
        
        Returns:
        --------
        go.Figure
            共変量バランスのPlotly図
        """
        if not self.is_trained or not self.balance_stats:
            raise ValueError("モデルが学習されていないか、バランス評価結果がありません")
        
        # 全体評価を除外
        balance_data = {k: v for k, v in self.balance_stats.items() if k != 'overall'}
        
        # SMDでソート
        sorted_covariates = sorted(balance_data.items(), key=lambda x: x[1]['smd'], reverse=True)
        
        # 上位15個の共変量を表示
        top_covariates = dict(sorted_covariates[:15])
        
        # バランス閾値
        threshold = 0.1
        
        # 棒グラフの作成
        fig = go.Figure()
        
        # SMDの棒グラフ
        fig.add_trace(go.Bar(
            y=[k for k in top_covariates.keys()],
            x=[v['smd'] for v in top_covariates.values()],
            orientation='h',
            name='標準化平均差',
            marker=dict(
                color=['rgba(255, 0, 0, 0.7)' if v['smd'] >= threshold else 'rgba(0, 128, 0, 0.7)' 
                      for v in top_covariates.values()],
                line=dict(width=1)
            )
        ))
        
        # 閾値線
        fig.add_shape(
            type='line',
            x0=threshold,
            x1=threshold,
            y0=-0.5,
            y1=len(top_covariates) - 0.5,
            line=dict(
                color='black',
                width=1,
                dash='dash'
            )
        )
        
        # レイアウトの設定
        fig.update_layout(
            title='共変量バランス（標準化平均差）',
            xaxis=dict(
                title='標準化平均差 (SMD)',
                range=[0, max([v['smd'] for v in top_covariates.values()]) * 1.1]
            ),
            yaxis=dict(
                title='共変量',
                categoryorder='array',
                categoryarray=[k for k in top_covariates.keys()]
            ),
            height=500,
            width=700,
            template='plotly_white'
        )
        
        # 閾値のアノテーション
        fig.add_annotation(
            x=threshold,
            y=len(top_covariates) - 1,
            text='バランス閾値 (0.1)',
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=0
        )
        
        return fig
    
    def plot_causal_graph(self) -> go.Figure:
        """
        因果グラフを可視化する
        
        Returns:
        --------
        go.Figure
            因果グラフのPlotly図
        """
        if not self.is_trained or self.causal_graph is None:
            raise ValueError("モデルが学習されていないか、因果グラフがありません")
        
        G = self.causal_graph
        
        # ノードの位置を計算（層状レイアウト）
        pos = {}
        
        # 共変量を左側に配置
        covariates = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'covariate']
        n_covariates = len(covariates)
        
        for i, node in enumerate(covariates):
            pos[node] = (-1, (i - n_covariates/2) / max(1, n_covariates/2))
        
        # 処理変数を中央に配置
        pos[self.treatment_col] = (0, 0)
        
        # 結果変数を右側に配置
        pos[self.outcome_col] = (1, 0)
        
        # エッジの重みを取得
        edge_weights = [G.edges[edge].get('weight', 1.0) for edge in G.edges]
        max_weight = max(edge_weights) if edge_weights else 1.0
        
        # ノードの色を設定
        node_colors = []
        node_sizes = []
        
        for node in G.nodes:
            node_type = G.nodes[node].get('node_type', '')
            
            if node_type == 'treatment':
                node_colors.append('rgba(255, 0, 0, 0.8)')  # 赤色
                node_sizes.append(30)
            elif node_type == 'outcome':
                node_colors.append('rgba(0, 0, 255, 0.8)')  # 青色
                node_sizes.append(30)
            else:  # covariate
                importance = G.nodes[node].get('importance', 0.0)
                # 重要度に基づく緑色の濃さ
                node_colors.append(f'rgba(0, 128, 0, {0.3 + 0.7 * importance})')
                node_sizes.append(20 + 20 * importance)
        
        # ノードのトレース
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes],
            y=[pos[node][1] for node in G.nodes],
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black')
            ),
            text=[node for node in G.nodes],
            textposition='bottom center',
            hoverinfo='text',
            name='ノード'
        )
        
        # エッジのトレース
        edge_traces = []
        
        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            weight = G.edges[edge].get('weight', 1.0)
            effect = G.edges[edge].get('effect', None)
            
            # エッジの太さと色
            width = 1 + 5 * (weight / max_weight)
            
            if effect is not None:
                # 処理効果のエッジ
                if effect > 0:
                    color = 'rgba(0, 128, 0, 0.8)'  # 正の効果は緑
                else:
                    color = 'rgba(255, 0, 0, 0.8)'  # 負の効果は赤
            else:
                # その他のエッジ
                color = 'rgba(128, 128, 128, 0.6)'  # グレー
            
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f'{edge[0]} → {edge[1]}<br>重み: {weight:.3f}' + 
                     (f'<br>効果: {effect:.3f}' if effect is not None else ''),
                showlegend=False
            )
            
            edge_traces.append(edge_trace)
        
        # 図の作成
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # レイアウトの設定
        fig.update_layout(
            title='因果グラフ',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800,
            template='plotly_white'
        )
        
        return fig
    
    def save_model(self, filepath: str) -> None:
        """
        モデルを保存する
        
        Parameters:
        -----------
        filepath : str
            保存先のファイルパス
        """
        if not self.is_trained or self.causal_model is None:
            raise ValueError("保存するモデルがありません")
        
        # モデルと関連データを辞書に格納
        model_data = {
            'model': self.causal_model,
            'model_type': self.model_type,
            'treatment_col': self.treatment_col,
            'outcome_col': self.outcome_col,
            'covariate_cols': self.covariate_cols,
            'categorical_cols': self.categorical_cols,
            'evaluation_results': self.evaluation_results,
            'feature_importance': self.feature_importance,
            'balance_stats': self.balance_stats
        }
        
        try:
            # ファイルに保存
            save_pkl.save(filepath, model_data)
            logger.info(f"モデルを {filepath} に保存しました")
        except Exception as e:
            logger.error(f"モデル保存中にエラーが発生しました: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        保存されたモデルを読み込む
        
        Parameters:
        -----------
        filepath : str
            モデルファイルのパス
        """
        try:
            # ファイルから読み込み
            model_data = load_pkl.load(filepath)
            
            # モデルと関連データを復元
            self.causal_model = model_data['model']
            self.model_type = model_data['model_type']
            self.treatment_col = model_data['treatment_col']
            self.outcome_col = model_data['outcome_col']
            self.covariate_cols = model_data['covariate_cols']
            self.categorical_cols = model_data['categorical_cols']
            self.evaluation_results = model_data['evaluation_results']
            self.feature_importance = model_data['feature_importance']
            self.balance_stats = model_data['balance_stats']
            
            # 因果グラフの再構築
            self._build_causal_graph()
            
            self.is_trained = True
            logger.info(f"モデルを {filepath} から読み込みました")
            
        except Exception as e:
            logger.error(f"モデル読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.data = None
        self.causal_model = None
        self.treatment_effect = None
        self.causal_graph = None
        gc.collect()
