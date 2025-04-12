import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Generator
import logging
import json
import os
import requests
import time
import re
from datetime import datetime
import gc

# OpenAI API関連のインポート
try:
    import openai
except ImportError:
    logging.warning("openaiがインストールされていません。pip install openai")

# Anthropic API関連のインポート
try:
    import anthropic
except ImportError:
    logging.warning("anthropicがインストールされていません。pip install anthropic")

logger = logging.getLogger(__name__)

class LLMExplainer:
    """
    LLM（大規模言語モデル）を活用した説明機能を提供するクラス
    外部LLM APIとの連携、チャットインターフェース、質問タイプの認識と応答生成を実装
    """
    
    def __init__(self, 
                model_provider: str = "anthropic", 
                model_name: str = "claude-3-sonnet-20240229",
                api_key: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 1000):
        """
        LLMExplainerクラスの初期化
        
        Parameters:
        -----------
        model_provider : str
            使用するLLMプロバイダー ('anthropic' または 'openai')
        model_name : str
            使用するモデル名
            - anthropic: 'claude-3-sonnet-20240229', 'claude-3-opus-20240229' など
            - openai: 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo' など
        api_key : Optional[str]
            API キー（Noneの場合は環境変数から取得）
        temperature : float
            生成の温度パラメータ（0.0〜1.0）
        max_tokens : int
            生成する最大トークン数
        """
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_history = []
        self.data_context = None
        self.causal_context = None
        
        # APIキーの設定
        if api_key:
            self.api_key = api_key
        else:
            # 環境変数からAPIキーを取得
            if self.model_provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    logger.warning("ANTHROPIC_API_KEYが設定されていません。環境変数を設定するか、初期化時にapi_keyを指定してください。")
            elif self.model_provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("OPENAI_API_KEYが設定されていません。環境変数を設定するか、初期化時にapi_keyを指定してください。")
            else:
                raise ValueError(f"サポートされていないモデルプロバイダーです: {model_provider}")
        
        # APIクライアントの初期化
        self._initialize_client()
        
        logger.info(f"LLMExplainerを初期化しました: プロバイダー={model_provider}, モデル={model_name}")
    
    def _initialize_client(self) -> None:
        """
        APIクライアントを初期化する
        """
        if self.model_provider == "anthropic":
            if self.api_key:
                try:
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info("Anthropic APIクライアントを初期化しました")
                except Exception as e:
                    logger.error(f"Anthropic APIクライアントの初期化中にエラーが発生しました: {str(e)}")
                    self.client = None
            else:
                self.client = None
                
        elif self.model_provider == "openai":
            if self.api_key:
                try:
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logger.info("OpenAI APIクライアントを初期化しました")
                except Exception as e:
                    logger.error(f"OpenAI APIクライアントの初期化中にエラーが発生しました: {str(e)}")
                    self.client = None
            else:
                self.client = None
        else:
            self.client = None
    
    def set_data_context(self, data: pd.DataFrame, summary: Optional[Dict] = None) -> None:
        """
        データコンテキストを設定する
        
        Parameters:
        -----------
        data : pd.DataFrame
            コンテキストとして使用するデータフレーム
        summary : Optional[Dict]
            データの要約情報（オプション）
        """
        if data is None:
            logger.warning("データコンテキストにNoneが設定されました")
            self.data_context = None
            return
            
        # データの基本情報
        data_info = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "sample": data.head(5).to_dict(orient="records"),
            "missing_values": data.isnull().sum().to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 数値列の統計情報
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            data_info["numeric_stats"] = data[numeric_cols].describe().to_dict()
        
        # カテゴリ列の情報
        categorical_cols = data.select_dtypes(include=["category", "object"]).columns.tolist()
        if categorical_cols:
            cat_info = {}
            for col in categorical_cols:
                value_counts = data[col].value_counts().head(10).to_dict()
                cat_info[col] = {
                    "unique_values": data[col].nunique(),
                    "top_values": value_counts
                }
            data_info["categorical_stats"] = cat_info
        
        # 追加の要約情報があれば統合
        if summary:
            data_info["summary"] = summary
        
        self.data_context = data_info
        logger.info(f"データコンテキストを設定しました: {data.shape[0]}行 x {data.shape[1]}列")
    
    def set_causal_context(self, causal_model: Any, evaluation: Optional[Dict] = None) -> None:
        """
        因果モデルのコンテキストを設定する
        
        Parameters:
        -----------
        causal_model : Any
            因果モデルのインスタンス
        evaluation : Optional[Dict]
            モデル評価結果（オプション）
        """
        if causal_model is None:
            logger.warning("因果モデルコンテキストにNoneが設定されました")
            self.causal_context = None
            return
            
        # モデルの基本情報
        model_info = {
            "model_type": getattr(causal_model, "model_type", "unknown"),
            "treatment_col": getattr(causal_model, "treatment_col", "unknown"),
            "outcome_col": getattr(causal_model, "outcome_col", "unknown"),
            "is_trained": getattr(causal_model, "is_trained", False),
            "timestamp": datetime.now().isoformat()
        }
        
        # 共変量情報
        if hasattr(causal_model, "covariate_cols"):
            model_info["covariate_cols"] = causal_model.covariate_cols
            
        # 特徴量重要度
        if hasattr(causal_model, "feature_importance") and causal_model.feature_importance:
            # 上位10個の特徴量のみ
            top_features = dict(list(causal_model.feature_importance.items())[:10])
            model_info["feature_importance"] = top_features
            
        # 評価結果
        if hasattr(causal_model, "evaluation_results") and causal_model.evaluation_results:
            model_info["evaluation"] = causal_model.evaluation_results
        elif evaluation:
            model_info["evaluation"] = evaluation
            
        # バランス評価
        if hasattr(causal_model, "balance_stats") and causal_model.balance_stats:
            if "overall" in causal_model.balance_stats:
                model_info["balance_overall"] = causal_model.balance_stats["overall"]
        
        self.causal_context = model_info
        logger.info(f"因果モデルコンテキストを設定しました: {model_info['model_type']}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        チャット履歴にメッセージを追加する
        
        Parameters:
        -----------
        role : str
            メッセージの役割 ('user', 'assistant', 'system')
        content : str
            メッセージの内容
        """
        self.chat_history.append({"role": role, "content": content})
        
        # 履歴が長すぎる場合、古いメッセージを削除（システムメッセージは保持）
        if len(self.chat_history) > 20:
            # システムメッセージを保持
            system_messages = [msg for msg in self.chat_history if msg["role"] == "system"]
            # 最新の会話を保持（システムメッセージを除く）
            recent_messages = [msg for msg in self.chat_history if msg["role"] != "system"][-19:]
            # 履歴を更新
            self.chat_history = system_messages + recent_messages
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        チャット履歴をクリアする
        
        Parameters:
        -----------
        keep_system : bool
            システムメッセージを保持するかどうか
        """
        if keep_system:
            self.chat_history = [msg for msg in self.chat_history if msg["role"] == "system"]
        else:
            self.chat_history = []
        logger.info("チャット履歴をクリアしました")
    
    def _prepare_system_message(self) -> str:
        """
        システムメッセージを準備する
        
        Returns:
        --------
        str
            システムメッセージ
        """
        system_message = """
あなたはCausalViz Analytics Platformの一部として動作するAIアシスタントです。
データ分析と因果推論に関する質問に対して、明確で正確な回答を提供してください。

以下のタイプの質問に対応してください：
1. データの探索と分析に関する質問
2. 因果関係と処理効果に関する質問
3. 分析結果の解釈に関する質問
4. 統計的手法や因果推論手法に関する質問

回答の際は以下のガイドラインに従ってください：
- 専門用語を適切に使用しつつ、わかりやすく説明する
- 不確実性がある場合は、それを明示する
- データや分析に基づいた根拠を示す
- 複雑な概念は段階的に説明する
- 必要に応じて、さらなる分析や可視化の提案を行う

ユーザーの質問に対して、データコンテキストと因果モデルコンテキストを活用して、
具体的で実用的な回答を提供してください。
"""
        
        # データコンテキストの追加
        if self.data_context:
            data_info = f"""
現在のデータセット情報:
- 行数: {self.data_context['shape'][0]}
- 列数: {self.data_context['shape'][1]}
- 列名: {', '.join(self.data_context['columns'][:10])}{'...' if len(self.data_context['columns']) > 10 else ''}
"""
            system_message += "\n" + data_info
            
        # 因果モデルコンテキストの追加
        if self.causal_context and self.causal_context.get("is_trained", False):
            model_info = f"""
現在の因果モデル情報:
- モデルタイプ: {self.causal_context['model_type']}
- 処理変数: {self.causal_context['treatment_col']}
- 結果変数: {self.causal_context['outcome_col']}
"""
            if "evaluation" in self.causal_context and "ATE" in self.causal_context["evaluation"]:
                model_info += f"- 平均処理効果 (ATE): {self.causal_context['evaluation']['ATE']:.4f}\n"
                
            system_message += "\n" + model_info
        
        return system_message
    
    def _prepare_messages_for_anthropic(self) -> List[Dict]:
        """
        Anthropic API用のメッセージを準備する
        
        Returns:
        --------
        List[Dict]
            Anthropic API用のメッセージリスト
        """
        # システムメッセージがあるか確認
        has_system = any(msg["role"] == "system" for msg in self.chat_history)
        
        if not has_system:
            # システムメッセージを追加
            system_content = self._prepare_system_message()
            messages = [{"role": "system", "content": system_content}]
        else:
            messages = []
        
        # ユーザーとアシスタントのメッセージを追加
        for msg in self.chat_history:
            if msg["role"] == "system":
                # システムメッセージは既に追加済みの場合はスキップ
                if not has_system:
                    messages.append({"role": "system", "content": msg["content"]})
                    has_system = True
            elif msg["role"] in ["user", "assistant"]:
                # Anthropicの形式に変換
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        return messages
    
    def _prepare_messages_for_openai(self) -> List[Dict]:
        """
        OpenAI API用のメッセージを準備する
        
        Returns:
        --------
        List[Dict]
            OpenAI API用のメッセージリスト
        """
        # システムメッセージがあるか確認
        has_system = any(msg["role"] == "system" for msg in self.chat_history)
        
        if not has_system:
            # システムメッセージを追加
            system_content = self._prepare_system_message()
            messages = [{"role": "system", "content": system_content}]
        else:
            messages = []
        
        # ユーザーとアシスタントのメッセージを追加
        for msg in self.chat_history:
            if msg["role"] in ["system", "user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        return messages
    
    def _classify_question(self, question: str) -> str:
        """
        質問のタイプを分類する
        
        Parameters:
        -----------
        question : str
            ユーザーの質問
            
        Returns:
        --------
        str
            質問のタイプ ('direct_causality', 'factor_analysis', 'mechanism', 'counterfactual', 'data_exploration', 'general')
        """
        # 質問タイプのパターン
        patterns = {
            'direct_causality': [
                r'何が.*原因',
                r'何が.*引き起こ',
                r'何が.*影響',
                r'何が.*要因',
                r'what.*cause',
                r'what.*impact'
            ],
            'factor_analysis': [
                r'どの.*要因が.*影響',
                r'どの.*変数が.*重要',
                r'最も.*影響.*要因',
                r'最も.*重要.*要因',
                r'which.*factor',
                r'most.*important.*variable'
            ],
            'mechanism': [
                r'なぜ.*発生',
                r'どのように.*影響',
                r'どのような.*メカニズム',
                r'どのような.*プロセス',
                r'why.*happen',
                r'how.*affect'
            ],
            'counterfactual': [
                r'もし.*なければ',
                r'もし.*場合.*どう',
                r'もし.*変わったら',
                r'if.*not',
                r'what.*if',
                r'without.*would'
            ],
            'data_exploration': [
                r'データ.*特徴',
                r'データ.*傾向',
                r'データ.*分布',
                r'データ.*相関',
                r'data.*feature',
                r'data.*trend',
                r'data.*distribution',
                r'data.*correlation'
            ]
        }
        
        # 質問タイプの判定
        for q_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, question, re.IGNORECASE):
                    return q_type
        
        # どのパターンにも一致しない場合
        return 'general'
    
    def _enhance_question_with_context(self, question: str, question_type: str) -> str:
        """
        コンテキスト情報を使って質問を強化する
        
        Parameters:
        -----------
        question : str
            ユーザーの質問
        question_type : str
            質問のタイプ
            
        Returns:
        --------
        str
            強化された質問
        """
        enhanced_question = question
        
        # データコンテキストの追加
        if self.data_context:
            data_context = f"\n\nデータセット情報:\n"
            data_context += f"- 行数: {self.data_context['shape'][0]}\n"
            data_context += f"- 列数: {self.data_context['shape'][1]}\n"
            data_context += f"- 列名: {', '.join(self.data_context['columns'])}\n"
            
            # 質問タイプに応じた追加情報
            if question_type == 'data_exploration':
                # 数値列の統計情報
                if "numeric_stats" in self.data_context:
                    data_context += "\n数値列の統計情報:\n"
                    for col, stats in list(self.data_context["numeric_stats"].items())[:5]:
                        data_context += f"- {col}: 平均={stats.get('mean', 'N/A'):.2f}, 最小={stats.get('min', 'N/A'):.2f}, 最大={stats.get('max', 'N/A'):.2f}\n"
                
                # カテゴリ列の情報
                if "categorical_stats" in self.data_context:
                    data_context += "\nカテゴリ列の情報:\n"
                    for col, stats in list(self.data_context["categorical_stats"].items())[:3]:
                        data_context += f"- {col}: ユニーク値数={stats.get('unique_values', 'N/A')}\n"
            
            enhanced_question += data_context
        
        # 因果モデルコンテキストの追加
        if self.causal_context and self.causal_context.get("is_trained", False):
            causal_context = f"\n\n因果モデル情報:\n"
            causal_context += f"- モデルタイプ: {self.causal_context['model_type']}\n"
            causal_context += f"- 処理変数: {self.causal_context['treatment_col']}\n"
            causal_context += f"- 結果変数: {self.causal_context['outcome_col']}\n"
            
            # 質問タイプに応じた追加情報
            if question_type in ['direct_causality', 'factor_analysis', 'mechanism', 'counterfactual']:
                # 評価結果
                if "evaluation" in self.causal_context:
                    causal_context += "\n評価結果:\n"
                    for metric, value in list(self.causal_context["evaluation"].items())[:5]:
                        if isinstance(value, (int, float)):
                            causal_context += f"- {metric}: {value:.4f}\n"
                        else:
                            causal_context += f"- {metric}: {value}\n"
                
                # 特徴量重要度
                if "feature_importance" in self.causal_context:
                    causal_context += "\n特徴量重要度 (上位5):\n"
                    for feature, importance in list(self.causal_context["feature_importance"].items())[:5]:
                        causal_context += f"- {feature}: {importance:.4f}\n"
            
            enhanced_question += causal_context
        
        return enhanced_question
    
    def generate_response(self, question: str, streaming: bool = True) -> Union[str, Generator[str, None, None]]:
        """
        ユーザーの質問に対する応答を生成する
        
        Parameters:
        -----------
        question : str
            ユーザーの質問
        streaming : bool
            ストリーミングレスポンスを使用するかどうか
            
        Returns:
        --------
        Union[str, Generator[str, None, None]]
            生成された応答（ストリーミングの場合はジェネレータ）
        """
        if not self.client:
            error_msg = f"{self.model_provider} APIクライアントが初期化されていません。APIキーを確認してください。"
            logger.error(error_msg)
            return error_msg
        
        # 質問のタイプを分類
        question_type = self._classify_question(question)
        logger.info(f"質問タイプ: {question_type}")
        
        # 質問をコンテキストで強化
        enhanced_question = self._enhance_question_with_context(question, question_type)
        
        # ユーザーメッセージを追加
        self.add_message("user", enhanced_question)
        
        try:
            if self.model_provider == "anthropic":
                return self._generate_anthropic_response(streaming)
            elif self.model_provider == "openai":
                return self._generate_openai_response(streaming)
            else:
                error_msg = f"サポートされていないモデルプロバイダーです: {self.model_provider}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"応答生成中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _generate_anthropic_response(self, streaming: bool = True) -> Union[str, Generator[str, None, None]]:
        """
        Anthropic APIを使用して応答を生成する
        
        Parameters:
        -----------
        streaming : bool
            ストリーミングレスポンスを使用するかどうか
            
        Returns:
        --------
        Union[str, Generator[str, None, None]]
            生成された応答（ストリーミングの場合はジェネレータ）
        """
        messages = self._prepare_messages_for_anthropic()
        
        if streaming:
            # ストリーミングレスポンス
            response_stream = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            def response_generator():
                full_response = ""
                for chunk in response_stream:
                    if chunk.delta.text:
                        full_response += chunk.delta.text
                        yield chunk.delta.text
                
                # 完全な応答をチャット履歴に追加
                self.add_message("assistant", full_response)
            
            return response_generator()
        else:
            # 非ストリーミングレスポンス
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 応答をチャット履歴に追加
            self.add_message("assistant", response.content[0].text)
            
            return response.content[0].text
    
    def _generate_openai_response(self, streaming: bool = True) -> Union[str, Generator[str, None, None]]:
        """
        OpenAI APIを使用して応答を生成する
        
        Parameters:
        -----------
        streaming : bool
            ストリーミングレスポンスを使用するかどうか
            
        Returns:
        --------
        Union[str, Generator[str, None, None]]
            生成された応答（ストリーミングの場合はジェネレータ）
        """
        messages = self._prepare_messages_for_openai()
        
        if streaming:
            # ストリーミングレスポンス
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            def response_generator():
                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                
                # 完全な応答をチャット履歴に追加
                self.add_message("assistant", full_response)
            
            return response_generator()
        else:
            # 非ストリーミングレスポンス
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 応答をチャット履歴に追加
            self.add_message("assistant", response.choices[0].message.content)
            
            return response.choices[0].message.content
    
    def get_chat_history(self) -> List[Dict]:
        """
        チャット履歴を取得する
        
        Returns:
        --------
        List[Dict]
            チャット履歴
        """
        return self.chat_history
    
    def save_chat_history(self, filepath: str) -> None:
        """
        チャット履歴をJSONファイルに保存する
        
        Parameters:
        -----------
        filepath : str
            保存先のファイルパス
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            logger.info(f"チャット履歴を {filepath} に保存しました")
        except Exception as e:
            logger.error(f"チャット履歴の保存中にエラーが発生しました: {str(e)}")
            raise
    
    def load_chat_history(self, filepath: str) -> None:
        """
        チャット履歴をJSONファイルから読み込む
        
        Parameters:
        -----------
        filepath : str
            読み込むファイルのパス
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.chat_history = json.load(f)
            logger.info(f"チャット履歴を {filepath} から読み込みました")
        except Exception as e:
            logger.error(f"チャット履歴の読み込み中にエラーが発生しました: {str(e)}")
            raise
    
    def __del__(self):
        """
        デストラクタ - メモリの解放
        """
        self.chat_history = []
        self.data_context = None
        self.causal_context = None
        gc.collect()
