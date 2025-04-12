import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import logging
from datetime import datetime
import uuid
import re
from typing import List, Dict, Any, Optional, Union, Tuple

# LLM関連のライブラリ
import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMExplainer:
    """
    LLMエクスプレイナークラス
    外部LLM APIと連携して、データに関する質問に回答する
    """
    
    def __init__(self):
        """初期化"""
        self.anthropic_client = None
        self.openai_client = None
        self.preferred_provider = None
        self.data_context = None
        self.causal_context = None
        self.chat_history = []
        self.initialized = False
        self.streaming = True
        self.max_tokens = 4000
        self.temperature = 0.7
        self.system_prompt_template = """
        あなたはデータ分析と因果推論の専門家AIアシスタントです。
        ユーザーのデータに関する質問に対して、正確で洞察に富んだ回答を提供してください。
        
        以下のデータコンテキストが提供されています：
        {data_context}
        
        以下の因果モデルコンテキストが提供されています：
        {causal_context}
        
        回答の際は以下のガイドラインに従ってください：
        1. データに基づいた事実のみを述べる
        2. 不確かな場合は、その旨を明示する
        3. 複雑な概念は簡潔に説明する
        4. 必要に応じて追加の分析や可視化を提案する
        5. 因果関係に関する質問には特に注意して回答する
        
        質問のタイプに応じて適切な回答フォーマットを使用してください：
        - 直接的因果関係の質問：明確な証拠と共に原因を特定
        - 要因分析の質問：影響度と共に要因をランク付け
        - メカニズム探索の質問：因果チェーンを用いたプロセス説明
        - 反事実分析の質問：代替シナリオのシミュレーション結果
        
        専門用語を使う場合は、必ず簡潔な説明を添えてください。
        """
        
        # 質問タイプの認識パターン
        self.question_patterns = {
            "direct_causality": [
                r"何が.*原因",
                r"何が.*引き起こし",
                r"何が.*要因",
                r"何が.*影響",
                r"what is causing",
                r"what causes",
                r"why does",
                r"why is"
            ],
            "factor_analysis": [
                r"どの要因が.*影響",
                r"どの変数が.*重要",
                r"どの特徴が.*重要",
                r"which factors",
                r"most important variables",
                r"key determinants"
            ],
            "mechanism_exploration": [
                r"なぜ.*発生",
                r"どのように.*影響",
                r"どのような仕組み",
                r"how does",
                r"through what mechanism",
                r"what is the process"
            ],
            "counterfactual_analysis": [
                r"もし.*なければ",
                r"もし.*だったら",
                r"もし.*変わったら",
                r"what if",
                r"if we had",
                r"had we not"
            ]
        }
    
    def is_initialized(self) -> bool:
        """初期化されているかどうかを返す"""
        return self.initialized
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """APIキーを設定する"""
        if provider.lower() == 'anthropic':
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                if self.preferred_provider is None:
                    self.preferred_provider = 'anthropic'
                self.initialized = True
            except Exception as e:
                st.error(f"Anthropic APIの初期化エラー: {str(e)}")
        
        elif provider.lower() == 'openai':
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                if self.preferred_provider is None:
                    self.preferred_provider = 'openai'
                self.initialized = True
            except Exception as e:
                st.error(f"OpenAI APIの初期化エラー: {str(e)}")
    
    def set_preferred_provider(self, provider: str) -> None:
        """優先プロバイダーを設定する"""
        if provider.lower() in ['anthropic', 'openai']:
            self.preferred_provider = provider.lower()
    
    def set_data_context(self, data: pd.DataFrame) -> None:
        """データコンテキストを設定する"""
        if data is None:
            self.data_context = None
            return
        
        # データの基本情報
        data_info = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "summary": data.describe().to_dict(),
            "sample": data.head(5).to_dict(orient="records")
        }
        
        # データコンテキストの作成
        self.data_context = f"""
        データセット情報:
        - 行数: {data_info['shape'][0]}
        - 列数: {data_info['shape'][1]}
        - カラム: {', '.join(data_info['columns'])}
        
        データ型:
        {json.dumps(data_info['dtypes'], indent=2, ensure_ascii=False)}
        
        基本統計:
        {json.dumps(data_info['summary'], indent=2, ensure_ascii=False)}
        
        サンプルデータ:
        {json.dumps(data_info['sample'], indent=2, ensure_ascii=False)}
        """
    
    def set_causal_context(self, causal_model: Any) -> None:
        """因果モデルコンテキストを設定する"""
        if causal_model is None:
            self.causal_context = None
            return
        
        # 因果モデルの情報を抽出
        try:
            model_info = {
                "treatment_variable": getattr(causal_model, "treatment_name", "不明"),
                "outcome_variable": getattr(causal_model, "outcome_name", "不明"),
                "covariates": getattr(causal_model, "covariate_names", []),
                "model_type": causal_model.__class__.__name__,
                "effect_estimate": getattr(causal_model, "effect_estimate", None),
                "p_value": getattr(causal_model, "p_value", None)
            }
            
            # 因果コンテキストの作成
            self.causal_context = f"""
            因果モデル情報:
            - モデルタイプ: {model_info['model_type']}
            - 処理変数: {model_info['treatment_variable']}
            - 結果変数: {model_info['outcome_variable']}
            - 共変量: {', '.join(model_info['covariates']) if model_info['covariates'] else '不明'}
            
            効果推定:
            - 推定効果: {model_info['effect_estimate'] if model_info['effect_estimate'] is not None else '未計算'}
            - p値: {model_info['p_value'] if model_info['p_value'] is not None else '未計算'}
            """
        except Exception as e:
            self.causal_context = "因果モデルの情報を抽出できませんでした。"
    
    def detect_question_type(self, question: str) -> str:
        """質問のタイプを検出する"""
        question = question.lower()
        
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return q_type
        
        return "general"  # デフォルトは一般的な質問
    
    def get_system_prompt(self, question: str) -> str:
        """システムプロンプトを生成する"""
        question_type = self.detect_question_type(question)
        
        # 質問タイプに応じた追加指示
        type_specific_instructions = {
            "direct_causality": """
            これは直接的因果関係に関する質問です。
            回答では以下の点に注意してください：
            1. 明確な証拠と共に原因を特定する
            2. 因果関係の強さを示す
            3. 代替説明の可能性も考慮する
            4. 相関と因果の違いを明確にする
            """,
            
            "factor_analysis": """
            これは要因分析に関する質問です。
            回答では以下の点に注意してください：
            1. 影響度と共に要因をランク付けする
            2. 各要因の効果サイズを示す
            3. 要因間の相互作用を考慮する
            4. 定量的な分析結果を提供する
            """,
            
            "mechanism_exploration": """
            これはメカニズム探索に関する質問です。
            回答では以下の点に注意してください：
            1. 因果チェーンを用いたプロセス説明を提供する
            2. 各ステップの論理的つながりを示す
            3. 可能な限り図表を用いて説明する
            4. 理論的背景と実証的証拠を組み合わせる
            """,
            
            "counterfactual_analysis": """
            これは反事実分析に関する質問です。
            回答では以下の点に注意してください：
            1. 代替シナリオのシミュレーション結果を提供する
            2. 仮定と制約を明確にする
            3. 不確実性の範囲を示す
            4. 複数の可能性を検討する
            """
        }
        
        # 質問タイプに応じた追加指示を取得
        type_instruction = type_specific_instructions.get(question_type, "")
        
        # システムプロンプトの作成
        system_prompt = self.system_prompt_template.format(
            data_context=self.data_context if self.data_context else "データコンテキストは提供されていません。",
            causal_context=self.causal_context if self.causal_context else "因果モデルコンテキストは提供されていません。"
        )
        
        # 質問タイプに応じた追加指示を追加
        if type_instruction:
            system_prompt += f"\n\n{type_instruction}"
        
        return system_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.APIError, openai.APIError))
    )
    def generate_response(self, question: str) -> Union[str, anthropic.types.MessageStreamManager, openai.types.chat.completions.ChatCompletionChunk]:
        """LLMを使用して応答を生成する"""
        if not self.initialized:
            return "APIキーが設定されていません。設定ページでAPIキーを設定してください。"
        
        system_prompt = self.get_system_prompt(question)
        
        # チャット履歴の準備
        messages = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # 新しい質問を追加
        messages.append({"role": "user", "content": question})
        
        try:
            # Anthropic (Claude) APIを使用
            if self.preferred_provider == 'anthropic' and self.anthropic_client:
                if self.streaming:
                    return self.anthropic_client.messages.stream(
                        model="claude-3-sonnet-20240229",
                        system=system_prompt,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                else:
                    response = self.anthropic_client.messages.create(
                        model="claude-3-sonnet-20240229",
                        system=system_prompt,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    return response.content[0].text
            
            # OpenAI (GPT) APIを使用
            elif self.preferred_provider == 'openai' and self.openai_client:
                # システムプロンプトを最初のメッセージとして追加
                openai_messages = [{"role": "system", "content": system_prompt}]
                
                # ユーザーとアシスタントのメッセージを追加
                for msg in messages:
                    openai_messages.append(msg)
                
                if self.streaming:
                    return self.openai_client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=openai_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=True
                    )
                else:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=openai_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
            
            else:
                return "有効なLLMプロバイダーが設定されていません。"
        
        except Exception as e:
            return f"応答生成中にエラーが発生しました: {str(e)}"
    
    def add_to_chat_history(self, role: str, content: str) -> None:
        """チャット履歴に追加する"""
        self.chat_history.append({
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def clear_chat_history(self) -> None:
        """チャット履歴をクリアする"""
        self.chat_history = []
    
    def save_chat_history(self, file_path: str) -> bool:
        """チャット履歴を保存する"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"チャット履歴の保存中にエラーが発生しました: {str(e)}")
            return False
    
    def load_chat_history(self, file_path: str) -> bool:
        """チャット履歴を読み込む"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                return True
            return False
        except Exception as e:
            st.error(f"チャット履歴の読み込み中にエラーが発生しました: {str(e)}")
            return False
    
    def render_chat_ui(self) -> None:
        """チャットUIを描画する"""
        st.subheader("AIアシスタントとチャット")
        
        # チャット設定
        with st.expander("チャット設定", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # プロバイダー選択
                provider_options = []
                if self.anthropic_client:
                    provider_options.append("anthropic")
                if self.openai_client:
                    provider_options.append("openai")
                
                if provider_options:
                    selected_provider = st.selectbox(
                        "LLMプロバイダー",
                        options=provider_options,
                        index=provider_options.index(self.preferred_provider) if self.preferred_provider in provider_options else 0
                    )
                    self.set_preferred_provider(selected_provider)
                
                # ストリーミング設定
                self.streaming = st.checkbox("ストリーミングレスポンス", value=self.streaming)
            
            with col2:
                # 温度設定
                self.temperature = st.slider("温度 (創造性)", min_value=0.0, max_value=1.0, value=self.temperature, step=0.1)
                
                # 最大トークン数
                self.max_tokens = st.slider("最大トークン数", min_value=100, max_value=8000, value=self.max_tokens, step=100)
            
            # チャット履歴の管理
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("チャット履歴をクリア"):
                    self.clear_chat_history()
                    st.success("チャット履歴をクリアしました")
            
            with col2:
                # チャット履歴の保存と読み込み
                chat_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'chats')
                os.makedirs(chat_dir, exist_ok=True)
                
                chat_files = [f for f in os.listdir(chat_dir) if f.endswith('.json')]
                
                if chat_files:
                    selected_chat = st.selectbox("チャット履歴", options=["新規チャット"] + chat_files)
                    
                    if selected_chat != "新規チャット":
                        if st.button("選択したチャット履歴を読み込む"):
                            if self.load_chat_history(os.path.join(chat_dir, selected_chat)):
                                st.success(f"チャット履歴を読み込みました: {selected_chat}")
                
                chat_name = st.text_input("チャット名", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                if st.button("現在のチャット履歴を保存"):
                    if chat_name:
                        if not chat_name.endswith('.json'):
                            chat_name += '.json'
                        
                        if self.save_chat_history(os.path.join(chat_dir, chat_name)):
                            st.success(f"チャット履歴を保存しました: {chat_name}")
        
        # チャット履歴の表示
        st.subheader("チャット履歴")
        
        for message in self.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # 入力フォーム
        if prompt := st.chat_input("質問を入力してください..."):
            # ユーザーの質問を表示
            st.chat_message("user").write(prompt)
            
            # チャット履歴に追加
            self.add_to_chat_history("user", prompt)
            
            # AIの応答を生成
            with st.chat_message("assistant"):
                if self.streaming:
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # ストリーミングレスポンスの処理
                    response = self.generate_response(prompt)
                    
                    if isinstance(response, str):
                        # エラーメッセージなどの文字列の場合
                        response_placeholder.write(response)
                        full_response = response
                    else:
                        try:
                            # Anthropic (Claude) APIのストリーミングレスポンス
                            if self.preferred_provider == 'anthropic':
                                for chunk in response:
                                    if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                                        full_response += chunk.delta.text
                                        response_placeholder.write(full_response)
                            
                            # OpenAI (GPT) APIのストリーミングレスポンス
                            elif self.preferred_provider == 'openai':
                                for chunk in response:
                                    if chunk.choices[0].delta.content:
                                        full_response += chunk.choices[0].delta.content
                                        response_placeholder.write(full_response)
                        except Exception as e:
                            error_msg = f"ストリーミングレスポンスの処理中にエラーが発生しました: {str(e)}"
                            response_placeholder.write(error_msg)
                            full_response = error_msg
                else:
                    # 非ストリーミングレスポンス
                    full_response = self.generate_response(prompt)
                    st.write(full_response)
            
            # チャット履歴に追加
            self.add_to_chat_history("assistant", full_response)
        
        # データコンテキスト情報
        if st.session_state.data is not None:
            with st.expander("データコンテキスト情報", expanded=False):
                st.subheader("現在のデータセット情報")
                
                # データ基本情報
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("行数", len(st.session_state.data))
                with col2:
                    st.metric("列数", len(st.session_state.data.columns))
                with col3:
                    memory_usage = st.session_state.data.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("メモリ使用量", f"{memory_usage:.2f} MB")
                
                # カラム情報
                st.subheader("カラム情報")
                
                # カラムの型情報
                col_info = []
                for col in st.session_state.data.columns:
                    col_info.append({
                        "カラム名": col,
                        "データ型": str(st.session_state.data[col].dtype),
                        "欠損値": st.session_state.data[col].isna().sum(),
                        "ユニーク値": st.session_state.data[col].nunique() if st.session_state.data[col].dtype != 'object' or st.session_state.data[col].nunique() < 100 else "100+"
                    })
                
                st.dataframe(pd.DataFrame(col_info))
                
                # データプレビュー
                st.subheader("データプレビュー")
                st.dataframe(st.session_state.data.head(5))
        
        # 因果モデル情報
        if hasattr(st.session_state, 'causal_inference') and hasattr(st.session_state.causal_inference, 'model') and st.session_state.causal_inference.model is not None:
            with st.expander("因果モデル情報", expanded=False):
                st.subheader("現在の因果モデル情報")
                
                model = st.session_state.causal_inference.model
                
                # モデル基本情報
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("処理変数", getattr(model, "treatment_name", "不明"))
                    st.metric("結果変数", getattr(model, "outcome_name", "不明"))
                
                with col2:
                    st.metric("モデルタイプ", model.__class__.__name__)
                    effect_estimate = getattr(model, "effect_estimate", None)
                    if effect_estimate is not None:
                        st.metric("推定効果", f"{effect_estimate:.4f}")
                
                # 共変量情報
                covariates = getattr(model, "covariate_names", [])
                if covariates:
                    st.subheader("共変量")
                    st.write(", ".join(covariates))
                
                # モデル詳細情報
                if hasattr(model, "summary") and callable(model.summary):
                    st.subheader("モデル詳細")
                    st.text(model.summary())
