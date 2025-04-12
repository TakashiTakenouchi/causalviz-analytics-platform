# CausalViz Analytics Platform

CausalViz Analytics Platformは、Streamlit + AutoGluon + LLMを組み合わせた高度なデータ分析・因果推論プラットフォームです。インタラクティブなデータ表示、ピボットテーブル機能、多様な可視化機能、因果推論分析、AIアシスタント機能を提供します。

## 主な機能

### データ分析機能
- インタラクティブなデータ表示と編集
- ピボットテーブル機能
- 多様な可視化機能（折れ線グラフ、散布図、棒グラフ、パレート図など）

### 因果推論機能
- 処理効果の推定
- 共変量バランスの評価
- 因果グラフの可視化
- 特徴量重要度の分析

### AIアシスタント機能
- 外部LLM API（Claude、GPT）との連携
- チャットインターフェース
- 質問タイプの自動認識
- データと因果モデルのコンテキスト管理

## ディレクトリ構造

```
causalviz_platform/
├── app/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
├── modules/
│   ├── data_analyzer/
│   │   ├── data_analyzer.py
│   │   ├── interactive_data_editor.py
│   │   ├── pivot_table_builder.py
│   │   └── visualization.py
│   ├── causal_inference/
│   │   └── causal_inference.py
│   └── llm_explainer/
│       ├── llm_explainer.py
│       └── chat_interface.py
├── utils/
│   └── system_utils.py
├── data/
│   ├── samples/
│   │   ├── boston_housing.csv
│   │   └── iris.csv
│   ├── exports/
│   └── chats/
├── logs/
├── config/
├── app.py
├── requirements.txt
├── deployment_instructions.md
├── github_setup_instructions.md
├── render_deployment_instructions.md
└── pycharm_setup_guide.md
```

## インストール方法

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/causalviz_platform.git
cd causalviz_platform

# 仮想環境の作成（推奨）
python -m venv causalviz_env
source causalviz_env/bin/activate  # Linuxの場合
causalviz_env\Scripts\activate     # Windowsの場合

# 依存関係のインストール
pip install -r requirements.txt

# アプリケーションの起動
streamlit run app.py
```

詳細なインストール手順とデプロイメント方法については、[deployment_instructions.md](deployment_instructions.md)を参照してください。

## 開発環境のセットアップ

- [GitHub設定ガイド](github_setup_instructions.md)
- [PyCharm設定ガイド](pycharm_setup_guide.md)
- [Renderデプロイガイド](render_deployment_instructions.md)

## 依存関係

主な依存関係は以下の通りです：

- Streamlit: Webインターフェース
- Pandas/NumPy: データ処理
- Matplotlib/Plotly: データ可視化
- AutoGluon: 機械学習と因果推論
- EconML/CausalML: 因果推論
- OpenAI/Anthropic: LLM連携

すべての依存関係は[requirements.txt](requirements.txt)に記載されています。

## ライセンス

MIT License
