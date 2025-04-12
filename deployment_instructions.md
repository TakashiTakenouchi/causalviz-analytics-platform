# CausalViz Analytics Platform デプロイメントガイド

このドキュメントでは、CausalViz Analytics Platformのデプロイメント手順について説明します。ローカル環境でのセットアップから本番環境へのデプロイまで、段階的に解説します。

## 目次

1. [システム要件](#システム要件)
2. [インストール手順](#インストール手順)
3. [環境設定](#環境設定)
4. [アプリケーションの起動方法](#アプリケーションの起動方法)
5. [本番環境へのデプロイ](#本番環境へのデプロイ)
6. [トラブルシューティング](#トラブルシューティング)

## システム要件

CausalViz Analytics Platformを実行するには、以下のシステム要件を満たす必要があります：

- **オペレーティングシステム**: Windows 10/11、macOS 10.15以降、Ubuntu 20.04/22.04以降
- **Python**: 3.8以上（3.10推奨）
- **メモリ**: 最低8GB（16GB以上推奨）
- **ストレージ**: 最低5GB（データセットのサイズによって変動）
- **プロセッサ**: マルチコアCPU（AutoGluonの実行には4コア以上推奨）
- **インターネット接続**: LLM APIへの接続に必要

## インストール手順

### 1. Pythonのインストール

まず、Python 3.8以上がインストールされていることを確認してください。

```bash
python --version
```

Pythonがインストールされていない場合は、[Python公式サイト](https://www.python.org/downloads/)からダウンロードしてインストールしてください。

### 2. 仮想環境の作成（推奨）

プロジェクト用の仮想環境を作成することをお勧めします：

```bash
# venvを使用する場合
python -m venv causalviz_env
source causalviz_env/bin/activate  # Linuxの場合
causalviz_env\Scripts\activate     # Windowsの場合

# condaを使用する場合
conda create -n causalviz_env python=3.10
conda activate causalviz_env
```

### 3. リポジトリのクローン

GitHubリポジトリからプロジェクトをクローンします：

```bash
git clone https://github.com/yourusername/causalviz_platform.git
cd causalviz_platform
```

### 4. 依存関係のインストール

必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

メモリやディスク容量に制限がある場合は、コア依存関係のみをインストールすることもできます：

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn
```

AutoGluonとLLM連携機能を使用する場合は、追加の依存関係をインストールします：

```bash
pip install autogluon openai anthropic
```

## 環境設定

### 1. 設定ファイルの作成

初回起動時に設定ファイルが自動的に作成されますが、手動で作成することもできます：

```bash
mkdir -p config
touch config/settings.json
```

`settings.json`の基本構成：

```json
{
  "theme": "light",
  "language": "ja",
  "max_memory_percent": 80,
  "auto_cleanup": true,
  "show_memory_warning": true,
  "default_chart_height": 500,
  "default_chart_width": 800,
  "default_chart_template": "plotly_white",
  "default_color_scheme": "viridis",
  "api_keys": {
    "openai": "",
    "anthropic": ""
  }
}
```

### 2. APIキーの設定

AIアシスタント機能を使用するには、OpenAIまたはAnthropic（Claude）のAPIキーが必要です。

APIキーは以下の方法で設定できます：

1. アプリケーション内の「設定」ページから設定
2. 設定ファイル（`config/settings.json`）に直接記述
3. 環境変数として設定：

```bash
# OpenAI APIキー
export OPENAI_API_KEY=your_openai_api_key

# Anthropic APIキー
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. データディレクトリの準備

必要なディレクトリ構造を作成します：

```bash
mkdir -p data/samples data/exports data/chats logs
```

サンプルデータを`data/samples`ディレクトリに配置することで、アプリケーション起動時にサンプルデータを利用できます。

## アプリケーションの起動方法

### ローカル環境での起動

以下のコマンドでアプリケーションを起動します：

```bash
cd causalviz_platform
streamlit run app.py
```

デフォルトでは、アプリケーションは`http://localhost:8501`でアクセス可能になります。

### カスタムポートでの起動

特定のポートでアプリケーションを起動する場合：

```bash
streamlit run app.py --server.port 8080
```

### サーバーモードでの起動

本番環境に近い設定で起動する場合：

```bash
streamlit run app.py --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

## 本番環境へのデプロイ

### Streamlit Cloudへのデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud)にアクセスし、アカウントを作成またはログインします。
2. 「New app」をクリックし、GitHubリポジトリを連携します。
3. リポジトリ、ブランチ、メインPythonファイル（`app.py`）を指定します。
4. 必要に応じて環境変数（APIキーなど）を設定します。
5. 「Deploy」をクリックしてデプロイを開始します。

### Renderへのデプロイ

1. [Render](https://render.com/)にアカウントを作成またはログインします。
2. 「New Web Service」を選択します。
3. GitHubリポジトリを連携します。
4. 以下の設定を行います：
   - **Name**: causalviz-platform（任意）
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
5. 必要に応じて環境変数（APIキーなど）を設定します。
6. 「Create Web Service」をクリックしてデプロイを開始します。

### Dockerを使用したデプロイ

Dockerを使用してコンテナ化する場合は、以下の`Dockerfile`を使用できます：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

ビルドとデプロイ：

```bash
docker build -t causalviz-platform .
docker run -p 8501:8501 causalviz-platform
```

## トラブルシューティング

### 一般的な問題と解決策

#### 依存関係のインストールエラー

**問題**: `pip install -r requirements.txt`実行時にエラーが発生する。

**解決策**:
1. Pythonのバージョンが3.8以上であることを確認してください。
2. 個別にパッケージをインストールしてみてください：
   ```bash
   pip install streamlit pandas numpy matplotlib plotly scikit-learn
   pip install autogluon
   pip install openai anthropic
   ```
3. メモリ不足の場合は、仮想メモリを増やすか、より多くのRAMを搭載したマシンを使用してください。

#### アプリケーションが起動しない

**問題**: `streamlit run app.py`を実行してもアプリケーションが起動しない。

**解決策**:
1. Streamlitが正しくインストールされているか確認してください：
   ```bash
   pip install streamlit
   ```
2. ファイルパスが正しいか確認してください。プロジェクトのルートディレクトリで実行しているか確認してください。
3. ポートが他のアプリケーションで使用されていないか確認してください。別のポートを指定してみてください：
   ```bash
   streamlit run app.py --server.port 8080
   ```

#### メモリエラー

**問題**: アプリケーション実行中に「メモリ不足」エラーが発生する。

**解決策**:
1. 設定ページでメモリ使用率の上限を調整してください。
2. 大きなデータセットを扱う場合は、データのサブセットを使用するか、データの前処理を行ってサイズを削減してください。
3. AutoGluonの使用時は、モデルの複雑さを下げるパラメータを設定してください。

#### LLM APIエラー

**問題**: AIアシスタント機能が動作しない。

**解決策**:
1. APIキーが正しく設定されているか確認してください。
2. インターネット接続が利用可能か確認してください。
3. APIの利用制限に達していないか確認してください。
4. アプリケーションの設定ページでAPIキーを再設定してみてください。

### ログの確認

問題が発生した場合は、ログファイルを確認してください：

```bash
cat logs/app_*.log
```

より詳細なログを有効にするには、`app.py`の`setup_logging`関数の呼び出し部分を編集し、ログレベルを`DEBUG`に設定してください。

### サポートの利用

さらに支援が必要な場合は、以下の方法でサポートを受けることができます：

1. GitHubリポジトリのIssueを作成する
2. プロジェクトのメンテナーに連絡する
3. コミュニティフォーラムで質問する

---

このデプロイメントガイドが、CausalViz Analytics Platformの設定と実行に役立つことを願っています。さらに質問がある場合は、お気軽にお問い合わせください。
