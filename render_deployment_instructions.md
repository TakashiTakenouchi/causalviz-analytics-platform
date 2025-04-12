# Renderデプロイメントガイド - CausalViz Analytics Platform

このガイドでは、CausalViz Analytics PlatformをRenderにデプロイする方法について詳しく説明します。Renderは、Webアプリケーションを簡単にデプロイできるクラウドプラットフォームです。

## 目次

1. [Renderとは](#renderとは)
2. [前提条件](#前提条件)
3. [Renderアカウントのセットアップ](#renderアカウントのセットアップ)
4. [プロジェクトの準備](#プロジェクトの準備)
5. [Renderへのデプロイ手順](#renderへのデプロイ手順)
6. [環境変数の設定](#環境変数の設定)
7. [カスタムドメインの設定](#カスタムドメインの設定)
8. [継続的デプロイメントの設定](#継続的デプロイメントの設定)
9. [パフォーマンスの最適化](#パフォーマンスの最適化)
10. [トラブルシューティング](#トラブルシューティング)

## Renderとは

Renderは、静的サイト、Webアプリケーション、APIなどを簡単にデプロイできるクラウドプラットフォームです。GitHubやGitLabと連携して継続的デプロイメントを実現し、SSL証明書の自動発行、カスタムドメインのサポート、環境変数の管理などの機能を提供します。

Renderの主な特徴：
- シンプルなデプロイプロセス
- 自動的なHTTPS対応
- GitHubとの連携による継続的デプロイメント
- 無料プランの提供（制限あり）
- スケーラブルなリソース管理

## 前提条件

- GitHubアカウントとリポジトリ（[GitHub設定ガイド](github_setup_instructions.md)を参照）
- 完全なCausalViz Analytics Platformのソースコード
- 基本的なコマンドラインの知識

## Renderアカウントのセットアップ

1. [Render公式サイト](https://render.com/)にアクセスします。
2. 「Sign Up」をクリックしてアカウントを作成します。
3. GitHubアカウントでサインアップすることをお勧めします（連携が簡単になります）。
4. メールアドレスを確認し、アカウントを有効化します。
5. 必要に応じて二要素認証（2FA）を設定します。

## プロジェクトの準備

Renderにデプロイする前に、プロジェクトに以下のファイルが含まれていることを確認してください：

### 1. requirements.txt

すべての依存関係が記載されていることを確認します：

```
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.18.0
scikit-learn>=1.3.0
...
```

### 2. runtime.txt（オプション）

特定のPythonバージョンを指定する場合は、`runtime.txt`ファイルを作成します：

```
python-3.10.0
```

### 3. render.yaml（オプション）

複数のサービスを定義する場合や、より詳細な設定が必要な場合は、`render.yaml`ファイルを作成します：

```yaml
services:
  - type: web
    name: causalviz-platform
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
      - key: OPENAI_API_KEY
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
    autoDeploy: true
```

## Renderへのデプロイ手順

### 1. Webサービスの作成

1. Renderダッシュボードにログインします。
2. 「New +」ボタンをクリックし、「Web Service」を選択します。
3. GitHubアカウントを連携していない場合は、連携を行います。
4. デプロイするリポジトリを選択します。

### 2. サービス設定

以下の設定を行います：

- **Name**: causalviz-platform（任意の名前）
- **Environment**: Python
- **Region**: 最も近いリージョンを選択（例：Frankfurt, EU Central）
- **Branch**: main（または任意のブランチ）
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false`

### 3. プランの選択

- 開発やテスト目的の場合は「Free」プランを選択できます。
- 本番環境では、必要なリソースに応じて適切な有料プランを選択してください。

### 4. 詳細設定（Advanced）

必要に応じて以下の設定を行います：

- **Auto-Deploy**: GitHubリポジトリに変更がプッシュされたときに自動的にデプロイするかどうかを選択します。
- **Preview Branches**: プレビュー環境を作成するブランチパターンを指定します（例：`preview/*`）。

### 5. デプロイの開始

「Create Web Service」ボタンをクリックしてデプロイを開始します。デプロイには数分かかる場合があります。

## 環境変数の設定

APIキーなどの機密情報は、環境変数として設定することをお勧めします：

1. Renderダッシュボードでデプロイしたサービスを選択します。
2. 「Environment」タブをクリックします。
3. 「Add Environment Variable」をクリックします。
4. 以下の環境変数を追加します：
   - `OPENAI_API_KEY`: OpenAI APIキー
   - `ANTHROPIC_API_KEY`: Anthropic APIキー
   - `STREAMLIT_SERVER_ENABLE_STATIC_SERVING`: true
   - `STREAMLIT_SERVER_BASE_URL`: カスタムパスが必要な場合に設定（例：`/causalviz`）
5. 「Save Changes」をクリックします。

環境変数を更新すると、サービスは自動的に再デプロイされます。

## カスタムドメインの設定

独自のドメインを使用する場合は、以下の手順で設定します：

1. Renderダッシュボードでデプロイしたサービスを選択します。
2. 「Settings」タブをクリックします。
3. 「Custom Domain」セクションで「Add Custom Domain」をクリックします。
4. ドメイン名を入力します（例：`causalviz.yourdomain.com`）。
5. 表示されるDNS設定手順に従って、ドメインのDNSレコードを更新します。
6. DNSの変更が反映されるまで待ちます（最大48時間）。

Renderは自動的にSSL証明書を発行し、HTTPSを有効にします。

## 継続的デプロイメントの設定

GitHubリポジトリとの継続的デプロイメントを設定するには：

1. Renderダッシュボードでデプロイしたサービスを選択します。
2. 「Settings」タブをクリックします。
3. 「Build & Deploy」セクションで「Auto-Deploy」を有効にします。
4. 必要に応じて「Branch」を変更します（デフォルトは`main`）。

これにより、指定したブランチに変更がプッシュされるたびに、Renderは自動的に新しいバージョンをデプロイします。

## パフォーマンスの最適化

Renderでのパフォーマンスを最適化するためのヒント：

### 1. ビルド時間の短縮

- 不要なファイルを`.dockerignore`または`.renderignore`ファイルに追加します。
- 大きなデータファイルはGit LFSを使用するか、デプロイ後にダウンロードするようにします。

### 2. メモリ使用量の最適化

- アプリケーション内でメモリ使用量をモニタリングし、必要に応じてクリーンアップします。
- 大きなデータセットを扱う場合は、データのストリーミング処理を検討します。

### 3. コールドスタートの改善

- 無料プランではサービスがアイドル状態になると停止します。最初のリクエストが遅くなる場合があります。
- 有料プランでは「Always On」オプションを有効にすることで、コールドスタートを回避できます。

### 4. キャッシュの活用

- 静的アセットにはキャッシュヘッダーを設定します。
- 計算コストの高い処理結果をキャッシュします。

## トラブルシューティング

### デプロイエラー

**問題**: デプロイが失敗する。

**解決策**:
1. Renderダッシュボードでデプロイログを確認します。
2. 依存関係のインストールエラーがある場合は、`requirements.txt`を確認し、互換性のある依存関係バージョンを指定します。
3. メモリ不足エラーの場合は、より大きなインスタンスタイプにアップグレードします。

### アプリケーションエラー

**問題**: アプリケーションは起動するが、エラーが発生する。

**解決策**:
1. ログを確認して、エラーの原因を特定します。
2. 環境変数が正しく設定されているか確認します。
3. ローカル環境と本番環境の違いを確認します（ファイルパス、依存関係など）。

### パフォーマンスの問題

**問題**: アプリケーションの応答が遅い。

**解決策**:
1. より大きなインスタンスタイプにアップグレードします。
2. アプリケーションのパフォーマンスを最適化します（データのプリロード、キャッシュの活用など）。
3. 「Always On」オプションを有効にして、コールドスタートを回避します。

### ディスク容量の問題

**問題**: ディスク容量が不足している。

**解決策**:
1. 不要なファイルやログを定期的にクリーンアップします。
2. 大きなデータファイルは外部ストレージ（S3など）に保存します。
3. より大きなディスク容量を持つプランにアップグレードします。

## Renderサポートの利用

問題が解決しない場合は、Renderのサポートを利用できます：

1. [Renderドキュメント](https://render.com/docs)を参照します。
2. [Renderコミュニティフォーラム](https://community.render.com/)で質問します。
3. サポートチケットを作成します（有料プランのみ）。

---

このガイドに従うことで、CausalViz Analytics PlatformをRenderに簡単にデプロイできます。デプロイ後は、アプリケーションのパフォーマンスをモニタリングし、必要に応じて設定を調整してください。
