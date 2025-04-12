# GitHub リポジトリセットアップガイド - CausalViz Analytics Platform

このガイドでは、CausalViz Analytics Platformのソースコードを管理するためのGitHubリポジトリのセットアップ方法について説明します。

## 目次

1. [前提条件](#前提条件)
2. [GitHubアカウントの作成](#githubアカウントの作成)
3. [新しいリポジトリの作成](#新しいリポジトリの作成)
4. [ローカルリポジトリのセットアップ](#ローカルリポジトリのセットアップ)
5. [ブランチ戦略](#ブランチ戦略)
6. [コラボレーション設定](#コラボレーション設定)
7. [CI/CDの設定](#cicdの設定)
8. [リポジトリの管理とメンテナンス](#リポジトリの管理とメンテナンス)

## 前提条件

- Git（バージョン2.20以上）がインストールされていること
- コマンドラインの基本的な知識
- インターネット接続

Gitがインストールされているか確認するには、以下のコマンドを実行します：

```bash
git --version
```

インストールされていない場合は、[Git公式サイト](https://git-scm.com/downloads)からダウンロードしてインストールしてください。

## GitHubアカウントの作成

1. [GitHub](https://github.com/)にアクセスします。
2. 「Sign up」をクリックし、指示に従ってアカウントを作成します。
3. メールアドレスを確認し、アカウントを有効化します。
4. 二要素認証（2FA）を設定することを強く推奨します（Settings > Password and authentication）。

## 新しいリポジトリの作成

### ウェブインターフェースからの作成

1. GitHubにログインします。
2. 右上の「+」アイコンをクリックし、「New repository」を選択します。
3. リポジトリ名を入力します（例：`causalviz-analytics-platform`）。
4. 説明を追加します（例：「Streamlit + AutoGluon + LLMベースのデータ分析・因果推論プラットフォーム」）。
5. リポジトリの可視性を選択します（Public または Private）。
6. 「Initialize this repository with:」セクションで以下を選択します：
   - `README file`にチェックを入れる
   - `.gitignore`テンプレートとして「Python」を選択
   - ライセンスを選択（例：MIT License）
7. 「Create repository」をクリックします。

### リポジトリの設定

リポジトリが作成されたら、以下の設定を行います：

1. 「Settings」タブをクリックします。
2. 「General」セクションで、必要に応じて以下の設定を行います：
   - リポジトリ名や説明の編集
   - デフォルトブランチの変更（通常は`main`）
   - 機能の有効化/無効化（Issues, Projects, Wiki等）
3. 「Branches」セクションで、ブランチ保護ルールを設定します：
   - 「Add branch protection rule」をクリックします。
   - 「Branch name pattern」に`main`を入力します。
   - 必要に応じて以下の保護設定を有効にします：
     - 「Require pull request reviews before merging」
     - 「Require status checks to pass before merging」
     - 「Require signed commits」

## ローカルリポジトリのセットアップ

### 新しいプロジェクトの場合

1. ローカルマシンで、プロジェクトを保存するディレクトリに移動します。
2. GitHubリポジトリをクローンします：

```bash
git clone https://github.com/yourusername/causalviz-analytics-platform.git
cd causalviz-analytics-platform
```

3. プロジェクトファイルをこのディレクトリにコピーします。
4. ファイルをステージングし、コミットして、プッシュします：

```bash
git add .
git commit -m "Initial commit: Add project files"
git push origin main
```

### 既存のプロジェクトの場合

既にローカルにプロジェクトがある場合：

1. プロジェクトディレクトリに移動します：

```bash
cd /path/to/causalviz_platform
```

2. Gitリポジトリを初期化します：

```bash
git init
```

3. リモートリポジトリを追加します：

```bash
git remote add origin https://github.com/yourusername/causalviz-analytics-platform.git
```

4. ファイルをステージングし、コミットして、プッシュします：

```bash
git add .
git commit -m "Initial commit: Add project files"
git push -u origin main
```

### .gitignoreファイルの設定

プロジェクト用の`.gitignore`ファイルが適切に設定されていることを確認します。以下は推奨される設定です：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
causalviz_env/

# Jupyter Notebook
.ipynb_checkpoints

# Streamlit
.streamlit/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
logs/
data/exports/
data/chats/
config/settings.json
*.log
```

## ブランチ戦略

効果的な開発のために、以下のブランチ戦略を採用することをお勧めします：

### GitFlow ブランチモデル

- `main`: 本番環境用の安定したコード
- `develop`: 開発中のコード、次のリリースの準備
- `feature/*`: 新機能の開発用（例：`feature/causal-graph-visualization`）
- `bugfix/*`: バグ修正用
- `release/*`: リリース準備用
- `hotfix/*`: 本番環境の緊急修正用

### 基本的なワークフロー

1. 新機能の開発を始める場合：

```bash
git checkout develop
git pull
git checkout -b feature/new-feature-name
# 開発作業を行う
git add .
git commit -m "Add new feature: description"
git push origin feature/new-feature-name
```

2. GitHub上でPull Requestを作成し、`develop`ブランチにマージします。
3. リリース準備ができたら、`release`ブランチを作成し、テスト後に`main`と`develop`にマージします。

## コラボレーション設定

### コラボレーターの追加

1. リポジトリの「Settings」タブをクリックします。
2. 左側のメニューから「Collaborators」を選択します。
3. 「Add people」をクリックし、ユーザー名、フルネーム、またはメールアドレスを入力します。
4. 適切な権限レベルを選択します：
   - Read: 読み取り専用
   - Triage: イシューとプルリクエストの管理
   - Write: リポジトリへのコード変更
   - Maintain: リポジトリの管理（保護されたブランチへのプッシュを除く）
   - Admin: フル管理権限

### イシューテンプレートの設定

1. リポジトリのルートに`.github/ISSUE_TEMPLATE`ディレクトリを作成します：

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

2. バグレポート用のテンプレートを作成します：

```markdown
<!-- .github/ISSUE_TEMPLATE/bug_report.md -->
---
name: バグレポート
about: アプリケーションの問題を報告する
title: '[BUG] '
labels: bug
assignees: ''
---

## バグの説明
明確かつ簡潔にバグを説明してください。

## 再現手順
バグを再現する手順：
1. '...'に移動
2. '....'をクリック
3. '....'までスクロール
4. エラーを確認

## 期待される動作
何が起こるべきだったのかを明確かつ簡潔に説明してください。

## スクリーンショット
該当する場合、問題を説明するためのスクリーンショットを追加してください。

## 環境
 - OS: [例: Windows 10, macOS 12.0]
 - ブラウザ: [例: Chrome, Safari]
 - アプリケーションバージョン: [例: 1.0.0]
 - Python バージョン: [例: 3.10.0]

## 追加情報
問題に関するその他の情報をここに追加してください。
```

3. 機能リクエスト用のテンプレートも作成します：

```markdown
<!-- .github/ISSUE_TEMPLATE/feature_request.md -->
---
name: 機能リクエスト
about: このプロジェクトのアイデアを提案する
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## 機能リクエストは問題に関連していますか？
問題が何であるかを明確かつ簡潔に説明してください。例：私はいつも[...]のときにフラストレーションを感じています。

## 希望するソリューションの説明
実現したいことを明確かつ簡潔に説明してください。

## 検討した代替案の説明
検討した代替ソリューションや機能を明確かつ簡潔に説明してください。

## 追加情報
機能リクエストに関するその他の情報やスクリーンショットをここに追加してください。
```

### プルリクエストテンプレートの設定

リポジトリのルートに`.github/PULL_REQUEST_TEMPLATE.md`ファイルを作成します：

```markdown
## 変更の説明
変更内容を明確かつ簡潔に説明してください。

## 関連するイシュー
このPRが解決するイシューを記載してください（例：#123）。

## 変更の種類
- [ ] バグ修正
- [ ] 新機能
- [ ] パフォーマンス改善
- [ ] コードスタイルの更新（フォーマット、変数名など）
- [ ] リファクタリング（機能変更なし）
- [ ] ドキュメントの更新
- [ ] テストの追加
- [ ] その他（説明してください）：

## チェックリスト
- [ ] コードが自己レビュー済み
- [ ] コメントが追加/更新され、特に理解しにくい部分
- [ ] ドキュメントが更新された
- [ ] 変更によって新しい警告が発生していない
- [ ] テストが追加され、すべてのテストが成功している
- [ ] 依存関係の更新がある場合、requirements.txtが更新されている

## スクリーンショット（該当する場合）
変更を示すスクリーンショットを追加してください。

## 追加情報
レビュアーが理解するのに役立つその他の情報をここに追加してください。
```

## CI/CDの設定

### GitHub Actionsを使用した継続的インテグレーション

1. リポジトリのルートに`.github/workflows`ディレクトリを作成します：

```bash
mkdir -p .github/workflows
```

2. 基本的なCI設定ファイルを作成します：

```yaml
# .github/workflows/ci.yml
name: Python CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
        pip install -r requirements.txt
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Test with pytest
      run: |
        pip install pytest
        pytest
```

### Streamlit Cloudへの自動デプロイ

Streamlit Cloudは、GitHubリポジトリと直接連携できるため、コードがプッシュされると自動的にデプロイされます。Streamlit Cloudの設定については、[デプロイメントガイド](/deployment_instructions.md)を参照してください。

## リポジトリの管理とメンテナンス

### リリース管理

1. リポジトリの「Releases」セクションで「Draft a new release」をクリックします。
2. タグバージョンを入力します（例：`v1.0.0`）。
3. リリースタイトルを入力します。
4. リリースノートを記入します。
5. 必要に応じてバイナリファイルをアップロードします。
6. 「Publish release」をクリックします。

### セマンティックバージョニング

バージョン番号は`MAJOR.MINOR.PATCH`の形式に従うことをお勧めします：

- MAJOR: 互換性のない変更
- MINOR: 後方互換性のある機能追加
- PATCH: 後方互換性のあるバグ修正

### リポジトリのドキュメント

リポジトリには以下のドキュメントを含めることをお勧めします：

- `README.md`: プロジェクトの概要、インストール手順、基本的な使用方法
- `CONTRIBUTING.md`: 貢献ガイドライン
- `CODE_OF_CONDUCT.md`: コミュニティ行動規範
- `LICENSE`: ライセンス情報
- `CHANGELOG.md`: バージョン履歴と変更内容

### リポジトリの健全性の維持

- 定期的に依存関係を更新する
- 未使用のブランチを削除する
- イシューとプルリクエストを定期的に確認し、対応する
- コードの品質を維持するためにコードレビューを徹底する

---

このガイドに従うことで、CausalViz Analytics Platformのための効率的で管理しやすいGitHubリポジトリをセットアップできます。チームでの開発やオープンソースコミュニティとの協力が容易になり、プロジェクトの長期的な成功に貢献します。
