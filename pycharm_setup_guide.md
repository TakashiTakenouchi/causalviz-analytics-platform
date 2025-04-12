# PyCharm開発環境セットアップガイド - CausalViz Analytics Platform

このガイドでは、CausalViz Analytics Platformの開発環境をPyCharmでセットアップする方法について詳しく説明します。PyCharmは、Pythonプロジェクト向けの強力な統合開発環境（IDE）です。

## 目次

1. [PyCharmのインストール](#pycharmのインストール)
2. [プロジェクトのセットアップ](#プロジェクトのセットアップ)
3. [仮想環境の設定](#仮想環境の設定)
4. [依存関係のインストール](#依存関係のインストール)
5. [実行構成の設定](#実行構成の設定)
6. [デバッグ設定](#デバッグ設定)
7. [コード品質ツールの設定](#コード品質ツールの設定)
8. [バージョン管理の設定](#バージョン管理の設定)
9. [便利な機能とショートカット](#便利な機能とショートカット)
10. [トラブルシューティング](#トラブルシューティング)

## PyCharmのインストール

### 1. PyCharmのダウンロード

[JetBrains公式サイト](https://www.jetbrains.com/pycharm/download/)から、PyCharmをダウンロードします。

- **Community Edition**: 無料版。基本的な機能が含まれています。
- **Professional Edition**: 有料版。Webフレームワークのサポート、リモート開発、データベースツールなどの追加機能があります。

学生や教育機関、オープンソースプロジェクトの開発者は、Professional Editionを無料で利用できる場合があります。

### 2. インストール手順

#### Windows
1. ダウンロードしたインストーラー（.exe）を実行します。
2. インストールウィザードの指示に従います。
3. 必要に応じて、「Create Desktop Shortcut」や「Update PATH variable」などのオプションを選択します。

#### macOS
1. ダウンロードしたディスクイメージ（.dmg）を開きます。
2. PyCharmアイコンをApplicationsフォルダにドラッグします。
3. Launchpadから起動します。

#### Linux
1. ダウンロードしたアーカイブ（.tar.gz）を展開します：
   ```bash
   tar -xzf pycharm-*.tar.gz -C /opt/
   ```
2. 展開したディレクトリ内の`bin/pycharm.sh`を実行します：
   ```bash
   cd /opt/pycharm-*/bin
   ./pycharm.sh
   ```

## プロジェクトのセットアップ

### 1. 既存のプロジェクトを開く

GitHubからクローンしたプロジェクトを開く場合：

1. PyCharmを起動します。
2. 「Open」をクリックします。
3. クローンしたプロジェクトのディレクトリを選択します。
4. 「Trust Project」をクリックします（セキュリティプロンプトが表示された場合）。

### 2. 新しいプロジェクトの作成

新しくプロジェクトを作成する場合：

1. PyCharmを起動します。
2. 「New Project」をクリックします。
3. プロジェクトの場所とPythonインタープリターを選択します。
4. 「Create」をクリックします。

### 3. プロジェクト構造の確認

プロジェクトが正しく開かれたら、左側のプロジェクトツリーで以下のディレクトリ構造が表示されていることを確認します：

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
│   ├── causal_inference/
│   └── llm_explainer/
├── utils/
├── data/
│   ├── samples/
│   ├── exports/
│   └── chats/
├── docs/
├── tests/
├── logs/
├── config/
├── app.py
└── requirements.txt
```

## 仮想環境の設定

### 1. 新しい仮想環境の作成

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Project: causalviz_platform」→「Python Interpreter」を選択します。
3. 歯車アイコンをクリックし、「Add...」を選択します。
4. 「Virtualenv Environment」を選択します。
5. 「New environment」を選択し、以下を設定します：
   - **Location**: プロジェクトディレクトリ内の`venv`フォルダ（デフォルト）
   - **Base interpreter**: Python 3.10（推奨）
6. 「OK」をクリックします。

### 2. 既存の仮想環境の使用

既に仮想環境がある場合：

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Project: causalviz_platform」→「Python Interpreter」を選択します。
3. 歯車アイコンをクリックし、「Add...」を選択します。
4. 「Existing environment」を選択します。
5. 既存の仮想環境のインタープリターパスを指定します：
   - Windows: `<venv_path>\Scripts\python.exe`
   - macOS/Linux: `<venv_path>/bin/python`
6. 「OK」をクリックします。

## 依存関係のインストール

### 1. requirements.txtからのインストール

1. プロジェクトツリーで`requirements.txt`ファイルを右クリックします。
2. 「Install All Packages」を選択します。
3. インストールの進行状況がPyCharmの下部に表示されます。

または、ターミナルから直接インストールすることもできます：

1. PyCharm内でターミナルを開きます（「View」→「Tool Windows」→「Terminal」）。
2. 以下のコマンドを実行します：
   ```bash
   pip install -r requirements.txt
   ```

### 2. 個別パッケージのインストール

メモリやディスク容量に制限がある場合は、コア依存関係のみをインストールすることもできます：

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn
```

## 実行構成の設定

### 1. Streamlitアプリケーションの実行構成

1. 「Run」→「Edit Configurations...」を選択します。
2. 「+」ボタンをクリックし、「Python」を選択します。
3. 以下の設定を行います：
   - **Name**: `Streamlit App`
   - **Script path**: `-m` を選択
   - **Parameters**: `streamlit run app.py`
   - **Python interpreter**: プロジェクトの仮想環境
   - **Working directory**: プロジェクトのルートディレクトリ
4. 「OK」をクリックします。

これで、実行ボタン（緑の三角形）をクリックするだけでStreamlitアプリケーションを起動できるようになります。

### 2. カスタムポートでの実行構成

デフォルトとは異なるポートでアプリケーションを実行する場合：

1. 「Run」→「Edit Configurations...」を選択します。
2. 既存の「Streamlit App」構成を選択するか、新しい構成を作成します。
3. **Parameters**を以下のように変更します：
   ```
   streamlit run app.py --server.port 8080
   ```
4. 「OK」をクリックします。

### 3. 環境変数の設定

APIキーなどの環境変数を設定する場合：

1. 「Run」→「Edit Configurations...」を選択します。
2. 実行構成を選択します。
3. 「Environment variables」フィールドをクリックします。
4. 「+」ボタンをクリックし、以下の環境変数を追加します：
   - **OPENAI_API_KEY**: OpenAI APIキー
   - **ANTHROPIC_API_KEY**: Anthropic APIキー
5. 「OK」をクリックします。

## デバッグ設定

### 1. ブレークポイントの設定

1. デバッグしたいコード行の左側の余白をクリックして、ブレークポイントを設定します。
2. 赤い丸がその行に表示されます。

### 2. デバッグモードでの実行

1. 実行ボタンの横にあるデバッグボタン（虫のアイコン）をクリックします。
2. アプリケーションがデバッグモードで起動します。
3. ブレークポイントに到達すると、実行が一時停止します。

### 3. デバッグツールの使用

デバッグ中に以下のツールを使用できます：

- **Step Over (F8)**: 現在の行を実行し、次の行に進みます。
- **Step Into (F7)**: 関数呼び出しの中に入ります。
- **Step Out (Shift+F8)**: 現在の関数から抜けます。
- **Resume Program (F9)**: 次のブレークポイントまで実行を再開します。
- **Variables**: 現在のスコープの変数値を表示します。
- **Watches**: 特定の式の値を監視します。

## コード品質ツールの設定

### 1. コードインスペクション

PyCharmは自動的にコードを分析し、潜在的な問題を検出します。警告は黄色の波線、エラーは赤い波線で表示されます。

### 2. PEP 8スタイルガイド

PEP 8に準拠したコードを書くために：

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Editor」→「Inspections」→「Python」→「PEP 8 coding style violation」を選択します。
3. 必要に応じて設定を調整します。

### 3. 外部ツールの統合

#### Flake8

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Tools」→「External Tools」を選択します。
3. 「+」ボタンをクリックし、以下の設定を行います：
   - **Name**: `Flake8`
   - **Program**: `$PyInterpreterDirectory$/python`
   - **Arguments**: `-m flake8 $FileDir$`
   - **Working directory**: `$ProjectFileDir$`
4. 「OK」をクリックします。

#### Black

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Tools」→「External Tools」を選択します。
3. 「+」ボタンをクリックし、以下の設定を行います：
   - **Name**: `Black`
   - **Program**: `$PyInterpreterDirectory$/python`
   - **Arguments**: `-m black $FileDir$`
   - **Working directory**: `$ProjectFileDir$`
4. 「OK」をクリックします。

## バージョン管理の設定

### 1. Gitの設定

1. 「VCS」→「Enable Version Control Integration...」を選択します（まだ有効になっていない場合）。
2. 「Git」を選択し、「OK」をクリックします。

### 2. GitHubとの連携

1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Version Control」→「GitHub」を選択します。
3. 「+」ボタンをクリックし、GitHubアカウントを追加します。
4. 認証方法を選択し、指示に従います。

### 3. 変更のコミットとプッシュ

1. 変更したファイルは「Version Control」ツールウィンドウに表示されます。
2. コミットするファイルを選択し、「Commit」ボタンをクリックします。
3. コミットメッセージを入力し、「Commit」または「Commit and Push...」をクリックします。

## 便利な機能とショートカット

### 1. コードナビゲーション

- **Ctrl+クリック**（Windows/Linux）または**Cmd+クリック**（macOS）: 定義にジャンプ
- **Ctrl+B**（Windows/Linux）または**Cmd+B**（macOS）: 定義にジャンプ
- **Alt+F7**: 使用箇所の検索
- **Ctrl+F12**（Windows/Linux）または**Cmd+F12**（macOS）: ファイル構造のポップアップ表示

### 2. コード編集

- **Ctrl+Space**: コード補完
- **Ctrl+P**（Windows/Linux）または**Cmd+P**（macOS）: パラメータ情報の表示
- **Alt+Enter**: クイックフィックスの表示
- **Ctrl+Alt+L**（Windows/Linux）または**Cmd+Option+L**（macOS）: コードの整形

### 3. リファクタリング

- **Shift+F6**: 名前の変更
- **Ctrl+Alt+M**（Windows/Linux）または**Cmd+Option+M**（macOS）: メソッドの抽出
- **Ctrl+Alt+V**（Windows/Linux）または**Cmd+Option+V**（macOS）: 変数の抽出

### 4. 検索と置換

- **Ctrl+F**（Windows/Linux）または**Cmd+F**（macOS）: ファイル内検索
- **Ctrl+R**（Windows/Linux）または**Cmd+R**（macOS）: ファイル内置換
- **Ctrl+Shift+F**（Windows/Linux）または**Cmd+Shift+F**（macOS）: プロジェクト内検索
- **Ctrl+Shift+R**（Windows/Linux）または**Cmd+Shift+R**（macOS）: プロジェクト内置換

## トラブルシューティング

### 1. インタープリターの問題

**問題**: Pythonインタープリターが見つからない、または正しく設定されていない。

**解決策**:
1. 「File」→「Settings」（Windows/Linux）または「PyCharm」→「Preferences」（macOS）を開きます。
2. 「Project: causalviz_platform」→「Python Interpreter」を選択します。
3. 正しいインタープリターが選択されていることを確認します。
4. インタープリターが表示されない場合は、「Add Interpreter」をクリックして新しく追加します。

### 2. 依存関係のインストールエラー

**問題**: パッケージのインストール中にエラーが発生する。

**解決策**:
1. PyCharm内のターミナルを開きます。
2. 仮想環境が有効になっていることを確認します（プロンプトに`(venv)`などが表示されます）。
3. pip自体を更新します：
   ```bash
   pip install --upgrade pip
   ```
4. 個別にパッケージをインストールしてみます：
   ```bash
   pip install streamlit
   pip install pandas
   # 他のパッケージも同様に
   ```

### 3. 実行構成の問題

**問題**: アプリケーションが起動しない、または正しく実行されない。

**解決策**:
1. 実行構成の設定を確認します。
2. 作業ディレクトリが正しく設定されていることを確認します。
3. ターミナルから直接実行してみます：
   ```bash
   streamlit run app.py
   ```

### 4. インデックス再構築

**問題**: コード補完が機能しない、または遅い。

**解決策**:
1. 「File」→「Invalidate Caches / Restart...」を選択します。
2. 「Invalidate and Restart」をクリックします。
3. PyCharmが再起動し、インデックスが再構築されます。

### 5. メモリ設定の調整

**問題**: PyCharmが遅い、またはメモリ不足エラーが発生する。

**解決策**:
1. 「Help」→「Edit Custom VM Options...」を選択します。
2. メモリ割り当てを増やします（例：`-Xmx2048m`を`-Xmx4096m`に変更）。
3. PyCharmを再起動します。

---

このガイドに従うことで、CausalViz Analytics Platformの開発環境をPyCharmで効率的にセットアップできます。PyCharmの強力な機能を活用して、コーディング、デバッグ、バージョン管理を効率的に行いましょう。
