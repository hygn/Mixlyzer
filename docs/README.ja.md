# Mixlyzer

言語: [English](../README.md) | [한국어](README.ko.md)

## 概要

Mixlyzer は DJ とプロデューサーのためのトラック解析・編集ツールです。曲全体をスキャンして BPM/位相の変化を捉え、キー変調を追跡し、JumpCUE の候補を提示します。インタラクティブなエディタで結果を整え、外部プラットフォームへ書き出せます。

## 主な機能

- **静的ビートグリッド解析** - 曲全体の BPM と位相を推定し、ベースとなるグリッドを生成。  
- **動的ビートグリッド解析** - 複数セグメントに分割して BPM/位相変化を正確にマップ。  
- **動的キー解析** - 時間軸でのキー変調を追跡し、ミックス時のキー整合性を確保。  
- **JumpCUE 解析** - 前後ジャンプに適したキューペアを検出し、スムーズなジャンプを支援。  
- **高速インタラクティブ編集** - セグメントベースのグリッド編集、部分再解析、Undo/Redo で素早く調整。  
- **エクスポート** - 補正済みテンポセグメント、ダウンビート、JumpCUE を含む Rekordbox XML を出力。

## こんな方におすすめ

- テンポ変動のある楽曲を安定してプレイしたい DJ  
- アニソン/同人/ゲーム音楽/ボカロ/VTuber などテンポ・キー変化が多いシーンを扱う DJ  
- トラック再構築前に信頼できる解析データを求めるプロデューサー/リミキサー

## はじめ方

### 方法 A: uv を使う（開発向け推奨）
1. [uv](https://github.com/astral-sh/uv) をインストールします。  
2. リポジトリをクローンし、プロジェクトフォルダへ移動します。  
3. `uv sync` で依存関係と仮想環境を準備します（Python 3.12+）。  
4. `uv run python -m app.main` で Mixlyzer を起動します。

#### ネイティブオーディオエンジンのビルド（任意）
ビルドしない場合は Python エンジンで再生します。ネイティブエンジンは Windows で開発・テストしており、Linux と macOS のサポートは実験的です（ビルドはできますが、まだそれらのシステムでテストしていません）。`audioengine/native/bin` にビルドするには:
1. git と [Rust](https://rustup.rs)、C コンパイラをインストールします。
   - **Windows**: [llvm-mingw](https://github.com/mstorsjo/llvm-mingw)（Visual Studio 不要。`bin` フォルダを `PATH` に入れるか、`LLVM_MINGW` でフォルダを指定）と Rust ツールチェーン `rustup toolchain install stable-x86_64-pc-windows-gnullvm`。
   - **Linux**: gcc または clang、ALSA と JACK のヘッダー（`libasound2-dev libjack-jackd2-dev` / `alsa-lib-devel pipewire-jack-audio-connection-kit-devel`）。
   - **macOS**: Xcode のコマンドラインツール。
2. `uv sync --group native` の後、`uv run --group native python build/audio_engine.py` を実行します（VOLK をソースからビルドしてからエンジンをビルドします）。再ビルドするときは先に Mixlyzer を終了してください。実行中のアプリがエンジンのファイルを使用しています。

設定 > 再生でエンジン（**Native 優先**: 使えないときは Python にフォールバック、または **Python**。次回の起動から適用）、出力デバイス、エンジンごとのバッファサイズを選べます。Native エンジンのオーディオ API は **WASAPI**（Windows: 低レイテンシーの共有ストリームまたは排他モード。排他モードではサンプリングレートも指定でき、次に読み込む曲から適用）、**WASAPI (cpal)**、**ALSA** / **JACK**（Linux）、**CoreAudio**（macOS）から選べます。バッファメーターの下に実測の応答（操作から音まで）とデバイスのレイテンシーが表示されます。Native エンジンの優先度（既定: 高。Windows: プロセスの優先度クラス、Linux: スレッドのリアルタイムスケジューリングまたは nice、ユーザー権限でできなければ rtkit 経由、macOS: リアルタイムスレッド）と電力スロットリング（既定: オフ。Windows: EcoQoS、Linux: タイマースラックと uclamp、macOS: App Nap とタイマー合体）もここで設定でき、実際の状態はメーターの下に表示されます。

### 方法 B: 配布バイナリを使う
1. Releases ページから最新ビルドを取得します。  
2. 任意の場所へ展開します。  
3. 実行ファイルを起動します（例: Windows は `Mixlyzer.exe`）。  
4. 初回起動時にライブラリパスを設定し、そのまま解析・編集・書き出しを開始できます。  

## Typical Workflow

1. **トラックを読み込む** - ファイルをドラッグするか、ライブラリから選択します。  
2. **自動解析に任せる** - テンポセグメント、キー、JumpCUE が自動検出されます。  
3. **グリッドを整える** - Beatgrid エディタで BPM 調整、セグメント移動、部分再解析を実施します。  
4. **書き出す** - 編集結果を保存し、Rekordbox XML へ書き出してブースに持ち込みます。  
