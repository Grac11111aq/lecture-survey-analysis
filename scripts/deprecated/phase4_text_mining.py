#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小学校出前授業アンケート Phase 4: テキストマイニング分析
============================================

実施内容:
- 日本語テキストの前処理
- 頻度分析（単語頻度、TF-IDF）
- n-gram分析
- 感情分析（ポジティブ/ネガティブ）
- テキスト特徴量と理解度の関連分析

Author: Claude Code Analysis
Date: 2025-05-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
from datetime import datetime
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

class Phase4TextMiner:
    """Phase 4: テキストマイニング分析クラス"""
    
    def __init__(self, data_dir="data/analysis"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.before_df = None
        self.after_df = None
        self.comment_df = None
        self.paired_data = None
        
        # 日本語テキスト処理用の設定
        self.stop_words = self.create_stop_words()
        self.positive_words = self.create_positive_words()
        self.negative_words = self.create_negative_words()
        
    def load_data(self):
        """データの読み込み"""
        try:
            self.before_df = pd.read_csv(self.data_dir / "before_excel_compliant.csv")
            self.after_df = pd.read_csv(self.data_dir / "after_excel_compliant.csv")
            self.comment_df = pd.read_csv(self.data_dir / "comment.csv")
            
            print("✓ データ読み込み完了")
            print(f"  - 授業前: {len(self.before_df)} 行")
            print(f"  - 授業後: {len(self.after_df)} 行")
            print(f"  - 感想文: {len(self.comment_df)} 行")
            
            # ペアリング
            self.create_paired_dataset()
            
            # テキストデータの整理
            self.organize_text_data()
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def create_paired_dataset(self):
        """前後データのペアリング"""
        # Page_IDで結合
        before_unique = self.before_df.drop_duplicates(subset=['Page_ID'])
        after_unique = self.after_df.drop_duplicates(subset=['Page_ID'])
        
        self.paired_data = pd.merge(
            before_unique, 
            after_unique, 
            on='Page_ID', 
            suffixes=('_before', '_after')
        )
        
        print(f"✓ ペアリング完了: {len(self.paired_data)} ペア")
    
    def organize_text_data(self):
        """テキストデータの整理"""
        # 分析対象テキストカラムの特定
        self.text_columns = {
            'Q2_before': 'Q2_MisoSalty_Reason',  # みそ汁の理由（授業前）
            'Q2_after': 'Q2_MisoSaltyReason',    # みそ汁の理由（授業後）
            'Q5_after': 'Q5_NewLearningsRating', # 新しい学び（授業後のみ）
            'comments': None  # 感想文は別処理
        }
        
        # 感想文データの処理
        if not self.comment_df.empty and 'comment' in self.comment_df.columns:
            # 感想文をPage_IDごとに統合
            comment_grouped = self.comment_df.groupby('page-ID')['comment'].apply(
                lambda x: ' '.join(str(comment) for comment in x if pd.notna(comment))
            ).to_dict()
            
            # paired_dataに感想文を追加
            self.paired_data['comments'] = self.paired_data['Page_ID'].map(comment_grouped)
        
        print("✓ テキストデータ整理完了")
    
    def create_stop_words(self):
        """日本語ストップワードの作成"""
        stop_words = set([
            'の', 'に', 'を', 'は', 'が', 'と', 'で', 'て', 'だ', 'である', 'です', 'ます',
            'した', 'する', 'され', 'ある', 'いる', 'なる', 'れる', 'られる', 'せる',
            'その', 'この', 'それ', 'これ', 'あの', 'あれ', 'どの', 'どれ', 'から', 'まで',
            'より', 'など', 'として', 'という', 'について', 'において', 'による', 'により',
            'こと', 'もの', 'ところ', 'とき', 'とても', 'すごく', 'ちょっと', '少し',
            'たくさん', 'いろいろ', 'みんな', 'みなさん', 'ありがとう', 'ございました',
            '思う', '思った', '感じる', '感じた', '見る', '見た', '聞く', '聞いた',
            '年', '月', '日', '時', '分', '秒', '人', '名', '個', '回', '度', '番'
        ])
        return stop_words
    
    def create_positive_words(self):
        """ポジティブ語彙の作成"""
        positive_words = set([
            'すごい', 'すばらしい', 'きれい', '美しい', '楽しい', '面白い', 'おもしろい',
            '好き', '良い', 'よい', 'いい', '素晴らしい', 'わくわく', '嬉しい', 'うれしい',
            '感動', '驚き', 'びっくり', '発見', '勉強', '学習', '理解', 'わかる', '分かる',
            '印象', '興味', '関心', '魅力', '感謝', 'ありがとう', '楽しかった', '面白かった',
            'きれいでした', 'すごかった', '良かった', 'よかった', 'いろいろ', 'たくさん',
            '初めて', 'はじめて', '新しい', '素敵', 'かっこいい', 'かわいい', '美味しい',
            '成功', '達成', '向上', '改善', '進歩', '成長', '満足', '充実', '価値'
        ])
        return positive_words
    
    def create_negative_words(self):
        """ネガティブ語彙の作成"""
        negative_words = set([
            '難しい', 'むずかしい', '困った', '困る', '大変', 'たいへん', '心配', '不安',
            '悪い', 'わるい', '嫌', 'いや', '嫌い', 'きらい', 'つまらない', '退屈',
            '疲れる', 'つかれる', '痛い', 'いたい', '失敗', '間違い', 'まちがい',
            '分からない', 'わからない', '理解できない', '難解', '複雑', '問題', '課題',
            '残念', 'ざんねん', '惜しい', 'おしい', '足りない', '不足', '不十分',
            '諦める', 'あきらめる', '無理', 'できない', 'だめ', 'ダメ', '駄目'
        ])
        return negative_words
    
    def preprocess_text(self, text):
        """テキストの前処理"""
        if pd.isna(text) or text == '':
            return []
        
        # 文字列に変換
        text = str(text)
        
        # 基本的な前処理
        text = re.sub(r'[0-9０-９]+', '', text)  # 数字除去
        text = re.sub(r'[a-zA-Zａ-ｚＡ-Ｚ]+', '', text)  # アルファベット除去
        text = re.sub(r'[、。！？!?.,\n\r\t　]', ' ', text)  # 句読点をスペースに
        text = re.sub(r'\s+', ' ', text)  # 連続スペースを1つに
        text = text.strip()
        
        # 簡単な単語分割（スペース区切り + 文字種変化点）
        words = []
        
        # まずスペースで分割
        space_words = text.split()
        
        for word in space_words:
            # ひらがな・カタカナ・漢字の境界で分割
            current_word = ""
            current_type = None
            
            for char in word:
                char_type = self.get_char_type(char)
                
                if current_type is None:
                    current_type = char_type
                    current_word = char
                elif current_type == char_type:
                    current_word += char
                else:
                    if len(current_word) > 1:
                        words.append(current_word)
                    current_word = char
                    current_type = char_type
            
            if len(current_word) > 1:
                words.append(current_word)
        
        # ストップワード除去と長さフィルタ
        filtered_words = [
            word for word in words 
            if len(word) >= 2 and word not in self.stop_words
        ]
        
        return filtered_words
    
    def get_char_type(self, char):
        """文字種の判定"""
        if '\u3040' <= char <= '\u309F':  # ひらがな
            return 'hiragana'
        elif '\u30A0' <= char <= '\u30FF':  # カタカナ
            return 'katakana'
        elif '\u4E00' <= char <= '\u9FAF':  # 漢字
            return 'kanji'
        else:
            return 'other'
    
    def frequency_analysis(self):
        """頻度分析"""
        print("\n" + "="*50)
        print("頻度分析（単語頻度、TF-IDF）")
        print("="*50)
        
        freq_results = {}
        
        # 各テキスト項目の分析
        for text_key in ['Q2_before', 'Q2_after', 'comments']:
            print(f"\n{text_key}の分析")
            print("-" * 30)
            
            result = self.analyze_text_frequency(text_key)
            freq_results[text_key] = result
        
        # Q2の前後比較
        print("\nQ2: みそ汁の理由の変化分析")
        print("-" * 30)
        
        q2_comparison = self.compare_q2_before_after()
        freq_results['q2_comparison'] = q2_comparison
        
        self.results['frequency_analysis'] = freq_results
        return freq_results
    
    def analyze_text_frequency(self, text_key):
        """個別テキスト項目の頻度分析"""
        # テキストデータの取得
        if text_key == 'Q2_before':
            column = self.text_columns['Q2_before']
            texts = self.paired_data[column].dropna()
        elif text_key == 'Q2_after':
            column = self.text_columns['Q2_after']
            texts = self.paired_data[column].dropna()
        elif text_key == 'comments':
            texts = self.paired_data['comments'].dropna()
        else:
            return {"error": f"Unknown text key: {text_key}"}
        
        if len(texts) == 0:
            return {"error": "No text data available"}
        
        # 前処理
        processed_texts = []
        all_words = []
        
        for text in texts:
            words = self.preprocess_text(text)
            if words:
                processed_texts.append(' '.join(words))
                all_words.extend(words)
        
        if not all_words:
            return {"error": "No words after preprocessing"}
        
        # 単語頻度
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        # TF-IDF分析
        tfidf_result = self.calculate_tfidf(processed_texts)
        
        # n-gram分析
        bigrams = self.extract_ngrams(processed_texts, 2)
        trigrams = self.extract_ngrams(processed_texts, 3)
        
        result = {
            'n_texts': len(texts),
            'n_processed': len(processed_texts),
            'total_words': len(all_words),
            'unique_words': len(set(all_words)),
            'top_words': top_words,
            'tfidf_words': tfidf_result,
            'bigrams': bigrams[:10],
            'trigrams': trigrams[:10]
        }
        
        # 結果表示
        print(f"  テキスト数: {result['n_texts']} → 処理済み: {result['n_processed']}")
        print(f"  総語数: {result['total_words']}, 語彙数: {result['unique_words']}")
        print("  頻出語:")
        for word, freq in top_words[:10]:
            print(f"    {word}: {freq}")
        
        return result
    
    def calculate_tfidf(self, texts):
        """TF-IDF分析"""
        if len(texts) < 2:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=1,
                max_df=0.8,
                token_pattern=r'(?u)\b\w+\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 平均TF-IDF スコア
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # スコア順でソート
            sorted_indices = np.argsort(mean_scores)[::-1]
            
            tfidf_words = [
                (feature_names[i], float(mean_scores[i])) 
                for i in sorted_indices[:20]
            ]
            
            return tfidf_words
            
        except Exception as e:
            return []
    
    def extract_ngrams(self, texts, n):
        """n-gram抽出"""
        all_ngrams = []
        
        for text in texts:
            words = text.split()
            if len(words) >= n:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
        
        ngram_freq = Counter(all_ngrams)
        return ngram_freq.most_common(10)
    
    def compare_q2_before_after(self):
        """Q2（みそ汁の理由）の前後比較"""
        before_col = self.text_columns['Q2_before']
        after_col = self.text_columns['Q2_after']
        
        # 前後両方に回答があるケースを抽出
        both_answered = self.paired_data[
            (self.paired_data[before_col].notna()) & 
            (self.paired_data[after_col].notna())
        ]
        
        if len(both_answered) == 0:
            return {"error": "No paired Q2 responses"}
        
        # 前後の語彙変化分析
        before_words = []
        after_words = []
        
        for _, row in both_answered.iterrows():
            before_text_words = self.preprocess_text(row[before_col])
            after_text_words = self.preprocess_text(row[after_col])
            
            before_words.extend(before_text_words)
            after_words.extend(after_text_words)
        
        before_freq = Counter(before_words)
        after_freq = Counter(after_words)
        
        # 新出語彙（授業後のみ）
        new_words = set(after_words) - set(before_words)
        disappeared_words = set(before_words) - set(after_words)
        
        # 頻度変化が大きい語彙
        common_words = set(before_words) & set(after_words)
        frequency_changes = []
        
        for word in common_words:
            before_count = before_freq[word]
            after_count = after_freq[word]
            change = after_count - before_count
            if abs(change) > 0:
                frequency_changes.append((word, before_count, after_count, change))
        
        frequency_changes.sort(key=lambda x: abs(x[3]), reverse=True)
        
        result = {
            'n_paired': len(both_answered),
            'before_total_words': len(before_words),
            'after_total_words': len(after_words),
            'before_unique': len(set(before_words)),
            'after_unique': len(set(after_words)),
            'new_words': list(new_words)[:10],
            'disappeared_words': list(disappeared_words)[:10],
            'frequency_changes': frequency_changes[:10]
        }
        
        # 結果表示
        print(f"  ペア数: {result['n_paired']}")
        print(f"  語彙数の変化: {result['before_unique']} → {result['after_unique']}")
        print(f"  新出語彙: {result['new_words'][:5]}")
        print(f"  消失語彙: {result['disappeared_words'][:5]}")
        
        return result
    
    def sentiment_analysis(self):
        """感情分析"""
        print("\n" + "="*50)
        print("感情分析（ポジティブ/ネガティブ）")
        print("="*50)
        
        sentiment_results = {}
        
        # 各テキスト項目の感情分析
        for text_key in ['Q2_before', 'Q2_after', 'comments']:
            result = self.analyze_sentiment(text_key)
            sentiment_results[text_key] = result
        
        # 感情スコアと理解度の相関分析
        correlation_result = self.correlate_sentiment_with_understanding()
        sentiment_results['correlation_analysis'] = correlation_result
        
        self.results['sentiment_analysis'] = sentiment_results
        return sentiment_results
    
    def analyze_sentiment(self, text_key):
        """個別テキストの感情分析"""
        # テキストデータの取得
        if text_key == 'Q2_before':
            column = self.text_columns['Q2_before']
            texts = self.paired_data[column].dropna()
        elif text_key == 'Q2_after':
            column = self.text_columns['Q2_after']
            texts = self.paired_data[column].dropna()
        elif text_key == 'comments':
            texts = self.paired_data['comments'].dropna()
        else:
            return {"error": f"Unknown text key: {text_key}"}
        
        if len(texts) == 0:
            return {"error": "No text data available"}
        
        # 各テキストの感情スコア計算
        sentiment_scores = []
        detailed_results = []
        
        for i, text in enumerate(texts):
            words = self.preprocess_text(text)
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            total_words = len(words)
            
            # 感情スコア（-1 to 1）
            if total_words > 0:
                sentiment_score = (positive_count - negative_count) / total_words
            else:
                sentiment_score = 0
            
            sentiment_scores.append(sentiment_score)
            
            detailed_results.append({
                'text_index': i,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_words': total_words,
                'sentiment_score': sentiment_score
            })
        
        # 統計サマリー
        sentiment_array = np.array(sentiment_scores)
        
        result = {
            'n_texts': len(texts),
            'sentiment_scores': sentiment_scores,
            'mean_sentiment': float(np.mean(sentiment_array)),
            'std_sentiment': float(np.std(sentiment_array)),
            'median_sentiment': float(np.median(sentiment_array)),
            'positive_ratio': float(np.mean(sentiment_array > 0)),
            'negative_ratio': float(np.mean(sentiment_array < 0)),
            'neutral_ratio': float(np.mean(sentiment_array == 0)),
            'detailed_results': detailed_results
        }
        
        # 結果表示
        print(f"\n{text_key}の感情分析:")
        print(f"  テキスト数: {result['n_texts']}")
        print(f"  平均感情スコア: {result['mean_sentiment']:.3f} ± {result['std_sentiment']:.3f}")
        print(f"  ポジティブ率: {result['positive_ratio']:.3f}")
        print(f"  ネガティブ率: {result['negative_ratio']:.3f}")
        print(f"  中立率: {result['neutral_ratio']:.3f}")
        
        return result
    
    def correlate_sentiment_with_understanding(self):
        """感情スコアと理解度の相関分析"""
        # 感想文の感情スコアを計算
        if 'comments' not in self.paired_data.columns:
            return {"error": "No comments data"}
        
        sentiment_scores = []
        understanding_scores = []
        
        for _, row in self.paired_data.iterrows():
            # 感想文の感情スコア
            if pd.notna(row['comments']):
                words = self.preprocess_text(row['comments'])
                positive_count = sum(1 for word in words if word in self.positive_words)
                negative_count = sum(1 for word in words if word in self.negative_words)
                total_words = len(words)
                
                if total_words > 0:
                    sentiment_score = (positive_count - negative_count) / total_words
                    sentiment_scores.append(sentiment_score)
                    
                    # 理解度スコア
                    if pd.notna(row['Q6_DissolvingUnderstandingRating']):
                        understanding_scores.append(row['Q6_DissolvingUnderstandingRating'])
                    else:
                        sentiment_scores.pop()  # 理解度がないので感情スコアも除去
        
        if len(sentiment_scores) < 3:
            return {"error": "Insufficient paired data"}
        
        # 相関分析
        correlation = np.corrcoef(sentiment_scores, understanding_scores)[0, 1]
        
        result = {
            'n_pairs': len(sentiment_scores),
            'correlation': float(correlation),
            'sentiment_mean': float(np.mean(sentiment_scores)),
            'understanding_mean': float(np.mean(understanding_scores)),
            'sentiment_std': float(np.std(sentiment_scores)),
            'understanding_std': float(np.std(understanding_scores))
        }
        
        print(f"\n感情-理解度相関分析:")
        print(f"  ペア数: {result['n_pairs']}")
        print(f"  相関係数: {result['correlation']:.3f}")
        
        return result
    
    def create_visualizations(self):
        """可視化の作成"""
        print("\n" + "="*50)
        print("テキストマイニング結果の可視化")
        print("="*50)
        
        output_dir = Path("outputs/phase4_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 頻度分析の可視化
        self.plot_frequency_analysis(output_dir)
        
        # 感情分析の可視化
        self.plot_sentiment_analysis(output_dir)
        
        # Q2変化の可視化
        self.plot_q2_changes(output_dir)
        
        print(f"✓ 図表を {output_dir} に保存しました")
    
    def plot_frequency_analysis(self, output_dir):
        """頻度分析の可視化"""
        if 'frequency_analysis' not in self.results:
            return
        
        # 各テキスト項目の頻出語
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        text_keys = ['Q2_before', 'Q2_after', 'comments']
        labels = ['Q2授業前', 'Q2授業後', '感想文']
        
        for i, (text_key, label) in enumerate(zip(text_keys, labels)):
            if i >= len(axes):
                break
                
            result = self.results['frequency_analysis'].get(text_key, {})
            
            if 'error' in result or 'top_words' not in result:
                axes[i].text(0.5, 0.5, f'{label}\nデータなし', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue
            
            top_words = result['top_words'][:10]
            if top_words:
                words, freqs = zip(*top_words)
                
                axes[i].barh(range(len(words)), freqs, alpha=0.7)
                axes[i].set_yticks(range(len(words)))
                axes[i].set_yticklabels(words)
                axes[i].set_xlabel('Frequency')
                axes[i].set_title(f'{label} - 頻出語')
                axes[i].grid(True, alpha=0.3)
                axes[i].invert_yaxis()
        
        # Q2前後比較
        if 'q2_comparison' in self.results['frequency_analysis']:
            comp_result = self.results['frequency_analysis']['q2_comparison']
            
            if 'error' not in comp_result and 'new_words' in comp_result:
                new_words = comp_result['new_words'][:8]
                disappeared = comp_result['disappeared_words'][:8]
                
                # 新出語彙
                if new_words:
                    axes[3].bar(range(len(new_words)), [1]*len(new_words), 
                              alpha=0.7, color='green', label='新出語彙')
                    axes[3].set_xticks(range(len(new_words)))
                    axes[3].set_xticklabels(new_words, rotation=45, ha='right')
                    
                # 消失語彙
                if disappeared:
                    start_pos = len(new_words) + 1
                    axes[3].bar(range(start_pos, start_pos + len(disappeared)), 
                              [1]*len(disappeared), alpha=0.7, color='red', label='消失語彙')
                    
                    all_labels = new_words + [''] + disappeared
                    axes[3].set_xticks(range(len(all_labels)))
                    axes[3].set_xticklabels(all_labels, rotation=45, ha='right')
                
                axes[3].set_ylabel('Presence')
                axes[3].set_title('Q2: 語彙の変化')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "frequency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sentiment_analysis(self, output_dir):
        """感情分析の可視化"""
        if 'sentiment_analysis' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 各テキスト項目の感情分布
        text_keys = ['Q2_before', 'Q2_after', 'comments']
        labels = ['Q2授業前', 'Q2授業後', '感想文']
        
        for i, (text_key, label) in enumerate(zip(text_keys, labels)):
            if i >= 3:
                break
                
            result = self.results['sentiment_analysis'].get(text_key, {})
            
            if 'error' in result or 'sentiment_scores' not in result:
                axes[i//2, i%2].text(0.5, 0.5, f'{label}\nデータなし', 
                                    ha='center', va='center', transform=axes[i//2, i%2].transAxes)
                continue
            
            sentiment_scores = result['sentiment_scores']
            
            axes[i//2, i%2].hist(sentiment_scores, bins=10, alpha=0.7, edgecolor='black')
            axes[i//2, i%2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='中立')
            axes[i//2, i%2].axvline(x=result['mean_sentiment'], color='blue', 
                                  linestyle='-', alpha=0.7, label=f'平均: {result["mean_sentiment"]:.3f}')
            axes[i//2, i%2].set_xlabel('Sentiment Score')
            axes[i//2, i%2].set_ylabel('Frequency')
            axes[i//2, i%2].set_title(f'{label} - 感情分布')
            axes[i//2, i%2].legend()
            axes[i//2, i%2].grid(True, alpha=0.3)
        
        # 感情-理解度相関
        if 'correlation_analysis' in self.results['sentiment_analysis']:
            corr_result = self.results['sentiment_analysis']['correlation_analysis']
            
            if 'error' not in corr_result:
                # 相関係数を表示（実際のプロットデータは保存されていないため、情報のみ表示）
                axes[1, 1].text(0.5, 0.5, 
                               f'感情-理解度相関\nr = {corr_result["correlation"]:.3f}\nn = {corr_result["n_pairs"]}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1, 1].set_xticks([])
                axes[1, 1].set_yticks([])
                axes[1, 1].set_title('感情-理解度相関')
        
        plt.tight_layout()
        plt.savefig(output_dir / "sentiment_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_q2_changes(self, output_dir):
        """Q2変化の可視化"""
        if ('frequency_analysis' not in self.results or 
            'q2_comparison' not in self.results['frequency_analysis']):
            return
        
        comp_result = self.results['frequency_analysis']['q2_comparison']
        
        if 'error' in comp_result:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 語彙数の変化
        before_unique = comp_result['before_unique']
        after_unique = comp_result['after_unique']
        
        axes[0].bar(['授業前', '授業後'], [before_unique, after_unique], 
                   alpha=0.7, color=['lightblue', 'lightcoral'])
        axes[0].set_ylabel('語彙数')
        axes[0].set_title('Q2: 語彙数の変化')
        axes[0].grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        axes[0].text(0, before_unique + max(before_unique, after_unique)*0.02, 
                    str(before_unique), ha='center', va='bottom')
        axes[0].text(1, after_unique + max(before_unique, after_unique)*0.02, 
                    str(after_unique), ha='center', va='bottom')
        
        # 頻度変化の大きい語彙
        if 'frequency_changes' in comp_result and comp_result['frequency_changes']:
            freq_changes = comp_result['frequency_changes'][:8]
            words = [item[0] for item in freq_changes]
            changes = [item[3] for item in freq_changes]
            
            colors = ['green' if change > 0 else 'red' for change in changes]
            
            axes[1].bar(range(len(words)), changes, color=colors, alpha=0.7)
            axes[1].set_xticks(range(len(words)))
            axes[1].set_xticklabels(words, rotation=45, ha='right')
            axes[1].set_ylabel('頻度変化')
            axes[1].set_title('Q2: 語彙頻度の変化')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "q2_changes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Phase 4 レポート生成"""
        print("\n" + "="*50)
        print("Phase 4 レポート生成")
        print("="*50)
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # JSON形式で詳細結果を保存
        with open(output_dir / "phase4_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # テキスト形式でサマリーレポートを生成
        report_content = self.create_summary_report()
        
        with open(output_dir / "phase4_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ レポートを {output_dir} に保存しました")
        print(f"  - 詳細結果: phase4_detailed_results.json")
        print(f"  - サマリー: phase4_summary_report.txt")
        
        return report_content
    
    def create_summary_report(self):
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("小学校出前授業アンケート Phase 4 テキストマイニング結果")
        report.append("="*60)
        report.append(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # データ概要
        report.append("【データ概要】")
        report.append(f"ペア数: {len(self.paired_data)}")
        
        # 各テキストデータの概要
        if 'frequency_analysis' in self.results:
            for text_key, label in [('Q2_before', 'Q2授業前'), ('Q2_after', 'Q2授業後'), ('comments', '感想文')]:
                result = self.results['frequency_analysis'].get(text_key, {})
                if 'error' not in result and 'n_texts' in result:
                    report.append(f"  {label}: {result['n_texts']}件, 語彙数: {result.get('unique_words', 'N/A')}")
        
        report.append("")
        
        # 頻度分析結果
        if 'frequency_analysis' in self.results:
            report.append("【頻度分析結果】")
            
            for text_key, label in [('comments', '感想文'), ('Q2_after', 'Q2授業後')]:
                result = self.results['frequency_analysis'].get(text_key, {})
                if 'error' not in result and 'top_words' in result:
                    top_words = result['top_words'][:5]
                    if top_words:
                        words_str = ', '.join([f"{word}({freq})" for word, freq in top_words])
                        report.append(f"\n{label}の頻出語:")
                        report.append(f"  {words_str}")
            
            # Q2変化分析
            if 'q2_comparison' in self.results['frequency_analysis']:
                comp_result = self.results['frequency_analysis']['q2_comparison']
                if 'error' not in comp_result:
                    report.append(f"\nQ2（みそ汁の理由）変化:")
                    report.append(f"  ペア数: {comp_result['n_paired']}")
                    report.append(f"  語彙数変化: {comp_result['before_unique']} → {comp_result['after_unique']}")
                    
                    if comp_result['new_words']:
                        report.append(f"  新出語彙: {', '.join(comp_result['new_words'][:5])}")
            
            report.append("")
        
        # 感情分析結果
        if 'sentiment_analysis' in self.results:
            report.append("【感情分析結果】")
            
            for text_key, label in [('comments', '感想文'), ('Q2_after', 'Q2授業後')]:
                result = self.results['sentiment_analysis'].get(text_key, {})
                if 'error' not in result and 'mean_sentiment' in result:
                    sentiment_interpretation = self.interpret_sentiment(result['mean_sentiment'])
                    report.append(f"\n{label}:")
                    report.append(f"  平均感情スコア: {result['mean_sentiment']:.3f} ({sentiment_interpretation})")
                    report.append(f"  ポジティブ率: {result['positive_ratio']:.3f}")
            
            # 感情-理解度相関
            if 'correlation_analysis' in self.results['sentiment_analysis']:
                corr_result = self.results['sentiment_analysis']['correlation_analysis']
                if 'error' not in corr_result:
                    correlation_strength = self.interpret_correlation(corr_result['correlation'])
                    report.append(f"\n感情-理解度相関:")
                    report.append(f"  相関係数: {corr_result['correlation']:.3f} ({correlation_strength})")
                    report.append(f"  分析対象: {corr_result['n_pairs']}ペア")
            
            report.append("")
        
        # 主要な発見事項
        report.append("【主要な発見事項】")
        
        # テキストデータの豊富さ
        total_texts = 0
        if 'frequency_analysis' in self.results:
            for text_key in ['comments', 'Q2_after', 'Q2_before']:
                result = self.results['frequency_analysis'].get(text_key, {})
                if 'error' not in result and 'n_texts' in result:
                    total_texts += result['n_texts']
        
        if total_texts > 20:
            report.append("✓ 豊富なテキストデータが利用可能")
        else:
            report.append("⚠️  テキストデータが限定的")
        
        # 感情の傾向
        if 'sentiment_analysis' in self.results and 'comments' in self.results['sentiment_analysis']:
            comment_sentiment = self.results['sentiment_analysis']['comments']
            if 'error' not in comment_sentiment:
                if comment_sentiment['mean_sentiment'] > 0.1:
                    report.append("✓ 感想文は全体的にポジティブ")
                elif comment_sentiment['mean_sentiment'] < -0.1:
                    report.append("⚠️  感想文にネガティブな傾向")
                else:
                    report.append("• 感想文の感情は中立的")
        
        # Q2の変化
        if ('frequency_analysis' in self.results and 
            'q2_comparison' in self.results['frequency_analysis']):
            comp_result = self.results['frequency_analysis']['q2_comparison']
            if 'error' not in comp_result:
                vocab_change = comp_result['after_unique'] - comp_result['before_unique']
                if vocab_change > 2:
                    report.append("✓ Q2回答で語彙の多様化を確認")
                elif vocab_change < -2:
                    report.append("• Q2回答で語彙の簡素化を確認")
                else:
                    report.append("• Q2回答の語彙に大きな変化なし")
        
        # 相関の強さ
        if ('sentiment_analysis' in self.results and 
            'correlation_analysis' in self.results['sentiment_analysis']):
            corr_result = self.results['sentiment_analysis']['correlation_analysis']
            if 'error' not in corr_result:
                if abs(corr_result['correlation']) > 0.3:
                    report.append(f"✓ 感情と理解度に明確な関連性 (r={corr_result['correlation']:.3f})")
                else:
                    report.append("• 感情と理解度の関連性は限定的")
        
        report.append("")
        report.append("【Phase 5への推奨事項】")
        report.append("1. テキスト特徴量を予測モデルに統合")
        report.append("2. 定量・定性データの総合分析")
        
        if total_texts > 20:
            report.append("3. テキストデータの詳細な質的分析")
        
        if ('sentiment_analysis' in self.results and 
            'correlation_analysis' in self.results['sentiment_analysis'] and
            'error' not in self.results['sentiment_analysis']['correlation_analysis']):
            report.append("4. 感情要因を含む総合的な教育効果モデル")
        
        report.append("5. 教育実践への具体的提言の作成")
        
        return "\n".join(report)
    
    def interpret_sentiment(self, score):
        """感情スコアの解釈"""
        if score > 0.2:
            return "ポジティブ"
        elif score < -0.2:
            return "ネガティブ"
        else:
            return "中立"
    
    def interpret_correlation(self, correlation):
        """相関係数の解釈"""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            return "強い相関"
        elif abs_corr > 0.3:
            return "中程度の相関"
        elif abs_corr > 0.1:
            return "弱い相関"
        else:
            return "相関なし"
    
    def run_complete_analysis(self):
        """Phase 4 完全分析実行"""
        print("小学校出前授業アンケート Phase 4: テキストマイニング分析")
        print("実行開始:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("="*60)
        
        try:
            # データ読み込み
            self.load_data()
            
            # 頻度分析
            self.frequency_analysis()
            
            # 感情分析
            self.sentiment_analysis()
            
            # 可視化作成
            self.create_visualizations()
            
            # レポート生成
            summary_report = self.generate_report()
            
            print("\n" + "="*60)
            print("Phase 4 分析完了!")
            print("="*60)
            print(summary_report)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Phase 4 分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """メイン実行関数"""
    miner = Phase4TextMiner()
    results = miner.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()