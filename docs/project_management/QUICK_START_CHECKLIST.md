# 🚀 次回セッション開始チェックリスト
## 5分で完了する必須確認事項

**目的**: 効率的なセッション開始、重複作業防止、適切な作業継続  
**所要時間**: 5分以内

---

## ⚡ **即座実行（2分）**

### **1. 作業環境確認**
```bash
# 作業ディレクトリ移動
cd /home/grace/projects/social-implement/lecture-survey-analysis/lecture-survey-analysis

# 現在位置確認
pwd
# 期待値: /home/grace/projects/social-implement/lecture-survey-analysis/lecture-survey-analysis
```

### **2. 最新状況確認**
```bash
# 引き継ぎドキュメント確認
cat docs/project_management/SESSION_HANDOVER_CURRENT.md | head -20

# 完了済みタスク確認
cat docs/project_management/COMPLETED_TASKS_LIST.md | grep "完了済み"

# 凍結項目確認
cat docs/project_management/FROZEN_ITEMS.md | head -10
```

---

## 🔍 **状況判断（2分）**

### **3. 重複作業防止チェック**
```markdown
□ 今回実行予定タスクが完了済みリストにないことを確認
□ 今回実行予定タスクが凍結リストにないことを確認
□ 分析スクリプト（scripts/active/）の再実行でないことを確認
□ 既存データ（outputs/current/）の変更作業でないことを確認
```

### **4. 今回セッション目標設定**
```markdown
## 今回の目標設定
□ 実行予定タスク: [ A1 / A2 / A4 / A3 / B1 / B3 / B2 ]
□ 目標完了時間: [ ___時間 ]
□ 最低達成ライン: [ A1, A2 / その他 ]
□ 理想達成ライン: [ A1-A4 / A1-B2 / その他 ]
```

---

## 🎯 **実行準備（1分）**

### **5. 優先タスク確認**
```markdown
## 推奨実行順序（現在設定）
1. 🔴 A1: Q1解釈レポート（2時間）← 最優先
2. 🔴 A2: ステークホルダー要約（1.5時間）← 最優先  
3. 🔴 A4: クラス2詳細分析（2時間）← 高優先
4. 🔴 A3: 完了報告書統合（1時間）← 高優先
5. 🟡 B1: 批判的検討（1時間）← 中優先
6. 🟡 B3: 方法論考察（1時間）← 中優先
7. 🟡 B2: データ再検証（0.5時間）← 低優先

## 開始タスク決定
□ 今回開始タスク: [ A1 / A2 / A4 / その他 ]
```

---

## ⚠️ **緊急停止条件**

### **以下の場合は作業停止・状況確認**
```markdown
□ 完了済みタスクリストに今回予定タスクが記載されている
□ 凍結項目リストに今回予定タスクが記載されている  
□ データファイルが見つからない・破損している
□ 既存分析結果との重大な矛盾を発見
□ scripts/active/内のファイル実行を求められている
```

---

## 🔄 **タスク実行プロトコル**

### **タスク開始時**
```bash
# Todo更新（開始時）
# TodoWrite で該当タスクを "in_progress" に更新

# 開始時刻記録
echo "タスク[ID]開始: $(date +%Y-%m-%d\ %H:%M)" >> docs/project_management/session_log.txt
```

### **タスク完了時**
```bash
# Todo更新（完了時）
# TodoWrite で該当タスクを "completed" に更新

# 完了記録
echo "[TaskID],[TaskName],$(date +%Y-%m-%d\ %H:%M),completed,[実際時間],[推定時間],passed,[成果物場所]" >> docs/project_management/TASK_COMPLETION_LOG.csv

# アーカイブ作業
cp [成果物] docs/project_management/completed_tasks/$(date +%Y%m%d)/

# 完了リスト更新
echo "- **[TaskID]**: [TaskName] - 完了済み ($(date +%Y-%m-%d))" >> docs/project_management/COMPLETED_TASKS_LIST.md
```

---

## ✅ **チェック完了確認**

### **開始準備完了チェック**
```markdown
□ 作業ディレクトリ確認完了
□ 引き継ぎドキュメント確認完了
□ 重複作業防止チェック完了
□ 凍結項目チェック完了
□ 今回セッション目標設定完了
□ 開始タスク決定完了
□ 緊急停止条件確認完了
```

### **実行開始合図**
```bash
echo "=== セッション開始準備完了 ==="
echo "開始タスク: [TaskID]"
echo "目標時間: [X]時間"
echo "開始時刻: $(date +%Y-%m-%d\ %H:%M)"
echo "================================"
```

---

## 🚨 **トラブル時の対応**

### **よくある問題と対処法**
```markdown
## ファイルが見つからない
□ パス確認: pwd で現在位置確認
□ ファイル存在確認: ls -la [ファイルパス]
□ バックアップ確認: ls backup_*/

## 完了済みタスクを実行しようとしている  
□ 完了済みリスト再確認
□ 成果物の存在確認
□ 作業内容の再検討（重複回避）

## 凍結項目を実行しようとしている
□ 凍結理由確認
□ 解除条件確認
□ 代替作業の検討

## データ・結果に矛盾発見
□ 作業停止
□ 既存結果の再確認
□ 矛盾内容の記録
□ 対応方針の検討
```

---

## 📊 **セッション終了時タスク**

### **セッション完了時の必須作業**
```markdown
□ 完了タスクのアーカイブ作業
□ 完了記録の更新
□ 引き継ぎドキュメントの状況更新
□ 次回セッション推奨事項の記録
□ Todo状況の最終確認
```

---

**⏱️ チェック完了目安時間**: 5分  
**🎯 次のアクション**: タスクA1実行開始  
**📋 最終確認**: 重複防止✅ + 凍結回避✅ + 目標設定✅ = 実行開始Ready🚀