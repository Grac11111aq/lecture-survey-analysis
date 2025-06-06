# 分析対象データセット

このディレクトリには、小学校出前授業アンケート分析の対象となる3つのデータファイルが含まれています。

## ファイル構成

### comment.csv
- **説明**: 授業後の感想コメントデータ（OCR処理）
- **データソース**: OCR処理による文字認識結果
- **選択理由**: コメントはプレーンテキストが主体で、OCRの精度がExcel変換より高いため

### before_excel_compliant.csv  
- **説明**: 授業前アンケートデータ（Excel準拠形式）
- **データソース**: validation_data/before_excel.csv からコピー
- **選択理由**: 構造化データでExcel変換の精度が高い

### after_excel_compliant.csv
- **説明**: 授業後アンケートデータ（Excel準拠形式）  
- **データソース**: validation_data/after_excel.csv からコピー
- **選択理由**: 構造化データでExcel変換の精度が高い

## データ処理方針

1. **comment.csv**: OCR由来のテキストデータを直接利用
2. **before/after_excel_compliant.csv**: Excel変換処理済みの構造化データを利用
3. すべてのデータは匿名化済み（Page_IDによる識別）

## 注意事項

- すべてのデータは個人情報を含まない匿名化済みデータです
- Page_IDを使用して個人を特定することはできません
- データは研究・教育目的でのみ使用してください