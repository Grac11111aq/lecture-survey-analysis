#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全分析実行スクリプト
===================

小学校出前授業アンケート分析の全工程を正しい順序で実行する。

機能:
- 設定ファイル読み込み
- 環境チェック
- 依存関係確認
- 順次実行（02→05）
- 結果検証
- 統合レポート生成

Author: Claude Code Analysis (Master Controller)
Date: 2025-05-31
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

class AnalysisMasterController:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config" / "analysis_metadata.yaml"
        self.config = None
        self.execution_log = []
        
    def load_config(self):
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print("✓ 設定ファイル読み込み完了")
            return True
        except Exception as e:
            print(f"❌ 設定ファイル読み込み失敗: {e}")
            return False
    
    def check_environment(self):
        """環境チェック"""
        print("🔍 環境チェック中...")
        
        # Python環境確認
        python_version = sys.version_info
        required_version = tuple(map(int, self.config['environment']['python_version'].split('.')))
        
        if python_version[:2] < required_version:
            print(f"❌ Python バージョン不足: {python_version} < {required_version}")
            return False
        
        # データファイル存在確認
        data_dir = self.project_root / "data" / "analysis"
        required_files = [
            "before_excel_compliant.csv",
            "after_excel_compliant.csv"
        ]
        
        for file_name in required_files:
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"❌ 必須データファイルが見つかりません: {file_path}")
                return False
        
        # 出力ディレクトリ存在確認
        output_dirs = [
            "outputs/current/02_group_comparison",
            "outputs/current/05_final_report",
            "outputs/figures/current/02_group_comparison"
        ]
        
        for dir_path in output_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                print(f"❌ 出力ディレクトリが見つかりません: {full_path}")
                return False
        
        print("✓ 環境チェック完了")
        return True
    
    def validate_active_scripts(self):
        """有効スクリプトの存在確認"""
        print("📋 有効スクリプト確認中...")
        
        for script_info in self.config['active_scripts']:
            script_path = self.project_root / script_info['path']
            if not script_path.exists():
                print(f"❌ スクリプトが見つかりません: {script_path}")
                return False
            print(f"✓ {script_info['name']} 確認済み")
        
        return True
    
    def check_deprecated_usage(self):
        """廃止ファイルの誤用チェック"""
        print("⚠️  廃止ファイル使用チェック...")
        
        # より精密な廃止パターン（問題のある使用のみ検出）
        critical_patterns = [
            "McNemar検定",
            "対応のあるt検定", 
            "paired_ttest",
            "before_after_paired",
            "matched_pairs",
            "個人の変化を測定",
            "ペアデータとして",
            "同一人物の前後比較"
        ]
        
        # 有効スクリプト内で問題のあるパターンをチェック
        issues_found = []
        for script_info in self.config['active_scripts']:
            script_path = self.project_root / script_info['path']
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in critical_patterns:
                        if pattern in content:
                            issues_found.append(f"{script_info['name']}: {pattern}")
            except Exception as e:
                print(f"⚠️  {script_info['name']} チェック失敗: {e}")
        
        if issues_found:
            print("❌ 問題のある廃止パターンが検出されました:")
            for issue in issues_found:
                print(f"  - {issue}")
            return False
        
        print("✓ 廃止パターンチェック完了")
        return True
    
    def execute_script(self, script_info):
        """スクリプト実行"""
        script_path = self.project_root / script_info['path']
        script_name = script_info['name']
        
        print(f"🚀 {script_name} 実行開始...")
        start_time = datetime.now()
        
        try:
            # 仮想環境でPythonスクリプト実行
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分タイムアウト
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            execution_record = {
                'script': script_name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.execution_log.append(execution_record)
            
            if result.returncode == 0:
                print(f"✓ {script_name} 実行完了 ({duration:.1f}秒)")
                return True
            else:
                print(f"❌ {script_name} 実行失敗:")
                print(f"  Return code: {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {script_name} タイムアウト (10分)")
            return False
        except Exception as e:
            print(f"❌ {script_name} 実行エラー: {e}")
            return False
    
    def validate_outputs(self, script_info):
        """出力ファイルの検証"""
        print(f"🔍 {script_info['name']} 出力検証中...")
        
        for output_file in script_info['output_files']:
            # ワイルドカードを含む場合はディレクトリ存在チェックのみ
            if '*' in output_file:
                dir_path = Path(output_file).parent
                full_dir_path = self.project_root / dir_path
                if not full_dir_path.exists():
                    print(f"❌ 出力ディレクトリが見つかりません: {full_dir_path}")
                    return False
                print(f"✓ {dir_path} ディレクトリ存在確認")
                continue
            
            file_path = self.project_root / output_file
            if not file_path.exists():
                print(f"❌ 期待される出力ファイルが見つかりません: {file_path}")
                return False
            
            # JSONファイルの場合は構造チェック
            if output_file.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"✓ {file_path.name} JSON構造正常")
                except json.JSONDecodeError as e:
                    print(f"❌ {file_path.name} JSON構造エラー: {e}")
                    return False
            else:
                print(f"✓ {file_path.name} 存在確認")
        
        return True
    
    def generate_execution_report(self):
        """実行レポート生成"""
        report_path = self.project_root / "outputs" / "current" / "execution_log.json"
        
        execution_summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_version': self.config['project']['version'],
                'analysis_type': self.config['data_assumptions']['analysis_type']
            },
            'execution_log': self.execution_log,
            'summary': {
                'total_scripts': len(self.execution_log),
                'successful_scripts': sum(1 for log in self.execution_log if log['return_code'] == 0),
                'failed_scripts': sum(1 for log in self.execution_log if log['return_code'] != 0),
                'total_duration': sum(log['duration_seconds'] for log in self.execution_log)
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(execution_summary, f, ensure_ascii=False, indent=2)
        
        print(f"📊 実行レポートを保存: {report_path}")
        return execution_summary
    
    def run_complete_analysis(self):
        """完全分析実行"""
        print("="*60)
        print("小学校出前授業アンケート分析 - 完全実行")
        print("="*60)
        print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 設定読み込み
        if not self.load_config():
            return False
        
        # 2. 環境チェック
        if not self.check_environment():
            return False
        
        # 3. スクリプト存在確認
        if not self.validate_active_scripts():
            return False
        
        # 4. 廃止パターンチェック
        if not self.check_deprecated_usage():
            return False
        
        print()
        print("🎯 分析実行開始...")
        print()
        
        # 5. 順次実行
        for script_info in self.config['active_scripts']:
            # スクリプト実行
            if not self.execute_script(script_info):
                print(f"❌ {script_info['name']} で実行が停止されました")
                return False
            
            # 出力検証
            if not self.validate_outputs(script_info):
                print(f"❌ {script_info['name']} の出力検証に失敗しました")
                return False
            
            print()
        
        # 6. 実行レポート生成
        summary = self.generate_execution_report()
        
        print("="*60)
        print("🎉 分析完了!")
        print("="*60)
        print(f"実行終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"成功スクリプト: {summary['summary']['successful_scripts']}/{summary['summary']['total_scripts']}")
        print(f"総実行時間: {summary['summary']['total_duration']:.1f}秒")
        print()
        print("📁 主要な結果ファイル:")
        print("  - outputs/current/02_group_comparison/phase2_revised_results.json")
        print("  - outputs/current/05_final_report/integrated_final_report.json")
        print("  - docs/reports/comprehensive_final_report.md")
        print()
        print("⚠️  重要: この分析は独立群比較であり、個人の変化は測定していません")
        
        return True

def main():
    """メイン実行関数"""
    controller = AnalysisMasterController()
    
    try:
        success = controller.run_complete_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()