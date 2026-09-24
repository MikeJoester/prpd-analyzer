import os

file_path = "/home/dannt/Documents/PRPD Analyzer/app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    '"데이터 폴더"': '"Data Folder"',
    'f"검색된 파일: {len(files):,}개"': 'f"Discovered files: {len(files):,}"',
    '"전체 데이터 분석 (필터 무시)"': '"Analyze all data (ignore filters)"',
    '"고장 종류"': '"Fault Type"',
    '"데이터 그룹"': '"Data Group"',
    '"측정 날짜"': '"Measurement Date"',
    '"최대 샘플 수"': '"Max Samples"',
    '"AI 데이터셋 포함 그룹"': '"Include Groups for AI Dataset"',
    '"AI 데이터셋 출력 폴더"': '"AI Dataset Output Folder"',
    '"데이터 폴더를 찾을 수 없습니다. Data/by_type 또는 Data/by_date 폴더를 선택하세요."': '"Data folder not found. Please select Data/by_type or Data/by_date folder."',
    'f"전체 검색: {len(files):,}개 | 분석 대상: {len(file_pool):,}개 "': 'f"Total search: {len(files):,} | Analysis target: {len(file_pool):,} "',
    'f"| 모드: {\'전체 데이터\' if analyze_all else \'필터 적용\'} "': 'f"| Mode: {\'All data\' if analyze_all else \'Filter applied\'} "',
    'f"| 선택 그룹: {\', \'.join(selected_groups) or \'없음\'} "': 'f"| Selected groups: {\', \'.join(selected_groups) or \'None\'} "',
    '"| 각 파일은 256차원(mean 128 + max 128) Feature로 변환됩니다."': '"| Each file is converted into 256-dimensional (mean 128 + max 128) Features."',
    '"데이터 분석 및 t-SNE 실행"': '"Run Data Analysis and t-SNE"',
    '"t-SNE를 실행하려면 최소 3개의 유효한 파일이 필요합니다."': '"At least 3 valid files are required to run t-SNE."',
    'f"{len(file_pool):,}개 파일을 읽고 Feature와 t-SNE를 계산하는 중..."': 'f"Reading {len(file_pool):,} files and calculating Features and t-SNE..."',
    '"읽을 수 있는 .dat 파일이 3개 미만입니다. 오류 목록을 확인하세요."': '"Less than 3 readable .dat files found. Please check the error list."',
    '"왼쪽 조건을 설정한 뒤 `데이터 분석 및 t-SNE 실행`을 누르세요."': '"Set conditions on the left and click `Run Data Analysis and t-SNE`."',
    '["요약", "PRPD / Feature", "t-SNE", "데이터", "Category 리포트", "AI 데이터셋"]': '["Summary", "PRPD / Feature", "t-SNE", "Data", "Category Report", "AI Dataset"]',
    '"정상 분석 파일"': '"Valid Analyzed Files"',
    '"오류 파일"': '"Error Files"',
    '"고장 종류"': '"Fault Types"',  # Note: there is also "Fault Type" above, but it's used as "Fault Types" for metric
    '"데이터 그룹"': '"Data Groups"', # Note: there is also "Data Group" above, used as "Data Groups" for metric
    '"Feature 차원"': '"Feature Dimension"',
    '"데이터 그룹별 고장 종류 수량"': '"Fault Types Count by Data Group"',
    '"날짜별 데이터 수"': '"Data Count by Date"',
    '"Noise 그룹 비교"': '"Noise Group Comparison"',
    '"Lab Noise / Field Noise 날짜별 비교"': '"Lab Noise / Field Noise Comparison by Date"',
    '"상세 데이터 그룹"': '"Detailed Data Group"',
    'f"파일: {selected_row[\'file\']} | 날짜: {selected_row[\'date\']}"': 'f"File: {selected_row[\'file\']} | Date: {selected_row[\'date\']}"',
    'f"{detail_group} / {detail_label} 위상 평균값"': 'f"{detail_group} / {detail_label} Phase Mean Value"',
    'f"{detail_group} / {detail_label} 위상 최대값"': 'f"{detail_group} / {detail_label} Phase Max Value"',
    '"상세 파일"': '"Detailed File"',
    '"t-SNE 결과 CSV"': '"t-SNE Result CSV"',
    'f"읽기 오류 파일: {len(errors):,}개"': 'f"Read Error Files: {len(errors):,}"',
    '"Category 품질 검토 대상"': '"Category Quality Review Target"',
    '"점수는 자동 재분류가 아닌 수동 검토 우선순위입니다. "': '"Scores are for manual review prioritization, not automatic reclassification. "',
    '"(group, label) 셀 기준 pooled shrinkage Mahalanobis 거리이며, "': '"Based on pooled shrinkage Mahalanobis distance per (group, label) cell, "',
    '"임계값은 경험적 percentile입니다."': '"threshold is empirical percentile."',
    'f"기준 분포를 만들 수 없습니다: {error}"': 'f"Cannot build reference distribution: {error}"',
    '"점수 계산 가능"': '"Scorable"',
    '"기준 셀"': '"Reference Cell"',
    '"변환"': '"Transform"',
    '"검토 범위"': '"Review Scope"',
    '("Top 1% (d²)", "Top 5% (d²)", "다른 category에 더 가까움 (margin > 0)", "전체")': '("Top 1% (d²)", "Top 5% (d²)", "Closer to another category (margin > 0)", "All")',
    'startswith("다른")': 'startswith("Closer")',
    'f"검토 대상: {len(suspects):,}개 / 점수 계산 가능: {len(scorable):,}개"': 'f"Review Target: {len(suspects):,} / Scorable: {len(scorable):,}"',
    'f"적용 컷값 — {cut_text}"': 'f"Applied Cuts — {cut_text}"',
    'f"기준 표본 부족 — 점수 계산 불가 ({len(unscorable):,}개)"': 'f"Insufficient Reference Sample — Cannot calculate score ({len(unscorable):,})"',
    '"표본이 부족한 (group, label) 셀은 평균이 사실상 자기 자신이라 거리가 "': '"Cells with insufficient samples have their mean close to themselves, so the distance "',
    '"0에 가까워집니다. \'정상\'으로 오인하지 않도록 분리해 표시합니다."': '"becomes near 0. They are separated to prevent being mistaken as \'normal\'."',
    '"Category 리포트 CSV"': '"Category Report CSV"',
    '"Feature 품질 진단"': '"Feature Quality Diagnostics"',
    '"Mean / Max / Mean+Max가 PD class를 얼마나 분리하는지 비교하는 **진단 지표**입니다. "': '"This is a **diagnostic metric** comparing how well Mean / Max / Mean+Max separate PD classes. "',
    '"분류기가 아니며 위 의심 점수에도 쓰지 않습니다. "': '"It is not a classifier and is not used for the suspect scores above. "',
    '"silhouette은 이 데이터에서 max와 mean+max를 구분하지 못하므로 "': '"Since silhouette cannot distinguish max and mean+max in this data, "',
    '"날짜단위 CV macro F1을 함께 봅니다."': '"CV macro F1 by date is also considered."',
    '"Feature 품질 진단 실행"': '"Run Feature Quality Diagnostics"',
    '"feature set × 변환 × scope 조합을 교차검증하는 중..."': '"Cross-validating feature set × transform × scope combination..."',
    '"Feature 진단 CSV"': '"Feature Diagnostics CSV"',
    '"raw data 기반 AI 학습 데이터셋 생성"': '"Generate AI Training Dataset based on raw data"',
    '"256차원 Feature가 아닌 원본 128×3600 raw tensor를 저장합니다."': '"Saves the original 128×3600 raw tensor instead of 256-dimensional features."',
    'f"포함 그룹: {\', \'.join(ai_groups) if ai_groups else \'없음\'}"': 'f"Included Groups: {\', \'.join(ai_groups) if ai_groups else \'None\'}"',
    'f"출력 위치: {Path(ai_output_root_text).resolve()}"': 'f"Output Location: {Path(ai_output_root_text).resolve()}"',
    '"선택 그룹 AI 데이터셋 생성"': '"Generate AI Dataset for Selected Groups"',
    '"최소 하나의 그룹을 선택하세요."': '"Please select at least one group."',
    'f"AI 데이터셋 생성 완료: {output_folder}"': 'f"AI Dataset Generation Complete: {output_folder}"',
    'f"AI 데이터셋 생성 실패: {error}"': 'f"AI Dataset Generation Failed: {error}"',
}

# The replacements dictionary has exact string matches (including quotes).
for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete.")
