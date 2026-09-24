import os

replacements = {
    "부분방전 상태판정 문제": "PD_Status_Determination_Issue",
    "부분방전": "Partial_Discharge",
    "안인개폐소": "Anin_Switchyard",
    "당진화력": "Dangjin_Thermal_Power",
    "해남변환소": "Haenam_Converter_Station",
    "전력연구원": "KEPRI",
    "신당진": "Sindangjin",
    "신가평": "Singapyeong",
    "신경산": "Singyeongsan",
    "신중부": "Sinjungbu",
    "신안성": "Sinanseong",
    "신태백": "Sintaebaek",
    "균압콘덴서": "Grading_Capacitor",
    "지지애자": "Support_Insulator",
    "노이즈제거": "Noise_Removal",
    "인위생성": "Artificially_Generated",
    "임의생성": "Randomly_Generated",
    "미소방전": "Micro_Discharge",
    "상태판정": "Status_Determination",
    "개선후": "After_Improvement",
    "개선전": "Before_Improvement",
    "스페이서": "Spacer",
    "부싱측": "Bushing_Side",
    "선로 측": "Line_Side",
    "의심됨": "Suspected",
    "디에스": "DS",
    "코로나": "Corona",
    "파티클": "Particle",
    "플로팅": "Floating",
    "보이드": "Void",
    "오일": "Oil",
    "가보": "Gabo",
    "용성": "Yongseong",
    "신서": "Sinseo",
    "순창": "Sunchang",
    "유호": "Yuho",
    "한빛": "Hanbit",
    "아산": "Asan",
    "가압": "Energization",
    "현장": "Field",
    "문제": "Issue",
    "내장": "Internal",
    "외장": "External",
    "증설": "Expansion",
    "A상": "Phase_A",
    "B상": "Phase_B",
    "C상": "Phase_C",
    "2차": "2nd",
    "측": "Side",
    "랩": "Lab",
    "╖ª": "Lab"
}

def translate_name(name):
    new_name = name
    for k, v in replacements.items():
        new_name = new_name.replace(k, v)
    return new_name

dirs = ['Data/by_date', 'Data/by_type']

for d in dirs:
    for root, dirs_list, files in os.walk(d, topdown=False):
        for name in files:
            new_name = translate_name(name)
            if new_name != name:
                src = os.path.join(root, name)
                dst = os.path.join(root, new_name)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                else:
                    if os.path.isdir(src):
                        import shutil
                        for item in os.listdir(src):
                            try:
                                shutil.move(os.path.join(src, item), dst)
                            except Exception as e:
                                print(f"Error moving {item} to {dst}: {e}")
                        try:
                            os.rmdir(src)
                        except Exception:
                            pass
                    else:
                        print(f"File {dst} already exists.")
        
        for name in dirs_list:
            new_name = translate_name(name)
            if new_name != name:
                src = os.path.join(root, name)
                dst = os.path.join(root, new_name)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                else:
                    if os.path.isdir(src):
                        import shutil
                        for item in os.listdir(src):
                            try:
                                shutil.move(os.path.join(src, item), dst)
                            except Exception as e:
                                print(f"Error moving {item} to {dst}: {e}")
                        try:
                            os.rmdir(src)
                        except Exception:
                            pass
                    else:
                        print(f"Directory name collision: {dst} already exists.")

print("Renaming complete.")
