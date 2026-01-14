"""
법령 JSON을 Microsoft GraphRAG 입력 형식으로 변환하는 스크립트
"""
import json
import os
from pathlib import Path

def prepare_law_data(json_path: str, output_dir: str):
    """
    법령 JSON 파일을 GraphRAG가 처리할 수 있는 텍스트 파일들로 변환

    Args:
        json_path: 법령.json 파일 경로
        output_dir: 출력 디렉토리 (input 폴더)
    """
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        laws = json.load(f)

    print(f"총 {len(laws)}개의 법령 문서 발견")

    for i, law in enumerate(laws):
        doc_id = law.get('doc_id', f'law_{i:04d}')
        content = law.get('content', '')
        source_file = law.get('source_file', '')

        # 메타데이터를 콘텐츠 앞에 추가 (GraphRAG가 컨텍스트로 활용)
        full_content = f"""문서ID: {doc_id}
원본파일: {source_file}
문서유형: {law.get('source_type', '법령')}

{content}
"""

        # 파일명에서 특수문자 제거
        safe_filename = doc_id.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"{safe_filename}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(laws)} 처리 완료...")

    print(f"\n완료! {len(laws)}개 파일이 {output_dir}에 생성되었습니다.")
    return len(laws)


if __name__ == "__main__":
    # 경로 설정
    BASE_DIR = Path(__file__).parent.parent
    JSON_PATH = BASE_DIR / "법령.json"
    OUTPUT_DIR = Path(__file__).parent / "input"

    prepare_law_data(str(JSON_PATH), str(OUTPUT_DIR))
