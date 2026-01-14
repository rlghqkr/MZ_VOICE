"""
GraphRAG 인덱싱 및 쿼리 실행 스크립트

사용법:
1. 먼저 prepare_data.py를 실행하여 법령 데이터 준비
2. .env 파일에 GRAPHRAG_API_KEY 설정
3. 인덱싱: python run_graphrag.py index
4. 쿼리: python run_graphrag.py query "질문 내용"
"""
import subprocess
import sys
import os
from pathlib import Path

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 없으면 환경변수에서 직접 읽음


def run_indexing():
    """GraphRAG 인덱싱 실행 (CLI 사용)"""
    print("=" * 50)
    print("GraphRAG 인덱싱 시작")
    print("=" * 50)

    root_dir = Path(__file__).parent

    # GraphRAG CLI로 인덱싱 실행
    result = subprocess.run(
        ["graphrag", "index", "--root", str(root_dir)],
        cwd=str(root_dir),
        env={**os.environ, "GRAPHRAG_API_KEY": os.environ.get("GRAPHRAG_API_KEY", "")}
    )

    if result.returncode == 0:
        print("\n인덱싱 완료!")
        print(f"출력 위치: {root_dir / 'output'}")
    else:
        print(f"\n인덱싱 실패 (exit code: {result.returncode})")

    return result.returncode


def run_query(query: str, method: str = "local"):
    """GraphRAG 쿼리 실행 (CLI 사용)"""
    print(f"\n[{method.upper()} 검색] 질문: {query}\n")

    root_dir = Path(__file__).parent

    result = subprocess.run(
        ["graphrag", "query", "--root", str(root_dir), "--method", method, "--query", query],
        cwd=str(root_dir),
        env={**os.environ, "GRAPHRAG_API_KEY": os.environ.get("GRAPHRAG_API_KEY", "")}
    )

    return result.returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n사용 가능한 명령어:")
        print("  index              - 지식 그래프 인덱싱")
        print("  query \"질문\"       - 로컬 검색 (특정 엔티티 질문)")
        print("  global \"질문\"      - 글로벌 검색 (전체 요약 질문)")
        print("  init               - GraphRAG 초기화 (설정 파일 생성)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        # GraphRAG 초기화
        root_dir = Path(__file__).parent
        subprocess.run(["graphrag", "init", "--root", str(root_dir)])

    elif command == "index":
        run_indexing()

    elif command == "query" or command == "local":
        if len(sys.argv) < 3:
            print("사용법: python run_graphrag.py query \"질문 내용\"")
            sys.exit(1)
        query = sys.argv[2]
        run_query(query, "local")

    elif command == "global":
        if len(sys.argv) < 3:
            print("사용법: python run_graphrag.py global \"질문 내용\"")
            sys.exit(1)
        query = sys.argv[2]
        run_query(query, "global")

    else:
        print(f"알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: init, index, query, global")
        sys.exit(1)


if __name__ == "__main__":
    main()
