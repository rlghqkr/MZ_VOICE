"""
GraphRAG Local Search with Sources (v2.7.0 compatible)
답변과 함께 근거 문서(sources)를 확인할 수 있는 검색 스크립트

사용법:
    python search_with_sources.py "질문 내용"
    python search_with_sources.py "질문 내용" --method global
"""

import asyncio
import sys
import os
from pathlib import Path

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
import pandas as pd
from graphrag.api import local_search, global_search
from graphrag.config.create_graphrag_config import create_graphrag_config


def load_config(root_dir: Path):
    """GraphRAG 설정 로드"""
    # .env 파일 로드 (root_dir 기준)
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    settings_path = root_dir / "settings.yaml"
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    # 환경 변수 치환: ${VAR_NAME} 형식을 실제 값으로 변경
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if 'models' in settings:
        for model_name, model_config in settings['models'].items():
            if isinstance(model_config, dict) and 'api_key' in model_config:
                if model_config['api_key'] == '${GRAPHRAG_API_KEY}':
                    model_config['api_key'] = api_key

    config = create_graphrag_config(values=settings, root_dir=str(root_dir))
    return config


def load_data(root_dir: Path):
    """GraphRAG 출력 데이터 로드"""
    output_dir = root_dir / "output"

    entities = pd.read_parquet(output_dir / "entities.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")

    # covariates는 선택적
    covariates_path = output_dir / "covariates.parquet"
    covariates = pd.read_parquet(covariates_path) if covariates_path.exists() else None

    return entities, communities, community_reports, text_units, relationships, covariates


async def search_local_query(query: str, root_dir: Path):
    """Local Search 실행 및 결과 출력"""
    print("=" * 60)
    print(f"[LOCAL SEARCH] 질문: {query}")
    print("=" * 60)

    # 설정 및 데이터 로드
    config = load_config(root_dir)
    entities, communities, community_reports, text_units, relationships, covariates = load_data(root_dir)

    # 검색 실행
    response, context_data = await local_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=covariates,
        community_level=2,
        response_type="multiple paragraphs",
        query=query,
    )

    # 답변 출력
    print("\n[답변]")
    print("-" * 60)
    print(response)

    # Sources 출력
    print("\n" + "=" * 60)
    print("[근거 자료 (Sources)]")
    print("=" * 60)

    if isinstance(context_data, dict):
        # 관련 엔티티
        if 'entities' in context_data:
            print("\n[관련 엔티티]")
            print("-" * 40)
            entities_df = context_data['entities']
            if isinstance(entities_df, pd.DataFrame) and not entities_df.empty:
                for _, row in entities_df.head(10).iterrows():
                    title = row.get('title', row.get('name', row.get('entity', 'N/A')))
                    desc = str(row.get('description', ''))
                    desc = desc[:100] + '...' if len(desc) > 100 else desc
                    print(f"  - {title}: {desc}")
            else:
                print("  (없음)")

        # 관련 텍스트 청크
        if 'sources' in context_data or 'text_units' in context_data:
            print("\n[관련 텍스트 청크 - 원본 근거]")
            print("-" * 40)
            sources = context_data.get('sources', context_data.get('text_units'))
            if isinstance(sources, pd.DataFrame) and not sources.empty:
                for i, (_, row) in enumerate(sources.head(5).iterrows()):
                    text = str(row.get('text', ''))
                    text = text[:300] + '...' if len(text) > 300 else text
                    print(f"\n  [{i+1}] {text}")
            elif isinstance(sources, list):
                for i, source in enumerate(sources[:5]):
                    text = str(source)[:300] + '...' if len(str(source)) > 300 else str(source)
                    print(f"\n  [{i+1}] {text}")
            else:
                print("  (없음)")

        # 관련 커뮤니티 리포트
        if 'reports' in context_data or 'community_reports' in context_data:
            print("\n[관련 커뮤니티 리포트]")
            print("-" * 40)
            reports_data = context_data.get('reports', context_data.get('community_reports'))
            if isinstance(reports_data, pd.DataFrame) and not reports_data.empty:
                for _, row in reports_data.head(3).iterrows():
                    title = row.get('title', 'N/A')
                    summary = str(row.get('summary', ''))
                    summary = summary[:150] + '...' if len(summary) > 150 else summary
                    print(f"\n  - {title}")
                    print(f"    {summary}")
            else:
                print("  (없음)")

    elif isinstance(context_data, list):
        print("\n[컨텍스트 데이터]")
        for i, df in enumerate(context_data[:3]):
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"\n  [DataFrame {i+1}] columns: {list(df.columns)[:5]}")

    return response, context_data


async def search_global_query(query: str, root_dir: Path):
    """Global Search 실행 및 결과 출력"""
    print("=" * 60)
    print(f"[GLOBAL SEARCH] 질문: {query}")
    print("=" * 60)

    # 설정 및 데이터 로드
    config = load_config(root_dir)
    entities, communities, community_reports, text_units, relationships, covariates = load_data(root_dir)

    # 검색 실행
    response, context_data = await global_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="multiple paragraphs",
        query=query,
    )

    # 답변 출력
    print("\n[답변]")
    print("-" * 60)
    print(response)

    # Context 정보 출력
    if isinstance(context_data, dict):
        print("\n" + "=" * 60)
        print("[사용된 커뮤니티 리포트]")
        print("=" * 60)

        if 'reports' in context_data:
            reports_data = context_data['reports']
            if isinstance(reports_data, pd.DataFrame) and not reports_data.empty:
                for _, row in reports_data.head(5).iterrows():
                    title = row.get('title', 'N/A')
                    print(f"  - {title}")

    return response, context_data


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    method = "local"

    if len(sys.argv) >= 4 and sys.argv[2] == "--method":
        method = sys.argv[3]

    root_dir = Path(__file__).parent

    if method == "local":
        await search_local_query(query, root_dir)
    elif method == "global":
        await search_global_query(query, root_dir)
    else:
        print(f"알 수 없는 method: {method}")
        print("사용 가능: local, global")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
