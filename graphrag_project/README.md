# 법령 GraphRAG 프로젝트

한국 법령 데이터를 Microsoft GraphRAG로 처리하여 지식 그래프 기반 RAG를 구현합니다.

## 설치

```bash
pip install graphrag pandas pyarrow
```

## 사용 방법

### 1단계: 환경변수 설정

```bash
# Windows (PowerShell)
$env:GRAPHRAG_API_KEY = "your-openai-api-key"

# Windows (CMD)
set GRAPHRAG_API_KEY=your-openai-api-key

# Linux/Mac
export GRAPHRAG_API_KEY="your-openai-api-key"
```

### 2단계: 데이터 준비

```bash
cd graphrag_project
python prepare_data.py
```

이 스크립트는 `법령.json`을 읽어 각 법령을 별도의 텍스트 파일로 `input/` 폴더에 저장합니다.

### 3단계: 인덱싱 (지식 그래프 생성)

```bash
python run_graphrag.py index
```

또는 CLI 직접 사용:

```bash
graphrag index --root ./
```

**주의**: 인덱싱은 LLM API를 많이 호출하므로 비용이 발생합니다. 먼저 소규모 샘플로 테스트하세요.

### 4단계: 쿼리 실행

```bash
# 로컬 검색 (특정 엔티티/관계 질문)
python run_graphrag.py query "5·18민주유공자의 자격 요건은?"

# 글로벌 검색 (전체적인 요약/개요 질문)
python run_graphrag.py global "이 법령들의 주요 주제는 무엇인가?"
```

## 프로젝트 구조

```
graphrag_project/
├── input/              # 변환된 법령 텍스트 파일
├── output/             # 인덱싱 결과 (parquet 파일)
├── cache/              # LLM 응답 캐시
├── prompts/            # 한국어 맞춤 프롬프트
│   ├── entity_extraction.txt
│   ├── summarize_descriptions.txt
│   └── community_report.txt
├── settings.yaml       # GraphRAG 설정
├── prepare_data.py     # 데이터 변환 스크립트
├── run_graphrag.py     # 인덱싱/쿼리 실행 스크립트
└── README.md
```

## 검색 유형 설명

### Local Search (로컬 검색)
- 특정 엔티티나 관계에 대한 구체적인 질문
- 예: "국가보훈부의 역할은?", "제4조의 내용은?"

### Global Search (글로벌 검색)
- 전체 데이터셋에 걸친 요약이나 패턴 파악
- 예: "법령들의 공통적인 구조는?", "가장 많이 참조되는 법령은?"

## 비용 최적화 팁

1. **모델 선택**: `gpt-4o-mini` 사용으로 비용 절감
2. **샘플 테스트**: 전체 인덱싱 전에 10-20개 파일로 테스트
3. **캐시 활용**: 동일 쿼리는 캐시에서 응답
4. **청크 크기**: `settings.yaml`의 `chunks.size` 조정

## 커스터마이징

### 엔티티 타입 수정
`settings.yaml`의 `entity_extraction.entity_types`와
`prompts/entity_extraction.txt`를 수정하세요.

현재 설정된 엔티티 타입:
- 법령, 조항, 기관, 용어, 대상자, 요건, 벌칙, 날짜

### Azure OpenAI 사용
`settings.yaml`에서 LLM 설정 부분을 수정하세요.
