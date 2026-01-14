"""
로깅 유틸리티 - 색상 출력 지원
"""
import logging
import sys
from typing import Optional


def _is_console_handler(handler):
    """핸들러가 콘솔 출력용인지 확인"""
    return isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr)


def _log_with_color(logger: logging.Logger, message: str, color_code: str = ""):
    """
    콘솔에만 색상을 적용하여 로깅
    
    Args:
        logger: 로거 인스턴스
        message: 로그 메시지
        color_code: ANSI 색상 코드
    """
    # root logger의 handlers를 사용
    root_logger = logging.getLogger()
    all_handlers = root_logger.handlers if root_logger.handlers else logger.handlers
    
    # 콘솔 핸들러와 파일 핸들러 분리
    console_handlers = [h for h in all_handlers if _is_console_handler(h)]
    file_handlers = [h for h in all_handlers if not _is_console_handler(h)]
    
    RESET = '\033[0m'
    
    # 콘솔에는 색상 적용
    if console_handlers:
        colored_message = f"{color_code}{message}{RESET}" if color_code else message
        record = logger.makeRecord(
            logger.name, logging.INFO, "(logging_utils)", 0,
            colored_message, (), None
        )
        for handler in console_handlers:
            handler.emit(record)
    
    # 파일에는 색상 없이
    if file_handlers:
        record = logger.makeRecord(
            logger.name, logging.INFO, "(logging_utils)", 0,
            message, (), None
        )
        for handler in file_handlers:
            handler.emit(record)


def log_prompt(logger: logging.Logger, prompt_name: str, prompt_text: str, user_input: Optional[str] = None):
    """
    프롬프트를 색상과 함께 로깅합니다.
    
    Args:
        logger: 로거 인스턴스
        prompt_name: 프롬프트 이름
        prompt_text: 프롬프트 전체 텍스트
        user_input: 사용자 입력 (있는 경우)
    """
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    
    # 프롬프트 시작
    _log_with_color(logger, '='*60, CYAN)
    _log_with_color(logger, f"[PROMPT] {prompt_name}", CYAN + BOLD)
    _log_with_color(logger, '-'*60, CYAN)
    
    # 사용자 입력
    if user_input:
        _log_with_color(logger, "[User Input]", MAGENTA)
        _log_with_color(logger, user_input, MAGENTA)
        _log_with_color(logger, '-'*60, CYAN)
    
    # 프롬프트 내용 (Context 축약)
    _log_with_color(logger, "[Prompt Content]", CYAN)
    
    lines = prompt_text.strip().split('\n')
    in_context = False
    context_line_count = 0
    context_omitted = False
    
    for line in lines:
        # Context 섹션 감지
        if '## 컨텍스트' in line or 'Context:' in line or 'context:' in line:
            in_context = True
            _log_with_color(logger, line, CYAN)
            continue
        
        # Question/질문 섹션이 나오면 context 끝
        if in_context and ('## 질문' in line or 'Question:' in line or 'question:' in line):
            in_context = False
            if context_omitted:
                _log_with_color(logger, '...(로깅생략)...', CYAN)
                context_omitted = False
        
        # Context 내부는 일부만 출력
        if in_context:
            context_line_count += 1
            if context_line_count <= 5:
                _log_with_color(logger, line, CYAN)
            elif not context_omitted:
                _log_with_color(logger, '...(로깅생략)...', CYAN)
                context_omitted = True
        else:
            _log_with_color(logger, line, CYAN)
    
    # 프롬프트 종료
    _log_with_color(logger, '='*60, CYAN)


def log_llm_response(logger: logging.Logger, response: str, response_type: str = "LLM Response"):
    """
    LLM 응답을 색상과 함께 로깅합니다.
    
    Args:
        logger: 로거 인스턴스
        response: LLM 응답 텍스트
        response_type: 응답 타입 (기본: "LLM Response")
    """
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    
    _log_with_color(logger, '='*60, GREEN)
    _log_with_color(logger, f"[{response_type}]", GREEN + BOLD)
    _log_with_color(logger, '-'*60, GREEN)
    
    # 응답 내용 출력 (길이 제한)
    max_length = 500
    if len(response) > max_length:
        display_response = response[:max_length] + "..."
    else:
        display_response = response
    
    for line in display_response.strip().split('\n'):
        _log_with_color(logger, line, GREEN)
    
    _log_with_color(logger, '='*60, GREEN)
