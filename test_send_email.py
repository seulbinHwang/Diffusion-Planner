#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import getpass
import json
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class EmailSettings:
    """메일을 보내기 위한 설정값 묶음입니다.

    이 값들은 ~/.gpu_node_watcher/config.json 에 저장된 것을 그대로 읽어옵니다.

    Args:
        smtp_host (str): 메일을 보내는 서버 주소(예: smtp.gmail.com).
        smtp_port (int): 메일을 보내는 서버 포트(예: 587).
        smtp_username (str): 메일 서버 로그인 아이디(대부분 이메일 주소).
        from_email (str): 받는 사람이 보게 될 '보낸 사람' 주소.
        to_email (str): 기본 받는 사람 주소.
        use_starttls (bool): 안전한 연결로 바꿔서 보낼지 여부(보통 True).
    """
    smtp_host: str
    smtp_port: int
    smtp_username: str
    from_email: str
    to_email: str
    use_starttls: bool


def get_app_dir() -> Path:
    """이 프로그램이 참조할 폴더 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher
    """
    return Path.home() / ".gpu_node_watcher"


def get_config_path() -> Path:
    """gpu_node_watcher 설정 파일 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher/config.json
    """
    return get_app_dir() / "config.json"


def get_password_file_path() -> Path:
    """file 방식으로 저장된 비밀번호 파일 경로를 돌려줍니다.

    Returns:
        Path: 예) ~/.gpu_node_watcher/email_password.txt
    """
    return get_app_dir() / "email_password.txt"


def read_json(path: Path) -> dict[str, Any]:
    """JSON 파일을 읽어 dict로 돌려줍니다.

    Args:
        path (Path): JSON 파일 경로.

    Returns:
        dict[str, Any]: JSON 내용.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def set_private_permissions(path: Path) -> None:
    """파일 권한을 가능한 한 '나만 읽기/쓰기'로 제한합니다.

    Args:
        path (Path): 대상 파일 경로.
    """
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def load_email_settings() -> tuple[EmailSettings, str]:
    """기존 gpu_node_watcher 설정 파일에서 이메일 설정과 비밀번호 저장 방식을 읽습니다.

    Returns:
        tuple[EmailSettings, str]:
            - EmailSettings: 이메일 발송 설정
            - str: password_storage ("keyring" 또는 "file")
    """
    cfg_path = get_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {cfg_path} (gpu_node_watcher.py --init 먼저 실행해줘)")

    raw = read_json(cfg_path)
    email = EmailSettings(**raw["email"])
    password_storage = str(raw.get("password_storage", "keyring"))
    return email, password_storage


def try_keyring_get(service: str, username: str) -> Optional[str]:
    """OS 비밀번호 저장소(keyring)에서 비밀번호를 읽어옵니다.

    Args:
        service (str): 저장할 때 썼던 서비스 이름.
        username (str): 계정 이름(여기서는 이메일).

    Returns:
        Optional[str]: 비밀번호가 있으면 문자열, 없으면 None.
    """
    try:
        import keyring  # type: ignore
        return keyring.get_password(service, username)
    except Exception:
        return None


def try_keyring_set(service: str, username: str, password: str) -> bool:
    """OS 비밀번호 저장소(keyring)에 비밀번호를 저장합니다.

    Args:
        service (str): 서비스 이름.
        username (str): 계정 이름.
        password (str): 비밀번호.

    Returns:
        bool: 성공하면 True, 실패하면 False.
    """
    try:
        import keyring  # type: ignore
        keyring.set_password(service, username, password)
        return True
    except Exception:
        return False


def load_password_from_file(path: Path) -> Optional[str]:
    """file 방식 비밀번호 파일에서 비밀번호를 읽어옵니다.

    Args:
        path (Path): 비밀번호 파일 경로.

    Returns:
        Optional[str]: 비밀번호가 있으면 문자열, 없으면 None.
    """
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def save_password_to_file(path: Path, password: str) -> None:
    """file 방식 비밀번호 파일에 비밀번호를 저장합니다.

    Args:
        path (Path): 비밀번호 파일 경로.
        password (str): 비밀번호.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(password, encoding="utf-8")
    set_private_permissions(path)


def get_or_ask_password(password_storage: str, smtp_username: str) -> str:
    """비밀번호를 가져옵니다. 없으면 한 번만 입력받아 저장합니다.

    Args:
        password_storage (str): "keyring" 또는 "file".
        smtp_username (str): SMTP 로그인 아이디(대부분 이메일 주소).

    Returns:
        str: 비밀번호(보통 앱 비밀번호 16자리).
    """
    service_name = "gpu_node_watcher_email"

    if password_storage == "keyring":
        saved = try_keyring_get(service_name, smtp_username)
        if saved:
            return saved

        pw = getpass.getpass("앱 비밀번호(16자리) 입력: ").strip().replace(" ", "")
        ok = try_keyring_set(service_name, smtp_username, pw)
        if ok:
            print("키링에 비밀번호 저장 완료")
            return pw

        print("키링 저장이 실패해서, file 방식으로 저장합니다.")
        save_password_to_file(get_password_file_path(), pw)
        return pw

    # file 방식
    saved_file = load_password_from_file(get_password_file_path())
    if saved_file:
        return saved_file

    pw = getpass.getpass("앱 비밀번호(16자리) 입력: ").strip().replace(" ", "")
    save_password_to_file(get_password_file_path(), pw)
    print(f"파일에 비밀번호 저장 완료: {get_password_file_path()}")
    return pw


def send_test_email(settings: EmailSettings, smtp_password: str, to_email: str, subject: str, body: str) -> None:
    """테스트 이메일을 1통 보냅니다.

    Args:
        settings (EmailSettings): 이메일 설정.
        smtp_password (str): 앱 비밀번호(16자리).
        to_email (str): 받는 사람 이메일.
        subject (str): 이메일 제목.
        body (str): 이메일 본문.

    Raises:
        RuntimeError: 전송 실패 시 예외를 발생시킵니다.
    """
    msg = EmailMessage()
    msg["From"] = settings.from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if settings.use_starttls:
            context = ssl.create_default_context()
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(settings.smtp_username, smtp_password)
                server.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=30) as server:
                server.login(settings.smtp_username, smtp_password)
                server.send_message(msg)
    except Exception as e:
        raise RuntimeError(f"메일 전송 실패: {e}") from e


def main() -> None:
    """스크립트 실행 시작점입니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", default=None, help="받는 사람 이메일(기본값: config의 to_email)")
    parser.add_argument("--subject", default="[gpu_node_watcher] 테스트 메일", help="메일 제목")
    parser.add_argument("--body", default="이 메일은 gpu_node_watcher 메일 발송 테스트입니다.", help="메일 본문")
    args = parser.parse_args()

    email_settings, password_storage = load_email_settings()
    pw = get_or_ask_password(password_storage=password_storage, smtp_username=email_settings.smtp_username)

    to_email = args.to if args.to else email_settings.to_email
    send_test_email(
        settings=email_settings,
        smtp_password=pw,
        to_email=to_email,
        subject=args.subject,
        body=args.body,
    )
    print(f"테스트 메일 발송 성공 -> {to_email}")


if __name__ == "__main__":
    main()
