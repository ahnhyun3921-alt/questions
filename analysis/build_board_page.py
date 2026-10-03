"""board/index.src.html에 나눔스퀘어 폰트(board/fonts/*.woff2)를 data URI로 넣어 board/dist/index.html을 만든다.

아티팩트 CSP가 Google Fonts 밖의 폰트 파일을 막아서, 폰트는 페이지 안에 넣는다.
게시용 결과에서는 주석(JS·CSS·HTML)을 모두 지운다(analysis/strip_comments.js, typescript 필요).
사용법: python analysis/build_board_page.py
"""
import base64
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "board"
FACES = [("NanumSquareR.woff2", 400), ("NanumSquareB.woff2", 700), ("NanumSquareEB.woff2", 800)]


def main():
    css = []
    for fn, w in FACES:
        b64 = base64.b64encode((B / "fonts" / fn).read_bytes()).decode()
        css.append(f'@font-face{{font-family:"NanumSquare";font-weight:{w};font-style:normal;font-display:swap;'
                   f'src:url(data:font/woff2;base64,{b64}) format("woff2")}}')
    src = (B / "index.src.html").read_text(encoding="utf-8")
    assert "/*FONTS*/" in src
    out = B / "dist" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src.replace("/*FONTS*/", "\n".join(css), 1), encoding="utf-8")
    # 게시용에서는 주석을 모두 뺀다(원본 board/index.src.html에는 남김)
    env = {**os.environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()}
    subprocess.run(["node", str(ROOT / "analysis" / "strip_comments.js"), str(out), str(out)], check=True, env=env)
    print(f"{out} · {out.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
