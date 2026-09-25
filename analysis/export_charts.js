// 리포트 차트를 독립 SVG로 추출: output/report.html을 http://localhost:8765/_preview.html 로 띄운 뒤 NODE_PATH=$(npm root -g) node analysis/export_charts.js
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const OUT = path.resolve(__dirname, '../output/charts') + '/';
const CHARTS = [
  { sel: '#forest', file: '01_답변을_막는_문구특성_오즈비.svg',
    title: '무엇이 답변을 막는가: 문구 특성별 답변 오즈비',
    sub: '이항 GLM(관심사·레벨 통제), 노출 1회 이상 질문 748개 · 막대 = 오즈비, 선 = 95% CI · 1보다 왼쪽이면 답변을 줄임',
    legend: [['#d95926','유의하게 낮춤 (p<0.05)'],['#2a78d6','유의하게 높임'],['#9aa39e','유의하지 않음']] },
  { sel: '#cats', file: '02_관심사별_답변율과_노출.svg',
    title: '관심사별 답변율과 노출 쏠림',
    sub: '점 = 답변율, 선 = 95% Wilson CI, 점선 = 전체 평균 · 오른쪽 숫자 = 질문 1개당 평균 노출',
    legend: [['#2a78d6','답변율 (95% CI)']] },
  { sel: '#cvChart', file: '03_모델별_교차검증_예측개선.svg',
    title: '어떤 모델이 처음 보는 질문을 더 잘 맞히나',
    sub: '질문 단위 10겹 교차검증 × 5회 · 기준선(전체 평균) 대비 검증 로그손실 개선율',
    legend: [] },
  { sel: '#fmtChart', file: '04_질문형식별_답변율과_비중.svg',
    title: '질문 형식별 답변율과 비중',
    sub: '결론 난 노출(답변+교체) 기준 답변율과 95% CI, 점선 = 전체 평균 · 오른쪽 숫자 = 전체 질문 중 비중',
    legend: [['#2a78d6','답변율 (95% CI)']] },
];
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 900, height: 900 }, colorScheme: 'light' });
  await p.goto('http://localhost:8765/_preview.html');
  await p.waitForTimeout(1200);
  for (const c of CHARTS) {
    const svg = await p.evaluate((c) => {
      const src = document.querySelector(c.sel + ' svg');
      const vb = src.viewBox.baseVal;
      const clone = src.cloneNode(true);
      // 계산된 스타일을 속성으로 고정 (CSS 변수·클래스 제거)
      const a = [src, ...src.querySelectorAll('*')], z = [clone, ...clone.querySelectorAll('*')];
      a.forEach((el, i) => {
        const cs = getComputedStyle(el), t = z[i];
        const tag = el.tagName.toLowerCase();
        if (tag === 'rect' && el.getAttribute('fill') === 'transparent') { t.remove(); return; }
        if (tag !== 'svg' && tag !== 'g') {
          if (cs.fill && cs.fill !== 'none') t.setAttribute('fill', cs.fill); else if (tag !== 'text') t.setAttribute('fill', 'none');
          if (cs.stroke && cs.stroke !== 'none') t.setAttribute('stroke', cs.stroke);
          if (cs.opacity !== '1') t.setAttribute('opacity', cs.opacity);
        }
        if (tag === 'text') {
          t.setAttribute('font-size', cs.fontSize);
          t.setAttribute('font-weight', cs.fontWeight);
          t.setAttribute('font-family', /Mono/.test(cs.fontFamily) ? "'IBM Plex Mono', Menlo, Consolas, monospace" : "'IBM Plex Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif");
        }
        t.removeAttribute('class'); t.removeAttribute('style'); t.removeAttribute('data-i');
      });
      // 제목·설명·범례 영역 추가
      const pad = 24, head = 70 + (c.legend.length ? 24 : 0), foot = 28;
      const W = vb.width + pad * 2, H = vb.height + head + foot;
      const NS = 'http://www.w3.org/2000/svg';
      const out = document.createElementNS(NS, 'svg');
      out.setAttribute('xmlns', NS); out.setAttribute('viewBox', `0 0 ${W} ${H}`);
      out.setAttribute('width', W); out.setAttribute('height', H);
      out.setAttribute('role', 'img'); out.setAttribute('aria-label', c.title);
      const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      const sans = "'IBM Plex Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif";
      let hdr = `<title>${esc(c.title)}</title><rect width="${W}" height="${H}" fill="#ffffff"/>`
        + `<text x="${pad}" y="34" font-family="${sans}" font-size="18" font-weight="600" fill="#18201d">${esc(c.title)}</text>`
        + `<text x="${pad}" y="56" font-family="${sans}" font-size="11.5" fill="#6b746f">${esc(c.sub)}</text>`;
      let lx = pad;
      c.legend.forEach(([col, lab]) => { hdr += `<rect x="${lx}" y="72" width="10" height="10" rx="2" fill="${col}"/><text x="${lx + 16}" y="81" font-family="${sans}" font-size="11.5" fill="#4d5853">${esc(lab)}</text>`; lx += 30 + lab.length * 11; });
      hdr += `<text x="${pad}" y="${H - 10}" font-family="${sans}" font-size="10.5" fill="#8a938e">나답 데일리 질문 진단 · 데이터 2026-08-27 ~ 2026-09-25</text>`;
      out.innerHTML = hdr;
      const g = document.createElementNS(NS, 'g');
      g.setAttribute('transform', `translate(${pad - vb.x},${head - vb.y})`);
      [...clone.childNodes].forEach(n => g.appendChild(n));
      out.appendChild(g);
      return '<?xml version="1.0" encoding="UTF-8"?>\n' + out.outerHTML;
    }, c);
    fs.writeFileSync(OUT + c.file, svg);
    console.log('wrote', c.file, svg.length);
  }
  // 확인용 렌더
  const q = await b.newPage({ viewport: { width: 860, height: 600 } });
  const imgs = CHARTS.map(c => `<img src="file://${OUT}${encodeURIComponent(c.file)}" style="display:block;margin:8px;border:1px solid #ddd">`).join('');
  fs.writeFileSync(OUT + '_check.html', '<meta charset="utf-8"><body style="background:#eee">' + imgs);
  await q.goto('file://' + OUT + '_check.html'); await q.waitForTimeout(500);
  await q.screenshot({ path: OUT + '_check.png', fullPage: true });
  await b.close();
})();
