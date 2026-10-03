// 게시용 HTML에서 주석을 지운다: <script> 안 JS 주석(TypeScript 스캐너로 문자열·정규식 보존), <style> 안 CSS 주석, HTML 주석
const fs = require('fs');
const ts = require('typescript');
const [src, dst] = process.argv.slice(2);
let html = fs.readFileSync(src, 'utf8');
html = html.replace(/<!--[\s\S]*?-->/g, '');
html = html.replace(/(<style[^>]*>)([\s\S]*?)(<\/style>)/g, (m, a, css, b) => a + css.replace(/\/\*[\s\S]*?\*\//g, '') + b);
html = html.replace(/(<script(?![^>]*\bsrc=)[^>]*>)([\s\S]*?)(<\/script>)/g, (m, a, js, b) => {
  const out = ts.transpileModule(js, { compilerOptions: { removeComments: true, target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.None } }).outputText;
  return a + out.replace(/^"use strict";\n?/, '').replace(/\n?export \{\};\s*$/, '') + b;
});
fs.writeFileSync(dst, html);
