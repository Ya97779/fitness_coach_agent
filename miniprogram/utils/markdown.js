/**
 * Markdown 解析器（带容错）
 * 处理标准格式 + LLM 常见错误
 */

function parseMarkdown(text) {
  if (!text) return ''

  let html = text

  // 转义 HTML 特殊字符
  html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  // 表格：标准格式 | header | header |
  html = html.replace(/((?:\|.+\|[\r\n]?)+)/g, function(match) {
    const lines = match.trim().split('\n').filter(line => line.trim())
    if (lines.length < 2) return match

    // 检查第二行是否是分隔行 (|---|---|)
    if (!/^\|[\s\-:|]+\|$/.test(lines[1].trim())) return match

    let table = '<table>'

    // 表头
    const headers = parseTableRow(lines[0])
    table += '<thead><tr>'
    headers.forEach(h => { table += '<th>' + h.trim() + '</th>' })
    table += '</tr></thead>'

    // 表体
    table += '<tbody>'
    for (let i = 2; i < lines.length; i++) {
      const cells = parseTableRow(lines[i])
      table += '<tr>'
      cells.forEach(c => { table += '<td>' + c.trim() + '</td>' })
      table += '</tr>'
    }
    table += '</tbody></table>'

    return table
  })

  // 标题：容错处理 # 后有无空格都行
  html = html.replace(/^#{4}\s+(.+)$/gm, '<h4>$1</h4>')
  html = html.replace(/^#{3}\s+(.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^#{2}\s+(.+)$/gm, '<h2>$1</h2>')
  html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>')
  // 处理 # 后无空格的情况（如 ###标题）
  html = html.replace(/^#{4}(\S.+)$/gm, '<h4>$1</h4>')
  html = html.replace(/^#{3}(\S.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^#{2}(\S.+)$/gm, '<h2>$1</h2>')

  // 加粗：标准格式 **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')

  // 斜体：标准格式 *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')

  // 无序列表：容错处理 - 后有无空格都行
  html = html.replace(/^[-·]\s+(.+)$/gm, '<li>$1</li>')
  html = html.replace(/^[-·](\S.+)$/gm, '<li>$1</li>')
  // 连续 <li> 包裹成 <ul>
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')

  // 有序列表：容错处理 1. 后有无空格都行
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
  html = html.replace(/^\d+\.(\S.+)$/gm, '<li>$1</li>')

  // 换行：两个换行变段落，一个换行变 <br>
  html = html.replace(/\n\n/g, '</p><p>')
  html = html.replace(/\n/g, '<br>')

  // 包裹段落
  html = '<p>' + html + '</p>'

  // 清理：把块级元素从 <p> 中释放，清理多余的 <br>
  html = html.replace(/<p><\/p>/g, '')
  html = html.replace(/<p>(<h[1-6]>)/g, '$1')
  html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1')
  html = html.replace(/<p>(<ul>)/g, '$1')
  html = html.replace(/(<\/ul>)<\/p>/g, '$1')
  html = html.replace(/<p>(<table>)/g, '$1')
  html = html.replace(/(<\/table>)<\/p>/g, '$1')
  html = html.replace(/<br>(<\/?(?:h[1-6]|ul|table|li)>)/g, '$1')
  html = html.replace(/(<\/?(?:h[1-6]|ul|table|li)>)<br>/g, '$1')

  return html
}

function parseTableRow(line) {
  return line.replace(/^\||\|$/g, '').split('|')
}

module.exports = { parse: parseMarkdown }
