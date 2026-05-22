/**
 * Markdown 解析器（带容错）
 * 处理标准格式 + LLM 常见错误
 */

function parseMarkdown(text) {
  if (!text) return ''

  // 统一换行符
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

  // 预处理：在 markdown 语法前插入换行（处理 LLM 不换行的情况）
  // 只在非行首的 # 前插入换行
  text = text.replace(/([^\n])(#{1,4}\s)/g, '$1\n$2')

  const lines = text.split('\n').map(l => l.trim()).filter(l => l)
  const blocks = []
  let tableLines = []
  let listItems = []

  function flushTable() {
    if (tableLines.length < 2) {
      tableLines.forEach(l => blocks.push('<p>' + inlineFormat(escapeHtml(l)) + '</p>'))
      tableLines = []
      return
    }
    const sepLine = tableLines[1].replace(/^\||\|$/g, '')
    const sepCells = sepLine.split('|').map(s => s.trim())
    if (!sepCells.every(c => /^[-:]+$/.test(c))) {
      tableLines.forEach(l => blocks.push('<p>' + inlineFormat(escapeHtml(l)) + '</p>'))
      tableLines = []
      return
    }
    let table = '<table><thead><tr>'
    parseTableRow(tableLines[0]).forEach(h => { table += '<th>' + escapeHtml(h.trim()) + '</th>' })
    table += '</tr></thead><tbody>'
    for (let i = 2; i < tableLines.length; i++) {
      table += '<tr>'
      parseTableRow(tableLines[i]).forEach(c => { table += '<td>' + escapeHtml(c.trim()) + '</td>' })
      table += '</tr>'
    }
    table += '</tbody></table>'
    blocks.push(table)
    tableLines = []
  }

  function flushList() {
    if (listItems.length === 0) return
    blocks.push('<ul>' + listItems.map(item => '<li>' + item + '</li>').join('') + '</ul>')
    listItems = []
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // 跳过纯分隔符行和单个 #
    if (/^[-*·]{3,}$/.test(line) || /^={3,}$/.test(line) || /^#$/.test(line)) continue

    // 表格行：包含 | 且有内容
    if (line.includes('|') && line.replace(/\|/g, '').trim().length > 0) {
      const stripped = line.replace(/^\||\|$/g, '')
      const cells = stripped.split('|').map(s => s.trim())
      // 检查是否是分隔行（只包含 -、:、空格）
      const isSepRow = cells.every(c => /^[\s\-:]+$/.test(c)) && cells.some(c => c.includes('-'))
      if (isSepRow) {
        tableLines.push(line)
        continue
      }
      flushList()
      tableLines.push(line)
      continue
    }

    if (tableLines.length > 0) {
      flushTable()
    }

    // 标题
    const headerMatch = line.match(/^(#{1,4})\s*(.+)$/)
    if (headerMatch) {
      flushList()
      const level = headerMatch[1].length
      let headerText = headerMatch[2].trim()

      // 容错：标题后紧跟正文（超过20字符可能包含正文）
      if (headerText.length > 20) {
        const splitPatterns = [
          /^(.{2,15}?[：:。！？\s])(.+)$/,
          /^(.{2,15}?[。！？])(.+)$/,
          /^(.{2,15}?[\s])(.+)$/
        ]
        let split = false
        for (const pattern of splitPatterns) {
          const match = headerText.match(pattern)
          if (match && match[2].length > 5) {
            blocks.push('<h' + level + '>' + inlineFormat(escapeHtml(match[1].trim())) + '</h' + level + '>')
            blocks.push('<p>' + inlineFormat(escapeHtml(match[2].trim())) + '</p>')
            split = true
            break
          }
        }
        if (!split) {
          blocks.push('<h' + level + '>' + inlineFormat(escapeHtml(headerText)) + '</h' + level + '>')
        }
        continue
      }
      blocks.push('<h' + level + '>' + inlineFormat(escapeHtml(headerText)) + '</h' + level + '>')
      continue
    }

    // 无序列表：- 或 * 开头（但不是 ** 加粗标记）
    const listMatch = line.match(/^([-·]|\*(?!\*))\s*(.+)$/)
    if (listMatch && !line.startsWith('**')) {
      flushTable()
      listItems.push(inlineFormat(escapeHtml(listMatch[2])))
      continue
    }

    // 有序列表：1. 开头
    const orderedMatch = line.match(/^\d+\.\s*(.+)$/)
    if (orderedMatch) {
      flushTable()
      listItems.push(inlineFormat(escapeHtml(orderedMatch[1])))
      continue
    }

    // 普通文本
    flushList()
    blocks.push('<p>' + inlineFormat(escapeHtml(line)) + '</p>')
  }

  flushTable()
  flushList()

  return blocks.join('')
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inlineFormat(text) {
  // 加粗：**text**
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // 斜体：*text*（避免匹配列表标记和加粗标记）
  text = text.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
  return text
}

function parseTableRow(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|')
}

module.exports = { parse: parseMarkdown }
