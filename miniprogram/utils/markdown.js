/**
 * 轻量 markdown 解析器
 * 只处理常用语法：标题、加粗、斜体、列表、换行、代码
 */

function parseMarkdown(text) {
  if (!text) return ''

  let html = text

  // 转义 HTML 特殊字符
  html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  // 代码块 ```
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')

  // 行内代码 `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')

  // 标题 ### / ## / #
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>')
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>')

  // 加粗 **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')

  // 斜体 *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')

  // 无序列表 - item 或 · item
  html = html.replace(/^[\-\·] (.+)$/gm, '<li>$1</li>')
  // 连续 <li> 包裹成 <ul>
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')

  // 有序列表 1. item
  html = html.replace(/^\d+\.\s(.+)$/gm, '<li>$1</li>')

  // 换行：两个换行变段落，一个换行变 <br>
  html = html.replace(/\n\n/g, '</p><p>')
  html = html.replace(/\n/g, '<br>')

  // 包裹段落
  html = '<p>' + html + '</p>'

  // 清理空段落
  html = html.replace(/<p><\/p>/g, '')
  html = html.replace(/<p>(<h[1-6]>)/g, '$1')
  html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1')
  html = html.replace(/<p>(<ul>)/g, '$1')
  html = html.replace(/(<\/ul>)<\/p>/g, '$1')
  html = html.replace(/<p>(<pre>)/g, '$1')
  html = html.replace(/(<\/pre>)<\/p>/g, '$1')

  return html
}

module.exports = { parse: parseMarkdown }
