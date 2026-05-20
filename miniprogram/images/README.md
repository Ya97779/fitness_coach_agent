# 图片资源说明

## TabBar 图标（必须，81x81 像素 PNG）

| 文件名 | 说明 |
|--------|------|
| tab-home.png | 首页图标（未选中，灰色） |
| tab-home-active.png | 首页图标（选中，绿色 #4CAF50） |
| tab-chat.png | 聊天图标（未选中） |
| tab-chat-active.png | 聊天图标（选中） |
| tab-timer.png | 计时器图标（未选中） |
| tab-timer-active.png | 计时器图标（选中） |
| tab-guide.png | 动作指导图标（未选中） |
| tab-guide-active.png | 动作指导图标（选中） |
| tab-profile.png | 我的图标（未选中） |
| tab-profile-active.png | 我的图标（选中） |

## 其他资源

| 文件名 | 说明 |
|--------|------|
| default-avatar.png | 默认头像（120x120） |

## 动作演示图

动作演示图（GIF/JPG/PNG）存放在后端服务器 `backend/static/guide/` 目录，通过 `https://gzyhm.xyz/guide/` 访问。
数据文件中 `cover` 字段直接使用完整 URL，不打包进小程序代码包（超出 2MB 限制）。
