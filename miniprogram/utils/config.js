const PROD_API_BASE_URL = 'https://gzyapi.gzyhm.xyz'

function resolveApiBaseUrl() {
  try {
    const accountInfo = wx.getAccountInfoSync()
    const envVersion = accountInfo && accountInfo.miniProgram
      ? accountInfo.miniProgram.envVersion
      : 'release'
    const devOverride = wx.getStorageSync('DEV_API_BASE_URL')

    // 仅开发版允许覆盖，体验版和正式版始终使用生产地址。
    if (envVersion === 'develop' && typeof devOverride === 'string' && devOverride.trim()) {
      return devOverride.trim().replace(/\/+$/, '')
    }
  } catch (e) {
    // 获取运行环境失败时安全回退生产地址。
  }
  return PROD_API_BASE_URL
}

const API_BASE_URL = resolveApiBaseUrl()
const IMG_BASE_URL = `${API_BASE_URL}/guide`

module.exports = {
  API_BASE_URL,
  IMG_BASE_URL,
  PROD_API_BASE_URL
}
