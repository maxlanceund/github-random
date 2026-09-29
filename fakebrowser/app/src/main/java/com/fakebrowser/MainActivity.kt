package com.fakebrowser

import android.app.Activity
import android.app.AlertDialog
import android.content.SharedPreferences
import android.os.Bundle
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.*
import org.json.JSONArray

class MainActivity : Activity() {
    private lateinit var urlBar: EditText
    private lateinit var webView: WebView
    private lateinit var prefs: SharedPreferences
    private var desktopUA = true
    private val desktopUAString = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    private val mobileUAString = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("fakebrowser", MODE_PRIVATE)
        desktopUA = prefs.getBoolean("desktop_ua", true)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        urlBar = EditText(this).apply {
            hint = "输入网址或搜索"
            imeOptions = EditorInfo.IME_ACTION_GO
            setSingleLine()
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_GO) {
                    handleInput(text.toString())
                    true
                } else false
            }
        }
        topBar.addView(urlBar, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        val menuBtn = Button(this).apply {
            text = "⋮"
            setOnClickListener { showMenu() }
        }
        topBar.addView(menuBtn, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        layout.addView(topBar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.userAgentString = if (desktopUA) desktopUAString else mobileUAString
            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                    val url = request.url.toString()
                    if (!url.startsWith("http://") && !url.startsWith("https://")) {
                        return true
                    }
                    return false
                }
            }
        }
        layout.addView(webView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
        ))

        setContentView(layout)
        loadUrl("https://www.bing.com")
    }

    private fun isUrl(input: String): Boolean {
        val s = input.trim()
        if (s.startsWith("http://") || s.startsWith("https://")) return true
        if (s.contains(".") && !s.contains(" ")) return true
        return false
    }

    private fun handleInput(input: String) {
        val s = input.trim()
        if (s.isEmpty()) return
        if (isUrl(s)) {
            var url = s
            if (!url.startsWith("http")) url = "https://$url"
            loadUrl(url)
        } else {
            val query = java.net.URLEncoder.encode(s, "UTF-8")
            loadUrl("https://www.bing.com/search?q=$query")
        }
    }

    private fun loadUrl(url: String) {
        urlBar.setText(url)
        webView.loadUrl(url)
    }

    private fun showMenu() {
        val options = arrayOf(
            "主页",
            "后退",
            "前进",
            "添加书签",
            "查看书签",
            if (desktopUA) "切换为手机UA" else "切换为桌面UA"
        )
        AlertDialog.Builder(this)
            .setTitle("菜单")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> loadUrl("https://www.bing.com")
                    1 -> if (webView.canGoBack()) webView.goBack()
                    2 -> if (webView.canGoForward()) webView.goForward()
                    3 -> addBookmark()
                    4 -> showBookmarks()
                    5 -> toggleUA()
                }
            }
            .show()
    }

    private fun addBookmark() {
        val title = webView.title ?: "无标题"
        val url = webView.url ?: return
        val arr = JSONArray(prefs.getString("bookmarks", "[]"))
        arr.put("$title|$url")
        prefs.edit().putString("bookmarks", arr.toString()).apply()
        Toast.makeText(this, "已收藏", Toast.LENGTH_SHORT).show()
    }

    private fun showBookmarks() {
        val arr = JSONArray(prefs.getString("bookmarks", "[]"))
        if (arr.length() == 0) {
            Toast.makeText(this, "无书签", Toast.LENGTH_SHORT).show()
            return
        }
        val items = Array(arr.length()) { i ->
            val parts = arr.getString(i).split("|", limit = 2)
            parts[0]
        }
        AlertDialog.Builder(this)
            .setTitle("书签")
            .setItems(items) { _, which ->
                val parts = arr.getString(which).split("|", limit = 2)
                loadUrl(parts[1])
            }
            .setNeutralButton("清空") { _, _ ->
                prefs.edit().putString("bookmarks", "[]").apply()
            }
            .show()
    }

    private fun toggleUA() {
        desktopUA = !desktopUA
        prefs.edit().putBoolean("desktop_ua", desktopUA).apply()
        webView.settings.userAgentString = if (desktopUA) desktopUAString else mobileUAString
        Toast.makeText(this, if (desktopUA) "已切桌面版" else "已切手机版", Toast.LENGTH_SHORT).show()
        webView.reload()
    }
}
