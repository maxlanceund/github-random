package com.qs

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import androidx.viewpager2.widget.ViewPager2
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var viewPager: ViewPager2
    private lateinit var emptyView: TextView
    private lateinit var pickBtn: Button
    private lateinit var historyStore: HistoryStore
    private val prefs by lazy { getSharedPreferences("app", MODE_PRIVATE) }
    private val handler = Handler(Looper.getMainLooper())
    private val hideBtnRunnable = Runnable { pickBtn.visibility = View.GONE }

    private val pickFolder = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            prefs.edit().putString("folder", uri.toString()).apply()
            emptyView.visibility = View.GONE
            loadVideos(uri)
        } else {
            // 如果用户按了返回键取消了，且原本没选过文件夹，显示按钮让他重选
            if (prefs.getString("folder", null) == null) {
                emptyView.text = "请点击下方按钮选择视频文件夹"
                emptyView.visibility = View.VISIBLE
                pickBtn.visibility = View.VISIBLE
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewPager = findViewById(R.id.viewPager)
        emptyView = findViewById(R.id.emptyView)
        pickBtn = findViewById(R.id.pickBtn)
        historyStore = HistoryStore(this)

        viewPager.offscreenPageLimit = 1
        pickBtn.setOnClickListener { pickFolder.launch(null) }

        val savedUri = prefs.getString("folder", null)
        if (savedUri != null) {
            // 有记录，直接秒进，绝不显示按钮
            pickBtn.visibility = View.GONE
            emptyView.visibility = View.GONE
            loadVideos(Uri.parse(savedUri))
        } else {
            // 第一次打开，没有记录，直接自动呼出系统选择器！
            emptyView.text = "请选择包含视频的文件夹"
            emptyView.visibility = View.VISIBLE
            pickBtn.visibility = View.GONE
            pickFolder.launch(null)
        }

        viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                (viewPager.adapter as? VideoAdapter)?.setCurrent(position)
            }
        })
    }

    private fun loadVideos(treeUri: Uri) {
        Executors.newSingleThreadExecutor().execute {
            val root = DocumentFile.fromTreeUri(this, treeUri)
            val videos = mutableListOf<Uri>()
            if (root != null) collectVideos(root, videos)
            runOnUiThread {
                if (videos.isEmpty()) {
                    emptyView.text = "文件夹里没有视频\n请长按屏幕或点击按钮重新选择"
                    emptyView.visibility = View.VISIBLE
                    pickBtn.visibility = View.VISIBLE
                    viewPager.visibility = View.GONE
                } else {
                    emptyView.visibility = View.GONE
                    pickBtn.visibility = View.GONE
                    viewPager.visibility = View.VISIBLE
                    
                    viewPager.adapter = VideoAdapter(this, videos, historyStore) {
                        // 长按回调：弹出按钮3秒
                        pickBtn.visibility = View.VISIBLE
                        Toast.makeText(this, "点击下方按钮可更换文件夹", Toast.LENGTH_SHORT).show()
                        handler.removeCallbacks(hideBtnRunnable)
                        handler.postDelayed(hideBtnRunnable, 3000)
                    }
                }
            }
        }
    }

    private fun collectVideos(dir: DocumentFile, out: MutableList<Uri>) {
        val exts = listOf(".mp4", ".webm", ".mkv", ".mov", ".m4v")
        for (f in dir.listFiles()) {
            if (f.isDirectory) {
                collectVideos(f, out)
            } else if (f.isFile && exts.any { f.name?.lowercase()?.endsWith(it) == true }) {
                out.add(f.uri)
            }
        }
    }

    override fun onPause() {
        super.onPause()
        (viewPager.adapter as? VideoAdapter)?.saveCurrentPosition()
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(hideBtnRunnable)
        (viewPager.adapter as? VideoAdapter)?.releaseAll()
    }
}
