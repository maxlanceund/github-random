package com.qs

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView
import kotlin.math.abs

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    // 播放器池
    private val playerPool = SparseArray<ExoPlayer>()
    private var currentPos = 0

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val playerView: PlayerView = view.findViewById(R.id.playerView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_video, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val uri = videos[position]
        val player = getOrCreatePlayer(position, uri)

        holder.playerView.player = player

        // 预渲染：只要不是当前页，静音偷偷播第一帧
        if (position == currentPos) {
            player.volume = 1f
            player.play()
        } else {
            player.volume = 0f
            player.play()
        }

        holder.itemView.setOnClickListener {
            if (player.isPlaying) player.pause() else player.play()
        }

        holder.itemView.setOnLongClickListener {
            onLongPress()
            true
        }
    }

    private fun getOrCreatePlayer(position: Int, uri: Uri): ExoPlayer {
        playerPool.get(position)?.let { return it } // 池中有，直接复用

        val renderersFactory = DefaultRenderersFactory(context)
            .setEnableDecoderFallback(true)
            .setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER)

        val loadControl = DefaultLoadControl.Builder()
            .setBufferDurationsMs(30000, 120000, 200, 1000)
            .build()

        val player = ExoPlayer.Builder(context, renderersFactory)
            .setLoadControl(loadControl)
            .build()

        player.setMediaItem(MediaItem.fromUri(uri))
        player.repeatMode = Player.REPEAT_MODE_ONE
        player.prepare()

        val saved = historyStore.getPosition(uri.toString())
        if (saved > 0) player.seekTo(saved)

        player.addListener(object : Player.Listener {
            override fun onPlayerError(error: PlaybackException) {
                Log.e("QsPlayer", "播放失败: ${error.errorCodeName}", error)
            }
        })

        playerPool.put(position, player)
        return player
    }

    override fun onViewRecycled(holder: VH) {
        super.onViewRecycled(holder)
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        // 离开的页面：保存进度、暂停、静音
        playerPool.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
            it.volume = 0f
        }

        currentPos = position

        // 进入的页面：恢复声音、播放
        playerPool.get(position)?.let {
            it.volume = 1f
            it.play()
        }

        // 【保洁员逻辑】清理掉离当前位置超过 2 页的播放器，防止内存爆炸
        val keysToRemove = mutableListOf<Int>()
        for (i in 0 until playerPool.size()) {
            val key = playerPool.keyAt(i)
            if (abs(key - currentPos) > 2) {
                playerPool.valueAt(i).release()
                keysToRemove.add(key)
            }
        }
        keysToRemove.forEach { playerPool.remove(it) }
    }

    fun saveCurrentPosition() {
        playerPool.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until playerPool.size()) {
            val p = playerPool.valueAt(i)
            historyStore.savePosition(videos[playerPool.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        playerPool.clear()
    }
}
