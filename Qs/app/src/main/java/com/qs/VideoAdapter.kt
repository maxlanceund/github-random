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

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    // 播放器池：Key 是位置，Value 是播放器实例
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

        if (position == currentPos) {
            player.volume = 1f
            player.play()
        } else {
            // 预渲染：静音播放第一帧，让画面提前就绪
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
        // 注意：这里不释放播放器，因为池中可能还需要它
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
